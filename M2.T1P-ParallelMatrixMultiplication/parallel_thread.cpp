// ============================================================================
// FILE        : parallel_thread.cpp
// TASK        : M2.T1P — Parallel Matrix Multiplication (std::thread)
// UNIT        : SIT315 Programming Paradigms, Trimester 4, 2024-25
// AUTHOR      : Jahan Garg | Roll: 2410994805
// DESCRIPTION : Parallelises C = A x B using C++11 std::thread by dividing
//               the output matrix C into contiguous row bands, one per thread
//               (output data decomposition). Each thread computes its
//               assigned rows independently — no mutex required because
//               threads write to disjoint memory regions. A join() barrier
//               ensures all threads complete before timing stops and results
//               are consumed.
// COMPILE     : g++ -std=c++11 -O2 parallel_thread.cpp -o parallel_thread
// RUN         : ./parallel_thread <N> <threads>    e.g.  ./parallel_thread 512 4
// OUTPUT      : C_thread.txt  (row-major result matrix, one row per line)
// ============================================================================

// ── Standard library headers ─────────────────────────────────────────────────
#include <iostream>   // std::cout — console output for results
#include <vector>     // std::vector — heap-allocated matrices and thread pool
#include <thread>     // std::thread — C++11 portable threading
#include <chrono>     // high_resolution_clock — microsecond-precision timing
#include <cstdlib>    // rand(), srand(), atoi() — RNG and argument parsing
#include <fstream>    // std::ofstream — writing result matrix to file

using namespace std;
using namespace std::chrono;

// ── Function: fillMatrixWithRandomValues ────────────────────────────────────
// Fills every element of the flat row-major matrix vector with a random
// integer in [0, 99]. Called once before threading starts using the shared
// srand(0) seed. rand() is NOT thread-safe, so this must remain sequential.
// Parameters:
//   matrix     — reference to the flat vector to fill
//   matrixSize — N, the side length of the square matrix
void fillMatrixWithRandomValues(vector<int> &matrix, int matrixSize)
{
    for (int i = 0; i < matrixSize * matrixSize; i++)
    {
        matrix[i] = rand() % 100;   // value in [0, 99]
    }
}

// ── Function: multiplyMatrixRows ────────────────────────────────────────────
// THREAD WORKER — computes a contiguous band of rows [startRow, endRow) of
// the output matrix C = A × B. Each thread receives its own non-overlapping
// row range, so no two threads ever write to the same element of C.
// This eliminates any data race on C without requiring a mutex or lock.
//
// Why row partitioning?
//   - Each row of C is fully determined by one row of A and all of B.
//   - Rows are independent — computing C[i][*] does not affect C[i'][*].
//   - Contiguous row bands maximise cache locality for accesses to matrixA.
//
// Parameters:
//   matrixA    — const ref to input matrix A (read-only, shared across threads)
//   matrixB    — const ref to input matrix B (read-only, shared across threads)
//   matrixC    — ref to output matrix C (each thread writes a disjoint band)
//   matrixSize — N (full matrix dimension)
//   startRow   — first row this thread is responsible for (inclusive)
//   endRow     — one past the last row this thread handles (exclusive)
void multiplyMatrixRows(const vector<int> &matrixA,
                        const vector<int> &matrixB,
                        vector<int>       &matrixC,
                        int                matrixSize,
                        int                startRow,
                        int                endRow)
{
    // Iterate only over the rows assigned to this thread
    for (int i = startRow; i < endRow; i++)
    {
        // For each column j in the assigned row i
        for (int j = 0; j < matrixSize; j++)
        {
            int cellSum = 0;   // accumulator for dot product A[i][*] · B[*][j]

            // Inner k-loop: dot product of row i of A with column j of B
            for (int k = 0; k < matrixSize; k++)
            {
                // A[i][k] = matrixA[i*N + k]  (row-major)
                // B[k][j] = matrixB[k*N + j]  (column access — sequential k)
                cellSum += matrixA[i * matrixSize + k] * matrixB[k * matrixSize + j];
            }

            // Write dot product result — this element is unique to this thread
            matrixC[i * matrixSize + j] = cellSum;
        }
    }
    // Thread function returns here — join() in main will detect completion
}

// ── Function: calculateChecksum ─────────────────────────────────────────────
// Sums all elements of result matrix C as a 64-bit integer.
// Used post-join to verify that parallel output matches the sequential result.
// Matching checksums across all three programs prove correctness.
long long calculateChecksum(const vector<int> &matrixC)
{
    long long totalChecksum = 0;
    for (int val : matrixC)
        totalChecksum += val;
    return totalChecksum;
}

// ── main ─────────────────────────────────────────────────────────────────────
int main(int argc, char *argv[])
{
    // ── Argument validation ──────────────────────────────────────────────────
    // Require N (matrix size) and T (thread count) as command-line arguments
    if (argc < 3)
    {
        cout << "Usage: ./parallel_thread N threads\n";
        cout << "  N       — square matrix dimension (1 to 2000)\n";
        cout << "  threads — number of parallel threads (1 to 8)\n";
        return 1;
    }

    int matrixSize = atoi(argv[1]);
    int numThreads = atoi(argv[2]);

    if (matrixSize <= 0 || matrixSize > 2000 || numThreads <= 0 || numThreads > 8)
    {
        cout << "Error: N must be 1-2000, threads must be 1-8\n";
        return 1;
    }

    // ── Matrix allocation ────────────────────────────────────────────────────
    // Flat 1-D vectors in row-major order. matrixC is pre-zeroed.
    vector<int> matrixA(matrixSize * matrixSize);
    vector<int> matrixB(matrixSize * matrixSize);
    vector<int> matrixC(matrixSize * matrixSize, 0);

    // ── Matrix initialisation ────────────────────────────────────────────────
    // Fixed seed srand(0) ensures identical A and B as sequential program
    srand(0);
    fillMatrixWithRandomValues(matrixA, matrixSize);
    fillMatrixWithRandomValues(matrixB, matrixSize);

    // ── Thread pool preparation ──────────────────────────────────────────────
    // Pre-allocate vector capacity to avoid reallocation during emplace_back
    vector<thread> workers;
    workers.reserve(numThreads);

    // Compute base rows per thread and the remainder (for uneven division)
    // e.g., N=512, T=3: rowsPerThread=170, extraRows=2
    //   Thread 0 gets 171 rows, thread 1 gets 171 rows, thread 2 gets 170
    int rowsPerThread = matrixSize / numThreads;
    int extraRows     = matrixSize % numThreads;   // first extraRows threads get +1 row
    int startRow      = 0;

    // ── Timed parallel computation ───────────────────────────────────────────
    // Timer starts before thread creation so that thread spawning cost is
    // included in the parallel execution time measurement
    auto start = high_resolution_clock::now();

    // Spawn T threads, each assigned a contiguous band of rows
    for (int t = 0; t < numThreads; t++)
    {
        // First extraRows threads get one extra row to handle remainder evenly
        int rows   = rowsPerThread + (t < extraRows ? 1 : 0);
        int endRow = startRow + rows;

        // Spawn thread: pass A and B as const references (read-only, zero copy)
        // Pass C as mutable reference (each thread writes its own disjoint band)
        workers.emplace_back(multiplyMatrixRows,
                             cref(matrixA), cref(matrixB), ref(matrixC),
                             matrixSize, startRow, endRow);

        startRow = endRow;   // advance start for next thread's band
    }

    // ── Join barrier ─────────────────────────────────────────────────────────
    // Block main thread until every worker finishes its row band.
    // join() is the synchronisation point — matrixC is only safe to read
    // after all joins complete.
    for (auto &th : workers)
        th.join();

    // Timer stops after all threads have joined — full wall-clock parallel time
    auto end = high_resolution_clock::now();

    long long executionTimeMicroseconds =
        duration_cast<microseconds>(end - start).count();

    // ── Write result matrix to file ──────────────────────────────────────────
    // Row-major format (one row per line) — diff with C_seq.txt for full verify
    ofstream outputFile("C_thread.txt");
    if (!outputFile)
    {
        cout << "Error: Could not create C_thread.txt\n";
        return 1;
    }
    for (int i = 0; i < matrixSize; i++)
    {
        for (int j = 0; j < matrixSize; j++)
        {
            outputFile << matrixC[i * matrixSize + j] << " ";
        }
        outputFile << "\n";
    }
    outputFile.close();

    // ── Checksum and results ─────────────────────────────────────────────────
    long long resultChecksum = calculateChecksum(matrixC);

    cout << "N               = " << matrixSize << "\n";
    cout << "Threads         = " << numThreads << "\n";
    cout << "Method          = std::thread\n";
    cout << "Execution time  = " << executionTimeMicroseconds
         << " us (" << executionTimeMicroseconds / 1000.0 << " ms)\n";
    cout << "Checksum        = " << resultChecksum << "\n";
    cout << "Output          = C_thread.txt\n";

    return 0;
}
