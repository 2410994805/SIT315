// FILE: parallel_thread.cpp
// TASK: M2.T1P - Parallel Matrix Multiplication using std::thread
// UNIT: SIT315 Programming Paradigms, Trimester 4, 2024-25
// AUTHOR: Jahan Garg | Roll: 2410994805
//
// Parallelises C = A x B using C++11 std::thread.
// The output matrix C is divided into contiguous row bands, one per thread
// (output data decomposition). Each thread computes its assigned rows
// independently with no mutex needed because threads write to disjoint
// memory regions. A join() barrier waits for all threads before timing stops.
//
// Compile: g++ -std=c++11 -O2 parallel_thread.cpp -o parallel_thread
// Run:     ./parallel_thread <N> <threads>    e.g. ./parallel_thread 512 4
// Output:  C_thread.txt (result matrix, one row per line)

#include <iostream>   // std::cout
#include <vector>     // std::vector
#include <thread>     // std::thread
#include <chrono>     // high_resolution_clock
#include <cstdlib>    // rand(), srand(), atoi()
#include <fstream>    // std::ofstream

using namespace std;
using namespace std::chrono;

// Fills every element of a flat NxN matrix vector with a random int in [0, 99].
// rand() is NOT thread-safe (it uses shared internal state), so this function
// must be called sequentially before any threads are spawned.
// srand(0) in main ensures the same matrices are produced every run.
void fillMatrixWithRandomValues(vector<int> &matrix, int matrixSize)
{
    for (int i = 0; i < matrixSize * matrixSize; i++)
    {
        matrix[i] = rand() % 100;
    }
}

// Thread worker function. Computes rows [startRow, endRow) of output matrix C.
// Each thread gets a non-overlapping row range, so no two threads write to
// the same element of C. This means no mutex or atomic operations are needed.
//
// Why row partitioning?
//   Each row of C depends only on one row of A and the entire read-only matrix B.
//   Rows of C are fully independent of each other, so splitting by row
//   achieves zero inter-thread data dependency with no synchronisation overhead.
//
// matrixA and matrixB are passed as const references (read-only, shared safely).
// matrixC is passed as a mutable reference; each thread writes only its band.
void multiplyMatrixRows(const vector<int> &matrixA,
                        const vector<int> &matrixB,
                        vector<int>       &matrixC,
                        int                matrixSize,
                        int                startRow,
                        int                endRow)
{
    for (int i = startRow; i < endRow; i++)
    {
        for (int j = 0; j < matrixSize; j++)
        {
            int cellSum = 0;

            // Dot product of row i from A and column j from B
            for (int k = 0; k < matrixSize; k++)
            {
                cellSum += matrixA[i * matrixSize + k] * matrixB[k * matrixSize + j];
            }

            // This element belongs exclusively to this thread's row band
            matrixC[i * matrixSize + j] = cellSum;
        }
    }
}

// Sums all elements of C into a 64-bit integer for correctness verification.
// A matching checksum between this program and sequential.cpp confirms that
// the parallel decomposition produced the correct result.
long long calculateChecksum(const vector<int> &matrixC)
{
    long long total = 0;
    for (int val : matrixC)
        total += val;
    return total;
}

int main(int argc, char *argv[])
{
    // Validate command-line arguments
    if (argc < 3)
    {
        cout << "Usage: ./parallel_thread N threads\n";
        cout << "  N       - square matrix dimension (1 to 2000)\n";
        cout << "  threads - number of parallel threads (1 to 8)\n";
        return 1;
    }

    int matrixSize = atoi(argv[1]);
    int numThreads = atoi(argv[2]);

    if (matrixSize <= 0 || matrixSize > 2000 || numThreads <= 0 || numThreads > 8)
    {
        cout << "Error: N must be 1-2000, threads must be 1-8\n";
        return 1;
    }

    // Allocate matrices. matrixC is zero-initialised before workers write to it.
    vector<int> matrixA(matrixSize * matrixSize);
    vector<int> matrixB(matrixSize * matrixSize);
    vector<int> matrixC(matrixSize * matrixSize, 0);

    // Seed and fill A and B sequentially. rand() is not thread-safe so this
    // must complete before any thread is spawned. srand(0) matches the seed
    // used in sequential.cpp so checksums can be compared directly.
    srand(0);
    fillMatrixWithRandomValues(matrixA, matrixSize);
    fillMatrixWithRandomValues(matrixB, matrixSize);

    // Pre-allocate the thread vector to avoid reallocation inside the spawn loop
    vector<thread> workers;
    workers.reserve(numThreads);

    // Divide N rows into T bands. If N is not evenly divisible by T,
    // the remainder rows (N % T) are distributed one extra row per thread
    // to the first (N % T) threads, ensuring all rows are always covered.
    // Example: N=512, T=3 -> rowsPerThread=170, extraRows=2
    //          Thread 0 gets 171 rows, thread 1 gets 171, thread 2 gets 170.
    int rowsPerThread = matrixSize / numThreads;
    int extraRows     = matrixSize % numThreads;
    int startRow      = 0;

    // Timer starts before thread creation so spawning cost is included.
    // This gives a fair wall-clock parallel time consistent with OpenMP timing.
    auto startTime = high_resolution_clock::now();

    for (int t = 0; t < numThreads; t++)
    {
        int rows   = rowsPerThread + (t < extraRows ? 1 : 0);
        int endRow = startRow + rows;

        // cref wraps A and B as const references (no copy, safe to share).
        // ref wraps C as a mutable reference (each thread's band is disjoint).
        workers.emplace_back(multiplyMatrixRows,
                             cref(matrixA), cref(matrixB), ref(matrixC),
                             matrixSize, startRow, endRow);

        startRow = endRow;
    }

    // Wait for every thread to finish. join() is the synchronisation barrier.
    // matrixC is only safe to read after all joins complete.
    for (auto &th : workers)
        th.join();

    auto endTime = high_resolution_clock::now();
    long long elapsedMicroseconds =
        duration_cast<microseconds>(endTime - startTime).count();

    // Write result matrix to file for diff-based verification against C_seq.txt
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

    long long checksum = calculateChecksum(matrixC);

    cout << "N               = " << matrixSize << "\n";
    cout << "Threads         = " << numThreads << "\n";
    cout << "Method          = std::thread\n";
    cout << "Execution time  = " << elapsedMicroseconds
         << " us (" << elapsedMicroseconds / 1000.0 << " ms)\n";
    cout << "Checksum        = " << checksum << "\n";
    cout << "Output          = C_thread.txt\n";

    return 0;
}
