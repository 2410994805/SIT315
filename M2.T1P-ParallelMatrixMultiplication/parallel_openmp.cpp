// ============================================================================
// FILE        : parallel_openmp.cpp
// TASK        : M2.T1P — Parallel Matrix Multiplication (OpenMP)
// UNIT        : SIT315 Programming Paradigms, Trimester 4, 2024-25
// AUTHOR      : Jahan Garg | Roll: 2410994805
// DESCRIPTION : Parallelises C = A x B using OpenMP's #pragma omp parallel
//               for with collapse(2) and static scheduling. Unlike std::thread,
//               OpenMP uses a pre-initialised thread pool (no creation cost
//               per call) and exposes N^2 independent tasks (via collapse)
//               rather than N row-level tasks, yielding finer load balance.
//               No race condition exists: each (i,j) pair writes exclusively
//               to C[i*N+j] — a unique memory location.
// COMPILE     : g++ -std=c++11 -O2 -fopenmp parallel_openmp.cpp -o parallel_openmp
// RUN         : ./parallel_openmp <N> <threads>   e.g.  ./parallel_openmp 512 4
// OUTPUT      : C_openmp.txt  (row-major result matrix, one row per line)
// ============================================================================

// ── Standard library headers ─────────────────────────────────────────────────
#include <iostream>   // std::cout — console output for results
#include <vector>     // std::vector — heap-allocated matrices
#include <chrono>     // high_resolution_clock — microsecond-precision timing
#include <cstdlib>    // rand(), srand(), atoi() — RNG and argument parsing
#include <fstream>    // std::ofstream — writing result matrix to file
#include <omp.h>      // OpenMP API — omp_set_num_threads(), #pragma omp directives

using namespace std;
using namespace std::chrono;

// ── Function: fillMatrixWithRandomValues ────────────────────────────────────
// Fills every element of the flat row-major matrix vector with a random
// integer in [0, 99]. Called sequentially before the parallel region because
// rand() is NOT thread-safe (shared internal state). Fixed seed srand(0)
// in main guarantees identical matrices A and B across all three programs.
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

// ── Function: calculateChecksum ─────────────────────────────────────────────
// Sums all elements of result matrix C as a 64-bit integer.
// Computed after the OpenMP parallel region to verify correctness against
// the sequential and std::thread implementations.
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
    if (argc < 3)
    {
        cout << "Usage: ./parallel_openmp N threads\n";
        cout << "  N       — square matrix dimension (1 to 2000)\n";
        cout << "  threads — number of OpenMP threads (1 to 8)\n";
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
    // Flat 1-D vectors in row-major order. matrixC pre-zeroed.
    vector<int> matrixA(matrixSize * matrixSize);
    vector<int> matrixB(matrixSize * matrixSize);
    vector<int> matrixC(matrixSize * matrixSize, 0);

    // ── Matrix initialisation ────────────────────────────────────────────────
    // Sequential: rand() is not thread-safe; srand(0) ensures reproducibility
    srand(0);
    fillMatrixWithRandomValues(matrixA, matrixSize);
    fillMatrixWithRandomValues(matrixB, matrixSize);

    // ── Set OpenMP thread count ──────────────────────────────────────────────
    // Must be called before the parallel region. Overrides the default (which
    // would use OMP_NUM_THREADS environment variable or hardware concurrency).
    omp_set_num_threads(numThreads);

    // ── Timed parallel computation ───────────────────────────────────────────
    // Timer starts before the pragma so that OpenMP region entry overhead is
    // included in the measurement (consistent with std::thread timing)
    auto start = high_resolution_clock::now();

    // ── OpenMP parallel region ───────────────────────────────────────────────
    // parallel for — distributes loop iterations across numThreads threads
    //                from the pre-initialised OpenMP thread pool (no creation
    //                cost, unlike std::thread which constructs threads here)
    //
    // collapse(2) — fuses the outer i-loop (N iterations) and middle j-loop
    //               (N iterations) into a single loop of N^2 iterations.
    //               Each of the N^2 (i,j) pairs is an independent task.
    //               At N=512, this exposes 262,144 tasks vs. 512 with row-only
    //               partitioning — finer granularity improves load balance.
    //
    // schedule(static) — divides the N^2 tasks into equal-sized contiguous
    //                    chunks, assigned to threads at compile/entry time.
    //                    Minimal scheduling overhead; optimal for uniform work.
    //
    // No race condition: C[i*N+j] is written by exactly one (i,j) combination.
    // cellSum is a private variable — each thread iteration has its own copy.
    // matrixA and matrixB are read-only shared data — no write conflict.
#pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < matrixSize; i++)
    {
        for (int j = 0; j < matrixSize; j++)
        {
            int cellSum = 0;   // private accumulator — unique per (i,j) iteration

            // Inner k-loop: dot product of row i of A and column j of B
            // Kept sequential: parallelising k would require atomic reduction,
            // adding synchronisation overhead that outweighs any benefit here
            for (int k = 0; k < matrixSize; k++)
            {
                // A[i][k] = matrixA[i*N + k]  (row-major, cache-friendly)
                // B[k][j] = matrixB[k*N + j]  (column access across k)
                cellSum += matrixA[i * matrixSize + k] * matrixB[k * matrixSize + j];
            }

            // Write result: this element is exclusively owned by this (i,j) task
            matrixC[i * matrixSize + j] = cellSum;
        }
    }
    // Implicit barrier at end of parallel for — all threads complete before
    // execution continues past this point. Timer stops after this barrier.

    auto end = high_resolution_clock::now();
    long long executionTimeMicroseconds =
        duration_cast<microseconds>(end - start).count();

    // ── Write result matrix to file ──────────────────────────────────────────
    // Row-major format (one row per line) — diff with C_seq.txt for full verify
    ofstream outputFile("C_openmp.txt");
    if (!outputFile)
    {
        cout << "Error: Could not create C_openmp.txt\n";
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
    cout << "Method          = OpenMP\n";
    cout << "Execution time  = " << executionTimeMicroseconds
         << " us (" << executionTimeMicroseconds / 1000.0 << " ms)\n";
    cout << "Checksum        = " << resultChecksum << "\n";
    cout << "Output          = C_openmp.txt\n";

    return 0;
}
