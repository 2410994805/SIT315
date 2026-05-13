// FILE: parallel_openmp.cpp
// TASK: M2.T1P - Parallel Matrix Multiplication using OpenMP
// UNIT: SIT315 Programming Paradigms, Trimester 4, 2024-25
// AUTHOR: Jahan Garg | Roll: 2410994805
//
// Parallelises C = A x B using OpenMP's #pragma omp parallel for
// with collapse(2) and static scheduling.
//
// Unlike std::thread, OpenMP uses a pre-initialised thread pool so there is
// no per-call thread creation cost. collapse(2) fuses the outer i and j loops
// into N^2 independent tasks rather than N row-level tasks, giving the
// scheduler finer granularity and better load balance.
//
// No race condition exists: each (i,j) pair writes only to C[i*N+j],
// a unique memory location not shared with any other iteration.
//
// Compile: g++ -std=c++11 -O2 -fopenmp parallel_openmp.cpp -o parallel_openmp
// Run:     ./parallel_openmp <N> <threads>    e.g. ./parallel_openmp 512 4
// Output:  C_openmp.txt (result matrix, one row per line)

#include <iostream>   // std::cout
#include <vector>     // std::vector
#include <chrono>     // high_resolution_clock
#include <cstdlib>    // rand(), srand(), atoi()
#include <fstream>    // std::ofstream
#include <omp.h>      // omp_set_num_threads() and OpenMP directives

using namespace std;
using namespace std::chrono;

// Fills every element of a flat NxN matrix vector with a random int in [0, 99].
// rand() is NOT thread-safe, so this must run sequentially before the OpenMP
// parallel region. srand(0) in main ensures the same matrices are generated
// every run, matching sequential.cpp and parallel_thread.cpp for comparison.
void fillMatrixWithRandomValues(vector<int> &matrix, int matrixSize)
{
    for (int i = 0; i < matrixSize * matrixSize; i++)
    {
        matrix[i] = rand() % 100;
    }
}

// Sums all elements of C into a 64-bit integer for correctness verification.
// Called after the parallel region completes. A matching checksum with
// sequential.cpp confirms the parallel result is numerically identical.
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
        cout << "Usage: ./parallel_openmp N threads\n";
        cout << "  N       - square matrix dimension (1 to 2000)\n";
        cout << "  threads - number of OpenMP threads (1 to 8)\n";
        return 1;
    }

    int matrixSize = atoi(argv[1]);
    int numThreads = atoi(argv[2]);

    if (matrixSize <= 0 || matrixSize > 2000 || numThreads <= 0 || numThreads > 8)
    {
        cout << "Error: N must be 1-2000, threads must be 1-8\n";
        return 1;
    }

    // Allocate matrices as flat vectors in row-major order.
    // matrixC is zero-initialised before the parallel region writes to it.
    vector<int> matrixA(matrixSize * matrixSize);
    vector<int> matrixB(matrixSize * matrixSize);
    vector<int> matrixC(matrixSize * matrixSize, 0);

    // Fill A and B sequentially. rand() is not thread-safe so this must
    // complete before entering the parallel region.
    srand(0);
    fillMatrixWithRandomValues(matrixA, matrixSize);
    fillMatrixWithRandomValues(matrixB, matrixSize);

    // Set the thread count explicitly before the parallel region.
    // Without this call, OpenMP would use OMP_NUM_THREADS or the hardware
    // concurrency default, which may not match the command-line argument.
    omp_set_num_threads(numThreads);

    // Timer starts before the pragma so that OpenMP region entry cost
    // (thread pool wake-up, work distribution) is included in the measurement.
    // This keeps timing consistent with how parallel_thread.cpp is measured.
    auto startTime = high_resolution_clock::now();

    // Parallelise the outer two loops over the N^2 output elements.
    //
    // parallel for: distributes loop iterations across the OpenMP thread pool.
    //   The pool is pre-initialised, so threads are reused across calls with
    //   no creation overhead (unlike std::thread which constructs new threads).
    //
    // collapse(2): fuses the i-loop (N iters) and j-loop (N iters) into one
    //   combined loop of N^2 iterations. At N=512 this exposes 262,144
    //   independent tasks instead of 512 row-level tasks, giving better
    //   load balance across threads especially when N % numThreads != 0.
    //
    // schedule(static): divides the N^2 tasks into equal contiguous chunks
    //   assigned to threads before execution starts. Minimal overhead, and
    //   correct because every (i,j) task does the same amount of work.
    //
    // cellSum is private by default (declared inside the loop body),
    //   so each iteration has its own accumulator with no sharing needed.
    // matrixA and matrixB are read-only shared data with no write conflicts.
    // matrixC writes are conflict-free: each (i,j) maps to a unique index.
#pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < matrixSize; i++)
    {
        for (int j = 0; j < matrixSize; j++)
        {
            int cellSum = 0;

            // Dot product of row i from A and column j from B.
            // The inner k-loop is kept sequential: parallelising it would
            // require an atomic reduction on cellSum, adding overhead that
            // exceeds any gain for the matrix sizes used here.
            for (int k = 0; k < matrixSize; k++)
            {
                cellSum += matrixA[i * matrixSize + k] * matrixB[k * matrixSize + j];
            }

            matrixC[i * matrixSize + j] = cellSum;
        }
    }
    // The implicit barrier at the end of "parallel for" ensures all threads
    // have finished writing before execution proceeds past this point.

    auto endTime = high_resolution_clock::now();
    long long elapsedMicroseconds =
        duration_cast<microseconds>(endTime - startTime).count();

    // Write result matrix to file for diff-based verification against C_seq.txt
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

    long long checksum = calculateChecksum(matrixC);

    cout << "N               = " << matrixSize << "\n";
    cout << "Threads         = " << numThreads << "\n";
    cout << "Method          = OpenMP\n";
    cout << "Execution time  = " << elapsedMicroseconds
         << " us (" << elapsedMicroseconds / 1000.0 << " ms)\n";
    cout << "Checksum        = " << checksum << "\n";
    cout << "Output          = C_openmp.txt\n";

    return 0;
}
