// ============================================================================
// FILE        : sequential.cpp
// TASK        : M2.T1P — Sequential Matrix Multiplication Baseline
// UNIT        : SIT315 Programming Paradigms, Trimester 4, 2024-25
// AUTHOR      : Jahan Garg | Roll: 2410994805
// DESCRIPTION : Computes C = A x B for two N×N integer matrices using a
//               classic triple-nested loop (O(N^3)). Serves as the sequential
//               baseline against which all parallel implementations are
//               compared. Execution time is measured via std::chrono and
//               result correctness is verified by a checksum.
// COMPILE     : g++ -std=c++11 -O2 sequential.cpp -o sequential
// RUN         : ./sequential <N>          e.g.  ./sequential 512
// OUTPUT      : C_seq.txt  (row-major result matrix, one row per line)
// ============================================================================

// ── Standard library headers ─────────────────────────────────────────────────
#include <iostream>   // std::cout — console output for results
#include <vector>     // std::vector — heap-allocated, dynamic-size arrays
#include <chrono>     // high_resolution_clock — microsecond-precision timing
#include <cstdlib>    // rand(), srand(), atoi() — RNG and argument parsing
#include <fstream>    // std::ofstream — writing result matrix to file

using namespace std;
using namespace std::chrono;

// ── Function: fillMatrixWithRandomValues ────────────────────────────────────
// Fills every element of a flat 1-D vector (representing an N×N matrix stored
// in row-major order) with a pseudo-random integer in the range [0, 99].
// Using the same seed (srand(0) in main) guarantees that all three programs
// (sequential, thread, OpenMP) produce identical matrices A and B, enabling
// direct checksum comparison for correctness verification.
// Parameters:
//   matrix     — reference to the flat vector to fill
//   matrixSize — N, the side length of the square matrix
void fillMatrixWithRandomValues(vector<int> &matrix, int matrixSize)
{
    // Iterate over all N*N elements in linear order and assign random values
    for (int i = 0; i < matrixSize * matrixSize; i++)
    {
        matrix[i] = rand() % 100;   // value in [0, 99]
    }
}

// ── Function: multiplyMatrices ──────────────────────────────────────────────
// Performs sequential matrix multiplication C = A × B using three nested
// loops (i, j, k). Time complexity: O(N^3). Space complexity: O(1) auxiliary.
//
// Memory layout: all matrices are stored as flat 1-D vectors in row-major
// order. Element at row i, column j is accessed as matrix[i * N + j].
// This layout ensures that the inner k-loop accesses matrixA in row-major
// order (sequential/cache-friendly), although matrixB is accessed column-
// major (potential cache miss). For this task the focus is parallelism
// correctness, not cache-oblivious optimisation.
//
// Parameters:
//   matrixA, matrixB — read-only input matrices (const reference)
//   matrixC          — output matrix (reference, pre-zeroed in main)
//   matrixSize       — N
void multiplyMatrices(const vector<int> &matrixA,
                      const vector<int> &matrixB,
                      vector<int>       &matrixC,
                      int                matrixSize)
{
    // Outer loop — iterates over each row i of output matrix C
    for (int i = 0; i < matrixSize; i++)
    {
        // Middle loop — iterates over each column j of output matrix C
        for (int j = 0; j < matrixSize; j++)
        {
            int cellSum = 0;   // accumulator for dot product of row i and col j

            // Inner loop — computes dot product: sum of A[i][k] * B[k][j]
            for (int k = 0; k < matrixSize; k++)
            {
                // Row-major index: A[i][k] = matrixA[i*N + k]
                //                  B[k][j] = matrixB[k*N + j]
                cellSum += matrixA[i * matrixSize + k] * matrixB[k * matrixSize + j];
            }

            // Store completed dot product in row-major position of C
            matrixC[i * matrixSize + j] = cellSum;
        }
    }
}

// ── Function: calculateChecksum ─────────────────────────────────────────────
// Computes a simple sum-of-all-elements checksum over the result matrix C.
// Using long long avoids integer overflow for large N (e.g., N=512 with
// values up to ~99*99*512 ≈ 5M per element, times 512*512 elements).
// The same fixed seed srand(0) guarantees the same checksum across all
// three implementations if their multiplication logic is correct.
// Parameters:
//   matrixC — const reference to the result matrix
// Returns:
//   64-bit integer checksum
long long calculateChecksum(const vector<int> &matrixC)
{
    long long totalChecksum = 0;
    for (int val : matrixC)
        totalChecksum += val;   // accumulate each element
    return totalChecksum;
}

// ── main ─────────────────────────────────────────────────────────────────────
int main(int argc, char *argv[])
{
    // ── Argument validation ──────────────────────────────────────────────────
    // Require exactly one command-line argument: matrix dimension N
    if (argc < 2)
    {
        cout << "Usage: ./sequential N\n";
        cout << "  N — square matrix dimension (1 to 2000)\n";
        return 1;
    }

    // Parse N from string argument; atoi returns 0 for non-numeric input
    int matrixSize = atoi(argv[1]);
    if (matrixSize <= 0 || matrixSize > 2000)
    {
        cout << "Error: N must be between 1 and 2000\n";
        return 1;
    }

    // ── Matrix allocation ────────────────────────────────────────────────────
    // Allocate three flat vectors: A (input), B (input), C (output, zeroed)
    // Size = N*N integers. Row-major layout: element [i][j] at index i*N+j
    vector<int> matrixA(matrixSize * matrixSize);
    vector<int> matrixB(matrixSize * matrixSize);
    vector<int> matrixC(matrixSize * matrixSize, 0);   // initialised to zero

    // ── Matrix initialisation ────────────────────────────────────────────────
    // Fixed seed ensures reproducibility and cross-program checksum matching
    srand(0);
    fillMatrixWithRandomValues(matrixA, matrixSize);
    fillMatrixWithRandomValues(matrixB, matrixSize);

    // ── Timed computation ────────────────────────────────────────────────────
    // Timer starts immediately before multiplication and stops immediately
    // after. Excludes matrix fill and file write as per task requirements.
    auto start = high_resolution_clock::now();
    multiplyMatrices(matrixA, matrixB, matrixC, matrixSize);
    auto end = high_resolution_clock::now();

    // Convert elapsed time from nanoseconds to microseconds
    long long executionTimeMicroseconds =
        duration_cast<microseconds>(end - start).count();

    // ── Write result matrix to file ──────────────────────────────────────────
    // Outputs C in row-major format (one row per line) for diff-based
    // correctness comparison against parallel implementations
    ofstream outputFile("C_seq.txt");
    if (!outputFile)
    {
        cout << "Error: Could not create C_seq.txt\n";
        return 1;
    }
    for (int i = 0; i < matrixSize; i++)
    {
        for (int j = 0; j < matrixSize; j++)
        {
            outputFile << matrixC[i * matrixSize + j] << " ";
        }
        outputFile << "\n";   // newline at end of each row
    }
    outputFile.close();

    // ── Checksum and results ─────────────────────────────────────────────────
    long long resultChecksum = calculateChecksum(matrixC);

    cout << "N               = " << matrixSize << "\n";
    cout << "Method          = Sequential\n";
    cout << "Execution time  = " << executionTimeMicroseconds
         << " us (" << executionTimeMicroseconds / 1000.0 << " ms)\n";
    cout << "Checksum        = " << resultChecksum << "\n";
    cout << "Output file     = C_seq.txt\n";

    return 0;
}
