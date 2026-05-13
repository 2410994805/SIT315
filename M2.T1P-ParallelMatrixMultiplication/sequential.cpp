// FILE: sequential.cpp
// TASK: M2.T1P - Sequential Matrix Multiplication Baseline
// UNIT: SIT315 Programming Paradigms, Trimester 4, 2024-25
// AUTHOR: Jahan Garg | Roll: 2410994805
//
// Computes C = A x B for two NxN integer matrices using a classic
// triple-nested loop (O(N^3)). This is the sequential baseline used
// to measure speedup for parallel implementations.
//
// Execution time is measured with std::chrono (microseconds) and
// wraps only the multiplication, not matrix fill or file write.
// A checksum of all elements in C is printed for correctness checks.
//
// Compile: g++ -std=c++11 -O2 sequential.cpp -o sequential
// Run:     ./sequential <N>       e.g. ./sequential 512
// Output:  C_seq.txt (result matrix, one row per line)

#include <iostream>   // std::cout
#include <vector>     // std::vector
#include <chrono>     // high_resolution_clock
#include <cstdlib>    // rand(), srand(), atoi()
#include <fstream>    // std::ofstream

using namespace std;
using namespace std::chrono;

// Fills every element of a flat NxN matrix vector with a random int in [0, 99].
// The matrix is stored in row-major order: element [i][j] lives at index i*N+j.
// srand(0) is called in main before this function so the same values are
// produced every run, which makes checksums comparable across all three programs.
void fillMatrixWithRandomValues(vector<int> &matrix, int matrixSize)
{
    for (int i = 0; i < matrixSize * matrixSize; i++)
    {
        matrix[i] = rand() % 100;
    }
}

// Computes C = A x B using three nested loops (i, j, k).
// Each element C[i][j] is the dot product of row i from A and column j from B.
// All three matrices are stored as flat vectors in row-major order.
// The inner k-loop accesses A sequentially (cache-friendly) and B
// by column (stride N), which is the standard trade-off in a naive triple loop.
// matrixC must be pre-zeroed before this call.
void multiplyMatrices(const vector<int> &matrixA,
                      const vector<int> &matrixB,
                      vector<int>       &matrixC,
                      int                matrixSize)
{
    for (int i = 0; i < matrixSize; i++)
    {
        for (int j = 0; j < matrixSize; j++)
        {
            int cellSum = 0;

            // Dot product: sum of A[i][k] * B[k][j] over all k
            for (int k = 0; k < matrixSize; k++)
            {
                cellSum += matrixA[i * matrixSize + k] * matrixB[k * matrixSize + j];
            }

            matrixC[i * matrixSize + j] = cellSum;
        }
    }
}

// Sums all elements of the result matrix C into a 64-bit integer.
// long long is used to avoid overflow at large N (N=512 can produce
// element values up to ~99*99*512 = ~5 million; total sum ~1.3 billion).
// Identical checksums across sequential, thread, and OpenMP runs confirm
// that all three programs computed the same result matrix.
long long calculateChecksum(const vector<int> &matrixC)
{
    long long total = 0;
    for (int val : matrixC)
        total += val;
    return total;
}

int main(int argc, char *argv[])
{
    // Validate command-line argument
    if (argc < 2)
    {
        cout << "Usage: ./sequential N\n";
        cout << "  N - square matrix dimension (1 to 2000)\n";
        return 1;
    }

    int matrixSize = atoi(argv[1]);
    if (matrixSize <= 0 || matrixSize > 2000)
    {
        cout << "Error: N must be between 1 and 2000\n";
        return 1;
    }

    // Allocate matrices as flat vectors in row-major order.
    // matrixC is zero-initialised so accumulated sums are correct.
    vector<int> matrixA(matrixSize * matrixSize);
    vector<int> matrixB(matrixSize * matrixSize);
    vector<int> matrixC(matrixSize * matrixSize, 0);

    // Seed the RNG once with a fixed value so A and B are identical
    // every run and across all three programs (enables checksum comparison).
    srand(0);
    fillMatrixWithRandomValues(matrixA, matrixSize);
    fillMatrixWithRandomValues(matrixB, matrixSize);

    // Time only the multiplication. Matrix fill and file write are excluded
    // so the timer reflects pure compute time, matching the parallel programs.
    auto startTime = high_resolution_clock::now();
    multiplyMatrices(matrixA, matrixB, matrixC, matrixSize);
    auto endTime = high_resolution_clock::now();

    long long elapsedMicroseconds =
        duration_cast<microseconds>(endTime - startTime).count();

    // Write the result matrix to a text file for diff-based correctness checks.
    // Format: one row per line, values separated by spaces.
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
        outputFile << "\n";
    }
    outputFile.close();

    long long checksum = calculateChecksum(matrixC);

    cout << "N               = " << matrixSize << "\n";
    cout << "Method          = Sequential\n";
    cout << "Execution time  = " << elapsedMicroseconds
         << " us (" << elapsedMicroseconds / 1000.0 << " ms)\n";
    cout << "Checksum        = " << checksum << "\n";
    cout << "Output file     = C_seq.txt\n";

    return 0;
}
