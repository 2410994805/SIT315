// Libraries
#include <iostream> // For console output
#include <vector>   // For dynamic arrays of matrices
#include <chrono>   // For execution time measurement
#include <cstdlib>  // For rand() and atoi()
#include <fstream>  // For result file output
#include <omp.h>    // For OpenMP parallel programming

using namespace std;
using namespace std::chrono;

// Fill matrix with random values 0-99
void fillMatrixWithRandomValues(vector<int> &matrix, int matrixSize)
{
    for (int i = 0; i < matrixSize * matrixSize; i++)
    {
        matrix[i] = rand() % 100;
    }
}

// Calculate checksum for result verification
long long calculateChecksum(const vector<int> &matrixC)
{
    long long totalChecksum = 0;
    for (int val : matrixC)
        totalChecksum += val;
    return totalChecksum;
}

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        cout << "Usage: ./parallel_openmp N threads\n";
        return 1;
    }

    int matrixSize = atoi(argv[1]);
    int numThreads = atoi(argv[2]);

    if (matrixSize <= 0 || matrixSize > 2000 || numThreads <= 0 || numThreads > 8)
    {
        cout << "N must be 1-2000, threads 1-8\n";
        return 1;
    }

    vector<int> matrixA(matrixSize * matrixSize), matrixB(matrixSize * matrixSize), matrixC(matrixSize * matrixSize, 0);

    srand(0);
    fillMatrixWithRandomValues(matrixA, matrixSize);
    fillMatrixWithRandomValues(matrixB, matrixSize);

    omp_set_num_threads(numThreads);

    auto start = high_resolution_clock::now();

// Parallelize outer two loops (i,j blocks) - perfect load balance
#pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < matrixSize; i++)
    {
        for (int j = 0; j < matrixSize; j++)
        {
            int cellSum = 0;
            for (int k = 0; k < matrixSize; k++)
            {
                cellSum += matrixA[i * matrixSize + k] * matrixB[k * matrixSize + j];
            }
            matrixC[i * matrixSize + j] = cellSum;
        }
    }

    auto end = high_resolution_clock::now();
    long long executionTimeMicroseconds = duration_cast<microseconds>(end - start).count();

    // Write result matrix to file (row-major format)
    ofstream outputFile("C_openmp.txt");
    if (!outputFile)
    {
        cout << "Could not create C_openmp.txt\n";
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

    long long resultChecksum = calculateChecksum(matrixC);

    cout << "N = " << matrixSize << '\n';
    cout << "Threads = " << numThreads << '\n';
    cout << "Method = OpenMP\n";
    cout << "Execution time = " << executionTimeMicroseconds << " us (" << executionTimeMicroseconds / 1000.0 << " ms)\n";
    cout << "Checksum = " << resultChecksum << '\n';
    cout << "Output = C_openmp.txt\n";

    return 0;
}