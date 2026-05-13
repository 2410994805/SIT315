// Libraries
#include <iostream> // For console output
#include <vector>   // For dynamic arrays of matrices
#include <thread>   // For std::thread parallel programming
#include <chrono>   // For execution time measurement
#include <cstdlib>  // For rand() and atoi()
#include <fstream>  // For result file output

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

// Each thread computes assigned rows of matrix C
void multiplyMatrixRows(const vector<int> &matrixA, const vector<int> &matrixB, vector<int> &matrixC,
                        int matrixSize, int startRow, int endRow)
{
    for (int i = startRow; i < endRow; i++)
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
        cout << "Usage: ./parallel_thread N threads\n";
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

    vector<thread> workers;

    int rowsPerThread = matrixSize / numThreads;
    int extraRows = matrixSize % numThreads;
    int startRow = 0;

    auto start = high_resolution_clock::now();

    for (int t = 0; t < numThreads; t++)
    {
        int rows = rowsPerThread + (t < extraRows ? 1 : 0);
        int endRow = startRow + rows;

        workers.emplace_back(multiplyMatrixRows, cref(matrixA), cref(matrixB), ref(matrixC),
                             matrixSize, startRow, endRow);
        startRow = endRow;
    }

    for (auto &th : workers)
        th.join();

    auto end = high_resolution_clock::now();

    long long executionTimeMicroseconds = duration_cast<microseconds>(end - start).count();

    // Write result matrix to file (row-major format)
    ofstream outputFile("C_thread.txt");
    if (!outputFile)
    {
        cout << "Could not create C_thread.txt\n";
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
    cout << "Method = std::thread\n";
    cout << "Execution time = " << executionTimeMicroseconds << " us (" << executionTimeMicroseconds / 1000.0 << " ms)\n";
    cout << "Checksum = " << resultChecksum << '\n';
    cout << "Output = C_thread.txt\n";

    return 0;
}