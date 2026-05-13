// Libraries
#include <iostream> // For console output
#include <vector>   // For dynamic arrays of matrices
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

// Sequential matrix multiplication C = A * B
void multiplyMatrices(const vector<int> &matrixA, const vector<int> &matrixB, vector<int> &matrixC, int matrixSize)
{
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
    if (argc < 2)
    {
        cout << "Usage: ./sequential N\n";
        return 1;
    }

    int matrixSize = atoi(argv[1]);
    if (matrixSize <= 0 || matrixSize > 2000)
    {
        cout << "N must be 1-2000\n";
        return 1;
    }

    vector<int> matrixA(matrixSize * matrixSize), matrixB(matrixSize * matrixSize), matrixC(matrixSize * matrixSize, 0);

    srand(0);
    fillMatrixWithRandomValues(matrixA, matrixSize);
    fillMatrixWithRandomValues(matrixB, matrixSize);

    auto start = high_resolution_clock::now();
    multiplyMatrices(matrixA, matrixB, matrixC, matrixSize);
    auto end = high_resolution_clock::now();

    long long executionTimeMicroseconds = duration_cast<microseconds>(end - start).count();

    // Write result matrix to file (row-major format)
    ofstream outputFile("C_seq.txt");
    if (!outputFile)
    {
        cout << "Could not create C_seq.txt\n";
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
    cout << "Method = Sequential\n";
    cout << "Execution time = " << executionTimeMicroseconds << " us (" << executionTimeMicroseconds / 1000.0 << " ms)\n";
    cout << "Checksum = " << resultChecksum << '\n';

    return 0;
}