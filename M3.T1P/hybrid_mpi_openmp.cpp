// Libraries
#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <fstream>

using namespace std;

// Fill matrix with random values 0-99
void fillMatrixWithRandomValues(vector<int> &matrix, int matrixSize)
{
    for (int i = 0; i < matrixSize * matrixSize; i++)
    {
        matrix[i] = rand() % 100;
    }
}

// Calculate checksum for correctness verification
long long calculateChecksum(const vector<int> &matrix)
{
    long long totalChecksum = 0;
    for (int value : matrix)
    {
        totalChecksum += value;
    }
    return totalChecksum;
}

// Write matrix to output file in row-major format
void writeMatrixToFile(const vector<int> &matrix, int matrixSize, const string &fileName)
{
    ofstream outputFile(fileName);
    if (!outputFile)
    {
        return;
    }

    for (int i = 0; i < matrixSize; i++)
    {
        for (int j = 0; j < matrixSize; j++)
        {
            outputFile << matrix[i * matrixSize + j] << " ";
        }
        outputFile << "\n";
    }

    outputFile.close();
}

// Build row counts and displacements for uneven data distribution
void buildDistribution(int matrixSize, int worldSize,
                       vector<int> &rowCounts,
                       vector<int> &sendCounts,
                       vector<int> &displacements)
{
    rowCounts.resize(worldSize);
    sendCounts.resize(worldSize);
    displacements.resize(worldSize);

    int baseRows = matrixSize / worldSize;
    int extraRows = matrixSize % worldSize;
    int currentDisplacement = 0;

    for (int rank = 0; rank < worldSize; rank++)
    {
        rowCounts[rank] = baseRows + (rank < extraRows ? 1 : 0);
        sendCounts[rank] = rowCounts[rank] * matrixSize;
        displacements[rank] = currentDisplacement;
        currentDisplacement += sendCounts[rank];
    }
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int worldRank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    if (argc < 3)
    {
        if (worldRank == 0)
        {
            cout << "Usage: mpirun -np <processes> ./hybrid_mpi_openmp N threads [write_output]\n";
        }
        MPI_Finalize();
        return 1;
    }

    int matrixSize = atoi(argv[1]);
    int numThreads = atoi(argv[2]);
    int writeOutput = (argc >= 4) ? atoi(argv[3]) : 0;

    if (matrixSize <= 0 || numThreads <= 0)
    {
        if (worldRank == 0)
        {
            cout << "N and threads must be > 0\n";
        }
        MPI_Finalize();
        return 1;
    }

    omp_set_num_threads(numThreads);

    vector<int> rowCounts, sendCounts, displacements;
    buildDistribution(matrixSize, worldSize, rowCounts, sendCounts, displacements);

    int localRows = rowCounts[worldRank];

    vector<int> matrixB(matrixSize * matrixSize);
    vector<int> localMatrixA(localRows * matrixSize);
    vector<int> localMatrixC(localRows * matrixSize, 0);

    vector<int> matrixA;
    vector<int> matrixC;

    if (worldRank == 0)
    {
        matrixA.resize(matrixSize * matrixSize);
        matrixC.resize(matrixSize * matrixSize, 0);

        srand(0);
        fillMatrixWithRandomValues(matrixA, matrixSize);
        fillMatrixWithRandomValues(matrixB, matrixSize);
    }

    // Broadcast full matrix B to all processes
    MPI_Bcast(matrixB.data(), matrixSize * matrixSize, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double startTime = MPI_Wtime();

    // Scatter rows of matrix A across processes
    MPI_Scatterv(worldRank == 0 ? matrixA.data() : nullptr,
                 sendCounts.data(),
                 displacements.data(),
                 MPI_INT,
                 localMatrixA.data(),
                 localRows * matrixSize,
                 MPI_INT,
                 0,
                 MPI_COMM_WORLD);

    // Use OpenMP inside each MPI process
#pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < localRows; i++)
    {
        for (int j = 0; j < matrixSize; j++)
        {
            int cellSum = 0;
            for (int k = 0; k < matrixSize; k++)
            {
                cellSum += localMatrixA[i * matrixSize + k] * matrixB[k * matrixSize + j];
            }
            localMatrixC[i * matrixSize + j] = cellSum;
        }
    }

    // Gather result rows back to root
    MPI_Gatherv(localMatrixC.data(),
                localRows * matrixSize,
                MPI_INT,
                worldRank == 0 ? matrixC.data() : nullptr,
                sendCounts.data(),
                displacements.data(),
                MPI_INT,
                0,
                MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double endTime = MPI_Wtime();

    double localElapsed = endTime - startTime;
    double maxElapsed = 0.0;
    MPI_Reduce(&localElapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (worldRank == 0)
    {
        long long checksum = calculateChecksum(matrixC);

        if (writeOutput == 1)
        {
            writeMatrixToFile(matrixC, matrixSize, "C_hybrid_openmp.txt");
        }

        cout << "N = " << matrixSize << '\n';
        cout << "Processes = " << worldSize << '\n';
        cout << "Threads per process = " << numThreads << '\n';
        cout << "Method = MPI + OpenMP\n";
        cout << "Execution time = " << (long long)(maxElapsed * 1000000.0)
             << " us (" << maxElapsed * 1000.0 << " ms)\n";
        cout << "Checksum = " << checksum << '\n';
        cout << "Output = " << (writeOutput == 1 ? "C_hybrid_openmp.txt" : "not written") << '\n';
    }

    MPI_Finalize();
    return 0;
}