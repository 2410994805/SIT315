#include <mpi.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace std;
using namespace std::chrono;

struct Point
{
    float x;
    float y;
};

struct PartialSums
{
    vector<double> sumX;
    vector<double> sumY;
    vector<int> count;

    explicit PartialSums(int k = 0) : sumX(k, 0.0), sumY(k, 0.0), count(k, 0) {}

    void reset()
    {
        fill(sumX.begin(), sumX.end(), 0.0);
        fill(sumY.begin(), sumY.end(), 0.0);
        fill(count.begin(), count.end(), 0);
    }
};

static inline double squaredDistance(const Point &a, const Point &b)
{
    const double dx = static_cast<double>(a.x) - static_cast<double>(b.x);
    const double dy = static_cast<double>(a.y) - static_cast<double>(b.y);
    return dx * dx + dy * dy;
}

vector<Point> generatePoints(int numPoints, int seed)
{
    vector<Point> points(numPoints);
    mt19937 generator(seed);
    uniform_real_distribution<float> distribution(0.0f, 1000.0f);

    for (int i = 0; i < numPoints; ++i)
    {
        points[i].x = distribution(generator);
        points[i].y = distribution(generator);
    }

    return points;
}

vector<Point> initializeCentroids(const vector<Point> &points, int numClusters)
{
    vector<Point> centroids(numClusters);
    for (int i = 0; i < numClusters; ++i)
    {
        centroids[i] = points[i % static_cast<int>(points.size())];
    }
    return centroids;
}

int findNearestCentroid(const Point &point, const vector<Point> &centroids)
{
    int nearestIndex = 0;
    double minimumDistance = numeric_limits<double>::max();

    for (int clusterIndex = 0; clusterIndex < static_cast<int>(centroids.size()); ++clusterIndex)
    {
        const double currentDistance = squaredDistance(point, centroids[clusterIndex]);
        if (currentDistance < minimumDistance)
        {
            minimumDistance = currentDistance;
            nearestIndex = clusterIndex;
        }
    }

    return nearestIndex;
}

bool compareCentroids(const vector<Point> &a, const vector<Point> &b, double tolerance)
{
    if (a.size() != b.size())
    {
        return false;
    }

    for (size_t i = 0; i < a.size(); ++i)
    {
        if (fabs(static_cast<double>(a[i].x) - static_cast<double>(b[i].x)) > tolerance ||
            fabs(static_cast<double>(a[i].y) - static_cast<double>(b[i].y)) > tolerance)
        {
            return false;
        }
    }

    return true;
}

vector<Point> runSequentialReference(const vector<Point> &points, int numClusters, int maxIterations)
{
    vector<Point> centroids = initializeCentroids(points, numClusters);
    vector<int> labels(points.size(), -1);

    for (int iteration = 0; iteration < maxIterations; ++iteration)
    {
        bool assignmentsChanged = false;
        vector<double> clusterSumX(numClusters, 0.0);
        vector<double> clusterSumY(numClusters, 0.0);
        vector<int> clusterCounts(numClusters, 0);

        for (size_t pointIndex = 0; pointIndex < points.size(); ++pointIndex)
        {
            const int assignedCluster = findNearestCentroid(points[pointIndex], centroids);
            if (labels[pointIndex] != assignedCluster)
            {
                labels[pointIndex] = assignedCluster;
                assignmentsChanged = true;
            }
            clusterSumX[assignedCluster] += points[pointIndex].x;
            clusterSumY[assignedCluster] += points[pointIndex].y;
            clusterCounts[assignedCluster]++;
        }

        for (int clusterIndex = 0; clusterIndex < numClusters; ++clusterIndex)
        {
            if (clusterCounts[clusterIndex] > 0)
            {
                centroids[clusterIndex].x = static_cast<float>(clusterSumX[clusterIndex] / clusterCounts[clusterIndex]);
                centroids[clusterIndex].y = static_cast<float>(clusterSumY[clusterIndex] / clusterCounts[clusterIndex]);
            }
        }

        if (!assignmentsChanged)
        {
            break;
        }
    }

    return centroids;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank = 0;
    int worldSize = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    int numPoints = 100000;
    int numClusters = 8;
    int maxIterations = 20;
    int seed = 42;
    int correctnessCheck = 1;

    if (argc > 1)
        numPoints = atoi(argv[1]);
    if (argc > 2)
        numClusters = atoi(argv[2]);
    if (argc > 3)
        maxIterations = atoi(argv[3]);
    if (argc > 4)
        seed = atoi(argv[4]);
    if (argc > 5)
        correctnessCheck = atoi(argv[5]);

    if (numPoints < worldSize)
    {
        if (rank == 0)
        {
            cerr << "Number of points must be at least the number of MPI processes." << endl;
        }
        MPI_Finalize();
        return 1;
    }

    vector<Point> globalPoints;
    vector<Point> referenceCentroids;
    vector<Point> centroids;
    vector<int> sendCounts(worldSize, 0);
    vector<int> displacements(worldSize, 0);

    if (rank == 0)
    {
        globalPoints = generatePoints(numPoints, seed);
        centroids = initializeCentroids(globalPoints, numClusters);
        if (correctnessCheck == 1 && numPoints <= 200000)
        {
            referenceCentroids = runSequentialReference(globalPoints, numClusters, maxIterations);
        }

        const int baseChunk = numPoints / worldSize;
        const int remainder = numPoints % worldSize;
        int offset = 0;
        for (int process = 0; process < worldSize; ++process)
        {
            const int localCount = baseChunk + (process < remainder ? 1 : 0);
            sendCounts[process] = localCount * static_cast<int>(sizeof(Point));
            displacements[process] = offset * static_cast<int>(sizeof(Point));
            offset += localCount;
        }
    }

    MPI_Bcast(sendCounts.data(), worldSize, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(displacements.data(), worldSize, MPI_INT, 0, MPI_COMM_WORLD);

    const int localBytes = sendCounts[rank];
    const int localCount = localBytes / static_cast<int>(sizeof(Point));
    vector<Point> localPoints(localCount);
    vector<int> localLabels(localCount, -1);

    MPI_Scatterv(
        rank == 0 ? reinterpret_cast<const void *>(globalPoints.data()) : nullptr,
        sendCounts.data(),
        displacements.data(),
        MPI_BYTE,
        reinterpret_cast<void *>(localPoints.data()),
        localBytes,
        MPI_BYTE,
        0,
        MPI_COMM_WORLD);

    if (rank != 0)
    {
        centroids.resize(numClusters);
    }

    MPI_Bcast(reinterpret_cast<void *>(centroids.data()), numClusters * static_cast<int>(sizeof(Point)), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    const auto startTime = high_resolution_clock::now();

    int iterationsCompleted = 0;
    for (int iteration = 0; iteration < maxIterations; ++iteration)
    {
        PartialSums localPartial(numClusters);
        int localChangedCount = 0;

        for (int pointIndex = 0; pointIndex < localCount; ++pointIndex)
        {
            const int assignedCluster = findNearestCentroid(localPoints[pointIndex], centroids);
            if (localLabels[pointIndex] != assignedCluster)
            {
                localLabels[pointIndex] = assignedCluster;
                localChangedCount++;
            }
            localPartial.sumX[assignedCluster] += localPoints[pointIndex].x;
            localPartial.sumY[assignedCluster] += localPoints[pointIndex].y;
            localPartial.count[assignedCluster]++;
        }

        vector<double> globalSumX(numClusters, 0.0);
        vector<double> globalSumY(numClusters, 0.0);
        vector<int> globalCount(numClusters, 0);
        int globalChangedCount = 0;

        MPI_Allreduce(localPartial.sumX.data(), globalSumX.data(), numClusters, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(localPartial.sumY.data(), globalSumY.data(), numClusters, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(localPartial.count.data(), globalCount.data(), numClusters, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&localChangedCount, &globalChangedCount, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        if (rank == 0)
        {
            for (int clusterIndex = 0; clusterIndex < numClusters; ++clusterIndex)
            {
                if (globalCount[clusterIndex] > 0)
                {
                    centroids[clusterIndex].x = static_cast<float>(globalSumX[clusterIndex] / globalCount[clusterIndex]);
                    centroids[clusterIndex].y = static_cast<float>(globalSumY[clusterIndex] / globalCount[clusterIndex]);
                }
            }
        }

        MPI_Bcast(reinterpret_cast<void *>(centroids.data()), numClusters * static_cast<int>(sizeof(Point)), MPI_BYTE, 0, MPI_COMM_WORLD);
        iterationsCompleted = iteration + 1;

        if (globalChangedCount == 0)
        {
            break;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const auto endTime = high_resolution_clock::now();
    const auto executionTimeMicroseconds = duration_cast<microseconds>(endTime - startTime).count();

    long long maxExecutionTimeMicroseconds = 0;
    MPI_Reduce(&executionTimeMicroseconds, &maxExecutionTimeMicroseconds, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        cout << fixed << setprecision(4);
        cout << "MPI K-Means completed." << endl;
        cout << "Processes: " << worldSize << endl;
        cout << "Number of points: " << numPoints << endl;
        cout << "Number of clusters: " << numClusters << endl;
        cout << "Maximum iterations: " << maxIterations << endl;
        cout << "Seed: " << seed << endl;
        cout << "Iterations completed: " << iterationsCompleted << endl;
        cout << "Execution time: " << maxExecutionTimeMicroseconds << " microseconds" << endl;

        for (int clusterIndex = 0; clusterIndex < numClusters; ++clusterIndex)
        {
            cout << "Cluster " << clusterIndex << ": ("
                 << centroids[clusterIndex].x << ", "
                 << centroids[clusterIndex].y << ")" << endl;
        }

        if (correctnessCheck == 1 && !referenceCentroids.empty())
        {
            const bool passed = compareCentroids(centroids, referenceCentroids, 0.01);
            cout << "Correctness check against sequential reference: "
                 << (passed ? "PASSED" : "FAILED") << endl;
        }
        else
        {
            cout << "Correctness check against sequential reference: SKIPPED" << endl;
        }
    }

    MPI_Finalize();
    return 0;
}