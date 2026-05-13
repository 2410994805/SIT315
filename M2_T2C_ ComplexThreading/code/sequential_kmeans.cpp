// Libraries
#include <iostream> // For console output
#include <vector>   // For dynamic arrays of points, labels, and centroids
#include <random>   // For generating the same random dataset in every run
#include <chrono>   // For execution time measurement
#include <limits>   // For numeric_limits<double>::max()

using namespace std;
using namespace std::chrono;

struct Point
{
    double x;
    double y;
};

double calculateSquaredDistance(const Point &firstPoint, const Point &secondPoint)
{
    double deltaX = firstPoint.x - secondPoint.x;
    double deltaY = firstPoint.y - secondPoint.y;
    return deltaX * deltaX + deltaY * deltaY;
}

vector<Point> generatePoints(int numPoints, int seed)
{
    vector<Point> points(numPoints);

    mt19937 randomGenerator(seed);
    uniform_real_distribution<double> coordinateDistribution(0.0, 1000.0);

    for (int pointIndex = 0; pointIndex < numPoints; pointIndex++)
    {
        points[pointIndex].x = coordinateDistribution(randomGenerator);
        points[pointIndex].y = coordinateDistribution(randomGenerator);
    }

    return points;
}

vector<Point> initializeCentroids(const vector<Point> &points, int numClusters)
{
    vector<Point> centroids;
    centroids.reserve(numClusters);

    for (int clusterIndex = 0; clusterIndex < numClusters; clusterIndex++)
    {
        centroids.push_back(points[clusterIndex]);
    }

    return centroids;
}

int findNearestCentroid(const Point &currentPoint, const vector<Point> &centroids)
{
    int nearestCentroidIndex = 0;
    double minimumDistance = numeric_limits<double>::max();

    for (int clusterIndex = 0; clusterIndex < static_cast<int>(centroids.size()); clusterIndex++)
    {
        double currentDistance = calculateSquaredDistance(currentPoint, centroids[clusterIndex]);

        if (currentDistance < minimumDistance)
        {
            minimumDistance = currentDistance;
            nearestCentroidIndex = clusterIndex;
        }
    }

    return nearestCentroidIndex;
}

void runKMeansSequential(const vector<Point> &points, int numClusters, int maxIterations)
{
    int numPoints = static_cast<int>(points.size());

    vector<int> labels(numPoints, -1);
    vector<Point> centroids = initializeCentroids(points, numClusters);

    auto startTime = high_resolution_clock::now();

    for (int iteration = 0; iteration < maxIterations; iteration++)
    {
        bool assignmentsChanged = false;

        vector<double> clusterSumX(numClusters, 0.0);
        vector<double> clusterSumY(numClusters, 0.0);
        vector<int> clusterCounts(numClusters, 0);

        for (int pointIndex = 0; pointIndex < numPoints; pointIndex++)
        {
            int assignedCluster = findNearestCentroid(points[pointIndex], centroids);

            if (labels[pointIndex] != assignedCluster)
            {
                labels[pointIndex] = assignedCluster;
                assignmentsChanged = true;
            }

            clusterSumX[assignedCluster] += points[pointIndex].x;
            clusterSumY[assignedCluster] += points[pointIndex].y;
            clusterCounts[assignedCluster]++;
        }

        for (int clusterIndex = 0; clusterIndex < numClusters; clusterIndex++)
        {
            if (clusterCounts[clusterIndex] > 0)
            {
                centroids[clusterIndex].x = clusterSumX[clusterIndex] / clusterCounts[clusterIndex];
                centroids[clusterIndex].y = clusterSumY[clusterIndex] / clusterCounts[clusterIndex];
            }
        }

        if (!assignmentsChanged)
        {
            cout << "Converged after " << iteration + 1 << " iterations." << endl;
            break;
        }

        if (iteration == maxIterations - 1)
        {
            cout << "Reached maximum iterations: " << maxIterations << endl;
        }
    }

    auto endTime = high_resolution_clock::now();
    auto executionTime = duration_cast<microseconds>(endTime - startTime);

    cout << "Sequential K-Means completed." << endl;
    cout << "Number of points: " << numPoints << endl;
    cout << "Number of clusters: " << numClusters << endl;
    cout << "Execution time: " << executionTime.count() << " microseconds" << endl;

    cout << "\nFinal centroids:" << endl;
    for (int clusterIndex = 0; clusterIndex < numClusters; clusterIndex++)
    {
        cout << "Cluster " << clusterIndex
             << ": (" << centroids[clusterIndex].x
             << ", " << centroids[clusterIndex].y << ")" << endl;
    }
}

int main()
{
    int numPoints = 1000000000;
    int numClusters = 8;
    int maxIterations = 20;
    int seed = 42;

    vector<Point> points = generatePoints(numPoints, seed);
    runKMeansSequential(points, numClusters, maxIterations);

    return 0;
}