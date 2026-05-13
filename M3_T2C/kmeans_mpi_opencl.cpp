#include <mpi.h>
#include <CL/cl.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace std;
using namespace std::chrono;

struct Point
{
    float x;
    float y;
};

static const char *kernelSource = R"CLC(
typedef struct {
    float x;
    float y;
} Point;

__kernel void assign_points(__global const Point* points,
                            __global const Point* centroids,
                            __global int* labels,
                            const int numClusters,
                            const int numPoints) {
    const int gid = get_global_id(0);
    if (gid >= numPoints) {
        return;
    }

    float bestDistance = FLT_MAX;
    int bestCluster = 0;
    const Point currentPoint = points[gid];

    for (int clusterIndex = 0; clusterIndex < numClusters; ++clusterIndex) {
        const float dx = currentPoint.x - centroids[clusterIndex].x;
        const float dy = currentPoint.y - centroids[clusterIndex].y;
        const float distance = dx * dx + dy * dy;
        if (distance < bestDistance) {
            bestDistance = distance;
            bestCluster = clusterIndex;
        }
    }

    labels[gid] = bestCluster;
}
)CLC";

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

string getBuildLog(cl_program program, cl_device_id device)
{
    size_t logSize = 0;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
    string buildLog(logSize, '\0');
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, buildLog.data(), nullptr);
    return buildLog;
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

    cl_int errorCode = CL_SUCCESS;
    cl_uint platformCount = 0;
    clGetPlatformIDs(0, nullptr, &platformCount);
    if (platformCount == 0)
    {
        if (rank == 0)
        {
            cerr << "No OpenCL platforms found." << endl;
        }
        MPI_Finalize();
        return 1;
    }

    vector<cl_platform_id> platforms(platformCount);
    clGetPlatformIDs(platformCount, platforms.data(), nullptr);

    cl_device_id device = nullptr;
    cl_platform_id selectedPlatform = nullptr;

    for (cl_platform_id platform : platforms)
    {
        cl_uint deviceCount = 0;
        if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &deviceCount) == CL_SUCCESS && deviceCount > 0)
        {
            vector<cl_device_id> devices(deviceCount);
            clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, deviceCount, devices.data(), nullptr);
            device = devices[0];
            selectedPlatform = platform;
            break;
        }
    }

    if (device == nullptr)
    {
        for (cl_platform_id platform : platforms)
        {
            cl_uint deviceCount = 0;
            if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &deviceCount) == CL_SUCCESS && deviceCount > 0)
            {
                vector<cl_device_id> devices(deviceCount);
                clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, deviceCount, devices.data(), nullptr);
                device = devices[0];
                selectedPlatform = platform;
                break;
            }
        }
    }

    if (device == nullptr)
    {
        if (rank == 0)
        {
            cerr << "No OpenCL devices available." << endl;
        }
        MPI_Finalize();
        return 1;
    }

    char platformName[256] = {0};
    char deviceName[256] = {0};
    cl_device_type deviceType = 0;
    clGetPlatformInfo(selectedPlatform, CL_PLATFORM_NAME, sizeof(platformName), platformName, nullptr);
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(deviceName), deviceName, nullptr);
    clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(deviceType), &deviceType, nullptr);

    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &errorCode);
    if (errorCode != CL_SUCCESS)
    {
        if (rank == 0)
        {
            cerr << "Failed to create OpenCL context." << endl;
        }
        MPI_Finalize();
        return 1;
    }

    cl_command_queue commandQueue = clCreateCommandQueue(context, device, 0, &errorCode);
    if (errorCode != CL_SUCCESS)
    {
        if (rank == 0)
        {
            cerr << "Failed to create OpenCL command queue." << endl;
        }
        clReleaseContext(context);
        MPI_Finalize();
        return 1;
    }

    const char *sources[] = {kernelSource};
    cl_program program = clCreateProgramWithSource(context, 1, sources, nullptr, &errorCode);
    if (errorCode != CL_SUCCESS)
    {
        if (rank == 0)
        {
            cerr << "Failed to create OpenCL program." << endl;
        }
        clReleaseCommandQueue(commandQueue);
        clReleaseContext(context);
        MPI_Finalize();
        return 1;
    }

    errorCode = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (errorCode != CL_SUCCESS)
    {
        if (rank == 0)
        {
            cerr << "Failed to build OpenCL program." << endl;
            cerr << getBuildLog(program, device) << endl;
        }
        clReleaseProgram(program);
        clReleaseCommandQueue(commandQueue);
        clReleaseContext(context);
        MPI_Finalize();
        return 1;
    }

    cl_kernel kernel = clCreateKernel(program, "assign_points", &errorCode);
    if (errorCode != CL_SUCCESS)
    {
        if (rank == 0)
        {
            cerr << "Failed to create OpenCL kernel." << endl;
        }
        clReleaseProgram(program);
        clReleaseCommandQueue(commandQueue);
        clReleaseContext(context);
        MPI_Finalize();
        return 1;
    }

    cl_mem pointsBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         sizeof(Point) * localPoints.size(), localPoints.data(), &errorCode);
    cl_mem centroidsBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                            sizeof(Point) * centroids.size(), nullptr, &errorCode);
    cl_mem labelsBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                         sizeof(int) * localLabels.size(), nullptr, &errorCode);

    MPI_Barrier(MPI_COMM_WORLD);
    const auto startTime = high_resolution_clock::now();

    int iterationsCompleted = 0;
    for (int iteration = 0; iteration < maxIterations; ++iteration)
    {
        errorCode = clEnqueueWriteBuffer(commandQueue, centroidsBuffer, CL_TRUE, 0,
                                         sizeof(Point) * centroids.size(), centroids.data(), 0, nullptr, nullptr);

        errorCode |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &pointsBuffer);
        errorCode |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &centroidsBuffer);
        errorCode |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &labelsBuffer);
        errorCode |= clSetKernelArg(kernel, 3, sizeof(int), &numClusters);
        errorCode |= clSetKernelArg(kernel, 4, sizeof(int), &localCount);

        if (errorCode != CL_SUCCESS)
        {
            if (rank == 0)
            {
                cerr << "Failed to set OpenCL kernel arguments." << endl;
            }
            break;
        }

        const size_t globalWorkSize = static_cast<size_t>(localCount);
        errorCode = clEnqueueNDRangeKernel(commandQueue, kernel, 1, nullptr, &globalWorkSize, nullptr, 0, nullptr, nullptr);
        if (errorCode != CL_SUCCESS)
        {
            if (rank == 0)
            {
                cerr << "Failed to enqueue OpenCL kernel." << endl;
            }
            break;
        }

        clFinish(commandQueue);
        errorCode = clEnqueueReadBuffer(commandQueue, labelsBuffer, CL_TRUE, 0,
                                        sizeof(int) * localLabels.size(), localLabels.data(), 0, nullptr, nullptr);
        if (errorCode != CL_SUCCESS)
        {
            if (rank == 0)
            {
                cerr << "Failed to read OpenCL labels." << endl;
            }
            break;
        }

        vector<double> localSumX(numClusters, 0.0);
        vector<double> localSumY(numClusters, 0.0);
        vector<int> localCountPerCluster(numClusters, 0);
        int localChangedCount = 0;
        static vector<int> previousLabels;
        if (previousLabels.size() != localLabels.size())
        {
            previousLabels.assign(localLabels.size(), -1);
        }

        for (int pointIndex = 0; pointIndex < localCount; ++pointIndex)
        {
            const int clusterIndex = localLabels[pointIndex];
            if (previousLabels[pointIndex] != clusterIndex)
            {
                localChangedCount++;
                previousLabels[pointIndex] = clusterIndex;
            }
            localSumX[clusterIndex] += localPoints[pointIndex].x;
            localSumY[clusterIndex] += localPoints[pointIndex].y;
            localCountPerCluster[clusterIndex]++;
        }

        vector<double> globalSumX(numClusters, 0.0);
        vector<double> globalSumY(numClusters, 0.0);
        vector<int> globalCount(numClusters, 0);
        int globalChangedCount = 0;

        MPI_Allreduce(localSumX.data(), globalSumX.data(), numClusters, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(localSumY.data(), globalSumY.data(), numClusters, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(localCountPerCluster.data(), globalCount.data(), numClusters, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
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
        cout << "MPI + OpenCL K-Means completed." << endl;
        cout << "Processes: " << worldSize << endl;
        cout << "Number of points: " << numPoints << endl;
        cout << "Number of clusters: " << numClusters << endl;
        cout << "Maximum iterations: " << maxIterations << endl;
        cout << "Seed: " << seed << endl;
        cout << "Iterations completed: " << iterationsCompleted << endl;
        cout << "Execution time: " << maxExecutionTimeMicroseconds << " microseconds" << endl;
        cout << "OpenCL platform: " << platformName << endl;
        cout << "OpenCL device: " << deviceName << endl;
        cout << "OpenCL device type: "
             << ((deviceType & CL_DEVICE_TYPE_GPU) ? "GPU" : ((deviceType & CL_DEVICE_TYPE_CPU) ? "CPU" : "OTHER"))
             << endl;

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

    clReleaseMemObject(pointsBuffer);
    clReleaseMemObject(centroidsBuffer);
    clReleaseMemObject(labelsBuffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(commandQueue);
    clReleaseContext(context);

    MPI_Finalize();
    return 0;
}