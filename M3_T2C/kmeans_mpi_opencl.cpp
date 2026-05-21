// MPI + OpenCL K-Means clustering
// - Rank 0 generates a synthetic dataset and initial centroids
// - Points are scattered across MPI ranks (1D block decomposition)
// - Each iteration:
//     E-step (OpenCL): each rank uses an OpenCL kernel to assign its local points
//                      to the nearest centroid in parallel (data-parallel over points)
//     M-step (MPI):    ranks reduce partial sums/counts via allreduce, rank 0
//                      updates centroids, then broadcasts them
// - Optional sequential reference on rank 0 for correctness checking

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

// Simple 2D point type for K-Means (host side)
struct Point
{
    float x;
    float y;
};

// OpenCL kernel for E-step: assign each point to nearest centroid
// Each work-item handles one point independently.
static const char *kernelSource = R"CLC(
typedef struct {
    float x;
    float y;
} Point;

__kernel void assign_points(__global const Point *points,
                            __global const Point *centroids,
                            __global int *labels,
                            const int numClusters,
                            const int numPoints)
{
    const int gid = get_global_id(0);
    if (gid >= numPoints) {
        return;
    }

    float bestDistance = FLT_MAX;
    int   bestCluster  = 0;
    const Point currentPoint = points[gid];

    for (int clusterIndex = 0; clusterIndex < numClusters; ++clusterIndex) {
        const float dx = currentPoint.x - centroids[clusterIndex].x;
        const float dy = currentPoint.y - centroids[clusterIndex].y;
        const float distance = dx * dx + dy * dy;

        if (distance < bestDistance) {
            bestDistance = distance;
            bestCluster  = clusterIndex;
        }
    }

    labels[gid] = bestCluster;
}
)CLC";

// Squared Euclidean distance between two points (host side)
static inline double squared_distance(const Point &a, const Point &b)
{
    const double dx = static_cast<double>(a.x) - static_cast<double>(b.x);
    const double dy = static_cast<double>(a.y) - static_cast<double>(b.y);
    return dx * dx + dy * dy;
}

// Generate synthetic dataset of num_points uniformly in [0, 1000] x [0, 1000]
vector<Point> generate_points(int num_points, int seed)
{
    vector<Point> points(num_points);

    mt19937 generator(seed);
    uniform_real_distribution<float> distribution(0.0f, 1000.0f);

    for (int i = 0; i < num_points; ++i)
    {
        points[i].x = distribution(generator);
        points[i].y = distribution(generator);
    }

    return points;
}

// Initialise centroids by reusing the first num_clusters points
vector<Point> initialize_centroids(const vector<Point> &points, int num_clusters)
{
    vector<Point> centroids(num_clusters);

    const int total_points = static_cast<int>(points.size());
    for (int cluster_index = 0; cluster_index < num_clusters; ++cluster_index)
    {
        centroids[cluster_index] = points[cluster_index % total_points];
    }

    return centroids;
}

// Find the index of the nearest centroid (used in sequential reference)
int find_nearest_centroid(const Point &point, const vector<Point> &centroids)
{
    int    nearest_index    = 0;
    double minimum_distance = numeric_limits<double>::max();

    const int num_clusters = static_cast<int>(centroids.size());
    for (int cluster_index = 0; cluster_index < num_clusters; ++cluster_index)
    {
        const double distance = squared_distance(point, centroids[cluster_index]);
        if (distance < minimum_distance)
        {
            minimum_distance = distance;
            nearest_index    = cluster_index;
        }
    }

    return nearest_index;
}

// Compare two sets of centroids with a tolerance on coordinates
bool compare_centroids(
    const vector<Point> &a,
    const vector<Point> &b,
    double               tolerance)
{
    if (a.size() != b.size())
    {
        return false;
    }

    const size_t n = a.size();
    for (size_t i = 0; i < n; ++i)
    {
        const double dx = fabs(static_cast<double>(a[i].x) - static_cast<double>(b[i].x));
        const double dy = fabs(static_cast<double>(a[i].y) - static_cast<double>(b[i].y));
        if (dx > tolerance || dy > tolerance)
        {
            return false;
        }
    }

    return true;
}

// Sequential K-Means used as a reference on rank 0 (for correctness checks)
vector<Point> run_sequential_reference(
    const vector<Point> &points,
    int                  num_clusters,
    int                  max_iterations)
{
    vector<Point> centroids = initialize_centroids(points, num_clusters);
    vector<int>   labels(points.size(), -1);

    for (int iteration = 0; iteration < max_iterations; ++iteration)
    {
        bool assignments_changed = false;

        vector<double> cluster_sum_x(num_clusters, 0.0);
        vector<double> cluster_sum_y(num_clusters, 0.0);
        vector<int>    cluster_counts(num_clusters, 0);

        const size_t num_points = points.size();
        for (size_t point_index = 0; point_index < num_points; ++point_index)
        {
            const int assigned_cluster =
                find_nearest_centroid(points[point_index], centroids);

            if (labels[point_index] != assigned_cluster)
            {
                labels[point_index]   = assigned_cluster;
                assignments_changed = true;
            }

            cluster_sum_x[assigned_cluster] += points[point_index].x;
            cluster_sum_y[assigned_cluster] += points[point_index].y;
            cluster_counts[assigned_cluster] += 1;
        }

        // Update centroids from accumulated sums
        for (int cluster_index = 0; cluster_index < num_clusters; ++cluster_index)
        {
            if (cluster_counts[cluster_index] > 0)
            {
                centroids[cluster_index].x =
                    static_cast<float>(cluster_sum_x[cluster_index] /
                                       cluster_counts[cluster_index]);
                centroids[cluster_index].y =
                    static_cast<float>(cluster_sum_y[cluster_index] /
                                       cluster_counts[cluster_index]);
            }
        }

        if (!assignments_changed)
        {
            // Converged early
            break;
        }
    }

    return centroids;
}

// Small helper to stringify OpenCL error codes when needed
string cl_error_to_string(cl_int error)
{
    // Minimal mapping for readability in messages (extended mappings can be added)
    switch (error)
    {
    case CL_SUCCESS: return "CL_SUCCESS";
    case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
    case CL_DEVICE_NOT_AVAILABLE: return "CL_DEVICE_NOT_AVAILABLE";
    case CL_COMPILER_NOT_AVAILABLE: return "CL_COMPILER_NOT_AVAILABLE";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
    case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
    case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
    default:
        return "CL_ERROR_" + to_string(error);
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank       = 0;
    int world_size = 0;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Default configuration (aligned with MPI-only version)
    int num_points        = 100000;
    int num_clusters      = 8;
    int max_iterations    = 20;
    int seed              = 42;
    int correctness_check = 1; // 1 = run sequential reference on small sizes

    // Optional command-line overrides:
    // argv[1] = num_points
    // argv[2] = num_clusters
    // argv[3] = max_iterations
    // argv[4] = seed
    // argv[5] = correctness_check (0 or 1)
    if (argc > 1) num_points        = atoi(argv[1]);
    if (argc > 2) num_clusters      = atoi(argv[2]);
    if (argc > 3) max_iterations    = atoi(argv[3]);
    if (argc > 4) seed              = atoi(argv[4]);
    if (argc > 5) correctness_check = atoi(argv[5]);

    if (num_points < world_size)
    {
        if (rank == 0)
        {
            cerr << "Number of points must be at least the number of MPI processes."
                 << endl;
        }
        MPI_Finalize();
        return 1;
    }

    // Global data (only fully populated on rank 0)
    vector<Point> global_points;
    vector<Point> reference_centroids;
    vector<Point> centroids;

    // Scatter layout arrays (same on all ranks after broadcast)
    vector<int> send_counts(world_size, 0);
    vector<int> displacements(world_size, 0);

    if (rank == 0)
    {
        // Phase 0: generate full dataset on rank 0
        global_points = generate_points(num_points, seed);

        // Initialise centroids on rank 0
        centroids = initialize_centroids(global_points, num_clusters);

        // Optional sequential reference for correctness (limit to avoid long runs)
        if (correctness_check == 1 && num_points <= 200000)
        {
            reference_centroids = run_sequential_reference(
                global_points, num_clusters, max_iterations);
        }

        // Compute a 1D block decomposition of the dataset:
        // send_counts[p] = bytes for rank p, displacements[p] = byte offset
        const int base_chunk = num_points / world_size;
        const int remainder  = num_points % world_size;

        int offset_points = 0;
        for (int process = 0; process < world_size; ++process)
        {
            const int local_count =
                base_chunk + (process < remainder ? 1 : 0);

            send_counts[process]   = local_count *
                                     static_cast<int>(sizeof(Point));
            displacements[process] = offset_points *
                                     static_cast<int>(sizeof(Point));

            offset_points += local_count;
        }
    }

    // Broadcast scatter layout to all ranks
    MPI_Bcast(send_counts.data(), world_size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(displacements.data(), world_size, MPI_INT, 0, MPI_COMM_WORLD);

    // Compute local number of bytes and points for this rank
    const int local_bytes = send_counts[rank];
    const int local_count = local_bytes / static_cast<int>(sizeof(Point));

    vector<Point> local_points(local_count);

    // Phase 1: scatter points so each rank owns a block of the dataset
    MPI_Scatterv(
        rank == 0 ? reinterpret_cast<void *>(global_points.data()) : nullptr,
        send_counts.data(),
        displacements.data(),
        MPI_BYTE,
        reinterpret_cast<void *>(local_points.data()),
        local_bytes,
        MPI_BYTE,
        0,
        MPI_COMM_WORLD);

    // All ranks must have centroids; non-root ranks allocate storage
    if (rank != 0)
    {
        centroids.resize(num_clusters);
    }

    // Broadcast initial centroids from rank 0
    MPI_Bcast(
        reinterpret_cast<void *>(centroids.data()),
        num_clusters * static_cast<int>(sizeof(Point)),
        MPI_BYTE,
        0,
        MPI_COMM_WORLD);

    // ---- OpenCL setup (one context and command queue per rank) ----

    cl_int error_code = CL_SUCCESS;

    cl_uint num_platforms = 0;
    error_code = clGetPlatformIDs(0, nullptr, &num_platforms);
    if (error_code != CL_SUCCESS || num_platforms == 0)
    {
        if (rank == 0)
        {
            cerr << "Failed to find any OpenCL platforms. Error: "
                 << cl_error_to_string(error_code) << endl;
        }
        MPI_Finalize();
        return 1;
    }

    vector<cl_platform_id> platforms(num_platforms);
    error_code = clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
    if (error_code != CL_SUCCESS)
    {
        if (rank == 0)
        {
            cerr << "Failed to get OpenCL platform IDs. Error: "
                 << cl_error_to_string(error_code) << endl;
        }
        MPI_Finalize();
        return 1;
    }

    cl_platform_id platform = platforms[0];

    // Prefer GPU if present, fall back to CPU (in this VM it will be CPU)
    cl_device_id device = nullptr;
    error_code = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    if (error_code != CL_SUCCESS)
    {
        error_code = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, nullptr);
        if (error_code != CL_SUCCESS)
        {
            if (rank == 0)
            {
                cerr << "Failed to find a suitable OpenCL device. Error: "
                     << cl_error_to_string(error_code) << endl;
            }
            MPI_Finalize();
            return 1;
        }
    }

    // Query basic platform and device information (for output later)
    char platform_name[256] = {};
    char device_name[256]   = {};
    cl_device_type device_type = 0;

    clGetPlatformInfo(platform, CL_PLATFORM_NAME,
                      sizeof(platform_name), platform_name, nullptr);
    clGetDeviceInfo(device, CL_DEVICE_NAME,
                    sizeof(device_name), device_name, nullptr);
    clGetDeviceInfo(device, CL_DEVICE_TYPE,
                    sizeof(device_type), &device_type, nullptr);

    // Create context and command queue for this rank
    cl_context_properties context_properties[] = {
        CL_CONTEXT_PLATFORM,
        reinterpret_cast<cl_context_properties>(platform),
        0
    };

    cl_context context = clCreateContext(
        context_properties, 1, &device, nullptr, nullptr, &error_code);
    if (error_code != CL_SUCCESS || context == nullptr)
    {
        if (rank == 0)
        {
            cerr << "Failed to create OpenCL context. Error: "
                 << cl_error_to_string(error_code) << endl;
        }
        MPI_Finalize();
        return 1;
    }

    cl_command_queue command_queue =
        clCreateCommandQueue(context, device, 0, &error_code);
    if (error_code != CL_SUCCESS || command_queue == nullptr)
    {
        if (rank == 0)
        {
            cerr << "Failed to create OpenCL command queue. Error: "
                 << cl_error_to_string(error_code) << endl;
        }
        clReleaseContext(context);
        MPI_Finalize();
        return 1;
    }

    // Build the OpenCL program and kernel
    const char *sources[] = { kernelSource };
    const size_t source_lengths[] = { strlen(kernelSource) };

    cl_program program = clCreateProgramWithSource(
        context, 1, sources, source_lengths, &error_code);
    if (error_code != CL_SUCCESS || program == nullptr)
    {
        if (rank == 0)
        {
            cerr << "Failed to create OpenCL program. Error: "
                 << cl_error_to_string(error_code) << endl;
        }
        clReleaseCommandQueue(command_queue);
        clReleaseContext(context);
        MPI_Finalize();
        return 1;
    }

    error_code = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (error_code != CL_SUCCESS)
    {
        if (rank == 0)
        {
            cerr << "Failed to build OpenCL program. Error: "
                 << cl_error_to_string(error_code) << endl;

            // Print build log for diagnostics
            size_t log_size = 0;
            clGetProgramBuildInfo(program, device,
                                  CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
            if (log_size > 1)
            {
                vector<char> build_log(log_size);
                clGetProgramBuildInfo(program, device,
                                      CL_PROGRAM_BUILD_LOG, log_size,
                                      build_log.data(), nullptr);
                cerr << "OpenCL build log:\n" << build_log.data() << endl;
            }
        }
        clReleaseProgram(program);
        clReleaseCommandQueue(command_queue);
        clReleaseContext(context);
        MPI_Finalize();
        return 1;
    }

    cl_kernel kernel = clCreateKernel(program, "assign_points", &error_code);
    if (error_code != CL_SUCCESS || kernel == nullptr)
    {
        if (rank == 0)
        {
            cerr << "Failed to create OpenCL kernel. Error: "
                 << cl_error_to_string(error_code) << endl;
        }
        clReleaseProgram(program);
        clReleaseCommandQueue(command_queue);
        clReleaseContext(context);
        MPI_Finalize();
        return 1;
    }

    // Create device buffers for this rank's points, centroids, and labels
    cl_mem points_buffer = clCreateBuffer(
        context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(Point) * local_points.size(),
        local_points.data(),
        &error_code);
    if (error_code != CL_SUCCESS || points_buffer == nullptr)
    {
        if (rank == 0)
        {
            cerr << "Failed to create OpenCL buffer for points. Error: "
                 << cl_error_to_string(error_code) << endl;
        }
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(command_queue);
        clReleaseContext(context);
        MPI_Finalize();
        return 1;
    }

    cl_mem centroids_buffer = clCreateBuffer(
        context,
        CL_MEM_READ_ONLY,
        sizeof(Point) * num_clusters,
        nullptr,
        &error_code);
    if (error_code != CL_SUCCESS || centroids_buffer == nullptr)
    {
        if (rank == 0)
        {
            cerr << "Failed to create OpenCL buffer for centroids. Error: "
                 << cl_error_to_string(error_code) << endl;
        }
        clReleaseMemObject(points_buffer);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(command_queue);
        clReleaseContext(context);
        MPI_Finalize();
        return 1;
    }

    vector<int> local_labels(local_count, -1);

    cl_mem labels_buffer = clCreateBuffer(
        context,
        CL_MEM_READ_WRITE,
        sizeof(int) * local_labels.size(),
        nullptr,
        &error_code);
    if (error_code != CL_SUCCESS || labels_buffer == nullptr)
    {
        if (rank == 0)
        {
            cerr << "Failed to create OpenCL buffer for labels. Error: "
                 << cl_error_to_string(error_code) << endl;
        }
        clReleaseMemObject(centroids_buffer);
        clReleaseMemObject(points_buffer);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(command_queue);
        clReleaseContext(context);
        MPI_Finalize();
        return 1;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Phase 2: distributed K-Means iterations (measure only this region)
    const auto start_time = high_resolution_clock::now();

    int iterations_completed = 0;

    // Track previous labels locally to count changes (convergence criterion)
    vector<int> previous_labels(local_count, -1);

    for (int iteration = 0; iteration < max_iterations; ++iteration)
    {
        // Copy current centroids to device (all ranks)
        error_code = clEnqueueWriteBuffer(
            command_queue,
            centroids_buffer,
            CL_TRUE,
            0,
            sizeof(Point) * centroids.size(),
            centroids.data(),
            0,
            nullptr,
            nullptr);
        if (error_code != CL_SUCCESS)
        {
            if (rank == 0)
            {
                cerr << "Failed to write centroids to OpenCL buffer. Error: "
                     << cl_error_to_string(error_code) << endl;
            }
            break;
        }

        // Set kernel arguments
        error_code  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &points_buffer);
        error_code |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &centroids_buffer);
        error_code |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &labels_buffer);
        error_code |= clSetKernelArg(kernel, 3, sizeof(int), &num_clusters);
        error_code |= clSetKernelArg(kernel, 4, sizeof(int), &local_count);

        if (error_code != CL_SUCCESS)
        {
            if (rank == 0)
            {
                cerr << "Failed to set OpenCL kernel arguments. Error: "
                     << cl_error_to_string(error_code) << endl;
            }
            break;
        }

        // Launch the kernel: one work-item per local point
        const size_t global_work_size = static_cast<size_t>(local_count);

        error_code = clEnqueueNDRangeKernel(
            command_queue,
            kernel,
            1,
            nullptr,
            &global_work_size,
            nullptr,
            0,
            nullptr,
            nullptr);
        if (error_code != CL_SUCCESS)
        {
            if (rank == 0)
            {
                cerr << "Failed to enqueue OpenCL kernel. Error: "
                     << cl_error_to_string(error_code) << endl;
            }
            break;
        }

        // Wait for kernel completion and read back labels
        clFinish(command_queue);

        error_code = clEnqueueReadBuffer(
            command_queue,
            labels_buffer,
            CL_TRUE,
            0,
            sizeof(int) * local_labels.size(),
            local_labels.data(),
            0,
            nullptr,
            nullptr);
        if (error_code != CL_SUCCESS)
        {
            if (rank == 0)
            {
                cerr << "Failed to read OpenCL labels. Error: "
                     << cl_error_to_string(error_code) << endl;
            }
            break;
        }

        // Local M-step: accumulate partial sums and counts, track label changes
        vector<double> local_sum_x(num_clusters, 0.0);
        vector<double> local_sum_y(num_clusters, 0.0);
        vector<int>    local_count_per_cluster(num_clusters, 0);
        int            local_changed_count = 0;

        for (int point_index = 0; point_index < local_count; ++point_index)
        {
            const int cluster_index = local_labels[point_index];

            if (cluster_index >= 0 && cluster_index < num_clusters)
            {
                if (previous_labels[point_index] != cluster_index)
                {
                    local_changed_count += 1;
                    previous_labels[point_index] = cluster_index;
                }

                local_sum_x[cluster_index] += local_points[point_index].x;
                local_sum_y[cluster_index] += local_points[point_index].y;
                local_count_per_cluster[cluster_index] += 1;
            }
        }

        // Global M-step: combine partial sums/counts and label changes
        vector<double> global_sum_x(num_clusters, 0.0);
        vector<double> global_sum_y(num_clusters, 0.0);
        vector<int>    global_count(num_clusters, 0);
        int            global_changed_count = 0;

        MPI_Allreduce(
            local_sum_x.data(),
            global_sum_x.data(),
            num_clusters,
            MPI_DOUBLE,
            MPI_SUM,
            MPI_COMM_WORLD);

        MPI_Allreduce(
            local_sum_y.data(),
            global_sum_y.data(),
            num_clusters,
            MPI_DOUBLE,
            MPI_SUM,
            MPI_COMM_WORLD);

        MPI_Allreduce(
            local_count_per_cluster.data(),
            global_count.data(),
            num_clusters,
            MPI_INT,
            MPI_SUM,
            MPI_COMM_WORLD);

        MPI_Allreduce(
            &local_changed_count,
            &global_changed_count,
            1,
            MPI_INT,
            MPI_SUM,
            MPI_COMM_WORLD);

        // Rank 0 updates centroids using the global sums and counts
        if (rank == 0)
        {
            for (int cluster_index = 0; cluster_index < num_clusters; ++cluster_index)
            {
                if (global_count[cluster_index] > 0)
                {
                    centroids[cluster_index].x =
                        static_cast<float>(global_sum_x[cluster_index] /
                                           global_count[cluster_index]);
                    centroids[cluster_index].y =
                        static_cast<float>(global_sum_y[cluster_index] /
                                           global_count[cluster_index]);
                }
            }
        }

        // Broadcast updated centroids to all ranks for the next iteration
        MPI_Bcast(
            reinterpret_cast<void *>(centroids.data()),
            num_clusters * static_cast<int>(sizeof(Point)),
            MPI_BYTE,
            0,
            MPI_COMM_WORLD);

        iterations_completed = iteration + 1;

        // No label changed on any rank → converged
        if (global_changed_count == 0)
        {
            break;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const auto end_time = high_resolution_clock::now();

    const auto execution_time_microseconds =
        duration_cast<microseconds>(end_time - start_time).count();

    // Use the maximum time across ranks as the overall distributed runtime
    long long max_execution_time_microseconds = 0;

    MPI_Reduce(
        &execution_time_microseconds,
        &max_execution_time_microseconds,
        1,
        MPI_LONG_LONG,
        MPI_MAX,
        0,
        MPI_COMM_WORLD);

    if (rank == 0)
    {
        cout << fixed << setprecision(4);

        cout << "Mode: MPI + OpenCL K-Means" << endl;
        cout << "Processes: "          << world_size          << endl;
        cout << "Number of points: "   << num_points          << endl;
        cout << "Number of clusters: " << num_clusters        << endl;
        cout << "Maximum iterations: " << max_iterations      << endl;
        cout << "Seed: "               << seed                << endl;
        cout << "Iterations completed: " << iterations_completed << endl;
        cout << "Execution time (max across ranks): "
             << max_execution_time_microseconds << " microseconds" << endl;

        cout << "OpenCL platform: " << platform_name << endl;
        cout << "OpenCL device: "   << device_name   << endl;
        cout << "OpenCL device type: "
             << ((device_type & CL_DEVICE_TYPE_GPU) ? "GPU"
                 : (device_type & CL_DEVICE_TYPE_CPU) ? "CPU"
                 : "OTHER")
             << endl;

        for (int cluster_index = 0; cluster_index < num_clusters; ++cluster_index)
        {
            cout << "Cluster " << cluster_index << ": ("
                 << centroids[cluster_index].x << ", "
                 << centroids[cluster_index].y << ")" << endl;
        }

        if (correctness_check == 1 && !reference_centroids.empty())
        {
            const bool passed = compare_centroids(
                centroids, reference_centroids, 0.01);

            cout << "Correctness check against sequential reference: "
                 << (passed ? "PASSED" : "FAILED") << endl;
        }
        else if (correctness_check == 1 && reference_centroids.empty())
        {
            cout << "Correctness check against sequential reference: SKIPPED "
                 << "(dataset too large for reference run)" << endl;
        }
        else
        {
            cout << "Correctness check against sequential reference: DISABLED"
                 << endl;
        }
    }

    // Clean up OpenCL resources
    clReleaseMemObject(labels_buffer);
    clReleaseMemObject(centroids_buffer);
    clReleaseMemObject(points_buffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);

    MPI_Finalize();
    return 0;
}
