// MPI-based K-Means clustering
// - Rank 0 generates a synthetic dataset and initial centroids
// - Points are scattered across MPI ranks (1D block decomposition)
// - Each iteration:
//     E-step: each rank assigns its local points to nearest centroid
//     M-step: global sums and counts are computed via allreduce and
//             rank 0 updates centroids, then broadcasts them
// - Optional sequential reference on rank 0 for correctness checking

#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

using namespace std;
using namespace std::chrono;

// Simple 2D point type for K-Means
struct Point
{
    float x;
    float y;
};

// Helper structure for local partial sums (one rank)
struct PartialSums
{
    vector<double> sum_x;
    vector<double> sum_y;
    vector<int>    count;

    explicit PartialSums(int num_clusters = 0)
        : sum_x(num_clusters, 0.0),
          sum_y(num_clusters, 0.0),
          count(num_clusters, 0)
    {
    }

    void reset()
    {
        fill(sum_x.begin(), sum_x.end(), 0.0);
        fill(sum_y.begin(), sum_y.end(), 0.0);
        fill(count.begin(), count.end(), 0);
    }
};

// Squared Euclidean distance between two points
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
        // Wrap around if num_clusters > total_points (defensive)
        centroids[cluster_index] = points[cluster_index % total_points];
    }

    return centroids;
}

// Find the index of the nearest centroid for a given point
int find_nearest_centroid(const Point &point, const vector<Point> &centroids)
{
    int    nearest_index   = 0;
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

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank       = 0;
    int world_size = 0;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Default configuration (matches your earlier experiments)
    int num_points       = 100000;
    int num_clusters     = 8;
    int max_iterations   = 20;
    int seed             = 42;
    int correctness_check = 1; // 1 = run sequential reference on small sizes

    // Optional command-line overrides:
    // argv[1] = num_points
    // argv[2] = num_clusters
    // argv[3] = max_iterations
    // argv[4] = seed
    // argv[5] = correctness_check (0 or 1)
    if (argc > 1) num_points       = atoi(argv[1]);
    if (argc > 2) num_clusters     = atoi(argv[2]);
    if (argc > 3) max_iterations   = atoi(argv[3]);
    if (argc > 4) seed             = atoi(argv[4]);
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

        // Initialise centroids on rank 0 (wrapped indexing if needed)
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
    vector<int>   local_labels(local_count, -1);

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

    MPI_Barrier(MPI_COMM_WORLD);

    // Phase 2: distributed K-Means iterations (measure only this region)
    const auto start_time = high_resolution_clock::now();

    int iterations_completed = 0;

    for (int iteration = 0; iteration < max_iterations; ++iteration)
    {
        // Local E-step: assign this rank's points to nearest centroid
        PartialSums local_partial(num_clusters);
        int         local_changed_count = 0;

        for (int point_index = 0; point_index < local_count; ++point_index)
        {
            const int assigned_cluster =
                find_nearest_centroid(local_points[point_index], centroids);

            if (local_labels[point_index] != assigned_cluster)
            {
                local_labels[point_index] = assigned_cluster;
                local_changed_count += 1;
            }

            local_partial.sum_x[assigned_cluster] += local_points[point_index].x;
            local_partial.sum_y[assigned_cluster] += local_points[point_index].y;
            local_partial.count[assigned_cluster] += 1;
        }

        // Global M-step: combine partial sums across ranks, then update centroids
        vector<double> global_sum_x(num_clusters, 0.0);
        vector<double> global_sum_y(num_clusters, 0.0);
        vector<int>    global_count(num_clusters, 0);
        int            global_changed_count = 0;

        // Allreduce combines per-rank contributions for centroids
        MPI_Allreduce(
            local_partial.sum_x.data(),
            global_sum_x.data(),
            num_clusters,
            MPI_DOUBLE,
            MPI_SUM,
            MPI_COMM_WORLD);

        MPI_Allreduce(
            local_partial.sum_y.data(),
            global_sum_y.data(),
            num_clusters,
            MPI_DOUBLE,
            MPI_SUM,
            MPI_COMM_WORLD);

        MPI_Allreduce(
            local_partial.count.data(),
            global_count.data(),
            num_clusters,
            MPI_INT,
            MPI_SUM,
            MPI_COMM_WORLD);

        // Allreduce on label changes to detect global convergence
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

        cout << "Mode: MPI K-Means" << endl;
        cout << "Processes: "        << world_size    << endl;
        cout << "Number of points: " << num_points    << endl;
        cout << "Number of clusters: " << num_clusters << endl;
        cout << "Maximum iterations: " << max_iterations << endl;
        cout << "Seed: "             << seed          << endl;
        cout << "Iterations completed: " << iterations_completed << endl;
        cout << "Execution time (max across ranks): "
             << max_execution_time_microseconds << " microseconds" << endl;

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

    MPI_Finalize();
    return 0;
}
