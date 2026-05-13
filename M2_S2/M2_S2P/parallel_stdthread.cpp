// SIT315 Programming Paradigms
// Task    : M2.S2P - Activity 2 | Parallel Vector Addition (std::thread)
// Author  : Jahan Garg (2410994805)
// Compile : g++ -std=c++11 -O2 parallel_stdthread.cpp -o parallel_stdthread
// Run     : ./parallel_stdthread              (auto-detects logical cores)
//           ./parallel_stdthread <num_threads> (manual override)
//
// Demonstrates the same output data decomposition as the pthread version
// using C++11 std::thread. Key advantages over pthreads:
//   - RAII thread management via std::vector<thread>
//   - hardware_concurrency() for automatic core detection
//   - Cleaner syntax; no void* casting needed

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <vector>
#include <ctime>

using namespace std;
using namespace std::chrono;

// Shared vectors: v1/v2 read-only after init; v3 write-partitioned per thread
int          *v1, *v2, *v3;
unsigned long totalSize;

// Thread worker: computes v3[i] = v1[i] + v2[i] for index range [start, end)
void addChunk(unsigned long start, unsigned long end)
{
    for (unsigned long i = start; i < end; i++)
        v3[i] = v1[i] + v2[i];
}

// Fills an integer array with pseudo-random values in [0, 99]
void randomVector(int vector[], unsigned long size)
{
    for (unsigned long i = 0; i < size; i++)
        vector[i] = rand() % 100;
}

int main(int argc, char *argv[])
{
    totalSize = 100000000;
    srand((unsigned)time(0));

    // Determine thread count: command-line argument or hardware auto-detection
    unsigned int numThreads;
    if (argc == 2)
    {
        numThreads = (unsigned int)atoi(argv[1]);
        if (numThreads == 0)
        {
            cerr << "Error: Thread count must be a positive integer." << endl;
            return 1;
        }
    }
    else
    {
        // hardware_concurrency returns logical processor count (cores x HT factor)
        numThreads = thread::hardware_concurrency();
        if (numThreads == 0)
            numThreads = 2; // Safe fallback for systems that cannot report core count
    }

    v1 = (int *)malloc(totalSize * sizeof(int));
    v2 = (int *)malloc(totalSize * sizeof(int));
    v3 = (int *)malloc(totalSize * sizeof(int));

    if (!v1 || !v2 || !v3)
    {
        cerr << "Error: Memory allocation failed." << endl;
        return 1;
    }

    // Populate input vectors before the timed parallel phase
    randomVector(v1, totalSize);
    randomVector(v2, totalSize);

    // Partition 100M elements into numThreads contiguous chunks
    unsigned long chunk     = totalSize / numThreads;
    unsigned long remainder = totalSize % numThreads;

    // STL vector of threads — RAII ensures clean destruction
    vector<thread> workers;
    workers.reserve(numThreads);

    // Begin timer — parallel computation phase only
    auto start = high_resolution_clock::now();

    // Spawn one thread per partition using emplace_back (avoids copy overhead)
    for (unsigned int t = 0; t < numThreads; t++)
    {
        unsigned long s = (unsigned long)t * chunk;
        unsigned long e = (t == numThreads - 1) ? s + chunk + remainder : s + chunk;
        workers.emplace_back(addChunk, s, e);
    }

    // Join all threads — equivalent to a pthread_join barrier
    for (auto &th : workers)
        th.join();

    // End timer after all threads have joined
    auto stop     = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);

    cout << "[std::thread] Threads used : " << numThreads << endl;
    cout << "[std::thread] Time taken   : " << duration.count() << " microseconds" << endl;

    // Spot-check first 10 elements to verify correctness
    bool correct = true;
    for (int i = 0; i < 10; i++)
        if (v3[i] != v1[i] + v2[i]) { correct = false; break; }

    cout << "[std::thread] Verification : " << (correct ? "PASS" : "FAIL") << endl;

    free(v1); free(v2); free(v3);

    return 0;
}
