// SIT315 Programming Paradigms
// Task    : M2.S2P - Activity 2 | Parallel Vector Addition (pthreads)
// Author  : Jahan Garg (2410994805)
// Compile : g++ -std=c++11 -O2 -pthread parallel_pthread.cpp -o parallel_pthread
// Run     : ./parallel_pthread <num_threads>
//           Example: ./parallel_pthread 4
//
// Parallelisation Strategy: Output Data Decomposition
//   v3[i] = v1[i] + v2[i] — each element is computed independently.
//   The output vector v3 is divided into N non-overlapping partitions,
//   one per thread. Threads write to disjoint memory regions, so no
//   mutex or synchronisation is required during computation.
//   A pthread_join barrier ensures all threads finish before timing stops.

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <pthread.h>
#include <ctime>

using namespace std;
using namespace std::chrono;

// Shared input and output vectors — accessible by all threads
// v1 and v2 are read-only after initialisation; v3 is write-partitioned
int          *v1, *v2, *v3;
unsigned long totalSize;

// Bundles per-thread work range into a single struct for pthread_create
struct ThreadArgs
{
    int           threadId; // Logical thread ID for debugging
    unsigned long start;    // Inclusive start index of this thread's partition
    unsigned long end;      // Exclusive end index of this thread's partition
};

// Thread worker function: computes v3[i] = v1[i] + v2[i] for [start, end)
// No race condition — each thread writes exclusively to its assigned slice
void *vectorAddChunk(void *arg)
{
    ThreadArgs *data = (ThreadArgs *)arg;

    for (unsigned long i = data->start; i < data->end; i++)
        v3[i] = v1[i] + v2[i];

    pthread_exit(NULL);
    return NULL;
}

// Fills an integer array with pseudo-random values in [0, 99]
void randomVector(int vector[], unsigned long size)
{
    for (unsigned long i = 0; i < size; i++)
        vector[i] = rand() % 100;
}

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        cout << "Usage: " << argv[0] << " <number_of_threads>" << endl;
        return 1;
    }

    int numThreads = atoi(argv[1]);
    if (numThreads <= 0)
    {
        cerr << "Error: Thread count must be a positive integer." << endl;
        return 1;
    }

    totalSize = 100000000;
    srand((unsigned)time(0));

    v1 = (int *)malloc(totalSize * sizeof(int));
    v2 = (int *)malloc(totalSize * sizeof(int));
    v3 = (int *)malloc(totalSize * sizeof(int));

    if (!v1 || !v2 || !v3)
    {
        cerr << "Error: Memory allocation failed." << endl;
        return 1;
    }

    // Initialise input vectors sequentially before the timed parallel phase
    randomVector(v1, totalSize);
    randomVector(v2, totalSize);

    // Compute base partition size; last thread absorbs integer-division remainder
    unsigned long chunk     = totalSize / numThreads;
    unsigned long remainder = totalSize % numThreads;

    pthread_t  *threads = new pthread_t[numThreads];
    ThreadArgs *args    = new ThreadArgs[numThreads];

    // Begin timer — covers parallel computation only
    auto start = high_resolution_clock::now();

    // Spawn one thread per data partition
    for (int t = 0; t < numThreads; t++)
    {
        args[t].threadId = t;
        args[t].start    = (unsigned long)t * chunk;
        args[t].end      = args[t].start + chunk;

        // Last thread absorbs any remainder to guarantee full coverage
        if (t == numThreads - 1)
            args[t].end += remainder;

        int rc = pthread_create(&threads[t], NULL, vectorAddChunk, (void *)&args[t]);
        if (rc)
        {
            cerr << "Error: pthread_create failed for thread " << t << endl;
            return 1;
        }
    }

    // Barrier: main thread blocks until all worker threads complete
    for (int t = 0; t < numThreads; t++)
        pthread_join(threads[t], NULL);

    // End timer after all threads have joined
    auto stop     = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);

    cout << "[pthread] Threads used : " << numThreads << endl;
    cout << "[pthread] Time taken   : " << duration.count() << " microseconds" << endl;

    // Spot-check first 10 elements to verify correctness
    bool correct = true;
    for (int i = 0; i < 10; i++)
        if (v3[i] != v1[i] + v2[i]) { correct = false; break; }

    cout << "[pthread] Verification : " << (correct ? "PASS" : "FAIL") << endl;

    free(v1); free(v2); free(v3);
    delete[] threads;
    delete[] args;

    return 0;
}
