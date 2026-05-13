// SIT315 Programming Paradigms
// Task    : M2.S2P - Activity 2 | Sequential Vector Addition (Baseline)
// Author  : Jahan Garg (2410994805)
// Compile : g++ -std=c++11 -O2 sequential.cpp -o sequential
// Run     : ./sequential

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <ctime>

using namespace std;
using namespace std::chrono;

// Fills an integer array with pseudo-random values in [0, 99]
void randomVector(int vector[], unsigned long size)
{
    for (unsigned long i = 0; i < size; i++)
        vector[i] = rand() % 100;
}

int main()
{
    // Problem size: 100 million elements per vector (~400 MB each)
    unsigned long size = 100000000;

    // Seed RNG with current time for non-deterministic inputs
    srand((unsigned)time(0));

    // Three heap-allocated integer vectors: v1, v2 (inputs), v3 (result)
    int *v1, *v2, *v3;

    v1 = (int *)malloc(size * sizeof(int));
    v2 = (int *)malloc(size * sizeof(int));
    v3 = (int *)malloc(size * sizeof(int));

    // Guard against allocation failure on memory-constrained systems
    if (!v1 || !v2 || !v3)
    {
        cerr << "Error: Memory allocation failed." << endl;
        return 1;
    }

    // Populate input vectors before the timed computation section
    randomVector(v1, size);
    randomVector(v2, size);

    // Begin high-resolution timer — measures addition loop only
    auto start = high_resolution_clock::now();

    // Sequential addition: v3[i] = v1[i] + v2[i] for every index i
    // Each iteration is fully independent — zero loop-carried dependency
    // This loop is the ideal target for data parallelism
    for (unsigned long i = 0; i < size; i++)
        v3[i] = v1[i] + v2[i];

    // End timer and compute elapsed duration
    auto stop     = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);

    cout << "[Sequential] Threads used : 1" << endl;
    cout << "[Sequential] Time taken   : " << duration.count() << " microseconds" << endl;

    // Release all heap memory
    free(v1);
    free(v2);
    free(v3);

    return 0;
}
