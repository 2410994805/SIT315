#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <vector>
#include <ctime>

using namespace std;
using namespace std::chrono;

void randomVector(int vector[], unsigned long size)
{
    for (unsigned long i = 0; i < size; i++)
    {
        vector[i] = rand() % 100;
    }
}

void addChunk(int *v1, int *v2, int *v3, unsigned long start, unsigned long end)
{
    for (unsigned long i = start; i < end; i++)
    {
        v3[i] = v1[i] + v2[i];
    }
}

int main()
{
    unsigned long size = 100000000;
    srand((unsigned)time(0));

    int *v1, *v2, *v3;
    v1 = (int *)malloc(size * sizeof(int));
    v2 = (int *)malloc(size * sizeof(int));
    v3 = (int *)malloc(size * sizeof(int));

    if (!v1 || !v2 || !v3)
    {
        cerr << "Memory allocation failed" << endl;
        return 1;
    }

    randomVector(v1, size);
    randomVector(v2, size);

    unsigned int numThreads = thread::hardware_concurrency();
    if (numThreads == 0)
        numThreads = 2;

    vector<thread> workers;
    workers.reserve(numThreads);

    unsigned long chunk = size / numThreads;

    auto start = high_resolution_clock::now();

    for (unsigned int t = 0; t < numThreads; t++)
    {
        unsigned long s = t * chunk;
        unsigned long e = (t == numThreads - 1) ? size : s + chunk;
        workers.emplace_back(addChunk, v1, v2, v3, s, e);
    }

    for (auto &th : workers)
    {
        th.join();
    }

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);

    cout << "Threads used: " << numThreads << endl;
    cout << "Time taken by parallel addition: " << duration.count() << " microseconds" << endl;

    free(v1);
    free(v2);
    free(v3);

    return 0;
}