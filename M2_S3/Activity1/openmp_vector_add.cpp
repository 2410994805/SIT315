#include <chrono>
#include <cstdlib>
#include <iostream>
#include <time.h>
#include <omp.h> // OpenMP header

using namespace std;
using namespace std::chrono;

// Initialises the given array with random integers in range [0, 99]
void randomVector(int vector[], unsigned long size)
{
    for (unsigned long i = 0; i < size; i++)
    {
        vector[i] = rand() % 100;
    }
}

int main()
{
    unsigned long size = 100000000; // Total number of elements
    srand(time(0));

    int *v1 = (int *)malloc(size * sizeof(int));
    int *v2 = (int *)malloc(size * sizeof(int));
    int *v3 = (int *)malloc(size * sizeof(int));

    if (!v1 || !v2 || !v3)
    {
        cerr << "Memory allocation failed." << endl;
        return 1;
    }

    randomVector(v1, size);
    randomVector(v2, size);

    auto start = high_resolution_clock::now();

// Parallel region with a work-sharing for loop
#pragma omp parallel
    {
#pragma omp for
        for (unsigned long i = 0; i < size; i++)
        {
            v3[i] = v1[i] + v2[i];
        }
    }

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);

    cout << "Threads used: " << omp_get_max_threads() << endl;
    cout << "Time taken by OpenMP vector addition: "
         << duration.count() << " microseconds" << endl;

    free(v1);
    free(v2);
    free(v3);
    return 0;
}