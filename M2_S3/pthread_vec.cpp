#include <chrono>
#include <cstdlib>
#include <iostream>
#include <pthread.h>
#include <time.h>

using namespace std::chrono;
using namespace std;

#define NUM_THREADS 8

struct ThreadData
{
    int *v1;
    int *v2;
    int *v3;
    unsigned long start;
    unsigned long end;
};

void *addChunk(void *arg)
{
    ThreadData *data = (ThreadData *)arg;
    for (unsigned long i = data->start; i < data->end; i++)
    {
        data->v3[i] = data->v1[i] + data->v2[i];
    }
    return NULL;
}

void randomVector(int vector[], unsigned long size)
{
    for (unsigned long i = 0; i < size; i++)
    {
        vector[i] = rand() % 100;
    }
}

int main()
{
    unsigned long size = 100000000;
    srand(time(0));

    int *v1 = (int *)malloc(size * sizeof(int));
    int *v2 = (int *)malloc(size * sizeof(int));
    int *v3 = (int *)malloc(size * sizeof(int));

    randomVector(v1, size);
    randomVector(v2, size);

    pthread_t threads[NUM_THREADS];
    ThreadData threadData[NUM_THREADS];

    auto start = high_resolution_clock::now();

    unsigned long chunk = size / NUM_THREADS;
    for (int i = 0; i < NUM_THREADS; i++)
    {
        threadData[i].v1 = v1;
        threadData[i].v2 = v2;
        threadData[i].v3 = v3;
        threadData[i].start = i * chunk;
        threadData[i].end = (i == NUM_THREADS - 1) ? size : (i + 1) * chunk;

        pthread_create(&threads[i], NULL, addChunk, (void *)&threadData[i]);
    }

    for (int i = 0; i < NUM_THREADS; i++)
    {
        pthread_join(threads[i], NULL);
    }

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);

    cout << "Time taken by pthread vector addition: "
         << duration.count() << " microseconds" << endl;

    free(v1);
    free(v2);
    free(v3);
    return 0;
}
