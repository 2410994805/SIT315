# SIT315 — M2.S2P: Parallel Vector Addition
**Author:** Jahan Garg | **Roll:** 2410994805

## Files
| File | Description |
|------|-------------|
| `sequential.cpp` | Baseline sequential vector addition |
| `parallel_pthread.cpp` | Parallel version using POSIX pthreads |
| `parallel_stdthread.cpp` | Parallel version using C++11 std::thread |
| `performance_test.bat` | Windows automated compile + benchmark script |

## Compile & Run
```bash
# Sequential
g++ -std=c++11 -O2 sequential.cpp -o sequential
./sequential

# pthread
g++ -std=c++11 -O2 -pthread parallel_pthread.cpp -o parallel_pthread
./parallel_pthread 4

# std::thread
g++ -std=c++11 -O2 parallel_stdthread.cpp -o parallel_stdthread
./parallel_stdthread 4
```

## Performance Summary (5M elements, 3-run mean)
| Threads | pthread (µs) | std::thread (µs) | Speedup | Efficiency |
|---------|-------------|------------------|---------|------------|
| 1       | 10,250      | 10,283           | 0.81×   | 81%        |
| 2       | 5,583       | 5,626            | 1.50×   | 75%        |
| 4       | 5,937       | 5,958            | 1.41×   | 35%        |
| 8       | 6,827       | 6,648            | 1.22×   | 15%        |

Sequential baseline: **8,352 µs**. Peak speedup at **2 threads (1.50×)** — limited by memory bandwidth saturation.
