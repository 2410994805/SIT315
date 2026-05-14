# M2.T3D — Traffic Control Simulator
**SIT315 Programming Paradigms | Semester-4**  
**Jahan Garg | Roll: 2410994805**

## Overview
A multi-threaded traffic signal monitoring system built in C++11 using the
bounded-buffer producer-consumer pattern with `std::thread`. Producer threads
read sensor records from a CSV file into a shared bounded buffer. Consumer
threads drain the buffer and compute per-hour Top-N congestion rankings for
each traffic light.

## Files
| File | Description |
|------|-------------|
| `traffic_simulator.cpp` | Main simulator — bounded-buffer producer-consumer |
| `generate_data.cpp` | Deterministic CSV input generator (`srand(0)`) |
| `traffic_large.csv` | Sample input: 50 signals × 24 hours = 14,400 records |

## Compile & Run
```bash
# Compile
g++ -std=c++11 -O2 -pthread traffic_simulator.cpp -o sim
g++ -std=c++11 -O2 generate_data.cpp -o gen

# Generate input files
./gen 50  24  traffic_large.csv
./gen 100 48  traffic_xlarge.csv
./gen 200 168 traffic_stress.csv

# Sequential baseline (P=1, C=1)
./sim traffic_large.csv 1 1 99999 5

# Optimal parallel (P=4 producers, C=2 consumers, buffer=500)
./sim traffic_large.csv 4 2 500 5

# Stress test
./sim traffic_stress.csv 4 2 500 5
```

## CSV Format
```
HH:MM,TLXXX,cars
00:00,TL001,70
00:05,TL001,145
```
- `HH:MM` — time-only timestamp (no date), 5-minute intervals
- `TLXXX` — traffic light ID (TL001 to TL200)
- `cars` — vehicle count per 5-minute window [20, 200]

## Thread Architecture
```
Producer Threads (P) --> BoundedBuffer --> Consumer Threads (C)
     fileMutex                               trackerMutex
```

## Key Synchronisation Primitives
| Primitive | Purpose |
|-----------|--------|
| `std::mutex` | Mutual exclusion — one thread at a time |
| `condition_variable` | Sleep/wake without busy-waiting |
| `unique_lock` | RAII lock releasable by `wait()` |
| `lock_guard` | Simpler RAII lock for full-scope use |
| `wait(lock, pred)` | Atomic release + sleep; predicate guards spurious wakeups |
| `notify_one/all` | Wake one or all waiting threads |

## Benchmark Results (B=500)
| Workload | Sequential | P=4, C=2 | Speedup |
|----------|-----------|---------|--------|
| Large (14,400 records) | 15.10 ms | 11.92 ms | 1.27× |
| XLarge (57,600 records) | 47.57 ms | 39.98 ms | 1.19× |
| Stress (403,200 records) | ~373 ms | ~140 ms | ~2.66× |
