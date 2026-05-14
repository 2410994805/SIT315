# M2.T3D — Traffic Control Simulator
**SIT315 Programming Paradigms | Trimester 4, 2024-25**  
**Jahan Garg | Roll: 2410994805**

## Overview
Multi-threaded bounded-buffer producer-consumer traffic simulator in C++11.  
Identifies Top-N most congested intersections per hour from sensor CSV data.

## Files
| File | Description |
|------|-------------|
| `traffic_simulator.cpp` | Main simulator — fully commented, line-by-line |
| `generate_data.cpp` | Deterministic CSV input generator (srand=0) |
| `traffic_large.csv` | Sample input: 50 signals × 24 hours = 14,400 records |

## Compile & Run
```bash
# Compile simulator
g++ -std=c++11 -O2 -pthread traffic_simulator.cpp -o sim

# Compile generator
g++ -std=c++11 -O2 generate_data.cpp -o gen

# Generate input files
./gen 50  24  traffic_large.csv
./gen 100 48  traffic_xlarge.csv
./gen 200 168 traffic_stress.csv

# Run — sequential
./sim traffic_large.csv 1 1 99999 5

# Run — optimal parallel (P=4 producers, C=2 consumers, buffer=500)
./sim traffic_large.csv 4 2 500 5

# Run — stress test
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
     fileMutex                                trackerMutex
```

## Key Results (B=500, verified benchmarks)
| Workload | Sequential | P=4,C=2 | Speedup |
|----------|-----------|---------|--------|
| Large (14,400) | 15.10 ms | 11.92 ms | 1.267× |
| XLarge (57,600) | 47.57 ms | 39.98 ms | 1.190× |
| Stress (403,200) | ~373 ms | ~140 ms | ~2.66× |
