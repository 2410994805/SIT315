# M2.T1P — Parallel Matrix Multiplication

**Unit:** SIT315 Programming Paradigms — Trimester 4, 2024-25  
**Author:** Jahan Garg | Roll: 2410994805

---

## Overview

This task implements and benchmarks **three approaches** to N×N matrix multiplication (C = A × B):

| File | Method | Parallelism |
|------|--------|-------------|
| `sequential.cpp` | Triple nested loop | None — baseline |
| `parallel_thread.cpp` | Row-band partitioning | C++11 `std::thread` |
| `parallel_openmp.cpp` | collapse(2) loop fusion | OpenMP `#pragma omp parallel for` |

All three use the same fixed seed `srand(0)` so checksums can be compared for correctness.

---

## Compile and Run

```bash
# Sequential
g++ -std=c++11 -O2 sequential.cpp       -o sequential
./sequential 512

# std::thread
g++ -std=c++11 -O2 parallel_thread.cpp  -o parallel_thread
./parallel_thread 512 4

# OpenMP
g++ -std=c++11 -O2 -fopenmp parallel_openmp.cpp -o parallel_openmp
./parallel_openmp 512 4
```

---

## Correctness Verification

All three programs write their result matrix to a text file. Matching checksums (printed to console) and identical files confirm correct parallelisation:

```bash
diff C_seq.txt C_thread.txt    # expect: no output
diff C_seq.txt C_openmp.txt    # expect: no output
```

---

## Verified Benchmark Results (mean of 3 runs, Ubuntu 22.04 sandbox)

### Sequential Baselines

| N | Time (µs) | Time (ms) | Checksum |
|---|-----------|-----------|----------|
| 128 | 1,600 | 1.60 | 5,145,407,666 |
| 256 | 13,239 | 13.24 | 41,096,351,553 |
| 512 | 129,277 | 129.28 | 328,794,973,398 |

### Speedup Summary (N = 512)

| Threads | std::thread Time (µs) | std::thread Speedup | OpenMP Time (µs) | OpenMP Speedup |
|---------|-----------------------|---------------------|------------------|----------------|
| 1 | 135,393 | 0.95x | 146,130 | 0.88x |
| 2 | 85,058 | 1.52x | 99,550 | 1.30x |
| 4 | 83,366 | 1.55x | 77,565 | 1.67x |
| 8 | 80,754 | 1.60x | 71,790 | **1.80x** |

> Peak speedup: **OpenMP at T=8 → 1.80x**  
> OpenMP outperforms std::thread at higher thread counts due to thread pool reuse and finer task granularity via `collapse(2)`.

---

## Key Concepts

- **Output Data Decomposition**: each C[i][j] is independent — no synchronisation required during computation
- **Row Partitioning** (std::thread): N rows split into T bands; `join()` acts as barrier
- **collapse(2)** (OpenMP): fuses i×j loops into N² independent tasks for better load balance
- **Amdahl's Law**: theoretical max speedup at T=4 with 5% sequential fraction = 3.48x; actual speedup limited by memory bandwidth and thread overhead

---

## Stress Test

Sequential at N=1500: **4,274,537 µs (4.3 seconds)**, checksum = 8,272,462,251,235.  
Parallel T=8 on 2-core sandbox: DNF (over-subscribed). On a 4-core machine, estimated ~2.7 seconds at T=4.
