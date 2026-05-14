// FILE: traffic_simulator.cpp
// TASK: M2.T3D - Traffic Control Simulator (Distinction)
// UNIT: SIT315 Programming Paradigms, Trimester 4, 2024-25
// AUTHOR: Jahan Garg | Roll: 2410994805
//
// Implements a multi-threaded traffic signal monitoring system using the
// classical bounded-buffer producer-consumer pattern with std::thread.
// Producer threads read sensor records from a CSV file and push them into
// a shared bounded buffer. Consumer threads drain the buffer and compute
// per-hour Top-N congestion rankings for each traffic light.
//
// Compile: g++ -std=c++11 -O2 -pthread traffic_simulator.cpp -o sim
// Run    : ./sim <csv_file> <producers> <consumers> <buf_size> <top_n>
// Example: ./sim traffic_large.csv 4 2 500 5

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <cstdlib>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <climits>

using namespace std;
using namespace std::chrono;

// Represents one sensor reading from the CSV input file.
// Treated as an immutable value: created by the producer, copied into
// the buffer, then read by the consumer. No synchronisation is needed
// on this struct because no thread modifies a record after creation.
struct TrafficRecord {
    string timestamp;    // time of reading in HH:MM format (time-only, no date)
    string lightID;      // traffic light identifier, e.g. "TL001"
    int    cars;         // number of cars counted in this 5-minute window

    // Extracts the hour (0-23) from the timestamp for per-hour bucketing.
    int getHour() const { return stoi(timestamp.substr(0, 2)); }
};

// Parses one comma-separated line from the CSV into a TrafficRecord.
// Called by producers after releasing fileMutex so all CPU-bound parsing
// runs in parallel across threads. Keeping the critical section to just
// getline() is the key Amdahl's Law technique: a shorter serial fraction
// means more of the work scales with thread count.
TrafficRecord parseLine(const string& line) {
    TrafficRecord r;
    stringstream  ss(line);
    string        part;
    getline(ss, r.timestamp, ',');   // column 1: HH:MM
    getline(ss, r.lightID,   ',');   // column 2: TLXXX
    getline(ss, part,        ',');   // column 3: car count as string
    r.cars = stoi(part);
    return r;
}

// Thread-safe FIFO queue with a fixed capacity limit.
// Models the bounded-buffer problem from SIT315 Module 2 using a mutex
// and two condition variables to coordinate producers and consumers
// without busy-waiting (spinning on a flag in a loop).
//
// Key synchronisation primitives used:
//   std::mutex              - mutual exclusion: only one thread holds it at a time.
//   std::condition_variable - lets a thread sleep and be woken by another thread.
//   unique_lock<mutex>      - RAII lock that can be atomically released by wait().
//   wait(lock, predicate)   - releases the lock, suspends the thread, and
//                             re-acquires it only when the predicate is true.
//                             The predicate also guards against spurious wakeups,
//                             which is an OS behaviour where a thread can wake
//                             without a corresponding notify() call. Without the
//                             predicate, a spurious wakeup could push or pop on
//                             a full or empty buffer respectively.
//   notify_one()            - wakes exactly one thread waiting on this CV.
//   notify_all()            - wakes all waiting threads, used on shutdown.
template<typename T>
class BoundedBuffer {
    queue<T>            buf;       // underlying FIFO container
    const size_t        capacity;  // maximum items allowed simultaneously
    mutex               mtx;       // protects buf, done, and size checks
    condition_variable  not_full;  // producer sleeps here when buf is at capacity
    condition_variable  not_empty; // consumer sleeps here when buf is empty
    bool                done;      // end-of-stream flag set by main

public:
    explicit BoundedBuffer(size_t cap) : capacity(cap), done(false) {}

    // Adds one item to the queue. Blocks if the buffer is full until a
    // consumer pops a record and signals not_full.
    void push(const T& item) {
        unique_lock<mutex> lock(mtx);
        not_full.wait(lock, [this]{ return buf.size() < capacity; });
        buf.push(item);
        not_empty.notify_one();    // wake one sleeping consumer
    }

    // Removes one item into 'out'. Blocks if empty until a producer pushes
    // or done is set. Returns false when empty AND done is true, which is
    // the end-of-stream signal for a consumer to exit its loop.
    bool pop(T& out) {
        unique_lock<mutex> lock(mtx);
        not_empty.wait(lock, [this]{ return !buf.empty() || done; });
        if (buf.empty()) return false;
        out = buf.front();
        buf.pop();
        not_full.notify_one();     // wake one sleeping producer
        return true;
    }

    // Called by main after all producers have joined. Sets the done flag
    // and wakes every sleeping consumer so each can check empty+done
    // and exit cleanly.
    void setDone() {
        unique_lock<mutex> lock(mtx);
        done = true;
        not_empty.notify_all();
    }
};

// Accumulates running statistics for one (hour, lightID) pair.
// Uses Welford's online algorithm to compute mean and variance in a
// single pass without storing all values. This is more numerically
// stable than (sum_sq/n) - mean^2 and avoids floating-point overflow
// for large datasets. update() is called inside a mutex-protected
// addRecord() so no internal synchronisation is needed here.
struct SignalStats {
    long long total = 0;        // cumulative car count across all readings
    int       count = 0;        // number of 5-minute readings recorded
    int       maxV  = 0;        // peak cars in a single reading
    int       minV  = INT_MAX;  // minimum cars in a single reading
    double    mean  = 0.0;      // running mean (Welford's algorithm)
    double    M2    = 0.0;      // running sum of squared deviations

    // Incorporates one new car-count observation using Welford's method.
    void update(int v) {
        total += v;
        count++;
        if (v > maxV) maxV = v;
        if (v < minV) minV = v;
        double delta = v - mean;         // deviation from current mean
        mean        += delta / count;    // update running mean
        M2          += delta * (v - mean); // update sum of squared deviations
    }

    // Returns the population standard deviation of recorded car counts.
    double stddev() const {
        return (count > 1) ? sqrt(M2 / count) : 0.0;
    }
};

// Stores and manages per-(hour, lightID) statistics across all consumers.
// All public methods are thread-safe. lock_guard<mutex> is a simpler RAII
// lock than unique_lock: it holds the mutex for the full scope and releases
// automatically on exit, even if an exception is thrown.
class CongestionTracker {
    map<pair<int,string>, SignalStats> stats;   // keyed by (hour, lightID)
    mutex tmtx;

public:
    // Thread-safe update for the record's (hour, lightID) pair.
    void addRecord(const TrafficRecord& r) {
        lock_guard<mutex> lock(tmtx);
        stats[{r.getHour(), r.lightID}].update(r.cars);
    }

    // Returns the count of distinct (hour, lightID) pairs processed.
    int uniquePairs() {
        lock_guard<mutex> lock(tmtx);
        return static_cast<int>(stats.size());
    }

    // Groups all entries by hour, sorts by total cars descending per hour,
    // and prints the top-N result. Called after all threads join so no
    // lock is needed (single-threaded output phase).
    void printTopN(int N) {
        map<int, vector<pair<string,long long>>> byHour;
        for (auto& kv : stats)
            byHour[kv.first.first].emplace_back(kv.first.second,
                                                kv.second.total);
        cout << "\nTop-" << N << " congested traffic lights per hour:\n";
        for (auto& hv : byHour) {
            auto& lights = hv.second;
            sort(lights.begin(), lights.end(),
                 [](const pair<string,long long>& a,
                    const pair<string,long long>& b){
                     return a.second > b.second;
                 });
            cout << "Hour " << setw(2) << setfill('0') << hv.first << ":00  ";
            int cnt = min(N, (int)lights.size());
            for (int i = 0; i < cnt; i++) {
                cout << (i+1) << ". " << lights[i].first
                     << " [" << lights[i].second << " cars]";
                if (i < cnt-1) cout << "  ";
            }
            cout << "\n";
        }
    }
};

// Producer thread: reads one line at a time from the shared input file
// under fileMutex, then releases the lock before parsing. Keeping the
// critical section to just getline() lets multiple producers parse and
// push records simultaneously, reducing the Amdahl serial fraction.
void producerThread(BoundedBuffer<TrafficRecord>& buffer,
                    ifstream& file, mutex& fileMutex) {
    while (true) {
        string line;
        {
            lock_guard<mutex> lk(fileMutex);    // hold only for the file read
            if (!getline(file, line)) break;    // EOF: exit cleanly
        }                                       // fileMutex released here
        if (line.empty()) continue;
        buffer.push(parseLine(line));           // parse outside the lock (parallel)
    }
}

// Consumer thread: repeatedly pops records from the buffer and updates
// the tracker. Exits when pop() returns false (buffer empty AND done set).
void consumerThread(BoundedBuffer<TrafficRecord>& buffer,
                    CongestionTracker& tracker) {
    TrafficRecord r;
    while (buffer.pop(r))
        tracker.addRecord(r);
}

// Orchestrates the full pipeline. Thread launch and join order is critical:
//   1. Start consumers first so they are ready to drain before data arrives.
//   2. Start producers to begin reading and pushing.
//   3. BARRIER 1: join all producers before calling setDone().
//      If setDone() fires while a producer is still mid-push, a consumer
//      can see empty+done and exit early, silently dropping records.
//   4. setDone() sends the end-of-stream signal to all consumers.
//   5. BARRIER 2: join all consumers before printing results to guarantee
//      every addRecord() call has completed before printTopN() reads data.
int main(int argc, char* argv[]) {
    if (argc < 6) {
        cerr << "Usage: ./sim <file> <producers> <consumers> <buf_size> <top_n>\n";
        cerr << "Example: ./sim traffic_large.csv 4 2 500 5\n";
        return 1;
    }

    string filename = argv[1];
    int numP = atoi(argv[2]);    // number of producer threads
    int numC = atoi(argv[3]);    // number of consumer threads
    int bufB = atoi(argv[4]);    // bounded buffer capacity
    int topN = atoi(argv[5]);    // top-N results to display per hour

    if (numP < 1 || numC < 1 || bufB < 1 || topN < 1) {
        cerr << "Error: all numeric arguments must be >= 1\n";
        return 1;
    }

    ifstream file(filename);
    if (!file) {
        cerr << "Error: cannot open file: " << filename << "\n";
        return 1;
    }

    cout << "Traffic Control Simulator  |  M2.T3D\n";
    cout << "Jahan Garg  |  Roll: 2410994805\n";
    cout << "File: " << filename
         << "  P=" << numP << "  C=" << numC
         << "  B=" << bufB << "  N=" << topN << "\n\n";

    BoundedBuffer<TrafficRecord> buffer(bufB);
    CongestionTracker tracker;
    mutex fileMutex;

    auto t0 = high_resolution_clock::now();

    vector<thread> consumers, producers;
    for (int i = 0; i < numC; i++)
        consumers.emplace_back(consumerThread, ref(buffer), ref(tracker));

    for (int i = 0; i < numP; i++)
        producers.emplace_back(producerThread, ref(buffer),
                               ref(file), ref(fileMutex));

    for (auto& p : producers) p.join();   // BARRIER 1: all records pushed
    buffer.setDone();                     // signal end-of-stream
    for (auto& c : consumers) c.join();   // BARRIER 2: all records processed

    auto t1 = high_resolution_clock::now();
    long long us = duration_cast<microseconds>(t1 - t0).count();

    cout << "Unique (hour, light) pairs : " << tracker.uniquePairs() << "\n";
    cout << "Execution time             : " << us << " us  ("
         << fixed << setprecision(3) << us / 1000.0 << " ms)\n";

    tracker.printTopN(topN);
    return 0;
}
