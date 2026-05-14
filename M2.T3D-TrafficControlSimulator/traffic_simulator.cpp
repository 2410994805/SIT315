// ============================================================
// FILE    : traffic_simulator.cpp
// TASK    : M2.T3D — Traffic Control Simulator (Distinction)
// UNIT    : SIT315 Programming Paradigms, Trimester 4, 2024-25
// AUTHOR  : Jahan Garg  |  Roll: 2410994805
// COMPILE : g++ -std=c++11 -O2 -pthread traffic_simulator.cpp -o sim
// RUN     : ./sim <csv_file> <producers> <consumers> <buf_size> <top_n>
// EXAMPLE : ./sim traffic_large.csv 4 2 500 5
//
// DESIGN  : Bounded-buffer producer-consumer pattern using std::thread.
//   Producers : read CSV lines → parse → push into BoundedBuffer
//   Buffer    : std::mutex + 2 std::condition_variable; blocks on full/empty
//   Consumers : pop records → update CongestionTracker (Welford statistics)
//   Main      : launch → join producers → setDone → join consumers → print
// ============================================================

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
#include <numeric>
#include <cmath>
#include <climits>

using namespace std;
using namespace std::chrono;

// ─────────────────────────────────────────────────────────────────────────────
// TrafficRecord: immutable value type representing one sensor reading.
// Copied into the buffer queue — no synchronisation required on this struct.
// ─────────────────────────────────────────────────────────────────────────────
struct TrafficRecord {
    string timestamp;    // "HH:MM" — time-only (no date), as per tutor guidance
    string lightID;      // e.g. "TL001" to "TL200"
    int    cars;         // cars counted in this 5-minute window [20, 200]

    // Extract hour integer (00..23) used for hourly Top-N aggregation
    int getHour() const { return stoi(timestamp.substr(0, 2)); }
};

// ─────────────────────────────────────────────────────────────────────────────
// parseLine: parse one CSV line into a TrafficRecord.
// Called by producers AFTER releasing fileMutex, so parsing runs in parallel.
// Keeping the critical section (file read) separate from CPU work (parsing)
// is the core Amdahl's Law technique: shorter serial fraction = more speedup.
// ─────────────────────────────────────────────────────────────────────────────
TrafficRecord parseLine(const string& line) {
    TrafficRecord r;
    stringstream  ss(line);
    string        part;
    getline(ss, r.timestamp, ',');   // column 1: "HH:MM"
    getline(ss, r.lightID,   ',');   // column 2: "TLXXX"
    getline(ss, part,        ',');   // column 3: car count as string
    r.cars = stoi(part);             // convert string to int
    return r;
}

// ─────────────────────────────────────────────────────────────────────────────
// BoundedBuffer<T>: thread-safe FIFO queue with fixed capacity.
//
// Key std::thread primitives used:
//   std::mutex              — mutual exclusion lock: only one thread at a time
//   std::condition_variable — allows a thread to sleep and be woken by another
//   unique_lock<mutex>      — RAII lock that can be temporarily released
//   wait(lock, predicate)   — atomically release lock + sleep until predicate true
//   notify_one()            — wake exactly one waiting thread
//   notify_all()            — wake ALL waiting threads (used for shutdown)
//
// The lambda predicate in wait() guards against SPURIOUS WAKEUPS — a POSIX
// behaviour where a sleeping thread may wake without a notify() call. The
// predicate re-checks the condition and re-sleeps if not truly satisfied.
// ─────────────────────────────────────────────────────────────────────────────
template<typename T>
class BoundedBuffer {
    queue<T>            buf;       // underlying FIFO container
    const size_t        capacity;  // maximum items allowed simultaneously
    mutex               mtx;       // protects buf, done, and size checks
    condition_variable  not_full;  // producer waits here when buf is at capacity
    condition_variable  not_empty; // consumer waits here when buf is empty
    bool                done;      // end-of-stream flag (set by main)

public:
    explicit BoundedBuffer(size_t cap) : capacity(cap), done(false) {}

    // push: add one item; BLOCKS if buffer is full (natural backpressure)
    void push(const T& item) {
        unique_lock<mutex> lock(mtx);              // acquire exclusive lock
        not_full.wait(lock, [this]{                // sleep while buf is full
            return buf.size() < capacity;          // wake only when space exists
        });
        buf.push(item);                            // enqueue the item
        not_empty.notify_one();                    // wake one sleeping consumer
    }                                              // lock auto-released (RAII)

    // pop: remove one item; BLOCKS when empty; returns false at end-of-stream
    bool pop(T& out) {
        unique_lock<mutex> lock(mtx);              // acquire exclusive lock
        not_empty.wait(lock, [this]{               // sleep while empty AND active
            return !buf.empty() || done;
        });
        if (buf.empty()) return false;             // end-of-stream: exit signal
        out = buf.front();                         // copy the front record
        buf.pop();                                 // remove from queue
        not_full.notify_one();                     // wake one sleeping producer
        return true;
    }

    // setDone: no more data will arrive; wake ALL consumers to exit cleanly
    void setDone() {
        unique_lock<mutex> lock(mtx);
        done = true;
        not_empty.notify_all();   // every blocked consumer must re-check
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// SignalStats: running statistics for one (hour, lightID) pair.
// Uses Welford's online algorithm to compute mean and variance incrementally.
// This makes consumer work O(1) per record and genuinely computation-heavy
// enough to demonstrate parallel speedup at larger workloads.
// ─────────────────────────────────────────────────────────────────────────────
struct SignalStats {
    long long total = 0;        // cumulative car count across all readings
    int       count = 0;        // number of 5-minute readings recorded
    int       maxV  = 0;        // peak cars in a single reading
    int       minV  = INT_MAX;  // minimum cars in a single reading
    double    mean  = 0.0;      // running mean (Welford online algorithm)
    double    M2    = 0.0;      // sum of squared deviations (Welford)

    // update: incorporate one new observation using Welford's method
    void update(int v) {
        total += v;                        // accumulate total
        count++;                           // increment reading count
        if (v > maxV) maxV = v;            // track peak
        if (v < minV) minV = v;            // track minimum
        double delta = v - mean;           // deviation from current mean
        mean        += delta / count;      // update running mean
        M2          += delta * (v - mean); // update sum of squared deviations
    }

    // stddev: return population standard deviation
    double stddev() const {
        return (count > 1) ? sqrt(M2 / count) : 0.0;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// CongestionTracker: aggregates per-(hour, lightID) statistics across threads.
// trackerMutex must be held for all map accesses because multiple consumers
// call addRecord() concurrently — classic shared-state concurrency pattern.
// ─────────────────────────────────────────────────────────────────────────────
class CongestionTracker {
    map<pair<int,string>, SignalStats> stats; // (hour, lightID) -> stats
    mutex tmtx;                               // protects stats map

public:
    // addRecord: thread-safe update for (hour, lightID) pair
    void addRecord(const TrafficRecord& r) {
        lock_guard<mutex> lock(tmtx);                        // auto-release on scope exit
        stats[{r.getHour(), r.lightID}].update(r.cars);     // O(log n) map lookup + update
    }

    // printTopN: sort per-hour by total cars (desc), print top-N.
    // Called only after all threads join (single-threaded — no lock needed).
    void printTopN(int N) {
        map<int, vector<pair<string,long long>>> byHour;
        for (auto& kv : stats)                               // group by hour
            byHour[kv.first.first].emplace_back(kv.first.second, kv.second.total);

        cout << "\n=== Top-" << N << " Congested Traffic Lights Per Hour ===\n";
        for (auto& hv : byHour) {
            auto& lights = hv.second;
            sort(lights.begin(), lights.end(),               // sort descending
                 [](const pair<string,long long>& a,
                    const pair<string,long long>& b){
                     return a.second > b.second;
                 });
            cout << "Hour " << setw(2) << setfill('0') << hv.first << ":00 | ";
            int cnt = min(N, (int)lights.size());
            for (int i = 0; i < cnt; i++) {
                cout << (i+1) << ". " << lights[i].first
                     << " [" << lights[i].second << "]";
                if (i < cnt-1) cout << "  ";
            }
            cout << "\n";
        }
    }

    // uniquePairs: return count of distinct (hour, lightID) pairs processed
    int uniquePairs() {
        lock_guard<mutex> l(tmtx);
        return static_cast<int>(stats.size());
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// producerThread: reads lines from shared file (critical section = getline only),
// then parses and pushes outside the lock — maximising parallel throughput.
// Multiple producers share one ifstream, protected by fileMutex.
// ─────────────────────────────────────────────────────────────────────────────
void producerThread(BoundedBuffer<TrafficRecord>& buffer,
                    ifstream& file, mutex& fileMutex) {
    while (true) {
        string line;
        {
            lock_guard<mutex> lk(fileMutex);    // hold only for file read
            if (!getline(file, line)) break;    // EOF -> exit cleanly
        }                                       // fileMutex released here
        if (line.empty()) continue;             // skip blank lines
        buffer.push(parseLine(line));           // parse + enqueue (no lock held)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// consumerThread: drains buffer and updates tracker until end-of-stream.
// pop() returns false when buffer is empty AND done=true.
// ─────────────────────────────────────────────────────────────────────────────
void consumerThread(BoundedBuffer<TrafficRecord>& buffer,
                    CongestionTracker& tracker) {
    TrafficRecord r;
    while (buffer.pop(r))         // block when empty; return false at end
        tracker.addRecord(r);     // thread-safe update inside addRecord()
}

// ─────────────────────────────────────────────────────────────────────────────
// main: orchestrate full pipeline.
// ORDERING GUARANTEE: producers.join() MUST precede setDone().
// If setDone() fires before all producers finish, a consumer can see
// empty+done and exit while a producer is still mid-push => silent data loss.
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    if (argc < 6) {
        cout << "Usage: ./sim <file> <P> <C> <buf_size> <top_n>\n";
        return 1;
    }
    string filename  = argv[1];
    int numP = atoi(argv[2]);    // number of producer threads
    int numC = atoi(argv[3]);    // number of consumer threads
    int bufB = atoi(argv[4]);    // bounded buffer capacity
    int topN = atoi(argv[5]);    // Top-N results per hour

    if (numP<1||numC<1||bufB<1||topN<1) {
        cerr << "Error: all numeric args must be >= 1\n"; return 1;
    }
    ifstream file(filename);
    if (!file) { cerr << "Error: cannot open: " << filename << "\n"; return 1; }

    cout << "==========================================\n";
    cout << "  M2.T3D  Traffic Control Simulator\n";
    cout << "  Jahan Garg  |  Roll: 2410994805\n";
    cout << "==========================================\n";
    cout << "File     : " << filename << "\n";
    cout << "P/C/B/N  : " << numP << " / " << numC
         << " / " << bufB << " / " << topN << "\n";
    cout << "------------------------------------------\n";

    BoundedBuffer<TrafficRecord> buffer(bufB);  // shared bounded buffer
    CongestionTracker            tracker;       // shared result accumulator
    mutex                        fileMutex;     // serialises file reads

    auto t0 = high_resolution_clock::now();     // start wall-clock timer

    // Launch consumers first: ready to drain immediately
    vector<thread> consumers, producers;
    for (int i=0;i<numC;i++)
        consumers.emplace_back(consumerThread, ref(buffer), ref(tracker));

    // Launch producers: begin reading and pushing
    for (int i=0;i<numP;i++)
        producers.emplace_back(producerThread, ref(buffer), ref(file), ref(fileMutex));

    for (auto& p : producers) p.join();   // BARRIER 1: all file data pushed
    buffer.setDone();                     // signal: no more data coming
    for (auto& c : consumers) c.join();   // BARRIER 2: all data processed

    auto t1 = high_resolution_clock::now();
    long long us = duration_cast<microseconds>(t1 - t0).count();

    cout << "Pairs processed : " << tracker.uniquePairs() << "\n";
    cout << "Time            : " << us << " us  ("
         << fixed << setprecision(3) << us/1000.0 << " ms)\n";
    cout << "TIME_US=" << us << "\n";
    cout << "==========================================\n";
    tracker.printTopN(topN);
    return 0;
}
