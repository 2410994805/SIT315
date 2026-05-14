// ============================================================
// FILE    : generate_data.cpp
// TASK    : M2.T3D — Traffic Input Data Generator
// UNIT    : SIT315 Programming Paradigms, Trimester 4, 2024-25
// AUTHOR  : Jahan Garg  |  Roll: 2410994805
// COMPILE : g++ -std=c++11 -O2 generate_data.cpp -o gen
// RUN     : ./gen <signals> <hours> <output_file>
// EXAMPLE : ./gen 50 24 traffic_large.csv
//
// Generates a deterministic CSV with time-only HH:MM timestamps.
// srand(0) ensures identical output on every machine — reproducible.
// One reading per 5 minutes per signal: 12 readings/hour/signal.
// Car counts: uniform random [20, 200] per 5-minute window.
// ============================================================

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <iomanip>
#include <sstream>
using namespace std;

int main(int argc, char* argv[]) {
    if (argc < 4) {
        cout << "Usage: ./gen <signals> <hours> <output_file>\n";
        cout << "Example: ./gen 200 168 traffic_stress.csv\n";
        return 1;
    }
    int    numSignals = atoi(argv[1]);   // number of traffic light signals
    int    numHours   = atoi(argv[2]);   // total hours to simulate
    string outFile    = argv[3];         // output CSV file path

    ofstream out(outFile);
    if (!out) { cerr << "Error: cannot create: " << outFile << "\n"; return 1; }

    // Fixed seed: guarantees reproducibility across all machines.
    // Sequential and parallel runs on the same file yield identical Top-N output.
    srand(0);

    long long total = 0;
    for (int h = 0; h < numHours; h++) {
        for (int m = 0; m < 60; m += 5) {            // 12 readings per hour
            for (int s = 1; s <= numSignals; s++) {   // one row per signal
                int cars = 20 + rand() % 181;         // uniform random [20, 200]

                // Time-only format: "HH:MM" — no date (per tutor guidance)
                out << setw(2) << setfill('0') << h << ":"
                    << setw(2) << setfill('0') << m << ","
                    << "TL" << setw(3) << setfill('0') << s << ","
                    << cars << "\n";
                total++;
            }
        }
    }
    out.close();

    cout << "Generated : " << total << " records\n";
    cout << "Signals   : " << numSignals << "  Hours: " << numHours << "\n";
    cout << "File      : " << outFile << "\n";
    return 0;
}
