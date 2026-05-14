// FILE: generate_data.cpp
// TASK: M2.T3D - Traffic Control Simulator (Distinction)
// UNIT: SIT315 Programming Paradigms, Semester-4
// AUTHOR: Jahan Garg | Roll: 2410994805
//
// Generates a synthetic CSV input file for the traffic simulator.
// Each row is one 5-minute sensor reading at one traffic light.
// Timestamps use time-only HH:MM format (no date) as per tutor guidance.
// A fixed seed (0) guarantees identical output on every run and machine,
// which is required so that sequential and parallel Top-N outputs can be
// compared for correctness verification.
//
// CSV format  : HH:MM,TLXXX,<cars>
// Example row : 08:30,TL012,143
//
// Compile: g++ -std=c++11 -O2 generate_data.cpp -o gen
// Run    : ./gen <num_signals> <num_hours> <output_file>
// Example: ./gen 50 24 traffic_large.csv

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <iomanip>
using namespace std;

int main(int argc, char* argv[]) {
    if (argc < 4) {
        cerr << "Usage: ./gen <signals> <hours> <output_file>\n";
        cerr << "Example: ./gen 200 168 traffic_stress.csv\n";
        return 1;
    }

    int    numSignals = atoi(argv[1]);   // number of distinct traffic lights
    int    numHours   = atoi(argv[2]);   // total hours of simulation
    string outFile    = argv[3];         // output CSV file path

    if (numSignals < 1 || numHours < 1) {
        cerr << "Error: signals and hours must both be >= 1\n";
        return 1;
    }

    ofstream out(outFile);
    if (!out) {
        cerr << "Error: cannot create file: " << outFile << "\n";
        return 1;
    }

    // Fixed seed ensures reproducibility: every run produces the same file.
    // Both the sequential (P=1,C=1) and parallel runs process identical data
    // so their Top-N outputs can be compared to verify correctness.
    srand(0);

    long long total = 0;

    for (int h = 0; h < numHours; h++) {
        for (int m = 0; m < 60; m += 5) {            // one reading every 5 minutes
            for (int s = 1; s <= numSignals; s++) {   // one row per traffic light

                // Time-only timestamp in HH:MM format, no date component
                out << setw(2) << setfill('0') << h << ":"
                    << setw(2) << setfill('0') << m << ",";

                // Traffic light ID zero-padded to 3 digits, e.g. TL001
                out << "TL" << setw(3) << setfill('0') << s << ",";

                // Car count: uniform random in [20, 200] per 5-minute window.
                // Range reflects real urban sensor data (SCATS: 10-250 veh/5 min).
                out << (20 + rand() % 181) << "\n";

                total++;
            }
        }
    }
    out.close();

    cout << "Generated : " << total      << " records\n";
    cout << "Signals   : " << numSignals << "\n";
    cout << "Hours     : " << numHours   << "\n";
    cout << "File      : " << outFile    << "\n";
    return 0;
}
