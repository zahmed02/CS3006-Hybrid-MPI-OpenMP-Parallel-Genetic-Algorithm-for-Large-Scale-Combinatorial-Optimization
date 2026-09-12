// Timing, logging, and CSV output helpers
#pragma once

#include <chrono>
#include <string>

using namespace std;

namespace pdc {

// Simple stopwatch
class Timer {
public:
    void   start();
    double elapsed_seconds() const;

private:
    chrono::high_resolution_clock::time_point t0_;
};

// Create a directory (and its parents) if it doesn't exist
void ensure_directory(const string& path);

// Append a row to a CSV file; writes a header if the file is new
void append_csv_row(const string& path,
                    const string& header_if_new,
                    const string& row);

// Print a message only from MPI rank 0
void log_rank0(const string& msg);

}