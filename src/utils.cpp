// Timer, CSV writer, and rank-0 logging
#include "utils.h"

#include <mpi.h>
#include <filesystem>
#include <fstream>
#include <iostream>

using namespace std;

namespace pdc {

void Timer::start() {
    t0_ = chrono::high_resolution_clock::now();
}

double Timer::elapsed_seconds() const {
    auto t1 = chrono::high_resolution_clock::now();
    return chrono::duration<double>(t1 - t0_).count();
}

void ensure_directory(const string& path) {
    filesystem::create_directories(path);
}

// Append a row to a CSV file. Adds a header if the file doesn't exist yet.
void append_csv_row(const string& path,
                    const string& header_if_new,
                    const string& row) {
    bool needs_header = !filesystem::exists(path);
    ofstream f(path, ios::app);
    if (needs_header) f << header_if_new << "\n";
    f << row << "\n";
}

void log_rank0(const string& msg) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) cout << msg << endl;
}

}