// ============================================================================
// utils.cpp — Timer, CSV, logging
// ============================================================================
#include "utils.h"

#include <mpi.h>

#include <filesystem>
#include <fstream>
#include <iostream>

namespace pdc {

void Timer::start() {
    t0_ = std::chrono::high_resolution_clock::now();
}

double Timer::elapsed_seconds() const {
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(t1 - t0_).count();
}

void ensure_directory(const std::string& path) {
    std::filesystem::create_directories(path);
}

void append_csv_row(const std::string& path,
                    const std::string& header_if_new,
                    const std::string& row) {
    bool needs_header = !std::filesystem::exists(path);
    std::ofstream f(path, std::ios::app);
    if (needs_header) f << header_if_new << "\n";
    f << row << "\n";
}

void log_rank0(const std::string& msg) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) std::cout << msg << std::endl;
}

}  // namespace pdc