// ============================================================================
// utils.h — Timing, logging, CSV output helpers
// ============================================================================
#pragma once

#include <chrono>
#include <string>

namespace pdc {

// Simple monotonic timer
class Timer {
public:
    void   start();
    double elapsed_seconds() const;

private:
    std::chrono::high_resolution_clock::time_point t0_;
};

// Ensure a directory exists (creates parents if needed)
void ensure_directory(const std::string& path);

// Append a CSV row; writes header on first creation
void append_csv_row(const std::string& path,
                    const std::string& header_if_new,
                    const std::string& row);

// Print a message only from rank 0
void log_rank0(const std::string& msg);

}  // namespace pdc