// ============================================================================
// fitness_check.cpp — Standalone harness to verify tour_length() against
//                      an external reference (e.g., NumPy).
//
// Usage:
//   ./fitness_check <coords_file> <tour_file>
//
// Inputs:
//   coords_file — plain text, one "x y" pair per line
//   tour_file   — plain text, one integer city index per line
//
// Output (stdout):
//   N=<num_cities>
//   TOUR_LENGTH=<value with 10 decimal places>
//
// Exit codes:
//   0 — success
//   1 — usage error, file error, or runtime exception
// ============================================================================
#include "fitness.h"

#include <cstdio>
#include <exception>
#include <fstream>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <coords_file> <tour_file>\n";
        return 1;
    }

    try {
        // ---- Load coordinates and precompute distances ----
        auto coords = pdc::load_tsp_coordinates(argv[1]);
        auto dist   = pdc::build_distance_matrix(coords);

        // ---- Read the tour ----
        std::ifstream tf(argv[2]);
        if (!tf.is_open()) {
            std::cerr << "Cannot open tour file: " << argv[2] << "\n";
            return 1;
        }

        std::vector<int> tour;
        int city = 0;
        while (tf >> city) tour.push_back(city);

        if (tour.empty()) {
            std::cerr << "Tour file is empty or malformed\n";
            return 1;
        }

        // ---- Sanity: every city index must be in [0, N) ----
        const std::size_t N = coords.size();
        for (int idx : tour) {
            if (idx < 0 || static_cast<std::size_t>(idx) >= N) {
                std::cerr << "Invalid city index in tour: " << idx
                          << " (N=" << N << ")\n";
                return 1;
            }
        }

        // ---- Compute and print ----
        const double len = pdc::tour_length(tour, dist);

        std::printf("N=%zu\n", N);
        std::printf("TOUR_LENGTH=%.10f\n", len);
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}