// ============================================================================
// fitness.cpp — TSP tour length computation + dataset I/O
//
// Provides the two foundational operations for the GA:
//   1. Loading / generating TSP coordinate sets
//   2. Computing a tour's total length from a precomputed distance matrix
//
// Design notes:
//   - The NxN distance matrix is computed ONCE at startup and reused across
//     every generation and every individual. This trades O(N^2) memory for
//     O(1) fitness evaluations per tour, which is critical because fitness
//     is the hot path of the genetic algorithm.
//   - build_distance_matrix() exploits symmetry (d[i][j] == d[j][i]) so only
//     half the matrix is computed. The outer loop is OpenMP-parallel.
// ============================================================================
#include "fitness.h"

#include <omp.h>

#include <cmath>
#include <cstddef>
#include <fstream>
#include <random>
#include <stdexcept>

namespace pdc {

// ----------------------------------------------------------------------------
// tour_length
//
// A TSP tour is a closed loop: it visits cities in the order given by
// `tour`, then returns from the last city back to the first. Total length
// is the sum of the N edges (N-1 forward + 1 closing edge).
//
// Complexity: O(N) — one distance-matrix lookup per edge.
// ----------------------------------------------------------------------------
double tour_length(const std::vector<int>& tour,
                   const std::vector<std::vector<double>>& dist_matrix) {
    const std::size_t n = tour.size();
    if (n < 2) return 0.0;

    double total = 0.0;
    int prev = tour[0];

    for (std::size_t i = 1; i < n; ++i) {
        const int curr = tour[i];
        total += dist_matrix[prev][curr];
        prev = curr;
    }

    // Close the loop: last city -> first city
    total += dist_matrix[tour[n - 1]][tour[0]];
    return total;
}

// ----------------------------------------------------------------------------
// load_tsp_coordinates
//
// Reads a plain-text file with one city per line: "x y".
// Whitespace-separated; blank lines tolerated.
//
// Throws std::runtime_error on missing file or empty result.
// ----------------------------------------------------------------------------
std::vector<std::pair<double, double>>
load_tsp_coordinates(const std::string& path) {
    std::ifstream in(path);
    if (!in.is_open()) {
        throw std::runtime_error("load_tsp_coordinates: cannot open '" + path + "'");
    }

    std::vector<std::pair<double, double>> coords;
    coords.reserve(1 << 16);  // ~65k cities by default, cheap

    double x = 0.0;
    double y = 0.0;
    while (in >> x >> y) {
        coords.emplace_back(x, y);
    }

    if (coords.empty()) {
        throw std::runtime_error("load_tsp_coordinates: no coordinates parsed from '" + path + "'");
    }
    return coords;
}

// ----------------------------------------------------------------------------
// build_distance_matrix
//
// Produces the full NxN Euclidean distance matrix from coordinates.
//
// Optimizations:
//   1. Only the upper triangle is computed (i < j) — the lower triangle is
//      filled by symmetry, halving the arithmetic work.
//   2. The outer loop is parallelized with OpenMP (rows are independent).
//   3. `schedule(static)` because every row's work is identical (N - i - 1
//      inner iterations), so equal-sized chunks are fair.
//
// Memory note: N=5000 → ~200 MB, N=10000 → ~800 MB (double precision).
// For very large N, consider switching to float or a flat 1D array.
// ----------------------------------------------------------------------------
std::vector<std::vector<double>>
build_distance_matrix(const std::vector<std::pair<double, double>>& coords) {
    const std::size_t n = coords.size();

    // Pre-allocate N x N with zeros
    std::vector<std::vector<double>> mat(n, std::vector<double>(n, 0.0));

    #pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
        const double xi = coords[i].first;
        const double yi = coords[i].second;

        for (std::size_t j = static_cast<std::size_t>(i) + 1; j < n; ++j) {
            const double dx = xi - coords[j].first;
            const double dy = yi - coords[j].second;
            const double d  = std::sqrt(dx * dx + dy * dy);

            mat[i][j] = d;
            mat[j][i] = d;   // symmetry
        }
    }
    return mat;
}

// ----------------------------------------------------------------------------
// generate_random_tsp
//
// Synthetic TSP instance: `num_cities` points uniformly distributed in the
// unit square [0,1]^2. Seeded for reproducibility — same seed always yields
// the same instance.
//
// This is the standard benchmark generator for TSP studies and makes the
// expected optimal tour length analytically predictable (Beardwood–Halton–
// Hammersley theorem ≈ 0.7124 * sqrt(N * A), with A = 1 here).
// ----------------------------------------------------------------------------
std::vector<std::pair<double, double>>
generate_random_tsp(int num_cities, unsigned seed) {
    if (num_cities <= 0) {
        throw std::runtime_error("generate_random_tsp: num_cities must be > 0");
    }

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> uni(0.0, 1.0);

    std::vector<std::pair<double, double>> coords;
    coords.reserve(static_cast<std::size_t>(num_cities));
    for (int i = 0; i < num_cities; ++i) {
        coords.emplace_back(uni(rng), uni(rng));
    }
    return coords;
}

}  // namespace pdc