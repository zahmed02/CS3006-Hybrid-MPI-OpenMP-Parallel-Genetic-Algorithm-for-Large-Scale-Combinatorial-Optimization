// ============================================================================
// fitness.h — TSP fitness evaluation (tour length) and dataset I/O
// ============================================================================
#pragma once

#include <string>
#include <utility>
#include <vector>

namespace pdc {

// Compute the total length of a tour given a precomputed distance matrix.
double tour_length(const std::vector<int>& tour,
                   const std::vector<std::vector<double>>& dist_matrix);

// Load TSP coordinates from a plain text file: "x y" per line.
std::vector<std::pair<double, double>>
load_tsp_coordinates(const std::string& path);

// Build the full NxN Euclidean distance matrix from coordinates.
std::vector<std::vector<double>>
build_distance_matrix(const std::vector<std::pair<double, double>>& coords);

// Generate a synthetic random TSP instance (uniform in [0,1]^2).
std::vector<std::pair<double, double>>
generate_random_tsp(int num_cities, unsigned seed);

}  // namespace pdc