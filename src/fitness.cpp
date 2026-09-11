// ============================================================================
// fitness.cpp — TSP tour length + dataset I/O
// ============================================================================
#include "fitness.h"

#include <cmath>
#include <fstream>
#include <random>
#include <stdexcept>

namespace pdc {

double tour_length(const std::vector<int>& tour,
                   const std::vector<std::vector<double>>& dist_matrix) {
    // TODO
    return 0.0;
}

std::vector<std::pair<double, double>>
load_tsp_coordinates(const std::string& path) {
    // TODO
    return {};
}

std::vector<std::vector<double>>
build_distance_matrix(const std::vector<std::pair<double, double>>& coords) {
    // TODO
    return {};
}

std::vector<std::pair<double, double>>
generate_random_tsp(int num_cities, unsigned seed) {
    // TODO
    return {};
}

}  // namespace pdc