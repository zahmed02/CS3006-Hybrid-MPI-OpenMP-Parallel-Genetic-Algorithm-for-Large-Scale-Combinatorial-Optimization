// TSP tour length, coordinate loading, distance matrix, random generator
#include "fitness.h"

#include <omp.h>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <random>
#include <stdexcept>

using namespace std;

namespace pdc {

// Total length of a closed tour. Loop from the first city back to it.
double tour_length(const vector<int>& tour,
                   const vector<vector<double>>& dist_matrix) {
    const size_t n = tour.size();
    if (n < 2) return 0.0;

    double total = 0.0;
    int prev = tour[0];

    for (size_t i = 1; i < n; ++i) {
        const int curr = tour[i];
        total += dist_matrix[prev][curr];
        prev = curr;
    }

    // Close the loop
    total += dist_matrix[tour[n - 1]][tour[0]];
    return total;
}

// Read "x y" per line from a text file.
vector<pair<double, double>> load_tsp_coordinates(const string& path) {
    ifstream in(path);
    if (!in.is_open()) {
        throw runtime_error("cannot open '" + path + "'");
    }

    vector<pair<double, double>> coords;
    coords.reserve(1 << 16);

    double x = 0.0, y = 0.0;
    while (in >> x >> y) {
        coords.emplace_back(x, y);
    }

    if (coords.empty()) {
        throw runtime_error("no coordinates in '" + path + "'");
    }
    return coords;
}

// Build the NxN Euclidean distance matrix.
// Uses symmetry: only i < j is computed; the rest is filled by mirroring.
// The outer loop runs in parallel with OpenMP.
vector<vector<double>> build_distance_matrix(const vector<pair<double, double>>& coords) {
    const size_t n = coords.size();
    vector<vector<double>> mat(n, vector<double>(n, 0.0));

    #pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); ++i) {
        const double xi = coords[i].first;
        const double yi = coords[i].second;

        for (size_t j = static_cast<size_t>(i) + 1; j < n; ++j) {
            const double dx = xi - coords[j].first;
            const double dy = yi - coords[j].second;
            const double d  = sqrt(dx * dx + dy * dy);

            mat[i][j] = d;
            mat[j][i] = d;
        }
    }
    return mat;
}

// Random TSP instance: `num_cities` points uniformly in [0,1]^2.
vector<pair<double, double>> generate_random_tsp(int num_cities, unsigned seed) {
    if (num_cities <= 0) {
        throw runtime_error("num_cities must be > 0");
    }

    mt19937_64 rng(seed);
    uniform_real_distribution<double> uni(0.0, 1.0);

    vector<pair<double, double>> coords;
    coords.reserve(static_cast<size_t>(num_cities));
    for (int i = 0; i < num_cities; ++i) {
        coords.emplace_back(uni(rng), uni(rng));
    }
    return coords;
}

}