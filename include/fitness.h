// TSP instance, tour length, dataset loading and neighbour lists
#pragma once

#include <cmath>
#include <string>
#include <utility>
#include <vector>

using namespace std;

namespace pdc {

struct Point {
    double x = 0.0;
    double y = 0.0;
};

// A TSP instance prepared for fast evaluation.
//
// Instead of an N x N distance matrix (200 MB for N = 5000, one copy per MPI
// rank, one cache miss per tour edge), distances are computed on the fly from
// the coordinates (80 KB for N = 5000, fits in L2 cache). Skylake's sqrt is
// pipelined, so this is far faster than walking the matrix.
//
// `neighbors` holds the K nearest cities of every city (row-major, N x K),
// sorted by increasing distance. The local search uses it.
struct TSPInstance {
    int           n = 0;
    int           k = 0;
    vector<Point> pts;
    vector<int>   neighbors;

    // Euclidean distance between two cities
    double dist(int a, int b) const {
        const double dx = pts[a].x - pts[b].x;
        const double dy = pts[a].y - pts[b].y;
        return sqrt(dx * dx + dy * dy);
    }

    const int* neighbor_row(int city) const {
        return &neighbors[static_cast<size_t>(city) * static_cast<size_t>(k)];
    }
};

// Build an instance and its K-nearest-neighbour lists (OpenMP-parallel).
// K is clamped to N-1.
TSPInstance build_instance(const vector<pair<double, double>>& coords, int k);

// Total length of a closed tour, distances computed from coordinates
double tour_length(const vector<int>& tour, const TSPInstance& inst);

// True if `tour` is a permutation of [0, n)
bool is_valid_tour(const vector<int>& tour, int n);

// ---- Matrix based helpers, kept for unit tests and the NumPy cross-check ----

// Total length of a closed tour, using a precomputed distance matrix
double tour_length(const vector<int>& tour,
                   const vector<vector<double>>& dist_matrix);

// Load coordinates from a text file with one "x y" pair per line
vector<pair<double, double>> load_tsp_coordinates(const string& path);

// Build the full NxN distance matrix from coordinates (small instances only)
vector<vector<double>> build_distance_matrix(const vector<pair<double, double>>& coords);

// Generate a random TSP instance with points in [0,1]^2
vector<pair<double, double>> generate_random_tsp(int num_cities, unsigned seed);

}  // namespace pdc