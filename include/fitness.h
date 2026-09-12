// TSP fitness evaluation and dataset loading
#pragma once

#include <string>
#include <utility>
#include <vector>

using namespace std;

namespace pdc {

// Total length of a closed tour, using a precomputed distance matrix
double tour_length(const vector<int>& tour,
                   const vector<vector<double>>& dist_matrix);

// Load coordinates from a text file with one "x y" pair per line
vector<pair<double, double>> load_tsp_coordinates(const string& path);

// Build the full NxN distance matrix from coordinates
vector<vector<double>> build_distance_matrix(const vector<pair<double, double>>& coords);

// Generate a random TSP instance with points in [0,1]^2
vector<pair<double, double>> generate_random_tsp(int num_cities, unsigned seed);

}