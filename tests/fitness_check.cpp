// Verify tour_length() against an external reference (e.g., NumPy).
//
// Usage:
//   ./fitness_check <coords_file> <tour_file>
//
// Prints:
//   N=<num_cities>
//   TOUR_LENGTH=<value with 10 decimals>
#include "fitness.h"

#include <cstdio>
#include <exception>
#include <fstream>
#include <iostream>
#include <vector>

using namespace std;

int main(int argc, char** argv) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <coords_file> <tour_file>\n";
        return 1;
    }

    try {
        // Load coordinates and build the distance matrix
        auto coords = pdc::load_tsp_coordinates(argv[1]);
        auto dist   = pdc::build_distance_matrix(coords);

        // Read the tour (one city index per line)
        ifstream tf(argv[2]);
        if (!tf.is_open()) {
            cerr << "Cannot open tour file: " << argv[2] << "\n";
            return 1;
        }

        vector<int> tour;
        int city = 0;
        while (tf >> city) tour.push_back(city);

        if (tour.empty()) {
            cerr << "Tour file is empty or malformed\n";
            return 1;
        }

        // Every city index must be in [0, N)
        const size_t N = coords.size();
        for (int idx : tour) {
            if (idx < 0 || static_cast<size_t>(idx) >= N) {
                cerr << "Invalid city index: " << idx << " (N=" << N << ")\n";
                return 1;
            }
        }

        const double len = pdc::tour_length(tour, dist);

        printf("N=%zu\n", N);
        printf("TOUR_LENGTH=%.10f\n", len);
        return 0;

    } catch (const exception& e) {
        cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}