// TSP tour length, coordinate loading, instance/neighbour lists, generator
#include "fitness.h"

#include <omp.h>
#include <algorithm>
#include <cstddef>
#include <fstream>
#include <limits>
#include <random>
#include <stdexcept>

using namespace std;

namespace pdc {

// ---------------------------------------------------------------------------
// Coordinate based evaluation (used by the GA)
// ---------------------------------------------------------------------------

// K nearest neighbours of every city by a single pass over all cities that
// keeps a tiny sorted array of the best K candidates (K is small, so the
// insertion is cheap and almost never taken). O(N^2) total, parallel over i.
TSPInstance build_instance(const vector<pair<double, double>>& coords, int k) {
    TSPInstance inst;
    inst.n = static_cast<int>(coords.size());
    inst.pts.resize(coords.size());
    for (size_t i = 0; i < coords.size(); ++i) {
        inst.pts[i] = {coords[i].first, coords[i].second};
    }

    inst.k = max(0, min(k, inst.n - 1));
    const int K = inst.k;
    inst.neighbors.assign(static_cast<size_t>(inst.n) * static_cast<size_t>(K), 0);
    if (K == 0) return inst;

    const int n = inst.n;

    #pragma omp parallel
    {
        vector<double> best_d(static_cast<size_t>(K));
        vector<int>    best_j(static_cast<size_t>(K));

        #pragma omp for schedule(static)
        for (int i = 0; i < n; ++i) {
            int filled = 0;
            const double xi = inst.pts[i].x;
            const double yi = inst.pts[i].y;

            for (int j = 0; j < n; ++j) {
                if (j == i) continue;
                const double dx = xi - inst.pts[j].x;
                const double dy = yi - inst.pts[j].y;
                const double d2 = dx * dx + dy * dy;

                if (filled == K && d2 >= best_d[K - 1]) continue;

                // Insertion into the sorted candidate array
                int p = (filled < K) ? filled++ : K - 1;
                while (p > 0 && best_d[p - 1] > d2) {
                    best_d[p] = best_d[p - 1];
                    best_j[p] = best_j[p - 1];
                    --p;
                }
                best_d[p] = d2;
                best_j[p] = j;
            }

            int* row = &inst.neighbors[static_cast<size_t>(i) * K];
            for (int m = 0; m < K; ++m) row[m] = best_j[m];
        }
    }
    return inst;
}

double tour_length(const vector<int>& tour, const TSPInstance& inst) {
    const size_t n = tour.size();
    if (n < 2) return 0.0;

    const Point* pts = inst.pts.data();
    const int*   t   = tour.data();

    double total = 0.0;
    Point  prev  = pts[t[n - 1]];   // closing edge first
    for (size_t i = 0; i < n; ++i) {
        const Point cur = pts[t[i]];
        const double dx = prev.x - cur.x;
        const double dy = prev.y - cur.y;
        total += sqrt(dx * dx + dy * dy);
        prev = cur;
    }
    return total;
}

bool is_valid_tour(const vector<int>& tour, int n) {
    if (static_cast<int>(tour.size()) != n) return false;
    vector<char> seen(static_cast<size_t>(n), 0);
    for (int c : tour) {
        if (c < 0 || c >= n || seen[c]) return false;
        seen[c] = 1;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Matrix based helpers (tests / NumPy cross-check)
// ---------------------------------------------------------------------------

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
    total += dist_matrix[tour[n - 1]][tour[0]];
    return total;
}

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

// Symmetric NxN matrix, outer loop in parallel. Only for small instances:
// the GA itself no longer uses it.
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

}  // namespace pdc