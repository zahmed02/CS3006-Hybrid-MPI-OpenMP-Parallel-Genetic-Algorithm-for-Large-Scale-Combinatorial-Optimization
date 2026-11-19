// Unit tests for the GA operators and the local search.
//
// Uses a CHECK macro instead of assert(): the Docker build uses
// CMAKE_BUILD_TYPE=Release, which defines NDEBUG and compiles every assert()
// away, so assert-based tests would silently test nothing.
#include "fitness.h"
#include "ga.h"

#include <omp.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

using namespace std;
using namespace pdc;

#define CHECK(cond)                                                         \
    do {                                                                    \
        if (!(cond)) {                                                      \
            cerr << "[FAIL] " << __FILE__ << ":" << __LINE__                \
                 << "  " #cond "\n";                                        \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

static vector<pair<double, double>> random_coords(int n, unsigned seed) {
    return generate_random_tsp(n, seed);
}

// Initialization should produce valid permutations (both modes)
static void test_initialize_population() {
    auto coords = random_coords(200, 1);
    TSPInstance inst = build_instance(coords, 8);

    for (InitMode mode : {InitMode::Random, InitMode::SpaceFilling}) {
        vector<Individual> pop(20);
        initialize_population(pop, inst, mode, /*local_search=*/false, 42);
        for (const auto& ind : pop) {
            CHECK(is_valid_tour(ind.genome, 200));
            CHECK(abs(ind.fitness - tour_length(ind.genome, inst)) < 1e-9);
        }
    }
    cout << "[PASS] initialize_population (random + Hilbert)\n";
}

// A space-filling-curve tour must be much shorter than a random one
static void test_sfc_quality() {
    auto coords = random_coords(1000, 2);
    TSPInstance inst = build_instance(coords, 8);

    vector<Individual> rnd(5), sfc(5);
    initialize_population(rnd, inst, InitMode::Random, false, 7);
    initialize_population(sfc, inst, InitMode::SpaceFilling, false, 7);
    CHECK(sfc[0].fitness < 0.2 * rnd[0].fitness);
    cout << "[PASS] Hilbert tours are far better than random tours\n";
}

// OX1 should always return a valid permutation
static void test_order_crossover() {
    mt19937_64 rng(7);
    Genome p1 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    Genome p2 = {9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
    vector<char> scratch;
    for (int i = 0; i < 200; ++i) {
        Genome child;
        order_crossover(p1, p2, child, rng, scratch);
        CHECK(is_valid_tour(child, 10));
    }
    // random parents of a larger size, 3-arg overload
    Genome a(300), b(300);
    iota(a.begin(), a.end(), 0);
    iota(b.begin(), b.end(), 0);
    shuffle(b.begin(), b.end(), rng);
    for (int i = 0; i < 200; ++i) {
        Genome child;
        order_crossover(a, b, child, rng);
        CHECK(is_valid_tour(child, 300));
    }
    cout << "[PASS] order_crossover produces valid permutations\n";
}

// Inversion keeps the permutation property and changes the tour
static void test_inversion_mutation() {
    mt19937_64 rng(11);
    int changed = 0;
    for (int trial = 0; trial < 100; ++trial) {
        Genome g(40);
        iota(g.begin(), g.end(), 0);
        Genome before = g;
        inversion_mutation(g, rng, 10);
        CHECK(is_valid_tour(g, 40));
        if (g != before) ++changed;
    }
    CHECK(changed > 90);

    // tiny genomes must not crash
    Genome tiny = {0, 1};
    inversion_mutation(tiny, rng);
    CHECK(is_valid_tour(tiny, 2));
    cout << "[PASS] inversion_mutation preserves permutation\n";
}

// With tournament size >> pop size the best individual wins nearly always;
// with size = 1 it is a plain random pick; the index must be in range.
static void test_tournament() {
    mt19937_64 rng(13);
    vector<Individual> pop(5);
    pop[0] = {{0, 1, 2}, 100.0};
    pop[1] = {{1, 0, 2},  50.0};
    pop[2] = {{2, 1, 0},  10.0};   // best
    pop[3] = {{0, 2, 1},  75.0};
    pop[4] = {{2, 0, 1},  25.0};

    int hits = 0;
    for (int i = 0; i < 1000; ++i) {
        const int idx = tournament_select(pop, 50, rng);
        CHECK(idx >= 0 && idx < 5);
        if (idx == 2) ++hits;
    }
    CHECK(hits > 990);   // P(miss) = 0.8^50 per draw
    for (int i = 0; i < 100; ++i) {
        const int idx = tournament_select(pop, 1, rng);
        CHECK(idx >= 0 && idx < 5);
    }
    cout << "[PASS] tournament_select\n";
}

// Coordinate based and matrix based tour_length agree and match analytic values
static void test_tour_length_known() {
    vector<pair<double, double>> coords = {
        {0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}
    };
    auto dist = build_distance_matrix(coords);
    TSPInstance inst = build_instance(coords, 2);

    Genome square = {0, 1, 2, 3};
    CHECK(abs(tour_length(square, dist) - 4.0) < 1e-9);
    CHECK(abs(tour_length(square, inst) - 4.0) < 1e-9);

    Genome bowtie = {0, 2, 1, 3};
    const double expected = 2.0 + 2.0 * sqrt(2.0);
    CHECK(abs(tour_length(bowtie, dist) - expected) < 1e-9);
    CHECK(abs(tour_length(bowtie, inst) - expected) < 1e-9);

    auto big = random_coords(300, 5);
    auto bigd = build_distance_matrix(big);
    TSPInstance bigi = build_instance(big, 8);
    mt19937_64 rng(3);
    Genome g(300);
    iota(g.begin(), g.end(), 0);
    shuffle(g.begin(), g.end(), rng);
    CHECK(abs(tour_length(g, bigd) - tour_length(g, bigi)) < 1e-9);

    cout << "[PASS] tour_length matches analytic values and the matrix version\n";
}

// KNN lists must match a brute-force sort
static void test_knn() {
    auto coords = random_coords(400, 9);
    const int K = 7;
    TSPInstance inst = build_instance(coords, K);
    CHECK(inst.k == K);

    for (int i = 0; i < inst.n; ++i) {
        vector<pair<double, int>> all;
        for (int j = 0; j < inst.n; ++j) {
            if (j != i) all.push_back({inst.dist(i, j), j});
        }
        sort(all.begin(), all.end());
        const int* row = inst.neighbor_row(i);
        for (int m = 0; m < K; ++m) {
            CHECK(abs(inst.dist(i, row[m]) - all[m].first) < 1e-12);
        }
    }

    // K larger than N-1 is clamped
    TSPInstance small = build_instance(random_coords(5, 1), 50);
    CHECK(small.k == 4);
    cout << "[PASS] build_instance neighbour lists match brute force\n";
}

// Local search: valid output, gain == actual length decrease, no worse.
// Run for 2-opt only and for 2-opt + Or-opt, on tiny and larger instances
// (tiny ones hit the degenerate cases: segment next to its own neighbours).
static void test_local_search() {
    mt19937_64 rng(21);
    LSScratch scratch;

    for (bool oropt : {false, true}) {
        for (int n : {8, 9, 10, 12, 15, 20, 50, 200, 500}) {
            const int trials = (n <= 50) ? 150 : 5;
            for (int trial = 0; trial < trials; ++trial) {
                auto coords = random_coords(n, 1000 + n + trial);
                TSPInstance inst = build_instance(coords, 10);

                Genome g(n);
                iota(g.begin(), g.end(), 0);
                shuffle(g.begin(), g.end(), rng);

                const double before = tour_length(g, inst);
                const double gain   = local_search_2opt(g, inst, scratch, nullptr, 0, oropt);
                const double after  = tour_length(g, inst);

                CHECK(is_valid_tour(g, n));
                CHECK(after <= before + 1e-9);
                CHECK(abs((before - after) - gain) < 1e-6);
            }
        }
    }

    // Or-opt finds strictly better tours than 2-opt alone on a larger instance
    {
        auto coords = random_coords(1000, 77);
        TSPInstance inst = build_instance(coords, 10);
        double sum2 = 0.0, sum3 = 0.0;
        for (int trial = 0; trial < 5; ++trial) {
            Genome g(1000);
            iota(g.begin(), g.end(), 0);
            shuffle(g.begin(), g.end(), rng);
            Genome h = g;
            local_search_2opt(g, inst, scratch, nullptr, 0, false);
            local_search_2opt(h, inst, scratch, nullptr, 0, true);
            sum2 += tour_length(g, inst);
            sum3 += tour_length(h, inst);
        }
        CHECK(sum3 < sum2);
    }

    // 2-opt alone: re-running on a local optimum finds nothing
    auto coords = random_coords(500, 4);
    TSPInstance inst = build_instance(coords, 10);
    Genome g(500);
    iota(g.begin(), g.end(), 0);
    shuffle(g.begin(), g.end(), rng);
    local_search_2opt(g, inst, scratch, nullptr, 0, false);
    CHECK(local_search_2opt(g, inst, scratch, nullptr, 0, false) < 1e-9);

    // Seeded run on a perturbed local optimum repairs it and stays valid
    inversion_mutation(g, rng, 30);
    vector<int> seeds(g.begin(), g.end());
    const double before = tour_length(g, inst);
    local_search_2opt(g, inst, scratch, seeds.data(), static_cast<int>(seeds.size()));
    CHECK(is_valid_tour(g, 500));
    CHECK(tour_length(g, inst) <= before + 1e-9);

    // Tiny instances are left alone
    TSPInstance tiny = build_instance(random_coords(4, 1), 3);
    Genome t = {0, 1, 2, 3};
    CHECK(local_search_2opt(t, tiny, scratch, nullptr, 0) == 0.0);

    cout << "[PASS] local_search_2opt / Or-opt: valid, exact gain, Or-opt helps\n";
}

// One full generation: valid tours, exact fitness, elites kept, no regression
static void test_evolve_generation() {
    auto coords = random_coords(300, 6);
    TSPInstance inst = build_instance(coords, 8);

    GAConfig cfg;
    cfg.population_size = 40;
    cfg.elitism_count   = 2;

    for (bool ls : {false, true}) {
        cfg.local_search = ls;
        vector<Individual> pop(40), next;
        initialize_population(pop, inst, InitMode::SpaceFilling, ls, 5);
        mt19937_64 rng(99);

        double best = (*min_element(pop.begin(), pop.end(),
            [](const Individual& a, const Individual& b) { return a.fitness < b.fitness; })).fitness;

        for (int g = 0; g < 10; ++g) {
            evolve_generation(pop, next, cfg, inst, rng);
            CHECK(static_cast<int>(pop.size()) == 40);
            double gen_best = 1e300;
            for (const auto& ind : pop) {
                CHECK(is_valid_tour(ind.genome, 300));
                CHECK(abs(ind.fitness - tour_length(ind.genome, inst)) < 1e-6);
                gen_best = min(gen_best, ind.fitness);
            }
            CHECK(gen_best <= best + 1e-9);   // elitism: never gets worse
            best = gen_best;
        }
    }
    cout << "[PASS] evolve_generation (with and without local search)\n";
}

// Same seed => same result, regardless of thread count
static void test_thread_determinism() {
    auto coords = random_coords(250, 8);
    TSPInstance inst = build_instance(coords, 8);
    GAConfig cfg;
    cfg.population_size = 30;

    auto run = [&](int threads) {
        omp_set_num_threads(threads);
        vector<Individual> pop(30), next;
        initialize_population(pop, inst, InitMode::SpaceFilling, true, 123);
        mt19937_64 rng(5);
        for (int g = 0; g < 5; ++g) evolve_generation(pop, next, cfg, inst, rng);
        double best = 1e300;
        for (const auto& ind : pop) best = min(best, ind.fitness);
        return best;
    };

    const double a = run(1);
    const double b = run(2);
    const double c = run(4);
    CHECK(a == b);
    CHECK(a == c);
    cout << "[PASS] results are identical for 1, 2 and 4 threads\n";
}

int main() {
    cout << "Running GA unit tests...\n";
    test_initialize_population();
    test_sfc_quality();
    test_order_crossover();
    test_inversion_mutation();
    test_tournament();
    test_tour_length_known();
    test_knn();
    test_local_search();
    test_evolve_generation();
    test_thread_determinism();
    cout << "\nAll tests passed.\n";
    return 0;
}