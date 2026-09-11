// ============================================================================
// test_ga.cpp — Assertion-based unit tests for GA operators
//
// No external test framework required — uses plain assert() and manual
// checks. Add gtest later if the project grows.
//
// Run via: ctest --output-on-failure
// ============================================================================
#include "fitness.h"
#include "ga.h"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>
#include <set>
#include <vector>

using namespace pdc;

static bool is_valid_permutation(const Genome& g, int n) {
    if (static_cast<int>(g.size()) != n) return false;
    std::set<int> seen(g.begin(), g.end());
    if (static_cast<int>(seen.size()) != n) return false;
    return *seen.begin() == 0 && *seen.rbegin() == n - 1;
}

static void test_initialize_population() {
    std::mt19937_64 rng(42);
    std::vector<Individual> pop(20);
    initialize_population(pop, 30, rng);
    for (const auto& ind : pop) {
        assert(is_valid_permutation(ind.genome, 30));
    }
    std::cout << "[PASS] initialize_population\n";
}

static void test_order_crossover() {
    std::mt19937_64 rng(7);
    Genome p1 = {0,1,2,3,4,5,6,7,8,9};
    Genome p2 = {9,8,7,6,5,4,3,2,1,0};
    for (int i = 0; i < 100; ++i) {
        Genome child;
        order_crossover(p1, p2, child, rng);
        assert(is_valid_permutation(child, 10));
    }
    std::cout << "[PASS] order_crossover produces valid permutations\n";
}

static void test_swap_mutation() {
    std::mt19937_64 rng(11);
    for (int trial = 0; trial < 50; ++trial) {
        Genome g(20);
        std::iota(g.begin(), g.end(), 0);
        swap_mutation(g, 0.5, rng);
        assert(is_valid_permutation(g, 20));
    }
    std::cout << "[PASS] swap_mutation preserves permutation\n";
}

static void test_tournament_selects_best() {
    std::mt19937_64 rng(13);
    std::vector<Individual> pop(5);
    pop[0] = { {0,1,2}, 100.0 };
    pop[1] = { {1,0,2},  50.0 };
    pop[2] = { {2,1,0},  10.0 };   // best
    pop[3] = { {0,2,1},  75.0 };
    pop[4] = { {2,0,1},  25.0 };

    // With tournament size = 5, we MUST pick the global best.
    for (int i = 0; i < 20; ++i) {
        Individual picked = tournament_select(pop, 5, rng);
        assert(picked.fitness == 10.0);
    }
    std::cout << "[PASS] tournament_select picks global best (tournament = pop size)\n";
}

static void test_tour_length_known() {
    // 4-city square: side length 1, diagonal sqrt(2)
    std::vector<std::pair<double, double>> coords = {
        {0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}
    };
    auto dist = build_distance_matrix(coords);

    // Tour 0->1->2->3->0 = 1 + 1 + 1 + 1 = 4
    Genome square = {0, 1, 2, 3};
    assert(std::abs(tour_length(square, dist) - 4.0) < 1e-9);

    // Tour 0->2->1->3->0 = sqrt(2) + 1 + sqrt(2) + 1 = 2 + 2*sqrt(2)
    Genome bowtie = {0, 2, 1, 3};
    double expected = 2.0 + 2.0 * std::sqrt(2.0);
    assert(std::abs(tour_length(bowtie, dist) - expected) < 1e-9);

    std::cout << "[PASS] tour_length matches analytic values\n";
}

int main() {
    std::cout << "Running GA unit tests...\n";
    test_initialize_population();
    test_order_crossover();
    test_swap_mutation();
    test_tournament_selects_best();
    test_tour_length_known();
    std::cout << "\nAll tests passed.\n";
    return 0;
}