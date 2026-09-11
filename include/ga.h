// ============================================================================
// ga.h — Core Genetic Algorithm data structures and operators
// ============================================================================
#pragma once

#include <cstdint>
#include <random>
#include <vector>

namespace pdc {

// A TSP tour is a permutation of city indices
using Genome = std::vector<int>;

struct Individual {
    Genome genome;
    double fitness = 0.0;   // lower = better for TSP (tour length)
};

struct GAConfig {
    int      population_size    = 1000;
    int      num_generations    = 500;
    double   crossover_rate     = 0.85;
    double   mutation_rate      = 0.02;
    int      tournament_size    = 5;
    int      elitism_count      = 2;
    int      migration_interval = 25;   // generations between migrations
    int      num_migrants       = 5;    // individuals exchanged per migration
    uint64_t seed               = 42;
};

// ---- GA Operators ----

// Fill `pop` with random permutations of [0, num_cities)
void initialize_population(std::vector<Individual>& pop,
                           int num_cities,
                           std::mt19937_64& rng);

// Compute fitness (tour length) for every individual — parallelized with OpenMP
void evaluate_population(std::vector<Individual>& pop,
                         const std::vector<std::vector<double>>& dist_matrix);

// Pick the best of `tournament_size` random individuals
Individual tournament_select(const std::vector<Individual>& pop,
                             int tournament_size,
                             std::mt19937_64& rng);

// Order Crossover (OX1) — preserves permutation validity
void order_crossover(const Genome& p1,
                     const Genome& p2,
                     Genome& child,
                     std::mt19937_64& rng);

// Swap mutation — swap two random positions
void swap_mutation(Genome& genome,
                   double mutation_rate,
                   std::mt19937_64& rng);

// Run one generation: selection -> crossover -> mutation -> elitism
void evolve_generation(std::vector<Individual>& pop,
                       const GAConfig& cfg,
                       const std::vector<std::vector<double>>& dist_matrix,
                       std::mt19937_64& rng);

}  // namespace pdc