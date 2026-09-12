// Core Genetic Algorithm data structures and operators
#pragma once

#include <cstdint>
#include <random>
#include <vector>

using namespace std;

namespace pdc {

// A TSP tour is a permutation of city indices
using Genome = vector<int>;

// One candidate solution with its fitness (tour length; lower is better)
struct Individual {
    Genome genome;
    double fitness = 0.0;
};

// Tunable GA settings
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

// Fill pop with random permutations of [0, num_cities)
void initialize_population(vector<Individual>& pop, int num_cities, mt19937_64& rng);

// Compute fitness for every individual (OpenMP-parallel)
void evaluate_population(vector<Individual>& pop,
                         const vector<vector<double>>& dist_matrix);

// Return the best of tournament_size random individuals
Individual tournament_select(const vector<Individual>& pop,
                             int tournament_size,
                             mt19937_64& rng);

// Order Crossover (OX1): keeps the child a valid permutation
void order_crossover(const Genome& p1, const Genome& p2,
                     Genome& child, mt19937_64& rng);

// Swap mutation: swap two random positions with the given probability
void swap_mutation(Genome& genome, double mutation_rate, mt19937_64& rng);

// One generation: sort, elitism, selection, crossover, mutation
void evolve_generation(vector<Individual>& pop,
                       const GAConfig& cfg,
                       const vector<vector<double>>& dist_matrix,
                       mt19937_64& rng);

}