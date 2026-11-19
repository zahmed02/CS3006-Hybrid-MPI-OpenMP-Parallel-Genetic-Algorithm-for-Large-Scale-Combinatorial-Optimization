// Core Genetic Algorithm data structures and operators (memetic variant:
// GA + neighbour-list 2-opt / Or-opt local search)
#pragma once

#include <cstdint>
#include <random>
#include <vector>

#include "fitness.h"

using namespace std;

namespace pdc {

// A TSP tour is a permutation of city indices
using Genome = vector<int>;

// One candidate solution with its fitness (tour length; lower is better)
struct Individual {
    Genome genome;
    double fitness = 0.0;
};

// How the starting population is built
enum class InitMode {
    Random,        // uniformly random permutations (very poor starting tours)
    SpaceFilling   // randomly shifted/flipped Hilbert-curve tours (~25% above optimal)
};

// Tunable GA settings
struct GAConfig {
    int      population_size    = 1000;   // LOCAL population (per MPI rank)
    int      num_generations    = 500;
    double   crossover_rate     = 0.85;
    double   mutation_rate      = 0.20;   // probability that a child gets an inversion mutation
    int      tournament_size    = 5;
    int      elitism_count      = 2;
    int      migration_interval = 25;     // generations between migrations
    int      num_migrants       = 5;      // individuals exchanged per migration
    bool     local_search       = true;   // local search on every new child
    bool     use_oropt          = true;   // add Or-opt (move 1-3 city segments) to the 2-opt
    InitMode init_mode          = InitMode::SpaceFilling;
    uint64_t seed               = 42;
};

// Reusable per-thread buffers for the local search (avoids malloc in hot loops)
struct LSScratch {
    vector<int>  pos;     // city -> position in the tour
    vector<int>  queue;   // ring buffer of cities still to be examined
    vector<char> inq;     // city already queued?
};

// Cheap, high-quality 64-bit mixer used to derive independent RNG streams
uint64_t splitmix64(uint64_t x);

// Fill pop with starting tours and evaluate them. Individual i is built from
// its own RNG stream derived from (seed, i), so the result does not depend on
// the number of OpenMP threads. Runs in parallel.
void initialize_population(vector<Individual>& pop,
                           const TSPInstance& inst,
                           InitMode mode,
                           bool apply_local_search,
                           uint64_t seed,
                           bool use_oropt = true);

// Compute fitness for every individual (OpenMP-parallel)
void evaluate_population(vector<Individual>& pop, const TSPInstance& inst);

// 2-opt (+ optional Or-opt) with neighbour lists and a work queue ("don't-look bits").
// If `seeds` is null every city is examined first; otherwise only the given
// cities (and whatever the moves touch). Returns the total length gained.
double local_search_2opt(Genome& tour,
                         const TSPInstance& inst,
                         LSScratch& scratch,
                         const int* seeds,
                         int num_seeds,
                         bool use_oropt = true);

// Return the INDEX of the best of tournament_size random individuals
int tournament_select(const vector<Individual>& pop,
                      int tournament_size,
                      mt19937_64& rng);

// Order Crossover (OX1): keeps the child a valid permutation.
// `used` is a scratch buffer (resized as needed) so no allocation happens
// per call when the caller reuses it.
void order_crossover(const Genome& p1, const Genome& p2,
                     Genome& child, mt19937_64& rng,
                     vector<char>& used);

// Convenience overload that allocates its own scratch buffer
void order_crossover(const Genome& p1, const Genome& p2,
                     Genome& child, mt19937_64& rng);

// Reverse one random segment of length 2..max_len (a random 2-opt move)
void inversion_mutation(Genome& genome, mt19937_64& rng, int max_len = 50);

// One generation. `next` is a caller-owned scratch population that is swapped
// with `pop` at the end, so genome buffers are recycled instead of
// reallocated. Children are evaluated here, so pop has valid fitness on return.
void evolve_generation(vector<Individual>& pop,
                       vector<Individual>& next,
                       const GAConfig& cfg,
                       const TSPInstance& inst,
                       mt19937_64& rng);

}  // namespace pdc