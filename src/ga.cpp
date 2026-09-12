// GA operators: initialization, fitness, selection, crossover, mutation
#include "ga.h"
#include <omp.h>
#include <algorithm>
#include <limits>
#include <numeric>

#include "fitness.h"

using namespace std;

namespace pdc {

// Random permutation of [0, num_cities) for each individual.
void initialize_population(vector<Individual>& pop,
                           int num_cities,
                           mt19937_64& rng) {
    for (auto& ind : pop) {
        ind.genome.resize(static_cast<size_t>(num_cities));
        iota(ind.genome.begin(), ind.genome.end(), 0);
        shuffle(ind.genome.begin(), ind.genome.end(), rng);
        ind.fitness = numeric_limits<double>::max();
    }
}

// Compute fitness for every individual. This is the hot path.
// Static schedule is fine because each individual costs the same O(N).
void evaluate_population(vector<Individual>& pop,
                         const vector<vector<double>>& dist_matrix) {
    const ptrdiff_t n = static_cast<ptrdiff_t>(pop.size());

    #pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < n; ++i) {
        pop[i].fitness = tour_length(pop[i].genome, dist_matrix);
    }
}

// Pick the best of `tournament_size` random individuals.
Individual tournament_select(const vector<Individual>& pop,
                             int tournament_size,
                             mt19937_64& rng) {
    uniform_int_distribution<size_t> pick(0, pop.size() - 1);

    Individual best = pop[pick(rng)];
    for (int k = 1; k < tournament_size; ++k) {
        const Individual& cand = pop[pick(rng)];
        if (cand.fitness < best.fitness) {
            best = cand;
        }
    }
    return best;
}

// Order Crossover (OX1). Copies a slice from p1 into the child, then
// fills the rest with p2's cities in order, skipping any already used.
void order_crossover(const Genome& p1, const Genome& p2,
                     Genome& child, mt19937_64& rng) {
    const size_t n = p1.size();
    if (n < 2) { child = p1; return; }

    child.assign(n, -1);

    uniform_int_distribution<size_t> pick_i(0, n - 2);
    const size_t i = pick_i(rng);
    uniform_int_distribution<size_t> pick_j(i + 1, n - 1);
    const size_t j = pick_j(rng);

    vector<char> used(n, 0);
    for (size_t k = i; k <= j; ++k) {
        child[k]    = p1[k];
        used[p1[k]] = 1;
    }

    size_t write = (j + 1) % n;
    for (size_t k = 0; k < n; ++k) {
        const size_t idx = (j + 1 + k) % n;
        const int city = p2[idx];
        if (used[city]) continue;
        child[write] = city;
        write = (write + 1) % n;
    }
}

// Swap two positions with probability mutation_rate per position.
void swap_mutation(Genome& genome, double mutation_rate, mt19937_64& rng) {
    const size_t n = genome.size();
    if (n < 2) return;

    uniform_real_distribution<double> roll(0.0, 1.0);
    uniform_int_distribution<size_t>  pick(0, n - 1);

    for (size_t k = 0; k < n; ++k) {
        if (roll(rng) < mutation_rate) {
            const size_t other = pick(rng);
            if (other != k) swap(genome[k], genome[other]);
        }
    }
}

// One generation:
//   1. Sort by fitness.
//   2. Copy the best few unchanged (elitism).
//   3. Fill the rest with children from tournament + crossover + mutation.
//   4. Swap in.
// Child creation runs in parallel; each thread has its own RNG seeded
// from a single base seed.
void evolve_generation(vector<Individual>& pop,
                       const GAConfig& cfg,
                       const vector<vector<double>>& dist_matrix,
                       mt19937_64& rng) {
    (void)dist_matrix;

    sort(pop.begin(), pop.end(),
         [](const Individual& a, const Individual& b) {
             return a.fitness < b.fitness;
         });

    const int pop_size    = cfg.population_size;
    const int elite_count = min(cfg.elitism_count, pop_size);

    vector<Individual> next(static_cast<size_t>(pop_size));

    for (int e = 0; e < elite_count; ++e) {
        next[e] = pop[e];
    }

    // One base seed; each thread derives its own RNG from it.
    const uint64_t base_seed = rng();

    #pragma omp parallel
    {
        mt19937_64 local_rng(
            base_seed +
            static_cast<uint64_t>(omp_get_thread_num()) * 0x9E3779B97F4A7C15ULL);

        uniform_real_distribution<double> coin(0.0, 1.0);

        #pragma omp for schedule(dynamic, 4)
        for (int i = elite_count; i < pop_size; ++i) {
            const Individual p1 = tournament_select(pop, cfg.tournament_size, local_rng);
            const Individual p2 = tournament_select(pop, cfg.tournament_size, local_rng);

            Individual child;
            child.genome.reserve(p1.genome.size());

            if (coin(local_rng) < cfg.crossover_rate) {
                order_crossover(p1.genome, p2.genome, child.genome, local_rng);
            } else {
                child.genome = p1.genome;
            }

            swap_mutation(child.genome, cfg.mutation_rate, local_rng);
            child.fitness = numeric_limits<double>::max();

            next[i] = move(child);
        }
    }

    pop = move(next);
}

}