// ============================================================================
// ga.cpp — Genetic algorithm operators for the TSP
//
// Implements a generational GA with:
//   - Random-permutation initialization
//   - OpenMP-parallel fitness evaluation (the hot path)
//   - k-way tournament selection
//   - Order Crossover (OX1) — permutation-preserving
//   - Swap mutation
//   - Elitism
//
// Design notes:
//   - The population is always sorted by fitness (ascending) after each
//     generation. Sorting is O(P log P), negligible compared to the
//     O(P * N) fitness evaluation where N = number of cities.
//   - Newly created children have fitness = +inf until evaluated. They are
//     evaluated in the next call to evaluate_population().
//   - The child-generation loop is sequential (RNG state is shared). Only
//     fitness evaluation is parallelized — this is the correct split, since
//     fitness evaluation dominates runtime (O(N) per individual vs O(1)
//     for each child-creation step).
// ============================================================================
#include "ga.h"
#include <omp.h>
#include <algorithm>
#include <limits>
#include <numeric>

#include "fitness.h"

namespace pdc {

// ----------------------------------------------------------------------------
// initialize_population
//
// Fills `pop` with random permutations of [0, num_cities). Each individual
// is an independent random shuffle of the identity permutation.
//
// Assumes `pop` was already sized to the desired population size.
// ----------------------------------------------------------------------------
void initialize_population(std::vector<Individual>& pop,
                           int num_cities,
                           std::mt19937_64& rng) {
    for (auto& ind : pop) {
        ind.genome.resize(static_cast<std::size_t>(num_cities));
        std::iota(ind.genome.begin(), ind.genome.end(), 0);
        std::shuffle(ind.genome.begin(), ind.genome.end(), rng);
        ind.fitness = std::numeric_limits<double>::max();
    }
}

// ----------------------------------------------------------------------------
// evaluate_population
//
// Computes the tour length for every individual. This is the hot path of
// the entire GA — parallelized with OpenMP.
//
// Schedule: static — every individual's tour length costs exactly O(N) and
// involves zero branching, so equal-sized chunks per thread are optimal.
//
// Thread safety: each iteration writes only to pop[i].fitness. No shared
// writes → no races → no synchronization needed.
// ----------------------------------------------------------------------------
void evaluate_population(std::vector<Individual>& pop,
                         const std::vector<std::vector<double>>& dist_matrix) {
    const std::ptrdiff_t n = static_cast<std::ptrdiff_t>(pop.size());

    #pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < n; ++i) {
        pop[i].fitness = tour_length(pop[i].genome, dist_matrix);
    }
}

// ----------------------------------------------------------------------------
// tournament_select
//
// Picks `tournament_size` individuals uniformly at random and returns a
// copy of the one with the smallest fitness (TSP is a minimization problem).
//
// Larger tournament_size → stronger selection pressure (faster convergence,
// higher risk of premature convergence).
// ----------------------------------------------------------------------------
Individual tournament_select(const std::vector<Individual>& pop,
                             int tournament_size,
                             std::mt19937_64& rng) {
    std::uniform_int_distribution<std::size_t> pick(0, pop.size() - 1);

    Individual best = pop[pick(rng)];
    for (int k = 1; k < tournament_size; ++k) {
        const Individual& cand = pop[pick(rng)];
        if (cand.fitness < best.fitness) {
            best = cand;
        }
    }
    return best;
}

// ----------------------------------------------------------------------------
// order_crossover (OX1)
//
// The classic Order Crossover. Guarantees the child is a valid permutation.
//
// Algorithm:
//   1. Pick two distinct cut points i < j in [0, N-1].
//   2. Copy p1[i..j] verbatim into child[i..j].
//   3. Walk p2 starting at position j+1 (wrapping around). For each city in
//      p2 that is NOT already in the child, place it in the next empty
//      position of the child (starting at j+1, wrapping around).
//
// Example (N=9):
//   p1     = [1 2 3 | 4 5 6 7 | 8 9]   (i=3, j=6)
//   p2     = [9 3 7 8 2 6 | 5 1 4]
//   child  = [3 8 2 | 4 5 6 7 | 1 9]
//                    ^^^^^^^ from p1
//            ^^^^^ ^^^ from p2, in order, skipping {4,5,6,7}
// ----------------------------------------------------------------------------
void order_crossover(const Genome& p1,
                     const Genome& p2,
                     Genome& child,
                     std::mt19937_64& rng) {
    const std::size_t n = p1.size();
    if (n < 2) {
        child = p1;
        return;
    }

    child.assign(n, -1);

    // Pick i < j (guaranteed distinct)
    std::uniform_int_distribution<std::size_t> pick_i(0, n - 2);
    const std::size_t i = pick_i(rng);
    std::uniform_int_distribution<std::size_t> pick_j(i + 1, n - 1);
    const std::size_t j = pick_j(rng);

    // Mark cities from p1[i..j] as used and copy them verbatim
    std::vector<char> used(n, 0);
    for (std::size_t k = i; k <= j; ++k) {
        child[k]   = p1[k];
        used[p1[k]] = 1;
    }

    // Fill remaining positions from p2, in cyclic order starting at j+1
    std::size_t write = (j + 1) % n;
    for (std::size_t k = 0; k < n; ++k) {
        const std::size_t idx = (j + 1 + k) % n;
        const int city = p2[idx];
        if (used[city]) continue;
        child[write] = city;
        write = (write + 1) % n;
    }
}

// ----------------------------------------------------------------------------
// swap_mutation
//
// For each position k in the genome, with probability mutation_rate,
// swap genome[k] with a uniformly random position.
//
// Note: even if the same position gets swapped multiple times, the result
// remains a valid permutation. The mutation rate therefore controls the
// *expected number* of swaps per tour, not the number of *distinct* cities
// affected.
// ----------------------------------------------------------------------------
void swap_mutation(Genome& genome,
                   double mutation_rate,
                   std::mt19937_64& rng) {
    const std::size_t n = genome.size();
    if (n < 2) return;

    std::uniform_real_distribution<double>  roll(0.0, 1.0);
    std::uniform_int_distribution<std::size_t> pick(0, n - 1);

    for (std::size_t k = 0; k < n; ++k) {
        if (roll(rng) < mutation_rate) {
            const std::size_t other = pick(rng);
            if (other != k) {
                std::swap(genome[k], genome[other]);
            }
        }
    }
}

// ----------------------------------------------------------------------------
// evolve_generation
//
// Produces the next generation in place:
//   1. Sort the current population by fitness (best first).
//   2. Copy the top `elitism_count` individuals to the next generation
//      unchanged (elitism preserves the best solution found so far).
//   3. Fill the rest of the next generation by:
//        a. Tournament-select two parents.
//        b. With probability crossover_rate, apply OX1.
//           Otherwise, clone parent 1.
//        c. Apply swap mutation to the child.
//   4. Replace `pop` with the new generation.
//
// The new individuals have their fitness set to +inf — they will be scored
// by the next call to evaluate_population().
// ----------------------------------------------------------------------------
void evolve_generation(std::vector<Individual>& pop,
                       const GAConfig& cfg,
                       const std::vector<std::vector<double>>& dist_matrix,
                       std::mt19937_64& rng) {
    // (dist_matrix is unused here — reserved for future local-search hooks.)
    (void)dist_matrix;

    // ---- Step 1: sort by fitness ascending (sequential, O(P log P)) ----
    std::sort(pop.begin(), pop.end(),
              [](const Individual& a, const Individual& b) {
                  return a.fitness < b.fitness;
              });

    const int pop_size    = cfg.population_size;
    const int elite_count = std::min(cfg.elitism_count, pop_size);

    std::vector<Individual> next(static_cast<std::size_t>(pop_size));

    // ---- Step 2: elitism (tiny, keep sequential) ----
    for (int e = 0; e < elite_count; ++e) {
        next[e] = pop[e];
    }

    // ---- Step 3: parallel child creation ----
    //
    // RNG design:
    //   The master `rng` is only used ONCE per generation to draw a
    //   `base_seed`. Each OpenMP thread derives its own mt19937_64 from
    //   `base_seed + thread_id * GOLDEN_RATIO`. This gives us:
    //     - No data races: each thread writes only to its own slice of
    //       `next[]`, and reads only from the immutable `pop`.
    //     - Independence: different threads explore different parts of
    //       the search space.
    //     - Determinism: results are reproducible given a fixed
    //       OMP_NUM_THREADS. (Reproducibility across *different* thread
    //       counts would require index-based seeding, which doubles the
    //       RNG construction cost per child.)
    //
    // Schedule: `dynamic, 4` — child creation cost varies (a crossover
    // loop can finish early on some inputs), so dynamic scheduling
    // reduces the tail effect where one thread finishes last.
    const uint64_t base_seed = rng();

    #pragma omp parallel
    {
        std::mt19937_64 local_rng(
            base_seed +
            static_cast<uint64_t>(omp_get_thread_num()) * 0x9E3779B97F4A7C15ULL);

        std::uniform_real_distribution<double> coin(0.0, 1.0);

        #pragma omp for schedule(dynamic, 4)
        for (int i = elite_count; i < pop_size; ++i) {
            const Individual p1 = tournament_select(pop, cfg.tournament_size, local_rng);
            const Individual p2 = tournament_select(pop, cfg.tournament_size, local_rng);

            Individual child;
            child.genome.reserve(p1.genome.size());

            if (coin(local_rng) < cfg.crossover_rate) {
                order_crossover(p1.genome, p2.genome, child.genome, local_rng);
            } else {
                child.genome = p1.genome;   // clone parent 1
            }

            swap_mutation(child.genome, cfg.mutation_rate, local_rng);
            child.fitness = std::numeric_limits<double>::max();

            next[i] = std::move(child);
        }
    }

    // ---- Step 4: swap in ----
    pop = std::move(next);
}

}  // namespace pdc