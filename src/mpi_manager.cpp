// ============================================================================
// mpi_manager.cpp — MPI island-model orchestration
//
// Implements the two cross-island communication primitives used by the
// hybrid GA:
//
//   1. migrate_ring()  — ring-topology exchange of the best individuals
//   2. global_best()   — reduction to find the single best individual
//                        across all MPI ranks
//
// Ring topology
// -------------
//   rank 0 -> rank 1 -> rank 2 -> ... -> rank (P-1) -> rank 0
//
// Each island sends its best `num_migrants` individuals to its successor
// and receives the same number from its predecessor. The received migrants
// replace the receiver's worst individuals (unconditional replacement).
//
// Deadlock avoidance
// ------------------
// MPI_Sendrecv is used for every exchange. Unlike paired MPI_Send/MPI_Recv,
// MPI_Sendrecv cannot deadlock even when all ranks simultaneously try to
// send to each other, because it uses internal buffering and completes the
// receive side in lock-step with the send side.
// ============================================================================
#include "mpi_manager.h"

#include <algorithm>
#include <cstddef>
#include <vector>

namespace pdc {

// ----------------------------------------------------------------------------
// migrate_ring
//
// Exchanges `num_migrants` top-fitness individuals between adjacent ranks
// in a ring topology.
//
// Steps per rank:
//   1. partial_sort the local population to bring the best migrants to the
//      front (O(P log m) — cheaper than a full sort).
//   2. Pack genomes + fitnesses into contiguous buffers.
//   3. Two MPI_Sendrecv calls: one for the genome ints, one for the fitness
//      doubles. (Packing into a single custom MPI datatype is possible but
//      adds complexity for negligible bandwidth gain — MPI doubles the
//      message for free.)
//   4. Full sort to find the worst individuals.
//   5. Overwrite the worst `num_migrants` with the received migrants.
//
// Design note: the migrants carry their *original* fitness values. This is
// correct because the fitness function (tour length) depends only on the
// genome, and the receiver is guaranteed to have the same distance matrix.
// ============================================================================
void migrate_ring(std::vector<Individual>& local_pop,
                  const MPIInfo& info,
                  int num_migrants) {
    // ---- Guards ----
    if (info.size < 2) return;         // single island, no migration
    if (num_migrants <= 0) return;
    if (local_pop.empty()) return;

    const int pop_size = static_cast<int>(local_pop.size());
    if (num_migrants > pop_size) {
        num_migrants = pop_size;       // clamp to avoid overflow
    }

    const int next_rank = (info.rank + 1) % info.size;
    const int prev_rank = (info.rank - 1 + info.size) % info.size;
    const int N = static_cast<int>(local_pop[0].genome.size());

    // ---- Step 1: bring the best `num_migrants` to the front ----
    std::partial_sort(
        local_pop.begin(),
        local_pop.begin() + num_migrants,
        local_pop.end(),
        [](const Individual& a, const Individual& b) {
            return a.fitness < b.fitness;   // minimization
        });

    // ---- Step 2: pack send buffers ----
    std::vector<int>    send_genome(static_cast<std::size_t>(num_migrants) * N);
    std::vector<double> send_fit(static_cast<std::size_t>(num_migrants));

    for (int k = 0; k < num_migrants; ++k) {
        const std::size_t offset = static_cast<std::size_t>(k) * N;
        std::copy(local_pop[k].genome.begin(), local_pop[k].genome.end(),
                  send_genome.begin() + offset);
        send_fit[k] = local_pop[k].fitness;
    }

    // ---- Step 3: receive buffers and MPI_Sendrecv ----
    std::vector<int>    recv_genome(static_cast<std::size_t>(num_migrants) * N);
    std::vector<double> recv_fit(static_cast<std::size_t>(num_migrants));

    // Tag 100: genome payload
    MPI_Sendrecv(send_genome.data(), num_migrants * N, MPI_INT,
                 next_rank, /*tag=*/100,
                 recv_genome.data(), num_migrants * N, MPI_INT,
                 prev_rank, /*tag=*/100,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Tag 101: fitness payload (separate to avoid custom MPI_Datatype)
    MPI_Sendrecv(send_fit.data(), num_migrants, MPI_DOUBLE,
                 next_rank, /*tag=*/101,
                 recv_fit.data(), num_migrants, MPI_DOUBLE,
                 prev_rank, /*tag=*/101,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // ---- Step 4: full sort to identify the worst individuals ----
    std::sort(local_pop.begin(), local_pop.end(),
              [](const Individual& a, const Individual& b) {
                  return a.fitness < b.fitness;
              });

    // ---- Step 5: overwrite the worst `num_migrants` ----
    // After sorting, worst individuals are at the tail (largest fitness).
    for (int k = 0; k < num_migrants; ++k) {
        Individual& victim = local_pop[pop_size - 1 - k];
        const std::size_t offset = static_cast<std::size_t>(k) * N;
        victim.genome.assign(recv_genome.begin() + offset,
                             recv_genome.begin() + offset + N);
        victim.fitness = recv_fit[k];
    }
}

// ----------------------------------------------------------------------------
// global_best
//
// Finds the single best individual across all MPI ranks and returns it on
// every rank. Uses a two-stage approach:
//
//   Stage 1 — MPI_Allreduce with MPI_MINLOC on a (fitness, rank) pair.
//             After this, every rank knows the global minimum fitness AND
//             the rank that owns it (ties broken by lowest rank).
//
//   Stage 2 — MPI_Bcast the winning rank's genome to everyone.
//
// Why not just use MPI_Allreduce on the fitness and then MPI_Bcast?
//   Because multiple ranks can have the same minimum fitness. Without
//   knowing *which* rank owns the best genome, a broadcast is ambiguous.
//   MPI_MINLOC disambiguates.
// ----------------------------------------------------------------------------
Individual global_best(const Individual& local_best, const MPIInfo& info) {
    // Single-process fallback
    if (info.size < 2) return local_best;

    // ---- Stage 1: find global minimum fitness + owning rank ----
    struct DoubleInt {
        double value;
        int    rank;
    };

    DoubleInt local_pair{local_best.fitness, info.rank};
    DoubleInt global_pair{0.0, -1};

    MPI_Allreduce(&local_pair, &global_pair, 1,
                  MPI_DOUBLE_INT, MPI_MINLOC,
                  MPI_COMM_WORLD);

    // ---- Stage 2: broadcast the winner's genome ----
    const int N = static_cast<int>(local_best.genome.size());
    std::vector<int> genome_buf(static_cast<std::size_t>(N));

    if (info.rank == global_pair.rank) {
        std::copy(local_best.genome.begin(), local_best.genome.end(),
                  genome_buf.begin());
    }

    MPI_Bcast(genome_buf.data(), N, MPI_INT,
              global_pair.rank, MPI_COMM_WORLD);

    Individual result;
    result.genome  = std::move(genome_buf);
    result.fitness = global_pair.value;
    return result;
}

}  // namespace pdc