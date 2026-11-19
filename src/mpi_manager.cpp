// MPI helpers: ring migration and global best reduction
#include "mpi_manager.h"

#include <algorithm>
#include <cstddef>
#include <vector>

using namespace std;

namespace pdc {

namespace {
bool better(const Individual& a, const Individual& b) {
    return a.fitness < b.fitness;
}
}  // namespace

// Exchange the best num_migrants individuals with the next and previous
// rank in a ring (MPI_Sendrecv, so no deadlock for any ring size).
// The incoming migrants replace the worst individuals of the local island.
void migrate_ring(vector<Individual>& local_pop,
                  const MPIInfo& info,
                  int num_migrants) {
    if (info.size < 2) return;
    if (num_migrants <= 0) return;
    if (local_pop.empty()) return;

    const int pop_size = static_cast<int>(local_pop.size());

    // The best m and the worst m must not overlap
    num_migrants = min(num_migrants, pop_size / 2);
    if (num_migrants <= 0) return;

    const int next_rank = (info.rank + 1) % info.size;
    const int prev_rank = (info.rank - 1 + info.size) % info.size;
    const int N = static_cast<int>(local_pop[0].genome.size());

    // Best num_migrants to the front (only that prefix is ordered)
    partial_sort(local_pop.begin(), local_pop.begin() + num_migrants,
                 local_pop.end(), better);

    // Pack send buffers
    vector<int>    send_genome(static_cast<size_t>(num_migrants) * N);
    vector<double> send_fit(static_cast<size_t>(num_migrants));

    for (int k = 0; k < num_migrants; ++k) {
        const size_t offset = static_cast<size_t>(k) * N;
        copy(local_pop[k].genome.begin(), local_pop[k].genome.end(),
             send_genome.begin() + offset);
        send_fit[k] = local_pop[k].fitness;
    }

    vector<int>    recv_genome(static_cast<size_t>(num_migrants) * N);
    vector<double> recv_fit(static_cast<size_t>(num_migrants));

    // Genome ints (tag 100), then fitness doubles (tag 101)
    MPI_Sendrecv(send_genome.data(), num_migrants * N, MPI_INT,
                 next_rank, 100,
                 recv_genome.data(), num_migrants * N, MPI_INT,
                 prev_rank, 100,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    MPI_Sendrecv(send_fit.data(), num_migrants, MPI_DOUBLE,
                 next_rank, 101,
                 recv_fit.data(), num_migrants, MPI_DOUBLE,
                 prev_rank, 101,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Put the num_migrants worst individuals at the tail (no full sort needed)
    nth_element(local_pop.begin() + num_migrants,
                local_pop.end() - num_migrants,
                local_pop.end(), better);

    for (int k = 0; k < num_migrants; ++k) {
        Individual& victim = local_pop[pop_size - 1 - k];
        const size_t offset = static_cast<size_t>(k) * N;
        victim.genome.assign(recv_genome.begin() + offset,
                             recv_genome.begin() + offset + N);
        victim.fitness = recv_fit[k];
    }
}

// Return the best individual across all ranks (same result on every rank).
// MPI_Allreduce with MINLOC finds the winner, then it broadcasts its tour.
Individual global_best(const Individual& local_best, const MPIInfo& info) {
    if (info.size < 2) return local_best;

    struct DoubleInt {
        double value;
        int    rank;
    };

    DoubleInt local_pair{local_best.fitness, info.rank};
    DoubleInt global_pair{0.0, -1};

    MPI_Allreduce(&local_pair, &global_pair, 1,
                  MPI_DOUBLE_INT, MPI_MINLOC,
                  MPI_COMM_WORLD);

    const int N = static_cast<int>(local_best.genome.size());
    vector<int> genome_buf(static_cast<size_t>(N));

    if (info.rank == global_pair.rank) {
        copy(local_best.genome.begin(), local_best.genome.end(),
             genome_buf.begin());
    }

    MPI_Bcast(genome_buf.data(), N, MPI_INT, global_pair.rank, MPI_COMM_WORLD);

    Individual result;
    result.genome  = move(genome_buf);
    result.fitness = global_pair.value;
    return result;
}

}  // namespace pdc