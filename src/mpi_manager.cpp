// MPI helpers: ring migration and global best reduction
#include "mpi_manager.h"

#include <algorithm>
#include <cstddef>
#include <vector>

using namespace std;

namespace pdc {

// Exchange the best num_migrants individuals with the next and previous
// rank in a ring. Uses MPI_Sendrecv for both genome and fitness payloads.
void migrate_ring(vector<Individual>& local_pop,
                  const MPIInfo& info,
                  int num_migrants) {
    if (info.size < 2) return;
    if (num_migrants <= 0) return;
    if (local_pop.empty()) return;

    const int pop_size = static_cast<int>(local_pop.size());
    if (num_migrants > pop_size) num_migrants = pop_size;

    const int next_rank = (info.rank + 1) % info.size;
    const int prev_rank = (info.rank - 1 + info.size) % info.size;
    const int N = static_cast<int>(local_pop[0].genome.size());

    // Bring the best num_migrants to the front
    partial_sort(
        local_pop.begin(),
        local_pop.begin() + num_migrants,
        local_pop.end(),
        [](const Individual& a, const Individual& b) {
            return a.fitness < b.fitness;
        });

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

    // Two Sendrecv calls: genome ints (tag 100) and fitness doubles (tag 101)
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

    // Sort so we can find the worst individuals at the tail
    sort(local_pop.begin(), local_pop.end(),
         [](const Individual& a, const Individual& b) {
             return a.fitness < b.fitness;
         });

    // Replace the worst num_migrants with the received ones
    for (int k = 0; k < num_migrants; ++k) {
        Individual& victim = local_pop[pop_size - 1 - k];
        const size_t offset = static_cast<size_t>(k) * N;
        victim.genome.assign(recv_genome.begin() + offset,
                             recv_genome.begin() + offset + N);
        victim.fitness = recv_fit[k];
    }
}

// Return the best individual across all ranks (same result on every rank).
// Uses MPI_Allreduce with MINLOC to find the winner, then broadcasts it.
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

}