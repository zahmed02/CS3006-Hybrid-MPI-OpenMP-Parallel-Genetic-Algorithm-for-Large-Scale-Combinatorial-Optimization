// ============================================================================
// mpi_manager.h — MPI island-model orchestration and migration
// ============================================================================
#pragma once

#include <mpi.h>

#include <vector>

#include "ga.h"

namespace pdc {

struct MPIInfo {
    int rank = 0;                    // this process's rank
    int size = 1;                    // total number of processes
    int local_population_size = 0;   // population_size / size
};

// Ring-topology migration: each island sends its best `num_migrants`
// individuals to the next rank and receives the same number from the
// previous rank. Migration is synchronous via MPI_Sendrecv.
void migrate_ring(std::vector<Individual>& local_pop,
                  const MPIInfo& info,
                  int num_migrants);

// Reduce the global best individual across all ranks.
// Returns the same best Individual on every rank.
Individual global_best(const Individual& local_best,
                       const MPIInfo& info);

}  // namespace pdc