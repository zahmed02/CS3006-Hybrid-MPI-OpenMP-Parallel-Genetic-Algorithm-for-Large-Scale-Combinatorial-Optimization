// MPI island-model migration and global best reduction
#pragma once

#include <mpi.h>
#include <vector>

#include "ga.h"

using namespace std;

namespace pdc {

// Per-rank MPI information
struct MPIInfo {
    int rank = 0;                    // this process's rank
    int size = 1;                    // total number of processes
    int local_population_size = 0;   // population_size / size
};

// Ring migration: each rank sends its best N individuals to the next
// rank and receives the same number from the previous rank.
void migrate_ring(vector<Individual>& local_pop,
                  const MPIInfo& info,
                  int num_migrants);

// Return the single best individual across all ranks
Individual global_best(const Individual& local_best, const MPIInfo& info);

}