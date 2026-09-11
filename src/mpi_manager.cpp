// ============================================================================
// mpi_manager.cpp — Ring migration + global best reduction
// ============================================================================
#include "mpi_manager.h"

#include <algorithm>

namespace pdc {

void migrate_ring(std::vector<Individual>& local_pop,
                  const MPIInfo& info,
                  int num_migrants) {
    // TODO: MPI_Sendrecv with neighbors in ring
}

Individual global_best(const Individual& local_best,
                       const MPIInfo& info) {
    // TODO: MPI_Allreduce on fitness, broadcast best genome
    return local_best;
}

}  // namespace pdc