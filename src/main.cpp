// ============================================================================
// main.cpp — Entry point: parses args, runs hybrid MPI+OpenMP GA
// ============================================================================
#include <mpi.h>
#include <omp.h>

#include <iostream>
#include <string>

#include "fitness.h"
#include "ga.h"
#include "mpi_manager.h"
#include "utils.h"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    pdc::MPIInfo info;
    MPI_Comm_rank(MPI_COMM_WORLD, &info.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &info.size);

    // ---- Parse command-line arguments ----
    std::string data_path = "data/raw/tsp_10000.txt";
    int  num_cities       = 10000;
    int  generations      = 500;
    int  pop_size         = 1000;
    bool generate_data    = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--data"        && i + 1 < argc) data_path   = argv[++i];
        if (arg == "--cities"      && i + 1 < argc) num_cities  = std::stoi(argv[++i]);
        if (arg == "--generations" && i + 1 < argc) generations = std::stoi(argv[++i]);
        if (arg == "--pop"         && i + 1 < argc) pop_size    = std::stoi(argv[++i]);
        if (arg == "--generate")                    generate_data = true;
    }

    if (info.rank == 0) {
        std::cout << "=== Parallel GA for TSP ===\n"
                  << "MPI ranks:      " << info.size << "\n"
                  << "OpenMP threads: " << omp_get_max_threads() << "\n"
                  << "Cities:         " << num_cities << "\n"
                  << "Generations:    " << generations << "\n"
                  << "Population:     " << pop_size << "\n"
                  << "Data path:      " << data_path << "\n";
    }

    // TODO: full GA loop will be wired here in the next step

    MPI_Finalize();
    return 0;
}