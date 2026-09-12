// Hybrid MPI + OpenMP GA driver: parses args, loads/generates the TSP
// instance, splits the population across MPI ranks, runs the main loop.
#include "fitness.h"
#include "ga.h"
#include "mpi_manager.h"
#include "utils.h"

#include <mpi.h>
#include <omp.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

namespace {

struct CLIOptions {
    string   data_path   = "data/raw/tsp_10000.txt";
    string   out_csv     = "results/benchmarks/trace.csv";
    int      num_cities  = 10000;
    int      generations = 500;
    int      pop_size    = 1000;
    uint64_t seed        = 42;
    bool     generate    = false;
    bool     verbose     = true;
    bool     show_help   = false;
};

void print_help() {
    cout <<
        "Usage: pdc_ga [options]\n\n"
        "  --data PATH          TSP coordinate file\n"
        "  --out PATH           Per-generation CSV trace\n"
        "  --cities N           Cities to generate (default 10000)\n"
        "  --generations G      GA generations (default 500)\n"
        "  --pop P              Global population across all ranks (default 1000)\n"
        "  --seed S             RNG seed (default 42)\n"
        "  --generate           Generate synthetic coordinates instead of loading\n"
        "  --quiet              Suppress progress output on rank 0\n"
        "  --help, -h           Show this help\n";
}

CLIOptions parse_args(int argc, char** argv) {
    CLIOptions o;
    for (int i = 1; i < argc; ++i) {
        string a = argv[i];
        if      (a == "--data"        && i+1 < argc) o.data_path   = argv[++i];
        else if (a == "--out"         && i+1 < argc) o.out_csv     = argv[++i];
        else if (a == "--cities"      && i+1 < argc) o.num_cities  = stoi(argv[++i]);
        else if (a == "--generations" && i+1 < argc) o.generations = stoi(argv[++i]);
        else if (a == "--pop"         && i+1 < argc) o.pop_size    = stoi(argv[++i]);
        else if (a == "--seed"        && i+1 < argc) o.seed        = stoull(argv[++i]);
        else if (a == "--generate")                  o.generate    = true;
        else if (a == "--quiet")                     o.verbose     = false;
        else if (a == "--help" || a == "-h")         o.show_help   = true;
    }
    return o;
}

// Split `global_pop` across ranks as evenly as possible.
int local_pop_size(int global_pop, int rank, int nranks) {
    const int base = global_pop / nranks;
    const int rem  = global_pop % nranks;
    return base + (rank < rem ? 1 : 0);
}

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    pdc::MPIInfo info;
    MPI_Comm_rank(MPI_COMM_WORLD, &info.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &info.size);

    CLIOptions opts = parse_args(argc, argv);

    if (opts.show_help) {
        if (info.rank == 0) print_help();
        MPI_Finalize();
        return 0;
    }

    if (opts.pop_size < info.size) {
        if (info.rank == 0) {
            cerr << "Error: --pop (" << opts.pop_size
                 << ") must be >= number of MPI ranks (" << info.size << ")\n";
        }
        MPI_Finalize();
        return 1;
    }

    // Load or generate coordinates
    vector<pair<double, double>> coords;
    try {
        if (opts.generate) {
            coords = pdc::generate_random_tsp(opts.num_cities,
                                              static_cast<unsigned>(opts.seed));
        } else {
            coords = pdc::load_tsp_coordinates(opts.data_path);
        }
    } catch (const exception& e) {
        cerr << "[rank " << info.rank << "] load error: " << e.what() << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const int N = static_cast<int>(coords.size());

    if (opts.verbose && info.rank == 0) {
        cout << "=== Parallel GA for TSP ===\n"
             << "MPI ranks:         " << info.size << "\n"
             << "OpenMP threads:    " << omp_get_max_threads() << "\n"
             << "Cities (N):        " << N << "\n"
             << "Generations:       " << opts.generations << "\n"
             << "Global pop:        " << opts.pop_size << "\n"
             << "Data path:         " << opts.data_path << "\n"
             << "Trace CSV:         " << opts.out_csv << "\n";
    }

    // Build distance matrix (OpenMP-parallel)
    pdc::Timer t_build;
    t_build.start();
    auto dist = pdc::build_distance_matrix(coords);
    const double build_time = t_build.elapsed_seconds();

    if (opts.verbose && info.rank == 0) {
        cout << "Distance matrix:   " << N << " x " << N
             << " built in " << fixed << setprecision(2)
             << build_time << " s\n";
    }

    info.local_population_size = local_pop_size(opts.pop_size, info.rank, info.size);

    // Each rank gets a distinct RNG so islands explore independently.
    mt19937_64 rng(opts.seed +
                   static_cast<uint64_t>(info.rank) * 0x9E3779B97F4A7C15ULL);

    // GA config uses the LOCAL population size
    pdc::GAConfig cfg;
    cfg.population_size = info.local_population_size;
    cfg.num_generations = opts.generations;
    cfg.seed            = opts.seed;

    // Initial population + first fitness evaluation
    vector<pdc::Individual> pop(static_cast<size_t>(info.local_population_size));
    pdc::initialize_population(pop, N, rng);
    pdc::evaluate_population(pop, dist);

    // Best individual on this rank
    pdc::Individual best_local = *min_element(
        pop.begin(), pop.end(),
        [](const pdc::Individual& a, const pdc::Individual& b) {
            return a.fitness < b.fitness;
        });

    pdc::Individual best_global = pdc::global_best(best_local, info);

    // Rank 0 writes a fresh CSV
    const string csv_header = "generation,global_best,rank0_avg,elapsed_sec";
    if (info.rank == 0) {
        pdc::ensure_directory("results/benchmarks");
        remove(opts.out_csv.c_str());

        ostringstream row0;
        row0 << "0," << fixed << setprecision(6)
             << best_global.fitness << ",,0.0";
        pdc::append_csv_row(opts.out_csv, csv_header, row0.str());
    }

    if (opts.verbose && info.rank == 0) {
        cout << "\nGen   | Global Best | Rank0 Avg | Elapsed (s)\n"
             << "------+-------------+-----------+------------\n";
    }

    // Main GA loop
    pdc::Timer t_loop;
    t_loop.start();

    const int log_every = max(1, opts.generations / 50);

    for (int gen = 1; gen <= opts.generations; ++gen) {
        pdc::evolve_generation(pop, cfg, dist, rng);
        pdc::evaluate_population(pop, dist);

        const pdc::Individual this_best = *min_element(
            pop.begin(), pop.end(),
            [](const pdc::Individual& a, const pdc::Individual& b) {
                return a.fitness < b.fitness;
            });
        if (this_best.fitness < best_local.fitness) {
            best_local = this_best;
        }

        // Ring migration every migration_interval generations
        if (info.size > 1 && (gen % cfg.migration_interval) == 0) {
            pdc::migrate_ring(pop, info, cfg.num_migrants);
        }

        // Log and sync the global best at boundaries
        if ((gen % log_every == 0) || gen == opts.generations) {
            best_global = pdc::global_best(best_local, info);
            const double elapsed = t_loop.elapsed_seconds();

            if (info.rank == 0) {
                double sum = 0.0;
                for (const auto& ind : pop) sum += ind.fitness;
                const double avg = sum / static_cast<double>(pop.size());

                if (opts.verbose) {
                    printf("%5d | %11.4f | %9.4f | %10.3f\n",
                           gen, best_global.fitness, avg, elapsed);
                }

                ostringstream row;
                row << gen << "," << fixed << setprecision(6)
                    << best_global.fitness << ","
                    << avg << ","
                    << setprecision(4) << elapsed;
                pdc::append_csv_row(opts.out_csv, csv_header, row.str());
            }
        }
    }

    // Final sync so rank 0 has the true global best
    best_global = pdc::global_best(best_local, info);

    const double loop_time = t_loop.elapsed_seconds();

    if (info.rank == 0) {
        cout << "\n=== Final Result ===\n"
             << "Best tour length:  " << fixed << setprecision(6)
             << best_global.fitness << "\n"
             << "Build time:        " << setprecision(3)
             << build_time << " s\n"
             << "Loop time:         " << loop_time << " s\n"
             << "Trace CSV:         " << opts.out_csv << "\n";
    }

    MPI_Finalize();
    return 0;
}