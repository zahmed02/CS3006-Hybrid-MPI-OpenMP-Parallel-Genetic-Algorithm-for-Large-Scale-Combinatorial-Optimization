// Hybrid MPI + OpenMP memetic GA driver: parses args, loads/generates the TSP
// instance, splits the population across MPI ranks (island model), runs the
// main loop and reports timings and the best tour.
#include "fitness.h"
#include "ga.h"
#include "mpi_manager.h"
#include "utils.h"

#include <mpi.h>
#include <omp.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
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
    string   tour_out;                       // optional: write best tour here
    int      num_cities  = 10000;
    int      generations = 500;
    int      pop_size    = 1000;
    int      knn         = 10;
    int      tournament  = 5;
    int      elite       = 2;
    int      mig_every   = 25;
    int      migrants    = 5;
    double   crossover   = 0.85;
    double   mutation    = 0.20;
    bool     local_search = true;
    bool     use_oropt    = true;
    pdc::InitMode init_mode = pdc::InitMode::SpaceFilling;
    uint64_t seed        = 42;
    bool     generate    = false;
    bool     verbose     = true;
    bool     show_help   = false;
};

void print_help() {
    cout <<
        "Usage: pdc_ga [options]\n\n"
        "  --data PATH            TSP coordinate file\n"
        "  --out PATH             Per-generation CSV trace\n"
        "  --tour-out PATH        Write the best tour (one city per line)\n"
        "  --cities N             Cities to generate (default 10000)\n"
        "  --generations G        GA generations (default 500)\n"
        "  --pop P                Global population across all ranks (default 1000)\n"
        "  --seed S               RNG seed (default 42)\n"
        "  --ls 0|1               2-opt local search on every child (default 1)\n"
        "  --oropt 0|1            Add Or-opt moves to the local search (default 1)\n"
        "  --init sfc|random      Starting tours (default sfc = Hilbert curve)\n"
        "  --knn K                Neighbour list size for the local search (default 10)\n"
        "  --crossover R          Crossover probability (default 0.85)\n"
        "  --mutation R           Per-child inversion probability (default 0.20)\n"
        "  --tournament T         Tournament size (default 5)\n"
        "  --elite E              Elites kept per rank (default 2)\n"
        "  --migrate-every G      Generations between migrations (default 25)\n"
        "  --migrants M           Individuals sent per migration (default 5)\n"
        "  --generate             Generate synthetic coordinates instead of loading\n"
        "  --quiet                Suppress progress output on rank 0\n"
        "  --help, -h             Show this help\n";
}

CLIOptions parse_args(int argc, char** argv) {
    CLIOptions o;
    for (int i = 1; i < argc; ++i) {
        const string a = argv[i];
        const bool has_val = (i + 1 < argc);
        if      (a == "--data"          && has_val) o.data_path    = argv[++i];
        else if (a == "--out"           && has_val) o.out_csv      = argv[++i];
        else if (a == "--tour-out"      && has_val) o.tour_out     = argv[++i];
        else if (a == "--cities"        && has_val) o.num_cities   = stoi(argv[++i]);
        else if (a == "--generations"   && has_val) o.generations  = stoi(argv[++i]);
        else if (a == "--pop"           && has_val) o.pop_size     = stoi(argv[++i]);
        else if (a == "--seed"          && has_val) o.seed         = stoull(argv[++i]);
        else if (a == "--ls"            && has_val) o.local_search = stoi(argv[++i]) != 0;
        else if (a == "--oropt"         && has_val) o.use_oropt    = stoi(argv[++i]) != 0;
        else if (a == "--init"          && has_val) {
            const string m = argv[++i];
            o.init_mode = (m == "random") ? pdc::InitMode::Random
                                          : pdc::InitMode::SpaceFilling;
        }
        else if (a == "--knn"           && has_val) o.knn          = stoi(argv[++i]);
        else if (a == "--crossover"     && has_val) o.crossover    = stod(argv[++i]);
        else if (a == "--mutation"      && has_val) o.mutation     = stod(argv[++i]);
        else if (a == "--tournament"    && has_val) o.tournament   = stoi(argv[++i]);
        else if (a == "--elite"         && has_val) o.elite        = stoi(argv[++i]);
        else if (a == "--migrate-every" && has_val) o.mig_every   = stoi(argv[++i]);
        else if (a == "--migrants"      && has_val) o.migrants     = stoi(argv[++i]);
        else if (a == "--generate")                 o.generate     = true;
        else if (a == "--quiet")                    o.verbose      = false;
        else if (a == "--help" || a == "-h")        o.show_help    = true;
    }
    return o;
}

// Split `global_pop` across ranks as evenly as possible.
int local_pop_size(int global_pop, int rank, int nranks) {
    const int base = global_pop / nranks;
    const int rem  = global_pop % nranks;
    return base + (rank < rem ? 1 : 0);
}

// Slowest rank defines the phase time
double max_over_ranks(double local) {
    double global = local;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return global;
}

bool by_fitness(const pdc::Individual& a, const pdc::Individual& b) {
    return a.fitness < b.fitness;
}

}  // namespace

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

    // ---- Load or generate coordinates ----
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
    if (N < 5) {
        if (info.rank == 0) cerr << "Error: need at least 5 cities\n";
        MPI_Finalize();
        return 1;
    }

    if (opts.verbose && info.rank == 0) {
        cout << "=== Parallel memetic GA for TSP ===\n"
             << "MPI ranks:         " << info.size << "\n"
             << "OpenMP threads:    " << omp_get_max_threads() << "\n"
             << "Cities (N):        " << N << "\n"
             << "Generations:       " << opts.generations << "\n"
             << "Global pop:        " << opts.pop_size << "\n"
             << "Local search:      " << (opts.local_search ? (opts.use_oropt ? "2-opt + Or-opt (neighbour lists)" : "2-opt (neighbour lists)") : "off") << "\n"
             << "Initialisation:    "
             << (opts.init_mode == pdc::InitMode::SpaceFilling ? "Hilbert curve" : "random") << "\n"
             << "Data path:         " << (opts.generate ? "(generated)" : opts.data_path) << "\n"
             << "Trace CSV:         " << opts.out_csv << "\n";
    }

    // ---- Phase 1: instance (coordinates + K-nearest-neighbour lists) ----
    MPI_Barrier(MPI_COMM_WORLD);
    pdc::Timer t_setup;
    t_setup.start();
    const pdc::TSPInstance inst = pdc::build_instance(coords, opts.knn);
    const double setup_time = max_over_ranks(t_setup.elapsed_seconds());

    info.local_population_size = local_pop_size(opts.pop_size, info.rank, info.size);

    // Rank-specific stream so the islands explore independently
    const uint64_t rank_seed = pdc::splitmix64(opts.seed + static_cast<uint64_t>(info.rank));
    mt19937_64 rng(rank_seed);

    pdc::GAConfig cfg;
    cfg.population_size    = info.local_population_size;
    cfg.num_generations    = opts.generations;
    cfg.crossover_rate     = opts.crossover;
    cfg.mutation_rate      = opts.mutation;
    cfg.tournament_size    = opts.tournament;
    cfg.elitism_count      = opts.elite;
    cfg.migration_interval = max(1, opts.mig_every);
    cfg.num_migrants       = opts.migrants;
    cfg.local_search       = opts.local_search;
    cfg.use_oropt          = opts.use_oropt;
    cfg.init_mode          = opts.init_mode;
    cfg.seed               = opts.seed;

    // ---- Phase 2: initial population (includes local search if enabled) ----
    MPI_Barrier(MPI_COMM_WORLD);
    pdc::Timer t_init;
    t_init.start();

    vector<pdc::Individual> pop(static_cast<size_t>(info.local_population_size));
    pdc::initialize_population(pop, inst, cfg.init_mode, cfg.local_search, rank_seed,
                               cfg.use_oropt);

    pdc::Individual best_local = *min_element(pop.begin(), pop.end(), by_fitness);
    pdc::Individual best_global = pdc::global_best(best_local, info);
    const double init_time = max_over_ranks(t_init.elapsed_seconds());

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
        cout << "Setup (KNN):       " << fixed << setprecision(3) << setup_time << " s\n"
             << "Init population:   " << init_time << " s  (best "
             << setprecision(4) << best_global.fitness << ")\n"
             << "\nGen   | Global Best | Rank0 Avg | Elapsed (s)\n"
             << "------+-------------+-----------+------------\n";
    }

    // ---- Phase 3: main GA loop ----
    MPI_Barrier(MPI_COMM_WORLD);
    pdc::Timer t_loop;
    t_loop.start();

    const int log_every = max(1, opts.generations / 50);
    vector<pdc::Individual> next_buffer;   // recycled every generation

    for (int gen = 1; gen <= opts.generations; ++gen) {
        pdc::evolve_generation(pop, next_buffer, cfg, inst, rng);

        const pdc::Individual& this_best = *min_element(pop.begin(), pop.end(), by_fitness);
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
    const double loop_time  = max_over_ranks(t_loop.elapsed_seconds());
    const double total_time = setup_time + init_time + loop_time;

    if (info.rank == 0) {
        // Independent check: recompute the length of the returned tour
        const bool   valid  = pdc::is_valid_tour(best_global.genome, N);
        const double recomp = valid ? pdc::tour_length(best_global.genome, inst) : -1.0;

        if (!opts.tour_out.empty() && valid) {
            ofstream f(opts.tour_out);
            for (int c : best_global.genome) f << c << "\n";
        }

        cout << "\n=== Final Result ===\n"
             << "Best tour length:  " << fixed << setprecision(6)
             << best_global.fitness << "\n"
             << "Recomputed length: " << recomp << "\n"
             << "Tour valid:        " << (valid ? "yes" : "NO") << "\n"
             << "Setup time:        " << setprecision(3) << setup_time << " s\n"
             << "Init time:         " << init_time << " s\n"
             << "Loop time:         " << loop_time << " s\n"
             << "Total time:        " << total_time << " s\n"
             << "Trace CSV:         " << opts.out_csv << "\n";
    }

    MPI_Finalize();
    return 0;
}