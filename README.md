# CS3006 Hybrid MPI OpenMP Parallel Genetic Algorithm for Large Scale Combinatorial Optimization

An island-model **Genetic Algorithm** for the **Traveling Salesman Problem**, implemented in modern C++17 with **hybrid MPI + OpenMP** parallelism.

The project distributes a global population across MPI ranks using a coarse-grained island model, and parallelizes both fitness evaluation and child creation within each rank using OpenMP threads. MPI ranks periodically exchange their best individuals through a ring-topology migration to preserve diversity and share search progress. The result is a parallel evolutionary optimizer designed to run efficiently on a multi-core CPU while improving solution quality relative to a single-rank baseline.

---

## Overview

The Traveling Salesman Problem is a classic combinatorial optimization problem. Given `N` cities and pairwise distances between them, the goal is to find the shortest closed tour that visits each city exactly once and returns to the starting city. The number of possible tours grows factorially with `N`, so exact algorithms become impractical beyond a few hundred cities. Genetic Algorithms offer a practical alternative by searching the space stochastically and converging to near-optimal tours.

This project builds a parallel Genetic Algorithm that scales on consumer multi-core hardware. It combines two parallel programming models:

- **MPI** for coarse-grained, distributed-memory parallelism across independent sub-populations.
- **OpenMP** for fine-grained, shared-memory parallelism within each sub-population.

The hybrid design allows the algorithm to use both memory separation and thread-level parallelism, which is important for a workload that is bound by memory bandwidth rather than raw compute.

---

## Architecture

```
+----------------------------------------------------------+
|                    MPI_COMM_WORLD                        |
+---------------+---------------+---------------+----------+
|   Island 0    |   Island 1    |   Island 2    |  Island  |
|  rank=0       |  rank=1       |  rank=2       |    P-1   |
|               |               |               |          |
|  Population   |  Population   |  Population   |  ...     |
|  |- Ind 0     |  |- Ind 0     |  |- Ind 0     |          |
|  |- Ind 1     |  |- Ind 1     |  |- Ind 1     |          |
|  |   ...      |  |   ...      |  |   ...      |          |
|  |- Ind M-1   |  |- Ind M-1   |  |- Ind M-1   |          |
|               |               |               |          |
|  OpenMP       |  OpenMP       |  OpenMP       |  OpenMP  |
|  threads      |  threads      |  threads      |  threads |
|  -> fitness   |  -> fitness   |  -> fitness   |  -> fit  |
+-------+-------+-------+-------+-------+-------+-----+----+
        |               |               |             |
        +--- Ring-topology migration (every K gens) ---+
             Best migrants sent to next rank,
             received from previous rank via MPI_Sendrecv
```

**Per generation, the algorithm runs these steps:**

1. `evolve_generation()`: selection, Order Crossover (OX1), swap mutation, and elitism. Child creation runs in parallel using OpenMP.
2. `evaluate_population()`: tour length computation for every individual. This step is OpenMP parallel.
3. Every `migration_interval` generations: ring migration through `MPI_Sendrecv`.
4. At log boundaries: `MPI_Allreduce` and `MPI_Bcast` for the global best individual.

---

## Algorithm Design

### Genetic Algorithm Operators

| Operator | Implementation | Notes |
|:---------|:--------------|:------|
| Representation | Permutation of city indices | Each genome is a valid TSP tour |
| Fitness | Euclidean closed-tour length | Lower is better; uses a precomputed distance matrix |
| Initialization | Random shuffle per individual | Independent across the initial population |
| Selection | k-way tournament (k = 5) | Minimization semantics |
| Crossover | Order Crossover (OX1) | Permutation-preserving; uses two distinct cut points |
| Mutation | Swap mutation (rate 0.02) | Per-position random swap |
| Elitism | Top-2 preserved per generation | Guarantees monotonic improvement of the best fitness |

### Fitness Function

The fitness of a tour `tau` is the total Euclidean length of the closed loop that visits cities in the order given by `tau` and returns to the first city.

```
L(tau) = sum over i in [0, N-1] of  d( tau[i], tau[(i+1) mod N] )
```

where `N` is the number of cities and `d(a, b)` is the Euclidean distance between cities `a` and `b`. Lower values mean better tours.

### Order Crossover (OX1)

OX1 preserves permutation validity by copying a contiguous slice from one parent into the child, then filling the remaining positions with cities from the other parent in their original relative order, skipping any city that is already in the child. This keeps the offspring a valid TSP tour without duplicates or missing entries.

### Selection Pressure

Tournament selection picks `k` individuals uniformly at random and returns the one with the smallest fitness. Increasing `k` increases selection pressure. Decreasing `k` keeps the population more diverse.

---

## Parallel Decomposition

### MPI Layer (coarse-grained, island model)

The global population `P` is split evenly across `R` MPI ranks. Each rank owns `P / R` individuals (with a remainder distributed to the first few ranks so every individual is accounted for). Each island evolves independently with its own random number generator, seeded from `seed + rank * golden_ratio`. Islands remain diverse until they communicate.

Every `migration_interval` generations, each rank sends its top `num_migrants` individuals to the next rank in a ring and receives the same number from the previous rank. Migration is synchronous and uses `MPI_Sendrecv`, which cannot deadlock even when all ranks try to send to each other simultaneously.

The global best individual is tracked with `MPI_Allreduce` using the `MPI_MINLOC` operation on a `(fitness, rank)` pair, followed by an `MPI_Bcast` that distributes the winning genome to all ranks.

### OpenMP Layer (fine-grained, within-rank)

Fitness evaluation is parallelized across the population:

```c
#pragma omp parallel for schedule(static)
for (int i = 0; i < pop_size; ++i) {
    pop[i].fitness = tour_length(pop[i].genome, dist_matrix);
}
```

Because every individual has identical work (one linear pass through the distance matrix), `schedule(static)` gives each thread an equal chunk of the population and avoids scheduling overhead.

Child creation is also parallelized. Each OpenMP thread derives its own `mt19937_64` generator from a single per-generation base seed. The loop uses `schedule(dynamic, 4)` to absorb the small variable cost of crossover and mutation:

```c
#pragma omp parallel
{
    std::mt19937_64 local_rng(base_seed + thread_id * GOLDEN_RATIO);
    #pragma omp for schedule(dynamic, 4)
    for (int i = elite_count; i < pop_size; ++i) {
        // tournament select, crossover, mutation
    }
}
```

### Why the Split Works

The two parallel layers target different bottlenecks:

- **MPI isolates memory.** Each rank holds a private copy of the distance matrix. Since the workload is memory-bound, this spreads the working set across cache regions and reduces bus contention.
- **OpenMP isolates fork/join overhead.** Within a rank, threads share the same matrix and require no serialization between parallel regions.

Combining the two lets the algorithm exploit both memory separation and thread-level parallelism at the same time.

---

## Amdahl's Law

Amdahl's Law bounds the maximum achievable speedup for a workload with a sequential fraction `f`:

```
S(N) = 1 / ( f + (1 - f) / N )
```

where `N` is the number of processing units. As `N` grows, speedup approaches `1 / f`, no matter how many cores are added. This is why identifying the sequential fraction matters.

In this project, the sequential parts of a generation are:

- Sorting the population by fitness, which is `O(P log P)`.
- The elitism copy of the top few individuals.
- MPI collective operations (`MPI_Allreduce`, `MPI_Bcast`).
- CSV logging.

Everything else, mainly fitness evaluation and child creation, is parallelizable. The analysis notebook back-solves `f` from the measured peak speedup, which gives a concrete estimate of how much of the runtime remains sequential on this hardware.

---

## Data Used

The project uses several plain-text files under `data/raw/`. All TSP files follow the same format: one city per line, with an `x` and a `y` coordinate separated by whitespace. All tour files contain one integer city index per line.

| File | Purpose | Format |
|:-----|:--------|:-------|
| `tsp_500.txt` | Small synthetic instance used for smoke testing and quick runs | One `x y` pair per line |
| `tsp_2000.txt` | Mid-size instance used for tuning and interactive runs | One `x y` pair per line |
| `tsp_5000.txt` | Main benchmark instance referenced by the sweep and the notebook | One `x y` pair per line |
| `tsp_verify_20.txt` | Coordinates for the small NumPy cross-check (20 cities) | One `x y` pair per line |
| `tsp_verify_100.txt` | Coordinates for the larger NumPy cross-check (100 cities) | One `x y` pair per line |
| `tour_verify_20.txt` | Tour file paired with the 20-city instance (identity permutation) | One integer per line |
| `tour_verify_100.txt` | Tour file paired with the 100-city instance (identity permutation) | One integer per line |

All TSP instances are generated by `scripts/generate_tsp.py`, which produces cities uniformly distributed in the unit square `[0, 1]^2`. The verification instances are produced by `scripts/verify_fitness.py` and are used only to cross-check the C++ tour length computation against a NumPy reference.

Results are written under `results/`:

- `results/benchmarks/speedup.csv` contains the per-configuration timing records produced by the benchmark sweep.
- `results/benchmarks/trace.csv` contains the per-generation convergence trace for the last executed configuration.
- `results/logs/` holds the raw stdout of each benchmark run, one file per configuration.

The analysis notebook reads `speedup.csv` and `trace.csv` directly, so it does not need to re-run the algorithm to produce its plots.

---

## Repository Structure

```
.
|-- src/                     C++ implementation
|   |-- main.cpp               Driver: CLI, GA loop, timing
|   |-- ga.cpp                 GA operators (init, select, OX1, mutate)
|   |-- fitness.cpp            TSP tour length, distance matrix
|   |-- mpi_manager.cpp        Ring migration, global best reduction
|   `-- utils.cpp              Timer, CSV output, logging
|-- include/                 Public headers
|-- tests/                   Unit tests and verification harnesses
|-- scripts/                 Python and bash utilities
|   |-- generate_tsp.py        Synthetic TSP instance generator
|   |-- run_experiments.sh     Benchmark sweep runner
|   |-- plot_results.py        Speedup and efficiency plot script
|   `-- verify_fitness.py      NumPy cross-check for tour_length()
|-- data/                    TSP instances (generated, git-ignored)
|-- results/                 Benchmarks, logs, plots
|-- docs/                    Analysis notebook and report figures
|-- docker/                  Reproducible build container
|-- CMakeLists.txt           Top-level build configuration
|-- README.md
`-- LICENSE
```

---

## Quick Start

### Requirements

- Ubuntu 24.04 (WSL2 supported)
- GCC 13+ with OpenMP
- OpenMPI 4.1+
- CMake 3.20+ and Ninja
- Python 3.12+ (for data generation, verification, and plotting)

### Build

```bash
git clone https://github.com/zahmed02/CS3006-Hybrid-MPI-OpenMP-Parallel-Genetic-Algorithm-for-Large-Scale-Combinatorial-Optimization.git
cd CS3006-Hybrid-MPI-OpenMP-Parallel-Genetic-Algorithm-for-Large-Scale-Combinatorial-Optimization
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

The build produces three executables in `build/bin/`:

- `pdc_ga`: the main hybrid MPI + OpenMP genetic algorithm
- `fitness_check`: a standalone verifier for the tour length function
- `pdc_tests`: unit tests for the GA operators

### Generate a TSP instance

```bash
python3 -m venv ~/.venvs/pdc
source ~/.venvs/pdc/bin/activate
pip install numpy matplotlib pandas seaborn scipy

python scripts/generate_tsp.py --cities 5000 --seed 42 --out data/raw/tsp_5000.txt
```

### Run the hybrid GA

```bash
OMP_NUM_THREADS=2 mpirun -np 2 --oversubscribe ./build/bin/pdc_ga \
    --data data/raw/tsp_5000.txt \
    --generations 200 \
    --pop 1000
```

### Run the benchmark sweep

The sweep runs six configurations (1x1, 2x1, 4x1, 1x2, 1x4, 2x2) and writes a summary CSV under `results/benchmarks/`:

```bash
bash scripts/run_experiments.sh
```

Plot the results:

```bash
python scripts/plot_results.py
```

### Reproducible run with Docker

```bash
docker build -t pdc-ga -f docker/Dockerfile .
docker run --rm pdc-ga
```

The image installs the toolchain, compiles the project, and runs a small self-contained benchmark.

---

## Verification

Two independent checks establish that the core fitness function is correct.

### Unit tests

`pdc_tests` contains five assertion-based tests covering initialization, crossover, mutation, selection, and tour length against analytic values. Run them with:

```bash
cd build && ctest --output-on-failure && cd ..
```

### Cross-check against NumPy

The C++ `tour_length()` implementation is compared against a NumPy reference on two small instances. Both executables print the tour length to ten decimal places. Any mismatch beyond floating-point rounding indicates a bug in the distance matrix construction or the tour length accumulation loop.

```bash
python scripts/verify_fitness.py --cities 100 --seed 7
./build/bin/fitness_check data/raw/tsp_verify_100.txt data/raw/tour_verify_100.txt
```

---

## Analysis Notebook

`docs/analysis.ipynb` reads the benchmark CSVs and produces the following figures, all saved under `docs/figures/`:

| Figure | What it shows |
|:-------|:--------------|
| `speedup.png` | Speedup versus cores, with the ideal linear reference |
| `efficiency.png` | Parallel efficiency versus cores |
| `walltime.png` | Wall-clock time per configuration, sorted from slowest to fastest |
| `strategy_comparison.png` | MPI-only versus OpenMP-only versus Hybrid at equal core counts |
| `convergence.png` | Best tour length over generations |
| `improvement_rate.png` | Per-generation improvement, with the 90 percent point highlighted |
| `amdahl.png` | Measured speedup versus the Amdahl's Law prediction |
| `final_quality.png` | Final solution quality |

The notebook also back-solves the sequential fraction `f` from the observed peak speedup and prints the theoretical speedup ceiling.

---

## Design Notes

### Memory layout

The `N x N` distance matrix dominates memory usage. For large `N`, it can exceed the size of the last-level cache, which makes `tour_length()` memory-bound rather than compute-bound. This drives several design choices:

- The matrix is computed once at startup using `#pragma omp parallel for schedule(static)` over rows and exploited symmetry (`d[i][j] == d[j][i]`), so only half of the entries are computed directly.
- Each MPI rank holds its own copy, which is why MPI reduces bus contention at equal core counts.
- Going beyond four cores on this workload would require a different layout (a flat one-dimensional array or cache-blocked access).

### Random number generator design

Per-thread RNGs are derived from a single per-generation base seed drawn from the master RNG. This gives three properties:

- **No data races.** Each thread writes only to its own slice of the next generation.
- **Independence.** Different threads explore different regions of the search space.
- **Determinism.** Results are reproducible given a fixed `OMP_NUM_THREADS`.

### Deadlock avoidance

Ring migration uses `MPI_Sendrecv` instead of paired `MPI_Send` and `MPI_Recv`. Unlike paired send and receive, `MPI_Sendrecv` cannot deadlock when all ranks attempt to send to each other at the same time, because the receive side progresses together with the send side.

---

## Documentation

- Analysis notebook: `docs/analysis.ipynb`
- Report figures: `docs/figures/`
- Benchmark summary: `results/benchmarks/speedup.csv`
- Convergence trace: `results/benchmarks/trace.csv`
- Per-configuration logs: `results/logs/`

---

## License

MIT. See [LICENSE](LICENSE) for details.
