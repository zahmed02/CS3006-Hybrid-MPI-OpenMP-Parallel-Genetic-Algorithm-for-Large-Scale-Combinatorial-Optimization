# CS3006 Hybrid MPI OpenMP Parallel Genetic Algorithm for Large Scale Combinatorial Optimization

[![Language](https://img.shields.io/badge/language-C%2B%2B17-blue)](https://en.cppreference.com/w/cpp/17)
[![Parallel](https://img.shields.io/badge/parallel-MPI%20%2B%20OpenMP-orange)](https://www.open-mpi.org/)
[![Build](https://img.shields.io/badge/build-CMake%20%2B%20Ninja-green)](https://cmake.org/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)
[![Peak Speedup](https://img.shields.io/badge/peak%20speedup-2.02%C3%97-brightgreen)](#-results)

An island-model **Genetic Algorithm** for the **Traveling Salesman Problem**, implemented in modern C++17 with **hybrid MPI + OpenMP** parallelism. Built for the CS3006 Parallel and Distributed Computing course.

---

## Results

**Peak speedup: 2.02× on 4 logical cores (50.6% parallel efficiency)**

Tested on N=5000 cities, population 1000, 200 generations.

| Cores | Configuration | Time (s) | Speedup | Efficiency |
|------:|:--------------|---------:|--------:|-----------:|
| 1     | 1 rank × 1 thread (baseline) | 60.87 | 1.00× | 100.0% |
| 2     | 2 ranks × 1 thread | 36.90 | 1.65× | 82.5% |
| 2     | 1 rank × 2 threads | 53.54 | 1.14× | 56.8% |
| 4     | 4 ranks × 1 thread | 34.31 | 1.77× | 44.3% |
| 4     | 1 rank × 4 threads | 39.25 | 1.55× | 38.8% |
| 4     | **2 ranks × 2 threads** ⭐ | **30.09** | **2.02×** | **50.6%** |

![Speedup and efficiency](results/plots/speedup_efficiency.png)

### Key findings

1. **MPI outperforms OpenMP at equal core counts**: each MPI rank holds a *private* copy of the distance matrix, spreading the working set across cache regions. OpenMP threads contend for the same shared matrix and saturate the memory bus.
2. **Hybrid wins overall**: combining 2 MPI ranks with 2 OpenMP threads per rank delivers the best of both worlds.
3. **Sub-linear scaling beyond 2 cores**: the WSL2 host's 4 logical cores are likely 2 physical + SMT, and the workload is memory-bandwidth-bound.
4. **Island-model migration improves solution quality**: the best 2-rank run found a tour ~1.5% shorter than the single-rank run, demonstrating that parallelization isn't only about speed.

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    MPI_COMM_WORLD                        │
├───────────────┬───────────────┬───────────────┬──────────┤
│   Island 0    │   Island 1    │   Island 2    │  Island  │
│  rank=0       │  rank=1       │  rank=2       │    P-1   │
│               │               │               │          │
│  Population   │  Population   │  Population   │  ...     │
│  ├─ Ind 0     │  ├─ Ind 0     │  ├─ Ind 0     │          │
│  ├─ Ind 1     │  ├─ Ind 1     │  ├─ Ind 1     │          │
│  │   ...      │  │   ...      │  │   ...      │          │
│  └─ Ind M-1   │  └─ Ind M-1   │  └─ Ind M-1   │          │
│               │               │               │          │
│  OpenMP       │  OpenMP       │  OpenMP       │  OpenMP  │
│  threads      │  threads      │  threads      │  threads │
│  └→ fitness   │  └→ fitness   │  └→ fitness   │  └→ fit  │
└───────┬───────┴───────┬───────┴───────┬───────┴─────┬────┘
        │               │               │             │
        └──── Ring-topology migration (every K gens) ─┘
              Best migrants sent to next rank,
              received from previous rank via MPI_Sendrecv
```

**Per generation:**
1. `evolve_generation()`: selection, OX1 crossover, swap mutation, elitism (OpenMP-parallel child creation)
2. `evaluate_population()`: tour length computation (OpenMP-parallel)
3. Every `migration_interval` generations: ring migration via `MPI_Sendrecv`
4. At log boundaries: `MPI_Allreduce` + `MPI_Bcast` for global best

---

## Quick Start

### Requirements

- Ubuntu 24.04 (WSL2 supported)
- GCC 13+ with OpenMP
- OpenMPI 4.1+
- CMake 3.20+ and Ninja
- Python 3.12+ (for data generation and plotting)

### Build

```bash
git clone https://github.com/zahmed02/CS3006-Hybrid-MPI-OpenMP-Parallel-Genetic-Algorithm-for-Large-Scale-Combinatorial-Optimization.git
cd CS3006-Hybrid-MPI-OpenMP-Parallel-Genetic-Algorithm-for-Large-Scale-Combinatorial-Optimization
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

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

```bash
bash scripts/run_experiments.sh
python scripts/plot_results.py
```

---

## Repository Structure

```
.
├── src/                     C++ implementation
│   ├── main.cpp               Driver: CLI, GA loop, timing
│   ├── ga.cpp                 GA operators (init, select, OX1, mutate)
│   ├── fitness.cpp            TSP tour length, distance matrix
│   ├── mpi_manager.cpp        Ring migration, global best reduction
│   └── utils.cpp              Timer, CSV output, logging
├── include/                 Public headers
├── tests/                   Unit tests + fitness cross-check harness
├── scripts/                 Python / bash utilities
│   ├── generate_tsp.py        Synthetic TSP instance generator
│   ├── run_experiments.sh     6-config benchmark sweep
│   ├── plot_results.py        Speedup / efficiency plot
│   └── verify_fitness.py      NumPy cross-check for tour_length()
├── data/                    TSP instances (generated)
├── results/                 Benchmarks, logs, plots
├── docker/                  Reproducible build container
└── docs/                    Report, proposal
```

---

## Verification

The `tour_length()` C++ implementation is cross-checked against a NumPy reference:

```bash
python scripts/verify_fitness.py --cities 100 --seed 7
./build/bin/fitness_check data/raw/tsp_verify_100.txt data/raw/tour_verify_100.txt
```

Both print the same value to 9+ decimal places.

---

## License

MIT see [LICENSE](LICENSE).
