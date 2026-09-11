# CS3006 PARALLEL & DISTRIBUTED COMPUTING Project

## Parallel Genetic Algorithm for Large Search Spaces
**Hybrid MPI + OpenMP implementation for the Traveling Salesman Problem**

### Overview
An island-model genetic algorithm that optimizes large TSP instances (10k+ cities) using:
- **MPI** — distributes independent populations (islands) across processes with ring-topology migration
- **OpenMP** — parallelizes fitness evaluation within each island

### Project Structure
```
├── src/          C++ source files
├── include/      C++ headers
├── tests/        Unit tests
├── data/         TSP instances (raw + processed)
├── scripts/      Python/bash utilities
├── docker/       Reproducible build environment
├── docs/         Report, proposal, figures
└── results/      Logs, benchmarks, plots
```

### Requirements
- Ubuntu 24.04 (WSL2 OK)
- GCC 13+, OpenMPI 4.1+, CMake 3.20+, Ninja
- Python 3.12+ with numpy, pandas, matplotlib, seaborn, scipy

### Build
```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

### Run
```bash
mpirun -np 4 --oversubscribe ./build/bin/pdc_ga \
    --cities 5000 --generations 200 --pop 1000
```

### Generate TSP Instance
```bash
~/.venvs/pdc/bin/python scripts/generate_tsp.py \
    --cities 10000 --out data/raw/tsp_10000.txt
```

### Benchmark
```bash
bash scripts/run_experiments.sh
```

### Plot
```bash
~/.venvs/pdc/bin/python scripts/plot_results.py \
    --csv results/benchmarks/speedup.csv
```