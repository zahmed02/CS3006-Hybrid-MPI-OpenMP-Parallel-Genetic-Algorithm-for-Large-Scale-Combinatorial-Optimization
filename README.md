# CS3006 — PDC Project

## Parallel Genetic Algorithm for Large Search Spaces
**Hybrid MPI + OpenMP implementation for the Traveling Salesman Problem**

![Project Status](https://img.shields.io/badge/status-in--development-yellow)
![Language](https://img.shields.io/badge/language-C%2B%2B17-blue)
![Parallel](https://img.shields.io/badge/parallel-MPI%20%2B%20OpenMP-orange)
![License](https://img.shields.io/badge/license-MIT-green)

---

### Overview

An **island-model genetic algorithm** that optimizes large TSP instances (10k+ cities) using:

- **MPI** — distributes independent populations (islands) across processes with
  ring-topology migration
- **OpenMP** — parallelizes fitness evaluation within each island

This project is the semester submission for **CS3006: Parallel and Distributed Computing**.

---

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

---

### Requirements

- **OS:** Ubuntu 24.04 (WSL2 supported)
- **Compiler:** GCC 13+
- **MPI:** OpenMPI 4.1+
- **Build:** CMake 3.20+, Ninja
- **Python:** 3.12+ with `numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy`

---

### Build

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

---

### Run

```bash
mpirun -np 4 --oversubscribe ./build/bin/pdc_ga \
    --cities 5000 --generations 200 --pop 1000
```

---

### Generate TSP Instance

```bash
~/.venvs/pdc/bin/python scripts/generate_tsp.py \
    --cities 10000 --out data/raw/tsp_10000.txt
```

---

### Benchmark & Plot

```bash
bash scripts/run_experiments.sh
~/.venvs/pdc/bin/python scripts/plot_results.py \
    --csv results/benchmarks/speedup.csv
```

---

### License

MIT — see [LICENSE](LICENSE) for details.