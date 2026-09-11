#!/usr/bin/env bash
# ============================================================================
# Benchmark runner: sweeps MPI ranks × OpenMP threads
# Produces results/benchmarks/speedup.csv
# ============================================================================
set -euo pipefail

BIN=./build/bin/pdc_ga
OUT=results/benchmarks/speedup.csv
LOG_DIR=results/logs

mkdir -p "$LOG_DIR" results/benchmarks
echo "cores,ranks,threads,time_sec,speedup,efficiency" > "$OUT"

# TODO: fill in after implementation is complete
echo "Runner ready. Implement the GA first, then fill in the sweep loop."