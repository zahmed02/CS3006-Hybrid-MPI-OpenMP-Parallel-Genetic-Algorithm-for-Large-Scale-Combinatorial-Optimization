#!/usr/bin/env bash
# ============================================================================
# Benchmark runner: sweeps MPI ranks × OpenMP threads
# Produces results/benchmarks/speedup.csv with timing/speedup/efficiency
#
# The baseline is 1 rank × 1 thread (sequential). All other configs are
# measured against it. Repeats take the MIN time (fastest) of each config
# to reduce noise from scheduler jitter.
# ============================================================================
set -euo pipefail

BIN="${BIN:-./build/bin/pdc_ga}"
DATA="${DATA:-data/raw/tsp_5000.txt}"
GENS="${GENS:-200}"
POP="${POP:-1000}"
OUT="${OUT:-results/benchmarks/speedup.csv}"
LOG_DIR="${LOG_DIR:-results/logs}"
REPEATS="${REPEATS:-1}"

mkdir -p "$LOG_DIR" "$(dirname "$OUT")"
echo "cores,ranks,threads,run,time_sec,speedup,efficiency" > "$OUT"

# ---- Sanity checks ----
if [[ ! -x "$BIN" ]]; then
    echo "Error: $BIN not found or not executable. Build first." >&2
    exit 1
fi
if [[ ! -f "$DATA" ]]; then
    echo "Error: data file '$DATA' not found." >&2
    exit 1
fi

# Configuration sweep: each entry is "ranks threads"
CONFIGS=(
    "1 1"
    "2 1"
    "4 1"
    "1 2"
    "1 4"
    "2 2"
)

echo "=== Benchmark sweep ==="
echo "Binary:       $BIN"
echo "Data:         $DATA"
echo "Generations:  $GENS"
echo "Population:   $POP"
echo "Repeats:      $REPEATS"
echo "Results:      $OUT"
echo

# ---- Baseline: 1 rank × 1 thread ----
BASE_TIME=""
for run in $(seq 1 "$REPEATS"); do
    LOG="$LOG_DIR/base_r1_t1_run${run}.log"
    echo "[baseline] 1 rank × 1 thread — run $run"
    OMP_NUM_THREADS=1 mpirun -np 1 --oversubscribe "$BIN" \
        --data "$DATA" --generations "$GENS" --pop "$POP" --quiet \
        > "$LOG" 2>&1
    T=$(grep "Loop time:" "$LOG" | awk '{print $3}')
    if [[ -z "$BASE_TIME" || $(echo "$T < $BASE_TIME" | bc -l) -eq 1 ]]; then
        BASE_TIME="$T"
    fi
done
echo "baseline_loop_time=$BASE_TIME"
echo "1,1,1,0,$BASE_TIME,1.000000,1.000000" >> "$OUT"
echo

# ---- Sweep the rest ----
for cfg in "${CONFIGS[@]}"; do
    read -r RANKS THREADS <<< "$cfg"
    if [[ "$RANKS" -eq 1 && "$THREADS" -eq 1 ]]; then
        continue   # already done as baseline
    fi
    CORES=$((RANKS * THREADS))

    BEST_TIME=""
    for run in $(seq 1 "$REPEATS"); do
        LOG="$LOG_DIR/r${RANKS}_t${THREADS}_run${run}.log"
        echo "[sweep] $RANKS rank(s) × $THREADS thread(s) — run $run"
        OMP_NUM_THREADS="$THREADS" mpirun -np "$RANKS" --oversubscribe "$BIN" \
            --data "$DATA" --generations "$GENS" --pop "$POP" --quiet \
            > "$LOG" 2>&1
        T=$(grep "Loop time:" "$LOG" | awk '{print $3}')
        if [[ -z "$BEST_TIME" || $(echo "$T < $BEST_TIME" | bc -l) -eq 1 ]]; then
            BEST_TIME="$T"
        fi
    done

    SPEEDUP=$(echo "scale=6; $BASE_TIME / $BEST_TIME" | bc -l)
    EFF=$(echo "scale=6; $SPEEDUP / $CORES" | bc -l)
    echo "$CORES,$RANKS,$THREADS,0,$BEST_TIME,$SPEEDUP,$EFF" >> "$OUT"
done

echo
echo "=== Done ==="
echo "Results: $OUT"
echo
column -t -s, "$OUT"