#!/usr/bin/env bash
# Run a benchmark sweep over MPI ranks x OpenMP threads.
#
# Writes:
#   results/benchmarks/speedup.csv   best-of-REPEATS time per config + speedup + quality
#   results/benchmarks/runs.csv      every individual run (for variance / error bars)
#   results/benchmarks/trace.csv     per-generation convergence of the fastest baseline run
#
# Baseline is 1 rank x 1 thread; all other configs are measured against it.
# The timed quantity is "Total time" = KNN setup + population init + GA loop,
# so nothing that runs in parallel is left out of the measurement.
#
# Notes for a 2-core / 4-thread laptop:
#   * Open MPI binds ranks to cores by default when np <= 2, which would pin all
#     OpenMP threads of a 1-rank run onto ONE physical core. We therefore pass
#     --bind-to none. Use --report-bindings (MPI_FLAGS) to check.
#   * mpi_yield_when_idle stops idle ranks from busy-polling when the machine
#     is oversubscribed (4 ranks on 2 cores).
#   * Close other programs and keep the laptop on mains power; the i7-6500U
#     throttles quickly. REPEATS >= 3 is recommended.
set -euo pipefail

BIN="${BIN:-./build/bin/pdc_ga}"
DATA="${DATA:-data/raw/tsp_5000.txt}"
GENS="${GENS:-200}"
POP="${POP:-1000}"
OUT="${OUT:-results/benchmarks/speedup.csv}"
RUNS_OUT="${RUNS_OUT:-results/benchmarks/runs.csv}"
LOG_DIR="${LOG_DIR:-results/logs}"
REPEATS="${REPEATS:-3}"
SEED="${SEED:-42}"
EXTRA_ARGS="${EXTRA_ARGS:-}"        # e.g. "--ls 0 --init random" for an ablation
MPI_FLAGS="${MPI_FLAGS:---oversubscribe --bind-to none --mca mpi_yield_when_idle 1}"

mkdir -p "$LOG_DIR" "$(dirname "$OUT")"
echo "cores,ranks,threads,run,time_sec,speedup,efficiency,best_length" > "$OUT"
echo "ranks,threads,run,total_sec,loop_sec,best_length" > "$RUNS_OUT"

# Sanity checks
if [[ ! -x "$BIN" ]]; then
    echo "Error: $BIN not found or not executable. Build first." >&2
    exit 1
fi
if [[ ! -f "$DATA" ]]; then
    echo "Error: data file '$DATA' not found." >&2
    exit 1
fi

# Each entry is "ranks threads". The first one must be the baseline.
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
echo "Seed:         $SEED"
echo "Extra args:   ${EXTRA_ARGS:-(none)}"
echo "MPI flags:    $MPI_FLAGS"
echo "Results:      $OUT"
echo

TRACE_OUT="${TRACE_OUT:-results/benchmarks/trace.csv}"

# Run one config REPEATS times; echo "best_total best_length best_run" (fastest run wins)
run_config() {
    local ranks="$1" threads="$2" prefix="$3"
    local best_t="" best_len="" best_run=""
    for run in $(seq 1 "$REPEATS"); do
        local log="$LOG_DIR/${prefix}_run${run}.log"
        echo "[run] $ranks rank(s) x $threads thread(s) - run $run" >&2
        # shellcheck disable=SC2086
        OMP_NUM_THREADS="$threads" mpirun -np "$ranks" $MPI_FLAGS "$BIN" \
            --data "$DATA" --generations "$GENS" --pop "$POP" --seed "$SEED" \
            --out "$LOG_DIR/${prefix}_run${run}_trace.csv" \
            --quiet $EXTRA_ARGS > "$log" 2>&1

        local total loop len
        total=$(awk '/^Total time:/ {print $3}' "$log")
        loop=$(awk '/^Loop time:/ {print $3}' "$log")
        len=$(awk '/^Best tour length:/ {print $4}' "$log")
        if [[ -z "$total" || -z "$len" ]]; then
            echo "Error: could not parse $log" >&2
            exit 1
        fi
        echo "$ranks,$threads,$run,$total,$loop,$len" >> "$RUNS_OUT"

        if [[ -z "$best_t" ]] || awk -v a="$total" -v b="$best_t" 'BEGIN{exit !(a<b)}'; then
            best_t="$total"
            best_len="$len"
            best_run="$run"
        fi
    done
    echo "$best_t $best_len $best_run"
}

BASE_TIME=""
for cfg in "${CONFIGS[@]}"; do
    read -r RANKS THREADS <<< "$cfg"
    CORES=$((RANKS * THREADS))
    if [[ "$RANKS" -eq 1 && "$THREADS" -eq 1 ]]; then
        PREFIX="base_r1_t1"
    else
        PREFIX="r${RANKS}_t${THREADS}"
    fi

    read -r BEST_TIME BEST_LEN BEST_RUN <<< "$(run_config "$RANKS" "$THREADS" "$PREFIX")"
    if [[ -z "$BEST_TIME" || -z "$BEST_LEN" ]]; then
        echo "Error: no usable result for ${RANKS}x${THREADS}" >&2
        exit 1
    fi

    if [[ -z "$BASE_TIME" ]]; then
        BASE_TIME="$BEST_TIME"
        echo "baseline_total_time=$BASE_TIME"
        # The convergence plots in the notebook read this file; make it the
        # trace of the same data set and the fastest baseline run.
        cp "$LOG_DIR/${PREFIX}_run${BEST_RUN}_trace.csv" "$TRACE_OUT"
    fi

    SPEEDUP=$(awk -v b="$BASE_TIME" -v t="$BEST_TIME" 'BEGIN{printf "%.6f", b/t}')
    EFF=$(awk -v s="$SPEEDUP" -v c="$CORES" 'BEGIN{printf "%.6f", s/c}')
    echo "$CORES,$RANKS,$THREADS,0,$BEST_TIME,$SPEEDUP,$EFF,$BEST_LEN" >> "$OUT"
done

echo
echo "=== Done ==="
echo "Results: $OUT   (raw runs: $RUNS_OUT)"
echo
column -t -s, "$OUT" 2>/dev/null || cat "$OUT"