#!/usr/bin/env python3
"""
Generate speedup/efficiency plots from results/benchmarks/speedup.csv.

Produces a 2-panel figure:
  - Left  : Speedup vs cores, per parallelization strategy
  - Right : Parallel efficiency vs cores, per strategy

Strategies are inferred from the (ranks, threads) columns:
  - MPI-only      : ranks > 1, threads == 1
  - OpenMP-only   : ranks == 1, threads > 1
  - Hybrid        : ranks > 1 and threads > 1
  - Baseline      : 1 × 1

Usage:
    python scripts/plot_results.py [--csv PATH] [--out PATH]
"""
import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def classify(row: pd.Series) -> str:
    r, t = row["ranks"], row["threads"]
    if r == 1 and t == 1:
        return "Baseline"
    if r > 1 and t == 1:
        return "MPI-only"
    if r == 1 and t > 1:
        return "OpenMP-only"
    return "Hybrid"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/benchmarks/speedup.csv")
    ap.add_argument("--out", default="results/plots/speedup_efficiency.png")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df["strategy"] = df.apply(classify, axis=1)
    df = df.sort_values("cores")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # ---- Shared style ----
    style = {
        "MPI-only":    {"marker": "o", "color": "#1f77b4", "label": "MPI-only (ranks > 1, 1 thread)"},
        "OpenMP-only": {"marker": "s", "color": "#ff7f0e", "label": "OpenMP-only (1 rank, threads > 1)"},
        "Hybrid":      {"marker": "^", "color": "#2ca02c", "label": "Hybrid (ranks > 1, threads > 1)"},
    }

    # ---- Panel 1: Speedup ----
    ax1.plot(df["cores"], df["cores"], "--",
             color="gray", label="Ideal linear", linewidth=1.2, alpha=0.6)

    for strategy, sty in style.items():
        sub = df[df["strategy"] == strategy]
        if not sub.empty:
            ax1.plot(sub["cores"], sub["speedup"],
                     marker=sty["marker"], color=sty["color"],
                     label=sty["label"], linewidth=2, markersize=9)

    # Annotate the hybrid best point
    best = df.loc[df["speedup"].idxmax()]
    ax1.annotate(f"Best: {best['speedup']:.2f}×",
                 xy=(best["cores"], best["speedup"]),
                 xytext=(best["cores"] + 0.15, best["speedup"] - 0.25),
                 fontsize=10, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="black", alpha=0.7))

    ax1.set_xlabel("Number of cores (MPI ranks × OpenMP threads)", fontsize=11)
    ax1.set_ylabel("Speedup vs sequential baseline", fontsize=11)
    ax1.set_title("Parallel speedup", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9, loc="upper left")
    ax1.set_xticks(sorted(df["cores"].unique()))

    # ---- Panel 2: Efficiency ----
    ax2.axhline(1.0, linestyle="--", color="gray",
                label="Ideal (100%)", linewidth=1.2, alpha=0.6)

    for strategy, sty in style.items():
        sub = df[df["strategy"] == strategy]
        if not sub.empty:
            ax2.plot(sub["cores"], sub["efficiency"],
                     marker=sty["marker"], color=sty["color"],
                     label=sty["label"], linewidth=2, markersize=9)

    ax2.set_xlabel("Number of cores (MPI ranks × OpenMP threads)", fontsize=11)
    ax2.set_ylabel("Parallel efficiency (speedup / cores)", fontsize=11)
    ax2.set_title("Parallel efficiency", fontsize=13, fontweight="bold")
    ax2.set_ylim(0, 1.1)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9, loc="lower left")
    ax2.set_xticks(sorted(df["cores"].unique()))

    plt.suptitle(
        "Hybrid MPI+OpenMP Parallel Genetic Algorithm — TSP Performance",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.out}")

    # ---- Print a markdown-ready table ----
    print("\n=== Summary Table ===")
    pretty = df[["cores", "ranks", "threads", "time_sec", "speedup", "efficiency", "strategy"]]
    print(pretty.to_string(index=False))


if __name__ == "__main__":
    main()