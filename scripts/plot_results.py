#!/usr/bin/env python3
"""Plot speedup and efficiency from results/benchmarks/speedup.csv.

Left panel:  speedup vs cores, split by strategy
Right panel: efficiency vs cores, split by strategy

Strategies are inferred from (ranks, threads):
  MPI-only    : ranks > 1, threads == 1
  OpenMP-only : ranks == 1, threads > 1
  Hybrid      : ranks > 1 and threads > 1
  Baseline    : 1 x 1
"""
import argparse
import os
import matplotlib.pyplot as plt
import pandas as pd


def classify(row):
    """Return the strategy name for one row of the CSV."""
    r, t = row["ranks"], row["threads"]
    if r == 1 and t == 1:
        return "Baseline"
    if r > 1 and t == 1:
        return "MPI-only"
    if r == 1 and t > 1:
        return "OpenMP-only"
    return "Hybrid"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/benchmarks/speedup.csv")
    ap.add_argument("--out", default="results/plots/speedup_efficiency.png")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df["strategy"] = df.apply(classify, axis=1)
    df = df.sort_values("cores")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Marker and color for each strategy
    style = {
        "MPI-only":    {"marker": "o", "color": "#1f77b4", "label": "MPI-only"},
        "OpenMP-only": {"marker": "s", "color": "#ff7f0e", "label": "OpenMP-only"},
        "Hybrid":      {"marker": "^", "color": "#2ca02c", "label": "Hybrid"},
    }

    # Left: speedup
    ax1.plot(df["cores"], df["cores"], "--", color="gray",
             label="Ideal", linewidth=1.2, alpha=0.6)

    for name, sty in style.items():
        sub = df[df["strategy"] == name]
        if not sub.empty:
            ax1.plot(sub["cores"], sub["speedup"],
                     marker=sty["marker"], color=sty["color"],
                     label=sty["label"], linewidth=2, markersize=9)

    # Highlight the best point
    best = df.loc[df["speedup"].idxmax()]
    ax1.annotate(f"Best: {best['speedup']:.2f}x",
                 xy=(best["cores"], best["speedup"]),
                 xytext=(best["cores"] + 0.15, best["speedup"] - 0.25),
                 fontsize=10, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="black", alpha=0.7))

    ax1.set_xlabel("Cores (MPI ranks x OpenMP threads)")
    ax1.set_ylabel("Speedup vs sequential")
    ax1.set_title("Parallel speedup")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9, loc="upper left")
    ax1.set_xticks(sorted(df["cores"].unique()))

    # Right: efficiency
    ax2.axhline(1.0, linestyle="--", color="gray",
                label="Ideal", linewidth=1.2, alpha=0.6)

    for name, sty in style.items():
        sub = df[df["strategy"] == name]
        if not sub.empty:
            ax2.plot(sub["cores"], sub["efficiency"],
                     marker=sty["marker"], color=sty["color"],
                     label=sty["label"], linewidth=2, markersize=9)

    ax2.set_xlabel("Cores (MPI ranks x OpenMP threads)")
    ax2.set_ylabel("Efficiency (speedup / cores)")
    ax2.set_title("Parallel efficiency")
    ax2.set_ylim(0, 1.1)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9, loc="lower left")
    ax2.set_xticks(sorted(df["cores"].unique()))

    plt.suptitle("Hybrid MPI+OpenMP GA - TSP Performance", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.out}")

    # Print a summary table
    print("\n=== Summary Table ===")
    cols = ["cores", "ranks", "threads", "time_sec", "speedup", "efficiency", "strategy"]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()