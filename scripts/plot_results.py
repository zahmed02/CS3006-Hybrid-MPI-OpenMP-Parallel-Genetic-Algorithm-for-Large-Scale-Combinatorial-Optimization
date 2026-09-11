#!/usr/bin/env python3
"""
Plot speedup / efficiency graphs from benchmark CSV output.
"""
import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot GA parallel speedup")
    parser.add_argument("--csv", default="results/benchmarks/speedup.csv")
    parser.add_argument("--out", default="results/plots/speedup.png")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df = pd.read_csv(args.csv)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ---- Speedup ----
    ax1.plot(df["cores"], df["speedup"], marker="o", label="Measured")
    ax1.plot(df["cores"], df["cores"], "--", label="Ideal linear")
    ax1.set_xlabel("Cores (MPI ranks × OpenMP threads)")
    ax1.set_ylabel("Speedup vs sequential")
    ax1.set_title("Parallel speedup")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ---- Efficiency ----
    ax2.plot(df["cores"], df["efficiency"], marker="s", color="orange")
    ax2.axhline(1.0, linestyle="--", color="gray", label="Ideal")
    ax2.set_xlabel("Cores")
    ax2.set_ylabel("Efficiency")
    ax2.set_title("Parallel efficiency")
    ax2.set_ylim(0, 1.1)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()