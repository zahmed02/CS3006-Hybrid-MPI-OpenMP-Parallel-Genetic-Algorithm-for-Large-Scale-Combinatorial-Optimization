#!/usr/bin/env python3
"""
Cross-check: verify that C++ tour_length() matches a NumPy reference
implementation for a small random TSP instance.

Workflow:
  1. Generate a tiny TSP instance (deterministic seed) and save coordinates.
  2. Compute the reference tour length in Python for a fixed tour (identity).
  3. Write a tour file so the C++ binary can process the same data.
  4. User runs the C++ binary with matching parameters and compares outputs.

Usage:
    python scripts/verify_fitness.py [--cities N] [--seed S] [--out-dir D]

Outputs:
    <out-dir>/tsp_verify_<N>.txt   — coordinates, "x y" per line
    <out-dir>/tour_verify_<N>.txt  — tour, one city index per line
    Prints the reference tour length to stdout.
"""
import argparse
import sys
from pathlib import Path

import numpy as np


def reference_tour_length(tour: np.ndarray, coords: np.ndarray) -> float:
    """
    Compute the closed-tour Euclidean length of `tour` through `coords`.

    The tour is treated as a loop: it visits cities in `tour` order, then
    returns from the last city back to the first.
    """
    ordered = coords[tour]
    # Append the first city at the end to close the loop
    closed = np.vstack([ordered, ordered[:1]])
    diffs = np.diff(closed, axis=0)
    return float(np.sqrt((diffs ** 2).sum(axis=1)).sum())


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Verify C++ tour_length() against a NumPy reference."
    )
    ap.add_argument("--cities", type=int, default=20,
                    help="Number of cities (default: 20)")
    ap.add_argument("--seed", type=int, default=42,
                    help="RNG seed for reproducibility (default: 42)")
    ap.add_argument("--out-dir", type=str, default="data/raw",
                    help="Output directory (default: data/raw)")
    args = ap.parse_args()

    if args.cities < 2:
        print("Error: --cities must be >= 2", file=sys.stderr)
        return 1

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- Generate coordinates (same algorithm as C++ generate_random_tsp) ----
    # Note: we use NumPy's PCG64 here, not std::mt19937_64. That's fine —
    # the C++ side will *read* these coordinates from disk. Only tour_length
    # is being cross-checked, not the RNG itself.
    rng = np.random.default_rng(args.seed)
    coords = rng.random((args.cities, 2))

    coords_file = out / f"tsp_verify_{args.cities}.txt"
    np.savetxt(coords_file, coords, fmt="%.10f")

    # ---- Fixed tour = identity permutation [0, 1, 2, ..., N-1] ----
    tour = np.arange(args.cities)
    ref_length = reference_tour_length(tour, coords)

    tour_file = out / f"tour_verify_{args.cities}.txt"
    np.savetxt(tour_file, tour, fmt="%d")

    # ---- Report ----
    print(f"Coordinates : {coords_file}")
    print(f"Tour file   : {tour_file}")
    print(f"Cities (N)  : {args.cities}")
    print(f"Reference tour length (NumPy): {ref_length:.10f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())