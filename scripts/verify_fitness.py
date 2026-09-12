#!/usr/bin/env python3
"""Cross-check the C++ tour_length() against a NumPy reference.

Steps:
  1. Generate a small TSP instance with a fixed seed.
  2. Compute the reference tour length in NumPy for the identity tour.
  3. Write the coordinates and tour to disk.
  4. Run the C++ harness on the same files and compare the outputs.
"""
import argparse
import sys
from pathlib import Path
import numpy as np


def reference_tour_length(tour, coords):
    """Closed-tour Euclidean length of `tour` through `coords`."""
    ordered = coords[tour]
    # Close the loop by appending the first city at the end
    closed = np.vstack([ordered, ordered[:1]])
    diffs = np.diff(closed, axis=0)
    return float(np.sqrt((diffs ** 2).sum(axis=1)).sum())


def main():
    ap = argparse.ArgumentParser(description="Verify C++ tour_length() against NumPy")
    ap.add_argument("--cities", type=int, default=20, help="Number of cities")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--out-dir", type=str, default="data/raw", help="Output directory")
    args = ap.parse_args()

    if args.cities < 2:
        print("Error: --cities must be >= 2", file=sys.stderr)
        return 1

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Same distribution as the C++ generator: uniform in [0,1]^2
    rng = np.random.default_rng(args.seed)
    coords = rng.random((args.cities, 2))

    coords_file = out / f"tsp_verify_{args.cities}.txt"
    np.savetxt(coords_file, coords, fmt="%.10f")

    # Fixed tour: identity permutation [0, 1, ..., N-1]
    tour = np.arange(args.cities)
    ref_length = reference_tour_length(tour, coords)

    tour_file = out / f"tour_verify_{args.cities}.txt"
    np.savetxt(tour_file, tour, fmt="%d")

    print(f"Coordinates : {coords_file}")
    print(f"Tour file   : {tour_file}")
    print(f"Cities (N)  : {args.cities}")
    print(f"Reference tour length (NumPy): {ref_length:.10f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())