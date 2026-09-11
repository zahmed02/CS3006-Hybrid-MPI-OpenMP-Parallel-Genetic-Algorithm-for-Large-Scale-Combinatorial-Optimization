#!/usr/bin/env python3
"""
Generate a synthetic random TSP instance (uniform in [0,1]^2)
and save as plain text with one "x y" pair per line.
"""
import argparse

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a random TSP instance")
    parser.add_argument("--cities", type=int, default=10000,
                        help="Number of cities to generate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--out", type=str, default="data/raw/tsp_10000.txt",
                        help="Output file path")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    coords = rng.random((args.cities, 2))
    np.savetxt(args.out, coords, fmt="%.6f")
    print(f"Wrote {args.cities} cities to {args.out}")


if __name__ == "__main__":
    main()