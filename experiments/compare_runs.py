"""
Overlay fitness curves from several runs, e.g. evolved vs control.

    python experiments/compare_runs.py <run_dir> <run_dir> [--labels evolved control] [--out compare.png]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from selfish_lora.runlog import plot_runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--out", default="compare.png")
    args = parser.parse_args()

    labels = args.labels or [os.path.basename(d.rstrip("/\\")).split("_", 1)[-1] for d in args.run_dirs]
    plot_runs(args.run_dirs, labels, args.out)


if __name__ == "__main__":
    main()
