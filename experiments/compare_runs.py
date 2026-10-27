"""
Overlay curves from several runs, e.g. evolved vs control. Runs given the
same label (e.g. several seeds) share a colour.

    python experiments/compare_runs.py <run_dir> <run_dir> --labels evolved control --out compare.png
    python experiments/compare_runs.py <run_dirs...> --metrics mean coop_rate invalid_rate --panels
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from selfish_lora.runlog import plot_runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--labels", nargs="+", help="One per run; defaults to the run directory name")
    parser.add_argument("--metrics", nargs="+", default=["best", "mean"], help="History fields (at most three unless --panels)")
    parser.add_argument("--ylabel", default="Fitness")
    parser.add_argument("--panels", action="store_true", help="One panel per metric")
    parser.add_argument("--title", default="Fitness by generation")
    parser.add_argument("--out", default="compare.png")
    args = parser.parse_args()

    labels = args.labels or [os.path.basename(d.rstrip("/\\")) for d in args.run_dirs]
    if len(labels) != len(args.run_dirs):
        parser.error("Give one label per run")
    plot_runs(args.run_dirs, labels, args.out, title=args.title, metrics=args.metrics, ylabel=args.ylabel,
              panels=args.panels)


if __name__ == "__main__":
    main()
