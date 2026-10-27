"""
Run a planned list of experiments, resuming after any interruption.

A plan is a JSON file listing runs:

    {"runs": [
        {"experiment": "countdown", "name": "std0.01_s0",
         "args": ["--generations", "5", "--mutation-std", "0.01"]}
    ]}

Each run gets a fixed directory (experiments/<experiment>/runs/<name>/) and
checkpoints after every generation, so re-running the same command skips
finished runs and resumes the interrupted one:

    python experiments/run_queue.py experiments/plans/<plan>.json
    python experiments/run_queue.py experiments/plans/<plan>.json --status

Console output for each run is appended to <run_dir>/console.log.
"""

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from selfish_lora.checkpoint import load_checkpoint
from selfish_lora.runlog import load_history

EXPERIMENTS_DIR = os.path.dirname(os.path.abspath(__file__))


def run_dir_for(run: dict) -> str:
    return os.path.join(EXPERIMENTS_DIR, run["experiment"], "runs", run["name"])


def run_status(run: dict) -> str:
    run_dir = run_dir_for(run)
    ckpt = load_checkpoint(run_dir)
    if ckpt is None:
        return "pending"
    if ckpt["done"]:
        return "done"
    return f"partial ({ckpt['next_generation']}/{ckpt['args']['generations']} generations)"


def print_status(runs: list[dict]):
    for run in runs:
        line = f"{run['experiment']:<18} {run['name']:<28} {run_status(run)}"
        history_path = os.path.join(run_dir_for(run), "history.jsonl")
        if os.path.exists(history_path) and os.path.getsize(history_path) > 0:
            last = load_history(run_dir_for(run))[-1]
            line += f"  last best {last['best']:.3f} mean {last['mean']:.3f}"
        print(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", help="Path to a plan JSON file")
    parser.add_argument("--status", action="store_true", help="Show progress and exit")
    args = parser.parse_args()

    with open(args.plan) as f:
        runs = json.load(f)["runs"]

    names = [(r["experiment"], r["name"]) for r in runs]
    if len(names) != len(set(names)):
        sys.exit("Plan has duplicate run names")

    if args.status:
        print_status(runs)
        return

    for i, run in enumerate(runs, start=1):
        status = run_status(run)
        label = f"[{i}/{len(runs)}] {run['experiment']}/{run['name']}"
        if status == "done":
            print(f"{label}: done, skipping")
            continue

        print(f"{label}: {'resuming' if status != 'pending' else 'starting'} ({status})", flush=True)
        run_dir = run_dir_for(run)
        os.makedirs(run_dir, exist_ok=True)
        script = os.path.join(EXPERIMENTS_DIR, run["experiment"], "run.py")
        with open(os.path.join(run_dir, "console.log"), "a") as log:
            result = subprocess.run(
                [sys.executable, "-u", script, "--run-name", run["name"], *run.get("args", [])],
                stdout=log, stderr=subprocess.STDOUT,
            )
        if result.returncode != 0:
            sys.exit(f"{label}: failed with exit code {result.returncode}; see {run_dir}/console.log. "
                     f"Fix and re-run this command to resume.")
        print(f"{label}: finished", flush=True)

    print("All runs finished.")


if __name__ == "__main__":
    main()
