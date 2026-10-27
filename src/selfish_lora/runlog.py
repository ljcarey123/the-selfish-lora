"""
Per-generation logging and fitness plots.

Each run writes to its own directory:
    config.json     settings for the run
    history.jsonl   one JSON line per generation
    best.pt         best genome from the final generation
    fitness.png     best and mean fitness by generation

plot_runs() overlays several runs (e.g. evolved vs control) on one chart.
"""

import json
import os
import time

import numpy as np

# Categorical slots in fixed order, plus chart chrome, from a validated default palette
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"


def make_run_dir(experiment_dir: str, tag: str) -> str:
    run_dir = os.path.join(experiment_dir, "runs", f"{time.strftime('%Y%m%d-%H%M%S')}_{tag}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


class RunLog:
    def __init__(self, run_dir: str, config: dict):
        self.run_dir = run_dir
        self.history_path = os.path.join(run_dir, "history.jsonl")
        self.start = time.time()
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

    def log(self, generation: int, fitness: list[float], **extra):
        """Record one generation's fitness scores plus any extra metrics."""
        record = {
            "generation": generation,
            "best": float(np.max(fitness)),
            "mean": float(np.mean(fitness)),
            "min": float(np.min(fitness)),
            "elapsed_s": round(time.time() - self.start, 1),
            **extra,
        }
        with open(self.history_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        shown = " ".join(f"{k}={v:.3f}" for k, v in extra.items() if isinstance(v, float))
        print(f"gen {generation:3d} | best {record['best']:.3f} mean {record['mean']:.3f} "
              f"| {shown} | {record['elapsed_s']:.0f}s")
        return record


def load_history(run_dir: str) -> list[dict]:
    with open(os.path.join(run_dir, "history.jsonl")) as f:
        return [json.loads(line) for line in f if line.strip()]


def plot_runs(run_dirs: list[str], labels: list[str], out_path: str, title: str = "Fitness by generation"):
    """
    Plot best (solid) and mean (dashed) fitness per generation for each run.

    One colour per run, so evolved vs control reads at a glance.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    for i, (run_dir, label) in enumerate(zip(run_dirs, labels)):
        history = load_history(run_dir)
        gens = [r["generation"] for r in history]
        color = SERIES_COLORS[i % len(SERIES_COLORS)]
        ax.plot(gens, [r["best"] for r in history], color=color, linewidth=2, label=f"{label} best")
        ax.plot(gens, [r["mean"] for r in history], color=color, linewidth=2, linestyle="--",
                label=f"{label} mean")
        ax.annotate(label, (gens[-1], history[-1]["best"]), xytext=(6, 0), textcoords="offset points",
                    va="center", color=TEXT_PRIMARY, fontsize=9)

    ax.set_title(title, color=TEXT_PRIMARY, loc="left")
    ax.set_xlabel("Generation", color=TEXT_SECONDARY)
    ax.set_ylabel("Fitness", color=TEXT_SECONDARY)
    ax.tick_params(colors=TEXT_SECONDARY)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.legend(frameon=False, labelcolor=TEXT_PRIMARY, fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved plot: {out_path}")
