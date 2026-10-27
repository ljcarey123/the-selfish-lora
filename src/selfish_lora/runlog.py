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


def make_run_dir(experiment_dir: str, name: str | None, tag: str) -> str:
    """
    Directory for a run. A given name makes the run resumable by re-running
    the same command; without one, a fresh timestamped directory is used.
    """
    run_dir = os.path.join(experiment_dir, "runs", name or f"{time.strftime('%Y%m%d-%H%M%S')}_{tag}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


class RunLog:
    def __init__(self, run_dir: str, config: dict, start_generation: int = 0):
        """
        Args:
            run_dir: Run directory
            config: Settings to record in config.json
            start_generation: Generation the run is starting from. When resuming,
                history from earlier generations is kept and anything later
                (a generation interrupted before its checkpoint) is dropped.
        """
        self.run_dir = run_dir
        self.history_path = os.path.join(run_dir, "history.jsonl")
        elapsed_before = 0.0

        if start_generation > 0 and os.path.exists(self.history_path):
            kept = [r for r in load_history(run_dir) if r["generation"] < start_generation]
            with open(self.history_path, "w") as f:
                f.writelines(json.dumps(r) + "\n" for r in kept)
            if kept:
                elapsed_before = kept[-1]["elapsed_s"]
            print(f"Resuming {run_dir} at generation {start_generation}")
        else:
            open(self.history_path, "w").close()
            with open(os.path.join(run_dir, "config.json"), "w") as f:
                json.dump(config, f, indent=2)

        self.start = time.time() - elapsed_before

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


LINE_STYLES = ["-", "--", ":"]


def plot_runs(run_dirs: list[str], labels: list[str], out_path: str, title: str = "Fitness by generation",
              metrics: tuple = ("best", "mean"), ylabel: str = "Fitness", panels: bool = False):
    """
    Plot history metrics per generation for each run.

    Colour identifies the label (runs sharing a label, e.g. several seeds of
    "evolved", share a colour). With panels=False, up to three metrics that
    share a unit go on one axis, told apart by line style. With panels=True,
    each metric gets its own panel and y-axis.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    if not panels and len(metrics) > len(LINE_STYLES):
        raise ValueError(f"At most {len(LINE_STYLES)} metrics on one axis; use panels=True")

    n_axes = len(metrics) if panels else 1
    fig, axes = plt.subplots(1, n_axes, figsize=(4 * n_axes + 1 if panels else 8, 4.5),
                             facecolor=SURFACE, squeeze=False)
    axes = axes[0]

    colors = {}
    for run_dir, label in zip(run_dirs, labels):
        color = colors.setdefault(label, SERIES_COLORS[len(colors) % len(SERIES_COLORS)])
        history = load_history(run_dir)
        gens = [r["generation"] for r in history]
        for i, metric in enumerate(metrics):
            ax = axes[i] if panels else axes[0]
            style = "-" if panels else LINE_STYLES[i]
            ax.plot(gens, [r.get(metric, float("nan")) for r in history], color=color, linewidth=2,
                    linestyle=style, label=label if panels else f"{label} {metric}")

    for i, ax in enumerate(axes):
        ax.set_facecolor(SURFACE)
        if panels:
            ax.set_title(metrics[i], color=TEXT_PRIMARY, loc="left", fontsize=10)
        ax.set_xlabel("Generation", color=TEXT_SECONDARY)
        ax.tick_params(colors=TEXT_SECONDARY)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
    if not panels:
        axes[0].set_ylabel(ylabel, color=TEXT_SECONDARY)

    # One legend entry per label (and metric), even with several seeds per label
    handles, names = axes[0].get_legend_handles_labels()
    unique = dict(zip(names, handles))
    axes[0].legend(unique.values(), unique.keys(), frameon=False, labelcolor=TEXT_PRIMARY, fontsize=9)

    fig.suptitle(title, color=TEXT_PRIMARY, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved plot: {out_path}")
