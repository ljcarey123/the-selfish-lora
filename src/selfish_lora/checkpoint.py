"""
Save and restore a run between generations so it can resume after a crash,
reboot or closed laptop lid.

A checkpoint holds everything needed to continue exactly where the run left
off: the run's arguments, the next generation to play, the population, the
fixed LoRA A matrices and all random number generator states. It is written
atomically after every generation, so at worst one generation is repeated.
"""

import os
import random

import numpy as np
import torch

CHECKPOINT_FILE = "checkpoint.pt"


def save_checkpoint(run_dir: str, args: dict, next_generation: int, population: list, fixed: dict,
                    done: bool = False, **extra):
    """
    Write a checkpoint to run_dir, replacing any previous one.

    Args:
        run_dir: Run directory
        args: The run's arguments (restored on resume so settings can't drift)
        next_generation: Generation to play next on resume
        population: Genomes to resume with
        fixed: Non-evolved LoRA tensors (the shared A matrices)
        done: True once the run has finished
        extra: Anything else the experiment needs on resume (e.g. base score)
    """
    state = {
        "args": args,
        "next_generation": next_generation,
        "done": done,
        "population": [{k: v.cpu() for k, v in g.items()} for g in population],
        "fixed": {k: v.cpu() for k, v in fixed.items()},
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        "extra": extra,
    }
    path = os.path.join(run_dir, CHECKPOINT_FILE)
    tmp_path = path + ".tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)


def load_checkpoint(run_dir: str) -> dict | None:
    """Load a run's checkpoint (tensors on CPU), or return None if it has none."""
    path = os.path.join(run_dir, CHECKPOINT_FILE)
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def restore_rng(state: dict):
    """Restore all random number generators from a checkpoint."""
    rng = state["rng"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if rng["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda"])
