"""
Pick a mutation size: for each noise std, sample mutants of the base genome
and measure how often they still answer validly and how often they behave
differently from the base model.

Evolution needs mutants that are mostly valid (too much noise breaks the
output format) but often different (too little noise and selection has
nothing to choose between).

    python experiments/calibrate_mutation.py --stds 0.005 0.01 0.02 0.05
"""

import argparse
import itertools
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "countdown"))
sys.path.insert(0, os.path.join(HERE, "prisoners_dilemma"))

from selfish_lora.model import load_model
from selfish_lora.genome import genome_keys, genome_from_model, load_genome
from selfish_lora.generate import generate

import task as countdown
import game as pd


def pd_histories(max_len: int = 3) -> list[tuple[list[str], list[str]]]:
    """Every history up to max_len rounds where each side played C or D."""
    histories = []
    for n in range(max_len + 1):
        for moves in itertools.product("CD", repeat=2 * n):
            histories.append((list(moves[:n]), list(moves[n:])))
    return histories


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stds", type=float, nargs="+", default=[0.005, 0.01, 0.02, 0.05])
    parser.add_argument("--mutants", type=int, default=8)
    parser.add_argument("--puzzles", type=int, default=32)
    args = parser.parse_args()

    model, tokenizer = load_model()
    torch.manual_seed(0)
    base = genome_from_model(model, genome_keys(model))

    puzzles = countdown.CountdownTask(size=args.puzzles, seed=123)
    idx = list(range(args.puzzles))
    histories = pd_histories(max_len=2)
    pd_prompts = [pd.build_prompt(mine, theirs) for mine, theirs in histories]

    def measure(genome):
        load_genome(model, genome)
        cd_out = generate(model, tokenizer, puzzles.prompts(idx), countdown.SYSTEM_PROMPT, max_new_tokens=24)
        cd_scores = puzzles.score(idx, cd_out)
        pd_moves = [pd.parse_move(t) for t in
                    generate(model, tokenizer, pd_prompts, pd.SYSTEM_PROMPT, max_new_tokens=3)]
        return cd_out, cd_scores, pd_moves

    base_cd, base_scores, base_moves = measure(base)
    print(f"Base: countdown score {np.mean(base_scores):.3f}, PD moves {''.join(m or '?' for m in base_moves)}")

    print(f"\n{'std':>7} | {'CD score':>8} {'CD valid':>8} {'CD changed':>10} | "
          f"{'PD valid':>8} {'PD changed':>10} {'PD coop':>8}")
    for std in args.stds:
        rows = []
        for _ in range(args.mutants):
            mutant = {k: v + torch.randn_like(v) * std for k, v in base.items()}
            cd_out, scores, moves = measure(mutant)
            rows.append((
                np.mean(scores),
                np.mean([s >= 0.05 for s in scores]),
                np.mean([a != b for a, b in zip(cd_out, base_cd)]),
                np.mean([m is not None for m in moves]),
                np.mean([a != b for a, b in zip(moves, base_moves)]),
                np.mean([m == "C" for m in moves]),
            ))
        r = np.mean(rows, axis=0)
        print(f"{std:7.3f} | {r[0]:8.3f} {r[1]:8.1%} {r[2]:10.1%} | {r[3]:8.1%} {r[4]:10.1%} {r[5]:8.1%}")


if __name__ == "__main__":
    main()
