"""
Show what strategy a genome actually plays.

Prints the genome's move for every history of up to two rounds and the
classic strategies it agrees with most. Compares the base model, and each
run's best genome, side by side.

    python experiments/prisoners_dilemma/fingerprint.py runs/panel_evolved_s0 runs/panel_control_s0
"""

import argparse
import itertools
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

from selfish_lora.model import load_model
from selfish_lora.genome import genome_keys, genome_from_model, load_genome, load_genome_file
from selfish_lora.checkpoint import load_checkpoint
from selfish_lora.generate import generate

from game import BOTS, SYSTEM_PROMPT, build_prompt, parse_move


def histories(max_len: int = 2) -> list[tuple[list[str], list[str]]]:
    """Every history up to max_len rounds where each side played C or D."""
    result = []
    for n in range(max_len + 1):
        for moves in itertools.product("CD", repeat=2 * n):
            result.append((list(moves[:n]), list(moves[n:])))
    return result


def describe(history: tuple[list[str], list[str]]) -> str:
    mine, theirs = history
    if not mine:
        return "first move"
    return " ".join(f"{a}{b}" for a, b in zip(mine, theirs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="*", help="Run directories (relative to this folder or absolute)")
    args = parser.parse_args()

    model, tokenizer = load_model()
    device = next(model.parameters()).device
    # Read the shared A matrices from a checkpoint so genomes are evaluated as they were evolved
    for run_dir in args.run_dirs:
        ckpt = load_checkpoint(os.path.join(HERE, run_dir))
        if ckpt:
            load_genome(model, {k: v.to(device) for k, v in ckpt["fixed"].items()})
            break

    players = {"base": genome_from_model(model, genome_keys(model))}
    for run_dir in args.run_dirs:
        players[os.path.basename(run_dir.rstrip("/\\"))] = load_genome_file(
            os.path.join(HERE, run_dir, "best.pt"), device)

    hs = histories()
    prompts = [build_prompt(mine, theirs) for mine, theirs in hs]
    moves = {}
    for name, genome in players.items():
        load_genome(model, genome)
        # Unbatched: in bf16, padding prompts of different lengths can flip borderline moves
        moves[name] = [parse_move(t) or "?" for t in
                       generate(model, tokenizer, prompts, SYSTEM_PROMPT, max_new_tokens=3, batch_size=1)]

    names = list(players)
    width = max(len(n) for n in names)
    print(f"{'history (you,opp per round)':<28}" + "".join(f"{n[:width]:>{width + 2}}" for n in names))
    for i, h in enumerate(hs):
        print(f"{describe(h):<28}" + "".join(f"{moves[n][i]:>{width + 2}}" for n in names))

    print("\nAgreement with classic strategies:")
    for name in names:
        scores = {}
        for bot, strategy in BOTS.items():
            if bot == "random":
                continue
            scores[bot] = sum(strategy(mine, theirs) == m for (mine, theirs), m in zip(hs, moves[name])) / len(hs)
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        print(f"  {name:<{width}}  " + "  ".join(f"{bot} {s:.0%}" for bot, s in ranked))


if __name__ == "__main__":
    main()
