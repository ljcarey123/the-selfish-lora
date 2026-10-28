"""
Evolve LoRA adapters that play the iterated prisoner's dilemma.

Each generation is a round-robin tournament: every genome plays every other
genome, plus a few fixed-strategy bots (tit-for-tat, always-defect, ...).
Fitness is a genome's average payoff per round across all its matches.
With --rounds-max, match lengths are drawn uniformly from [--rounds,
--rounds-max] each generation, so there is no fixed last round to exploit.
--lengths-per-gen draws several lengths and plays every pairing at each, so
fitness averages over lengths instead of chasing one generation's draw.

    python experiments/prisoners_dilemma/run.py --generations 30 --run-name evolved_s0
    python experiments/prisoners_dilemma/run.py --generations 30 --run-name control_s0 --control

A checkpoint is saved after every generation. Re-running a command with the
same --run-name resumes where it stopped (or exits if the run is finished).
"""

import argparse
import itertools
import os
import random
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from selfish_lora.model import load_model
from selfish_lora.genome import (genome_keys, fixed_keys, genome_from_model, load_genome, load_genome_file,
                                 save_genome, genome_norm)
from selfish_lora.evolution import EvolutionConfig, initial_population, next_generation, mean_pairwise_distance
from selfish_lora.generate import generate
from selfish_lora.runlog import RunLog, make_run_dir, plot_runs
from selfish_lora.checkpoint import save_checkpoint, load_checkpoint, restore_rng

from game import BOTS, DEFECT, SYSTEM_PROMPT, build_prompt, parse_move, score_round


def play_tournament(model, tokenizer, population, bot_names, lengths, batch_size, self_play=True):
    """
    Play all matches simultaneously, one round at a time.

    Players are genome indices (ints) or bot names (strings). Every pairing is
    played once at each length in `lengths`; matches run side by side and a
    match stops when it reaches its length. Each round, each genome is loaded
    once and answers every one of its unfinished matches in a batch. With
    self_play=False, genomes only play the bots.

    Returns:
        matches: list of dicts with players "a", "b", "length" and their move lists
        payoffs: dict player -> list of average payoff per round, one per match
            (so short and long matches count equally)
        invalid: dict genome index -> count of unparseable moves
        moves_made: dict genome index -> count of moves made
    """
    genome_ids = list(range(len(population)))
    pairs = [(g, b) for g in genome_ids for b in bot_names]
    if self_play:
        pairs += list(itertools.combinations(genome_ids, 2))
    matches = [{"a": a, "b": b, "length": n, "moves": {a: [], b: []}, "payoff": {a: 0, b: 0}}
               for a, b in pairs for n in lengths]

    by_player = defaultdict(list)
    for m in matches:
        by_player[m["a"]].append(m)
        by_player[m["b"]].append(m)

    invalid = defaultdict(int)
    moves_made = defaultdict(int)

    for round_idx in range(max(lengths)):
        active = [m for m in matches if round_idx < m["length"]]
        chosen = {}  # (match id, player) -> move
        for g in genome_ids:
            own = [m for m in by_player[g] if round_idx < m["length"]]
            prompts = [build_prompt(m["moves"][g], m["moves"][_opponent(m, g)]) for m in own]
            load_genome(model, population[g])
            completions = generate(model, tokenizer, prompts, SYSTEM_PROMPT, max_new_tokens=3,
                                   batch_size=batch_size)
            for m, text in zip(own, completions):
                move = parse_move(text)
                invalid[g] += move is None
                moves_made[g] += 1
                chosen[(id(m), g)] = move
        for m in active:
            for p in (m["a"], m["b"]):
                if isinstance(p, str):
                    chosen[(id(m), p)] = BOTS[p](m["moves"][p], m["moves"][_opponent(m, p)])

        for m in active:
            move_a, move_b = chosen[(id(m), m["a"])], chosen[(id(m), m["b"])]
            pay_a, pay_b = score_round(move_a, move_b)
            m["payoff"][m["a"]] += pay_a
            m["payoff"][m["b"]] += pay_b
            m["moves"][m["a"]].append(move_a or DEFECT)
            m["moves"][m["b"]].append(move_b or DEFECT)

    payoffs = defaultdict(list)
    for m in matches:
        for p in (m["a"], m["b"]):
            payoffs[p].append(m["payoff"][p] / m["length"])
    return matches, payoffs, invalid, moves_made


def _opponent(match, player):
    return match["b"] if match["a"] == player else match["a"]


def behaviour_metrics(matches, genome_ids, bot_names, best):
    """
    Population-level behaviour, so strategy shifts are visible even when fitness is flat.

    - coop_rate: share of C moves by genomes
    - retaliation: P(genome plays D | opponent played D last round)
    - coop_vs_<bot>: the population's cooperation rate against each bot
    - best_coop_vs_<bot>: the best genome's cooperation rate against each bot
    """
    coop, total, retaliate, provoked = 0, 0, 0, 0
    best_vs_bot = defaultdict(list)
    vs_bot = defaultdict(list)
    for m in matches:
        for p in (m["a"], m["b"]):
            if p not in genome_ids:
                continue
            mine, theirs = m["moves"][p], m["moves"][_opponent(m, p)]
            coop += mine.count("C")
            total += len(mine)
            for prev_theirs, now_mine in zip(theirs[:-1], mine[1:]):
                if prev_theirs == DEFECT:
                    provoked += 1
                    retaliate += now_mine == DEFECT
            if _opponent(m, p) in bot_names:
                rate = mine.count("C") / len(mine)
                vs_bot[f"coop_vs_{_opponent(m, p)}"].append(rate)
                if p == best:
                    best_vs_bot[f"best_coop_vs_{_opponent(m, p)}"].append(rate)

    return {
        "coop_rate": coop / max(total, 1),
        "retaliation": retaliate / provoked if provoked else float("nan"),
        **{k: float(np.mean(v)) for k, v in vs_bot.items()},
        **{k: float(np.mean(v)) for k, v in best_vs_bot.items()},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--population", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=8, help="Rounds per match (minimum if --rounds-max)")
    parser.add_argument("--rounds-max", type=int, help="Draw each generation's match length from [rounds, this]")
    parser.add_argument("--lengths-per-gen", type=int, default=1,
                        help="With --rounds-max: lengths drawn per generation; every pairing is played at each")
    parser.add_argument("--bots", default="tft,alld", help=f"Comma-separated from {list(BOTS)}, or ''")
    parser.add_argument("--self-play", action=argparse.BooleanOptionalAction, default=True,
                        help="Genomes play each other as well as the bots (--no-self-play: bots only)")
    parser.add_argument("--mutation-std", type=float, default=0.02, help="Noise added to each child")
    parser.add_argument("--initial-std", type=float, default=0.03, help="Noise used to diversify generation 0")
    parser.add_argument("--seed-genome", help="Genome file (.pt) to place in generation 0, e.g. a trained tit-for-tat")
    parser.add_argument("--seed-count", type=int, default=0, help="Exact copies of --seed-genome in generation 0")
    parser.add_argument("--weight-decay", type=float, default=0.0,
                        help="Pull each child this fraction of the way back to the base genome")
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--control", action="store_true", help="Random selection baseline")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", help="Fixed run directory name; re-running resumes it")
    args = parser.parse_args()

    tag = "control" if args.control else "evolved"
    run_dir = make_run_dir(os.path.dirname(__file__), args.run_name, tag)
    ckpt = load_checkpoint(run_dir)
    if ckpt and ckpt["done"]:
        print(f"Run already finished: {run_dir}")
        return
    if ckpt:
        # Saved settings win on resume; defaults fill options added since the run started
        args = argparse.Namespace(**{**vars(parser.parse_args([])), **ckpt["args"]})
    if args.rounds_max is not None and args.rounds_max < args.rounds:
        parser.error("--rounds-max must be at least --rounds")
    if args.seed_count and not args.seed_genome:
        parser.error("--seed-count needs --seed-genome")

    bot_names = [b for b in args.bots.split(",") if b]
    unknown = set(bot_names) - set(BOTS)
    if unknown:
        parser.error(f"Unknown bots: {unknown}")

    cfg = EvolutionConfig(population_size=args.population, mutation_std=args.mutation_std,
                          initial_std=args.initial_std, weight_decay=args.weight_decay)
    start = ckpt["next_generation"] if ckpt else 0
    log = RunLog(run_dir, {**vars(args), "evolution": cfg.to_dict()}, start_generation=start)

    # Seed after loading: load_model reseeds torch to fix the shared A matrices
    model, tokenizer = load_model(lora_rank=args.lora_rank)
    device = next(model.parameters()).device
    base = genome_from_model(model, genome_keys(model))

    if ckpt:
        load_genome(model, ckpt["fixed"])
        population = [{k: v.to(device) for k, v in g.items()} for g in ckpt["population"]]
        restore_rng(ckpt)
    else:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        population = initial_population(base, cfg)
        if args.seed_count:
            # Replace the noisiest members (the base genome stays at index 0) with exact copies
            seeded = load_genome_file(args.seed_genome, device)
            population[-args.seed_count:] = [{k: v.clone() for k, v in seeded.items()}
                                             for _ in range(args.seed_count)]
    fixed = genome_from_model(model, fixed_keys(model))
    genome_ids = list(range(len(population)))

    for gen in range(start, args.generations):
        if args.rounds_max:
            lengths = [random.randint(args.rounds, args.rounds_max) for _ in range(args.lengths_per_gen)]
        else:
            lengths = [args.rounds]
        matches, payoffs, invalid, moves_made = play_tournament(model, tokenizer, population, bot_names,
                                                                lengths, args.batch_size, args.self_play)
        fitness = [float(np.mean(payoffs[g])) for g in genome_ids]
        best = int(np.argmax(fitness))

        extra = {
            **behaviour_metrics(matches, genome_ids, bot_names, best),
            "invalid_rate": sum(invalid.values()) / sum(moves_made.values()),
            "diversity": mean_pairwise_distance(population),
            "best_norm": genome_norm(population[best]),
            "rounds": lengths[0] if len(lengths) == 1 else lengths,
        }
        for b in bot_names:
            extra[f"{b}_payoff"] = float(np.mean(payoffs[b]))
        log.log(gen, fitness, **extra)

        if gen == args.generations - 1:
            save_genome(population[best], os.path.join(run_dir, "best.pt"))
            save_checkpoint(run_dir, vars(args), gen + 1, population, fixed, done=True)
        else:
            population = next_generation(population, fitness, cfg, control=args.control, base=base)
            save_checkpoint(run_dir, vars(args), gen + 1, population, fixed)

    plot_runs([run_dir], [tag], os.path.join(run_dir, "fitness.png"),
              title="Prisoner's dilemma: payoff per round")
    print(f"Run saved to {run_dir}")


if __name__ == "__main__":
    main()
