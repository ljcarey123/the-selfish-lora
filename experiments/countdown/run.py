"""
Evolve LoRA adapters to solve Countdown puzzles.

Each generation, every genome answers the same fresh batch of puzzles, so
fitness differences come from the weights rather than puzzle luck. The best
genome is also scored on a fixed held-out set to check it generalises.

    python experiments/countdown/run.py --generations 30 --run-name evolved_s0
    python experiments/countdown/run.py --generations 30 --run-name control_s0 --control

A checkpoint is saved after every generation. Re-running a command with the
same --run-name resumes where it stopped (or exits if the run is finished).
    python experiments/compare_runs.py experiments/countdown/runs/<evolved> experiments/countdown/runs/<control>
"""

import argparse
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from selfish_lora.model import load_model
from selfish_lora.genome import genome_keys, fixed_keys, genome_from_model, load_genome, save_genome, genome_norm
from selfish_lora.evolution import EvolutionConfig, initial_population, next_generation, mean_pairwise_distance
from selfish_lora.generate import generate
from selfish_lora.runlog import RunLog, make_run_dir, plot_runs
from selfish_lora.checkpoint import save_checkpoint, load_checkpoint, restore_rng

from task import CountdownTask, SYSTEM_PROMPT

TRAIN_POOL = 5000
HELDOUT_SIZE = 100


def evaluate(model, tokenizer, genome, task, indices, batch_size):
    load_genome(model, genome)
    completions = generate(model, tokenizer, task.prompts(indices), SYSTEM_PROMPT,
                           max_new_tokens=24, batch_size=batch_size)
    scores = task.score(indices, completions)
    return float(np.mean(scores)), float(np.mean([s == 1.0 for s in scores])), completions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--population", type=int, default=16)
    parser.add_argument("--puzzles-per-gen", type=int, default=32)
    parser.add_argument("--mutation-std", type=float, default=0.01, help="Noise added to each child")
    parser.add_argument("--initial-std", type=float, default=0.015, help="Noise used to diversify generation 0")
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--heldout-every", type=int, default=5)
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
        args = argparse.Namespace(**ckpt["args"])  # Saved settings win on resume

    cfg = EvolutionConfig(population_size=args.population, mutation_std=args.mutation_std,
                          initial_std=args.initial_std)
    start = ckpt["next_generation"] if ckpt else 0
    log = RunLog(run_dir, {**vars(args), "evolution": cfg.to_dict()}, start_generation=start)

    # Seed after loading: load_model reseeds torch to fix the shared A matrices
    model, tokenizer = load_model(lora_rank=args.lora_rank)
    device = next(model.parameters()).device

    train = CountdownTask(size=TRAIN_POOL, seed=args.seed)
    heldout = CountdownTask(size=HELDOUT_SIZE, seed=10_000 + args.seed)
    heldout_idx = list(range(HELDOUT_SIZE))

    if ckpt:
        load_genome(model, ckpt["fixed"])
        population = [{k: v.to(device) for k, v in g.items()} for g in ckpt["population"]]
        base_score = ckpt["extra"]["base_score"]
        restore_rng(ckpt)
    else:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        base = genome_from_model(model, genome_keys(model))
        base_score, base_acc, samples = evaluate(model, tokenizer, base, heldout, heldout_idx, args.batch_size)
        print(f"Base model held-out: score {base_score:.3f}, accuracy {base_acc:.1%}")
        print("Sample completions:", samples[:3])
        population = initial_population(base, cfg)
    fixed = genome_from_model(model, fixed_keys(model))

    for gen in range(start, args.generations):
        indices = random.sample(range(TRAIN_POOL), args.puzzles_per_gen)
        results = [evaluate(model, tokenizer, g, train, indices, args.batch_size) for g in population]
        fitness = [r[0] for r in results]
        best = population[int(np.argmax(fitness))]

        extra = {
            "best_accuracy": max(r[1] for r in results),
            "diversity": mean_pairwise_distance(population),
            "best_norm": genome_norm(best),
        }
        if gen % args.heldout_every == 0 or gen == args.generations - 1:
            extra["heldout_score"], extra["heldout_accuracy"], _ = evaluate(
                model, tokenizer, best, heldout, heldout_idx, args.batch_size)
        log.log(gen, fitness, **extra)

        if gen == args.generations - 1:
            save_genome(best, os.path.join(run_dir, "best.pt"))
            save_checkpoint(run_dir, vars(args), gen + 1, population, fixed, done=True, base_score=base_score)
        else:
            population = next_generation(population, fitness, cfg, control=args.control)
            save_checkpoint(run_dir, vars(args), gen + 1, population, fixed, base_score=base_score)

    plot_runs([run_dir], [tag], os.path.join(run_dir, "fitness.png"),
              title=f"Countdown fitness (base held-out score {base_score:.3f})")
    print(f"Run saved to {run_dir}")


if __name__ == "__main__":
    main()
