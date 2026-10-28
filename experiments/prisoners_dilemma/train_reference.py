"""
Train a genome to play a hand-coded strategy with gradient descent.

This answers "can this LoRA represent the strategy at all?" separately from
"can evolution find it?". It uses the same rank, target modules and fixed A
matrices as the evolution runs and trains only the B matrices, so the result
is an ordinary genome: it can be fingerprinted, or seeded into a population.

    python experiments/prisoners_dilemma/train_reference.py --strategy tft
"""

import argparse
import os
import random
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

from selfish_lora.model import load_model
from selfish_lora.genome import genome_keys, genome_from_model, save_genome, genome_norm
from selfish_lora.generate import generate

from game import BOTS, SYSTEM_PROMPT, build_prompt, parse_move


def random_history(max_rounds: int) -> tuple[list[str], list[str]]:
    n = random.randint(0, max_rounds)
    return [random.choice("CD") for _ in range(n)], [random.choice("CD") for _ in range(n)]


def chat(tokenizer, prompt: str) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def agreement(model, tokenizer, strategy, histories) -> tuple[float, float]:
    """Share of histories where the greedy move matches the strategy, and the valid-move rate."""
    prompts = [build_prompt(mine, theirs) for mine, theirs in histories]
    moves = [parse_move(t) for t in generate(model, tokenizer, prompts, SYSTEM_PROMPT, max_new_tokens=3,
                                             batch_size=1)]
    target = [strategy(mine, theirs) for mine, theirs in histories]
    return float(np.mean([m == t for m, t in zip(moves, target)])), float(np.mean([m is not None for m in moves]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="tft", choices=[b for b in BOTS if b != "random"])
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-rounds", type=int, default=9, help="Longest training history")
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model, tokenizer = load_model(lora_rank=args.lora_rank)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    strategy = BOTS[args.strategy]

    keys = set(genome_keys(model))
    params = []
    for name, p in model.named_parameters():
        p.requires_grad = name in keys
        if "lora_" in name:
            p.data = p.data.float()  # LoRA in float32 for stable training; the frozen base stays bf16
        if name in keys:
            params.append(p)
    optimizer = torch.optim.Adam(params, lr=args.lr)

    target_ids = {m: tokenizer.encode(m, add_special_tokens=False)[0] for m in "CD"}
    device = next(model.parameters()).device
    test = [random_history(args.max_rounds) for _ in range(100)]
    print("Before training: agreement %.0f%%, valid %.0f%%" % tuple(100 * x for x in
          agreement(model, tokenizer, strategy, test)))

    model.train()
    for step in range(1, args.steps + 1):
        batch = [random_history(args.max_rounds) for _ in range(args.batch)]
        texts = [chat(tokenizer, build_prompt(mine, theirs)) for mine, theirs in batch]
        labels = torch.tensor([target_ids[strategy(mine, theirs)] for mine, theirs in batch], device=device)
        inputs = tokenizer(texts, return_tensors="pt", padding=True).to(device)
        logits = model(**inputs).logits[:, -1, :].float()
        loss = torch.nn.functional.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 25 == 0:
            print(f"step {step}: loss {loss.item():.3f}")
    model.eval()

    agree, valid = agreement(model, tokenizer, strategy, test)
    longer = [random_history(15) for _ in range(50)]
    agree_long, _ = agreement(model, tokenizer, strategy, longer)
    genome = genome_from_model(model, genome_keys(model))
    print(f"After training: agreement {agree:.0%} (held-out histories up to {args.max_rounds} rounds), "
          f"{agree_long:.0%} (up to 15 rounds), valid {valid:.0%}, genome norm {genome_norm(genome):.1f}")

    out_dir = os.path.join(HERE, "runs", f"reference_{args.strategy}")
    os.makedirs(out_dir, exist_ok=True)
    save_genome(genome, os.path.join(out_dir, "best.pt"))
    print(f"Saved genome to {out_dir}")


if __name__ == "__main__":
    main()
