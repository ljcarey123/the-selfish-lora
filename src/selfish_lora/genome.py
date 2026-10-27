"""
A genome is a dict mapping LoRA parameter names to float32 tensors.

By default only the lora_B matrices are evolved. The LoRA update is
(alpha / r) * B @ A, so with A fixed at a random init, a mutation to B changes
the model's behaviour directly. (The legacy gridworld evolved A and B from
zero, where a small mutation to both changes B @ A only to second order -
one reason its fitness barely moved.)
"""

import torch

Genome = dict[str, torch.Tensor]


def genome_keys(model, evolve_a: bool = False) -> list[str]:
    """Names of the LoRA parameters that make up a genome."""
    keys = []
    for name, _ in model.named_parameters():
        if "lora_B" in name or (evolve_a and "lora_A" in name):
            keys.append(name)
    return keys


def genome_from_model(model, keys: list[str]) -> Genome:
    """Copy the model's current LoRA parameters into a new genome."""
    params = dict(model.named_parameters())
    return {k: params[k].detach().float().clone() for k in keys}


def load_genome(model, genome: Genome):
    """Write a genome's tensors into the model in place."""
    params = dict(model.named_parameters())
    with torch.no_grad():
        for name, tensor in genome.items():
            params[name].data.copy_(tensor.to(params[name].dtype))


def genome_distance(g1: Genome, g2: Genome) -> float:
    """L2 distance between two genomes."""
    return sum(torch.sum((g1[k] - g2[k]) ** 2).item() for k in g1) ** 0.5


def genome_norm(genome: Genome) -> float:
    """L2 norm of a genome."""
    return sum(torch.sum(t ** 2).item() for t in genome.values()) ** 0.5


def save_genome(genome: Genome, path: str):
    torch.save({k: v.cpu() for k, v in genome.items()}, path)


def load_genome_file(path: str, device: str) -> Genome:
    return {k: v.to(device) for k, v in torch.load(path).items()}
