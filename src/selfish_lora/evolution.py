"""
Genetic algorithm over LoRA genomes: truncation selection, elitism,
uniform crossover and Gaussian mutation.

Setting control=True keeps everything identical but ignores fitness when
choosing elites and parents. Comparing a normal run against a control run
with the same budget shows whether selection is doing any work.
"""

import random
from dataclasses import dataclass, asdict

import numpy as np
import torch

from .genome import Genome, genome_distance


@dataclass
class EvolutionConfig:
    population_size: int = 16
    num_elites: int = 2             # Copied unchanged into the next generation
    parent_fraction: float = 0.25   # Top fraction allowed to breed
    mutation_std: float = 0.05      # Std of Gaussian noise added to every element
    initial_std: float = 0.05       # Std of noise used to diversify generation 0
    crossover: bool = True          # Uniform crossover; False = mutate one parent only
    weight_decay: float = 0.0       # Fraction of each child's distance from the base genome removed
                                    # before mutation; bounds the build-up of mutations over generations

    def to_dict(self) -> dict:
        return asdict(self)


def initial_population(base: Genome, cfg: EvolutionConfig) -> list[Genome]:
    """
    Generation 0: member 0 is the unmodified base genome, the rest are noisy copies.

    Keeping the base genome in the population means fitness can never start
    below the base model's own score.
    """
    population = [{k: v.clone() for k, v in base.items()}]
    for _ in range(cfg.population_size - 1):
        population.append({k: v + torch.randn_like(v) * cfg.initial_std for k, v in base.items()})
    return population


def next_generation(
    population: list[Genome],
    fitness: list[float],
    cfg: EvolutionConfig,
    control: bool = False,
    base: Genome | None = None,
) -> list[Genome]:
    """
    Breed the next generation from scored genomes.

    Args:
        population: Current genomes
        fitness: One score per genome (higher is better)
        cfg: Evolution settings
        control: If True, rank genomes randomly instead of by fitness
        base: Genome that weight decay pulls children towards (required if cfg.weight_decay > 0)

    Returns:
        New population of the same size
    """
    if cfg.weight_decay > 0 and base is None:
        raise ValueError("weight_decay needs the base genome")
    if control:
        order = list(np.random.permutation(len(population)))
    else:
        order = list(np.argsort(fitness)[::-1])
    ranked = [population[i] for i in order]

    elites = [{k: v.clone() for k, v in g.items()} for g in ranked[:cfg.num_elites]]
    parents = ranked[:max(2, int(len(ranked) * cfg.parent_fraction))]

    children = []
    while len(elites) + len(children) < cfg.population_size:
        if cfg.crossover:
            p1, p2 = random.sample(parents, 2)
            child = {k: torch.where(torch.rand_like(p1[k]) < 0.5, p1[k], p2[k]) for k in p1}
        else:
            child = {k: v.clone() for k, v in random.choice(parents).items()}
        for k in child:
            if cfg.weight_decay > 0:
                child[k] = base[k] + (1 - cfg.weight_decay) * (child[k] - base[k])
            child[k] += torch.randn_like(child[k]) * cfg.mutation_std
        children.append(child)

    return elites + children


def mean_pairwise_distance(population: list[Genome], max_pairs: int = 50) -> float:
    """Average L2 distance between sampled pairs of genomes (genetic diversity)."""
    if len(population) < 2:
        return 0.0
    pairs = [random.sample(population, 2) for _ in range(max_pairs)]
    return float(np.mean([genome_distance(a, b) for a, b in pairs]))
