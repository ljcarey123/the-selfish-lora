"""
Main entry point for the vectorized evolutionary simulation.

This script runs the vectorized simulation with:
- VectorizedSimulation orchestrator
- BatchInferenceEngine with per-agent LoRA weights
- VectorizedEnvironment with NumPy operations
- Sequential LLM inference to preserve genetic diversity
"""

import argparse
import json
import os
import torch
from Simulation.simulation_vectorized import VectorizedSimulation


def main(
    num_generations: int = 20,
    population_size: int = 32,
    batch_size: int = 8,
    turns_per_batch: int = 100,
    mutation_strength: float = 0.005,
    num_elites: int = 5,
    initial_noise: float = 0.01,
    sparse_mutation_rate: float = 0.2,
    weight_regularization: float = 0.0005,
    random_init: bool = False,
    rounds_per_generation: int = 2,
):
    """
    Run the vectorized evolutionary simulation.

    Args:
        num_generations: Number of generations to run
        population_size: Total agents per generation (must be divisible by batch_size)
        batch_size: Agents per batch
        turns_per_batch: Max turns per batch
        mutation_strength: Mutation strength
        num_elites: Elite agents to preserve
        initial_noise: Initial weight noise
        sparse_mutation_rate: Fraction of weights to mutate
        weight_regularization: Weight regularization strength
        random_init: Use random initialization
        rounds_per_generation: Number of rounds each agent competes per generation
    """
    print("="*60)
    print("VECTORIZED SELFISH LORA SIMULATION")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Generations: {num_generations}")
    print(f"  Population: {population_size} agents")
    print(f"  Batch size: {batch_size} agents ({population_size // batch_size} batches)")
    print(f"  Rounds per generation: {rounds_per_generation}")
    print(f"  Turns per batch: {turns_per_batch}")
    print(f"  Elites: {num_elites}")
    print(f"  Mutation strength: {mutation_strength}")
    print(f"  Sparse mutation rate: {sparse_mutation_rate}")
    print(f"  Weight regularization: {weight_regularization}")
    print(f"  Initial noise: {initial_noise}")
    print(f"  Initialization: {'RANDOM' if random_init else 'FINE-TUNED WEIGHTS'}")
    print()

    # Validate parameters
    if population_size % batch_size != 0:
        raise ValueError(f"population_size ({population_size}) must be divisible by batch_size ({batch_size})")

    # Initialize vectorized simulation
    sim = VectorizedSimulation(
        population_size=population_size,
        batch_size=batch_size,
        turns_per_batch=turns_per_batch,
        mutation_strength=mutation_strength,
        num_elites=num_elites,
        initial_noise=initial_noise,
        sparse_mutation_rate=sparse_mutation_rate,
        weight_regularization=weight_regularization,
        random_init=random_init,
        rounds_per_generation=rounds_per_generation,
    )

    # Run evolution
    for gen in range(num_generations):
        print(f"\n{'='*60}")
        print(f"GENERATION {gen + 1}/{num_generations}")
        print(f"{'='*60}")

        # Run generation
        sim.run_generation()

        # Show top agents
        print(f"\nTop 3 agents:")
        top3 = sim.get_top_agents(3)
        for idx, agent in enumerate(top3, 1):
            status = "✓ ALIVE" if agent.is_alive else "✗ DEAD"
            print(f"  #{idx}: Agent {agent.id}")
            print(f"       Energy: {agent.energy:.1f}")
            print(f"       Food collected: {agent.food_collected}")
            print(f"       Attacks made: {agent.combat_attacks}")
            print(f"       Attacked (victim): {agent.combat_victims}")
            print(f"       Status: {status}")

        # Evolve population (except on last generation)
        if gen < num_generations - 1:
            sim.evolve_population()

    # Run champion tournament (if multiple generations)
    if num_generations > 1:
        sim.run_champion_tournament()

    # Save final population weights
    print(f"\n{'='*60}")
    print("SAVING FINAL CHAMPION WEIGHTS")
    print(f"{'='*60}")

    sim.save_final_weights()

    # Show final champions
    top_champions = sim.get_top_agents(3)
    for i, champion in enumerate(top_champions, 1):
        print(f"\nChampion {i} (Agent {champion.id}):")
        print(f"  Energy: {champion.energy:.1f} ({champion.fitness_percentage:.1f}%)")
        print(f"  Food collected: {champion.food_collected}")
        print(f"  Attacks made: {champion.combat_attacks}")
        print(f"  Attacked (victim): {champion.combat_victims}")
        print(f"  Status: {'✓ ALIVE' if champion.is_alive else '✗ DEAD'}")

    print(f"\nVisualization data saved to: ./output/simulation_paths.json")
    print(f"Weights saved to: ./output/weights/")

    # Generate visualizations automatically
    print(f"\n{'='*60}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*60}")

    try:
        from visualisation.visualiser import Visualiser

        viz = Visualiser(data_path="./output/simulation_paths.json")

        # Generate all visualizations for the final generation
        final_gen = num_generations - 1

        print(f"\nGenerating heatmap for generation {final_gen}...")
        viz.plot_agent_heatmap(generation=final_gen, save_as_png=True)

        print(f"Generating action timelines for top 3 agents...")
        viz.plot_champion_action_timeline(generation=final_gen, n=3, save_as_png=True)

        print(f"Generating path animations for top 3 agents...")
        viz.animate_top_n_champion_paths(generation=final_gen, n=3, save_as_gif=True)

        if num_generations > 1:
            print(f"\nGenerating genetic diversity plot...")
            viz.plot_genetic_diversity(save_as_png=True)

            print(f"Generating behavioral metrics plot...")
            viz.plot_behavioural_metrics(save_as_png=True)

        print(f"\n✓ All visualizations generated successfully!")
        print(f"  - Heatmaps: ./output/agent_heatmap_gen_*.png")
        print(f"  - Timelines: ./output/action_timeline_gen_*.png")
        print(f"  - Animations: ./output/gifs/champion_paths_gen_*.gif")
        if num_generations > 1:
            print(f"  - Genetic diversity: ./output/genetic_diversity.png")
            print(f"  - Behavioral metrics: ./output/behavioural_metrics.png")

    except Exception as e:
        print(f"\n✗ Error generating visualizations: {e}")
        print(f"  You can manually generate them by running: python -m visualisation.visualiser")

    print(f"\n{'='*60}")
    print("SIMULATION COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the vectorized Selfish LoRA simulation.")
    parser.add_argument(
        "--generations",
        type=int,
        default=20,
        help="Number of generations to run (default: 20)",
    )
    parser.add_argument(
        "--population",
        type=int,
        default=32,
        help="Population size - must be divisible by batch size (default: 32)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for environment processing (default: 8)",
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=100,
        help="Turns per batch (default: 100)",
    )
    parser.add_argument(
        "--elites",
        type=int,
        default=5,
        help="Number of elite agents to preserve (default: 5)",
    )
    parser.add_argument(
        "--mutation",
        type=float,
        default=0.005,
        help="Mutation strength (default: 0.005)",
    )
    parser.add_argument(
        "--sparse-mutation",
        type=float,
        default=0.2,
        help="Sparse mutation rate (default: 0.2)",
    )
    parser.add_argument(
        "--regularization",
        type=float,
        default=0.0005,
        help="Weight regularization (default: 0.0005)",
    )
    parser.add_argument(
        "--initial-noise",
        type=float,
        default=0.01,
        help="Initial noise for weight initialization (default: 0.01)",
    )
    parser.add_argument(
        "--random-init",
        action="store_true",
        help="Use random initialization instead of fine-tuned weights",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=2,
        help="Rounds per generation for tournament-style evaluation (default: 2)",
    )

    args = parser.parse_args()

    main(
        num_generations=args.generations,
        population_size=args.population,
        batch_size=args.batch_size,
        turns_per_batch=args.turns,
        mutation_strength=args.mutation,
        num_elites=args.elites,
        initial_noise=args.initial_noise,
        sparse_mutation_rate=args.sparse_mutation,
        weight_regularization=args.regularization,
        random_init=args.random_init,
        rounds_per_generation=args.rounds,
    )
