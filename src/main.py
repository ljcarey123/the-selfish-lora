import argparse
import json
import os
import torch
from Simulation.simulation import Simulation
from visualization.visualizer import Visualizer


def main(num_generations: int = 25, fixed_environment: bool = False, population_size: int = 10, initial_noise: float = 0.01, sparse_mutation_rate: float = 0.1, weight_regularization: float = 0.001):
    """
    Main entry point for running the evolutionary simulation.
    
    Args:
        num_generations: Number of generations to run
        fixed_environment: If True, use same environment and elite positions each generation
        population_size: Number of agents in the population
        initial_noise: Noise added to fine-tuned weights at initialization (default: 0.01 = 1%)
        sparse_mutation_rate: Fraction of weights selected for Gaussian mutation (default: 0.1 = 10%)
        weight_regularization: Pull weights back toward init (default: 0.001 = 0.1%)
    """
    print("--- Selfish LoRA Simulation ---")
    print(f"Running for {num_generations} generations with population size {population_size}...")
    if fixed_environment:
        print("Using FIXED environment (same resources and elite positions each generation)")
    else:
        print("Using RANDOM environments (different resources each generation)")

    # Calculate number of elites (15% of population, minimum 2)
    num_elites = max(2, int(population_size * 0.15))

    # Initialize the simulation
    sim = Simulation(
        population_size=population_size,
        grid_size=30,  # Manageable grid size
        turns_per_generation=50,  # Reduced from 100 (faster with +5 scoring)
        mutation_strength=0.005,  # 0.5% mutation (balanced: preserves fine-tuning but allows exploration)
        num_elites=num_elites,
        fixed_environment=fixed_environment,
        initial_noise=initial_noise,
        sparse_mutation_rate=sparse_mutation_rate,
        weight_regularization=weight_regularization,
    )

    champion_paths_by_gen = {}

    # This block is moved up to handle periodic visualization inside the loop
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(num_generations):
        print(f"--- Running Generation {i + 1}/{num_generations} ---")
        sim.run_generation()

        # Get the latest performance metrics for the completed generation
        latest_metrics = sim.performance_history[-1]

        # Store the paths of the top champions for this generation
        # Get min(3, population_size) to handle small populations
        n_champions = min(3, population_size)
        top_champions = sim.get_top_agents(n=n_champions)
        champion_paths_by_gen[i] = [agent.path_history for agent in top_champions]

        # Get action distribution for this generation
        action_counts = sim.behavioral_history[-1] if sim.behavioral_history else {}
        total_actions = sum(action_counts.values()) if action_counts else 1
        action_pcts = {k: (v/total_actions)*100 for k, v in action_counts.items()}
        
        print(
            f"Generation {i + 1}/{num_generations} | "
            f"Max Fitness: {latest_metrics['max_fitness']:.2f} | "
            f"Avg Fitness: {latest_metrics['average_fitness']:.2f} | "
            f"Genetic Diversity: {latest_metrics['genetic_diversity']:.2f}"
        )
        print(f"  Population Actions: ", end="")
        for action in ['North', 'South', 'East', 'West', 'Gather', 'UNPARSEABLE']:
            if action in action_pcts:
                print(f"{action}={action_pcts[action]:.1f}% ", end="")
        print()  # Newline
        
        # Show action distribution for each elite
        for elite_idx, elite in enumerate(top_champions):
            elite_action_counts = {}
            for action in elite.action_history:
                elite_action_counts[action] = elite_action_counts.get(action, 0) + 1
            elite_total = sum(elite_action_counts.values()) if elite_action_counts else 1
            elite_action_pcts = {k: (v/elite_total)*100 for k, v in elite_action_counts.items()}
            
            print(f"    Elite {elite_idx+1} (fitness={elite.fitness_score:.1f}): ", end="")
            for action in ['North', 'South', 'East', 'West', 'Gather', 'UNPARSEABLE']:
                if action in elite_action_pcts:
                    print(f"{action}={elite_action_pcts[action]:.1f}% ", end="")
            print()  # Newline

        # Every 5 generations (but not the final one), save a snapshot and create the champion GIF
        if (i + 1) % 5 == 0 and (i + 1) < num_generations:
            n_to_show = min(3, len(champion_paths_by_gen[i]))
            print(f"\n--- Visualizing Top {n_to_show} Paths for Generation {i} ---")
            # Create a temporary data structure for the visualizer
            temp_data = {
                "champion_paths_by_gen": {str(i): champion_paths_by_gen[i]},
                "environment_history": {str(i): sim.environment_history[i]},
            }
            # The visualizer can be initialized with data directly
            temp_visualizer = Visualizer(data=temp_data, grid_size=sim.grid_size)
            temp_visualizer.animate_top_n_champion_paths(generation=i, n=n_to_show)

        # Evolve the population to create the next generation
        sim.evolve()

    # Collect paths from the final generation
    final_population_paths = [agent.path_history for agent in sim.population]

    print("--- Simulation Complete ---")
    print(f"Collected champion paths for {len(champion_paths_by_gen)} generations.")
    print(f"Collected {len(final_population_paths)} paths from the final generation.")

    # --- Save Top 3 Champion LoRA Weights ---
    print("\n--- Saving Top 3 Champion LoRA Weights ---")
    weights_dir = os.path.join(output_dir, "weights")
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    top_champions = sim.get_top_agents(n=3)
    for i, champion in enumerate(top_champions):
        champion_weights = champion.brain.get_weights()
        weights_path = os.path.join(weights_dir, f"champion_{i+1}_lora_weights.pth")
        torch.save(champion_weights, weights_path)
        print(f"Champion {i+1}'s LoRA weights saved to {weights_path}")

    # --- Save Full Simulation Data ---
    # Convert champion_paths_by_gen keys to strings for JSON compatibility
    champion_paths_str_keys = {str(k): v for k, v in champion_paths_by_gen.items()}
    
    simulation_data = {
        "performance_history": sim.performance_history,
        "genetic_history": sim.genetic_history,
        "behavioral_history": sim.behavioral_history,
        "environment_history": sim.environment_history,
        "initial_environment_history": sim.initial_environment_history,
        "champion_paths_by_gen": champion_paths_str_keys,
        "final_population_paths": final_population_paths,
    }

    output_path = os.path.join(output_dir, "simulation_paths.json")
    with open(output_path, "w") as f:
        json.dump(simulation_data, f, indent=4)
    print(f"Full simulation data saved to {output_path}")

    # --- Run Final Visualizations ---
    print("\n--- Running Final Visualizations ---")
    # Initialize the visualizer with the complete data file
    visualizer = Visualizer(data_path=output_path, grid_size=sim.grid_size)

    # Animate the top 3 champion paths for the final generation
    print(f"\nVisualizing Top 3 Paths for Final Generation ({num_generations - 1})...")
    visualizer.animate_top_n_champion_paths(generation=num_generations - 1, n=3)

    # Plot and save the performance metrics graph
    print("\nPlotting performance metrics...")
    visualizer.plot_performance_metrics()

    # Plot and save the genetic diversity graph
    print("\nPlotting genetic diversity...")
    visualizer.plot_genetic_diversity()

    # Plot and save the behavioral metrics graph
    print("\nPlotting behavioral metrics...")
    visualizer.plot_behavioral_metrics()

    # Plot agent movement heatmap for final generation
    print(f"\nGenerating agent movement heatmap for final generation ({num_generations - 1})...")
    visualizer.plot_agent_heatmap(generation=num_generations - 1)

    print("\n--- All tasks complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Selfish LoRA simulation.")
    parser.add_argument(
        "--num_generations",
        type=int,
        default=20,  # Increased from 5 for meaningful evolution
        help="The number of generations to run the simulation for.",
    )
    parser.add_argument(
        "--fixed-environment",
        action="store_true",
        help="Use the same environment layout and elite positions each generation (easier for initial testing).",
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=10,  # Reduced from 20 for faster testing
        help="Number of agents in the population (default: 10 for quick tests, use 20-50 for serious runs).",
    )
    parser.add_argument(
        "--initial-noise",
        type=float,
        default=0.01,  # 1% noise
        help="Noise added to fine-tuned weights at initialization (default: 0.01 = 1%%).",
    )
    parser.add_argument(
        "--sparse-mutation-rate",
        type=float,
        default=0.1,  # 10% of weights selected
        help="Fraction of weights selected for Gaussian-weighted mutation (default: 0.1 = 10%%).",
    )
    parser.add_argument(
        "--weight-regularization",
        type=float,
        default=0.001,  # 0.1% pull back
        help="Pull weights back toward initial fine-tuned values (default: 0.001 = 0.1%%). Prevents genetic drift.",
    )
    args = parser.parse_args()

    main(
        num_generations=args.num_generations,
        fixed_environment=args.fixed_environment,
        population_size=args.population_size,
        initial_noise=args.initial_noise,
        sparse_mutation_rate=args.sparse_mutation_rate,
        weight_regularization=args.weight_regularization
    )
