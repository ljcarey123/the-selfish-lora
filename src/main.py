import argparse
import json
import os
import torch
from Simulation.simulation import Simulation
from visualization.visualizer import Visualizer


def main(num_generations: int = 25):
    """
    Main entry point for running the evolutionary simulation.
    """
    print("--- Selfish LoRA Simulation ---")
    print(f"Running for {num_generations} generations...")

    # Initialize the simulation
    sim = Simulation(
        population_size=20,  # Sufficient diversity for evolution (was 5 - too small!)
        grid_size=30,  # Manageable grid size
        turns_per_generation=50,  # Reduced from 100 (faster with +5 scoring)
        mutation_strength=0.01,  # 1% mutation for exploration
        num_elites=3,  # Top 15% survive (3/20)
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
        top_champions = sim.get_top_agents(n=3)
        champion_paths_by_gen[i] = [agent.path_history for agent in top_champions]

        print(
            f"Generation {i + 1}/{num_generations} | "
            f"Max Fitness: {latest_metrics['max_fitness']:.2f} | "
            f"Avg Fitness: {latest_metrics['average_fitness']:.2f} | "
            f"Genetic Diversity: {latest_metrics['genetic_diversity']:.2f}"
        )

        # Every 5 generations (but not the final one), save a snapshot and create the champion GIF
        if (i + 1) % 5 == 0 and (i + 1) < num_generations:
            print(f"\n--- Visualizing Top 3 Paths for Generation {i} ---")
            # Create a temporary data structure for the visualizer
            temp_data = {
                "champion_paths_by_gen": {i: champion_paths_by_gen[i]},
                "environment_history": {i: sim.environment_history[i]},
            }
            # The visualizer can be initialized with data directly
            temp_visualizer = Visualizer(data=temp_data, grid_size=sim.grid_size)
            temp_visualizer.animate_top_n_champion_paths(generation=i, n=3)

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
    simulation_data = {
        "performance_history": sim.performance_history,
        "genetic_history": sim.genetic_history,
        "behavioral_history": sim.behavioral_history,
        "environment_history": sim.environment_history,
        "initial_environment_history": sim.initial_environment_history,
        "champion_paths_by_gen": champion_paths_by_gen,
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
    args = parser.parse_args()

    main(num_generations=args.num_generations)
