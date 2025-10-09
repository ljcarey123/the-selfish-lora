import json
import os
from Simulation.simulation import Simulation
from visualization.visualizer import Visualizer


def main():
    """
    Main entry point for running the evolutionary simulation.
    """
    num_generations = 50

    print("--- Selfish LoRA Simulation ---")
    print(f"Running for {num_generations} generations...")

    # Initialize the simulation
    sim = Simulation(population_size=50, grid_size=50, turns_per_generation=100)

    champion_paths_by_gen = {}

    for gen in range(num_generations):
        # Run the simulation for one generation
        sim.run_generation()

        # Get the latest performance metrics for the completed generation
        latest_metrics = sim.performance_history[-1]
        avg_fitness = latest_metrics["average_fitness"]
        max_fitness = latest_metrics["max_fitness"]

        # Store the paths of the top champions for this generation
        top_champions = sim.get_top_agents(n=3)
        champion_paths_by_gen[gen] = [agent.path_history for agent in top_champions]

        print(
            f"Generation {gen + 1}/{num_generations} | "
            f"Average Fitness: {avg_fitness:.2f} | "
            f"Max Fitness: {max_fitness}"
        )

        # Evolve the population to create the next generation
        sim.evolve()

    # Collect paths from the final generation
    final_population_paths = [agent.path_history for agent in sim.population]

    print("--- Simulation Complete ---")
    print(f"Collected champion paths for {len(champion_paths_by_gen)} generations.")
    print(f"Collected {len(final_population_paths)} paths from the final generation.")

    # Save the collected data to a file
    # Set the output directory to be inside the `src` folder
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    simulation_data = {
        "performance_history": sim.performance_history,
        "genetic_history": sim.genetic_history,
        "behavioral_history": sim.behavioral_history,
        "champion_paths_by_gen": champion_paths_by_gen,
        "final_population_paths": final_population_paths,
    }

    output_path = os.path.join(output_dir, "simulation_paths.json")
    with open(output_path, "w") as f:
        json.dump(simulation_data, f, indent=4)

    print(f"Simulation data saved to {output_path}")

    # Run the visualizer for the top champion of the final generation
    print("\n--- Running Visualization ---")
    visualizer = Visualizer(output_path, grid_size=sim.grid_size)
    print(
        f"Visualizing top champion path for final generation ({num_generations - 1})..."
    )
    visualizer.animate_champion_path(generation=num_generations - 1, champion_rank=0)

    # Plot and save the performance metrics graph
    print("\nPlotting performance metrics...")
    visualizer.plot_performance_metrics()

    # Plot and save the genetic diversity graph
    print("\nPlotting genetic diversity...")
    visualizer.plot_genetic_diversity()

    # Plot and save the behavioral metrics graph
    print("\nPlotting behavioral metrics...")
    visualizer.plot_behavioral_metrics()


if __name__ == "__main__":
    main()
