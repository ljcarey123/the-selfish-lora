import argparse
import json
import os
import torch
from Simulation.simulation import Simulation
from visualisation.visualiser import Visualiser
from visualisation.energy_visualiser import EnergyVisualiser


def main(num_generations: int = 25, fixed_environment: bool = False, population_size: int = 10, initial_noise: float = 0.01, sparse_mutation_rate: float = 0.1, weight_regularization: float = 0.001, random_init: bool = False, tournament_mode: bool = False, num_tournaments: int = 3, tournament_size: int = None, all_compete: bool = True, resource_regen_turns: int = 10):
    """
    Main entry point for running the evolutionary simulation.

    Args:
        num_generations: Number of generations to run
        fixed_environment: If True, use same environment and elite positions each generation
        population_size: Number of agents in the population
        initial_noise: Noise added to fine-tuned weights at initialization (default: 0.01 = 1%)
        sparse_mutation_rate: Fraction of weights selected for Gaussian mutation (default: 0.1 = 10%)
        weight_regularization: Pull weights back toward init (default: 0.001 = 0.1%)
        random_init: If True, use random initialization instead of loading fine-tuned weights
        tournament_mode: If True, use multi-agent competition for fitness evaluation
        num_tournaments: Number of tournaments to run per generation (fitness averaged)
        tournament_size: Number of agents in each tournament (None = all agents)
        all_compete: If True, all agents compete in each tournament; if False, random subsets
        resource_regen_turns: Number of turns before depleted resource regenerates (0 = disabled)
    """
    print("--- Selfish LoRA Simulation ---")
    print(f"Running for {num_generations} generations with population size {population_size}...")
    if random_init:
        print("Using RANDOM initialization (no fine-tuned weights)")
    else:
        print("Using FINE-TUNED initialization (loading weights from src/weights/)")

    if tournament_mode:
        print(f"Using TOURNAMENT MODE:")
        print(f"  - {num_tournaments} tournaments per generation (fitness averaged)")
        if all_compete:
            print(f"  - All {population_size} agents compete in each tournament")
        else:
            actual_size = tournament_size if tournament_size else population_size
            print(f"  - {actual_size} randomly selected agents per tournament")
        print(f"  - Multi-agent competition in shared environment")
        print(f"  - Truncation selection for parent selection (top 50%)")
    else:
        print("Using STANDARD MODE:")
        print("  - Isolated agents (no competition)")
        print("  - Truncation selection for parent selection (top 50%)")

    if fixed_environment:
        print("Using FIXED environment (same resources and elite positions each generation)")
    else:
        print("Using RANDOM environments (different resources each generation)")

    if resource_regen_turns > 0:
        print(f"Resource regeneration: ENABLED (regenerates after {resource_regen_turns} turns)")
    else:
        print("Resource regeneration: DISABLED")

    # Calculate number of elites (15% of population, minimum 2)
    num_elites = max(2, int(population_size * 0.15))

    # Initialize the simulation
    sim = Simulation(
        population_size=population_size,
        grid_size=30,  # Manageable grid size
        turns_per_generation=50,  # 50 steps per tournament
        mutation_strength=0.005,  # 0.5% mutation (balanced: preserves fine-tuning but allows exploration)
        num_elites=num_elites,
        fixed_environment=fixed_environment,
        initial_noise=initial_noise,
        sparse_mutation_rate=sparse_mutation_rate,
        weight_regularization=weight_regularization,
        random_init=random_init,
        tournament_mode=tournament_mode,
        num_tournaments=num_tournaments,
        tournament_size=tournament_size,
        all_compete=all_compete,
        resource_regen_turns=resource_regen_turns,
    )

    champion_paths_by_gen = {}

    # This block is moved up to handle periodic visualization inside the loop
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(num_generations):
        # Run the generation (logging is now inside run_generation/run_tournament_generation)
        if tournament_mode:
            sim.run_tournament_generation()
        else:
            sim.run_generation()

        # Get the latest performance metrics for the completed generation
        latest_metrics = sim.performance_history[-1]

        # Store the paths of the top champions for this generation
        # Get min(3, population_size) to handle small populations
        n_champions = min(3, population_size)
        top_champions = sim.get_top_agents(n=n_champions)
        champion_paths_by_gen[i] = [agent.path_history for agent in top_champions]

        # Get action distribution for this generation
        action_counts = sim.behavioural_history[-1] if sim.behavioural_history else {}
        total_actions = sum(action_counts.values()) if action_counts else 1
        action_pcts = {k: (v/total_actions)*100 for k, v in action_counts.items()}

        # Summary statistics
        print(f"\n--- Generation {i + 1} Summary ---")
        print(f"Genetic Diversity: {latest_metrics.get('genetic_diversity', 0):.2f}")
        print(f"Population Actions: ", end="")
        for action in ['move_north', 'move_south', 'move_east', 'move_west', 'gather', 'stand_still', 'UNPARSEABLE']:
            if action in action_pcts:
                print(f"{action}={action_pcts[action]:.1f}% ", end="")
        print()  # Newline

        # Show action distribution for top champions
        print(f"Top {len(top_champions)} Champions:")
        for elite_idx, elite in enumerate(top_champions):
            elite_action_counts = {}
            for action in elite.action_history:
                elite_action_counts[action] = elite_action_counts.get(action, 0) + 1
            elite_total = sum(elite_action_counts.values()) if elite_action_counts else 1
            elite_action_pcts = {k: (v/elite_total)*100 for k, v in elite_action_counts.items()}

            print(f"  #{elite_idx+1} (ID: {elite.id}, energy={elite.energy:.1f}): ", end="")
            for action in ['move_north', 'move_south', 'move_east', 'move_west', 'gather', 'stand_still', 'UNPARSEABLE']:
                if action in elite_action_pcts:
                    print(f"{action}={elite_action_pcts[action]:.1f}% ", end="")
            print()  # Newline

        # Every 5 generations (but not the final one), save a snapshot and create the champion GIF
        if (i + 1) % 5 == 0 and (i + 1) < num_generations:
            n_to_show = min(3, len(champion_paths_by_gen[i]))
            print(f"\n--- Visualizing Top {n_to_show} Paths for Generation {i} ---")
            # Create a temporary data structure for the visualiser
            temp_data = {
                "champion_paths_by_gen": {str(i): champion_paths_by_gen[i]},
                "environment_history": {str(i): sim.environment_history[i]},
            }
            # The visualiser can be initialised with data directly
            temp_visualiser = Visualiser(data=temp_data, grid_size=sim.grid_size)
            temp_visualiser.animate_top_n_champion_paths(generation=i, n=n_to_show)

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
        "behavioural_history": sim.behavioural_history,
        "environment_history": sim.environment_history,
        "initial_environment_history": sim.initial_environment_history,
        "champion_paths_by_gen": champion_paths_str_keys,
        "final_population_paths": final_population_paths,
        "tournament_mode": tournament_mode,
        "tournament_history": {str(k): v for k, v in sim.tournament_history.items()} if tournament_mode else {},
        "tournament_energy_variance": sim.tournament_energy_variance if tournament_mode else [],
    }

    output_path = os.path.join(output_dir, "simulation_paths.json")
    with open(output_path, "w") as f:
        json.dump(simulation_data, f, indent=4)
    print(f"Full simulation data saved to {output_path}")

    # --- Run Final Visualisations ---
    print("\n--- Running Final Visualisations ---")

    # Load the complete simulation data
    with open(output_path, 'r') as f:
        simulation_data = json.load(f)

    # Initialise visualisers
    visualiser = Visualiser(data=simulation_data, grid_size=sim.grid_size)
    energy_visualiser = EnergyVisualiser(simulation_data)

    # Generate energy visualisations
    print("\n--- Generating Energy Visualisations ---")
    energy_visualiser.generate_all_plots()

    # Animate the top 3 champion paths for the final generation
    print(f"\nVisualising Top 3 Paths for Final Generation ({num_generations - 1})...")
    visualiser.animate_top_n_champion_paths(generation=num_generations - 1, n=3)

    # Plot and save the genetic diversity graph
    print("\nPlotting genetic diversity...")
    visualiser.plot_genetic_diversity()

    # Plot and save the behavioural metrics graph
    print("\nPlotting behavioural metrics...")
    visualiser.plot_behavioural_metrics()

    # Plot agent movement heatmap for final generation
    print(f"\nGenerating agent movement heatmap for final generation ({num_generations - 1})...")
    visualiser.plot_agent_heatmap(generation=num_generations - 1)

    print("\n--- All tasks complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Selfish LoRA simulation.")
    parser.add_argument(
        "--num-generations",
        type=int,
        default=20,
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
        default=10, 
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
        default=0.2,  # 20% of weights selected
        help="Fraction of weights selected for Gaussian-weighted mutation (default: 0.2 = 20%%).",
    )
    parser.add_argument(
        "--weight-regularization",
        type=float,
        default=0.0005,  # 0.05% pull back (reduced since more weights mutating)
        help="Pull weights back toward initial fine-tuned values (default: 0.0005 = 0.05%%). Prevents genetic drift.",
    )
    parser.add_argument(
        "--random-init",
        action="store_true",
        help="Use random initialization instead of loading fine-tuned weights. Uses very small noise (0.001) to prevent gibberish outputs.",
    )
    parser.add_argument(
        "--tournament",
        action="store_true",
        help="Enable tournament mode (multi-agent competition in shared environment for fitness evaluation)",
    )
    parser.add_argument(
        "--num-tournaments",
        type=int,
        default=3,
        help="Number of tournaments per generation to average fitness over (default: 3)",
    )
    parser.add_argument(
        "--tournament-size",
        type=int,
        default=None,
        help="Number of agents in each tournament (default: None = all agents compete). Only used with --tournament.",
    )
    parser.add_argument(
        "--random-subsets",
        action="store_true",
        help="Use random subsets of agents in each tournament instead of all agents. Requires --tournament-size. Only used with --tournament.",
    )
    parser.add_argument(
        "--resource-regen-turns",
        type=int,
        default=10,
        help="Number of turns before a depleted resource regenerates (default: 10, set to 0 to disable regeneration).",
    )
    args = parser.parse_args()

    # Validate tournament arguments
    if args.random_subsets and not args.tournament:
        print("Warning: --random-subsets requires --tournament flag. Ignoring.")
        args.random_subsets = False

    if args.random_subsets and args.tournament_size is None:
        print("Warning: --random-subsets requires --tournament-size. Using all agents instead.")
        args.random_subsets = False

    main(
        num_generations=args.num_generations,
        fixed_environment=args.fixed_environment,
        population_size=args.population_size,
        initial_noise=args.initial_noise,
        sparse_mutation_rate=args.sparse_mutation_rate,
        weight_regularization=args.weight_regularization,
        random_init=args.random_init,
        tournament_mode=args.tournament,
        num_tournaments=args.num_tournaments,
        tournament_size=args.tournament_size,
        all_compete=not args.random_subsets,  # all_compete = True unless random_subsets is set
        resource_regen_turns=args.resource_regen_turns,
    )
