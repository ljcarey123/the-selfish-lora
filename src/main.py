from Simulation.simulation import Simulation


def main():
    """
    Main entry point for running the evolutionary simulation.
    """
    num_generations = 50

    print("--- Selfish LoRA Simulation ---")
    print(f"Running for {num_generations} generations...")

    # Initialize the simulation
    sim = Simulation(population_size=50, grid_size=50, turns_per_generation=100)

    for gen in range(num_generations):
        # Run the simulation for one generation
        sim.run_generation()

        # Get the average fitness for the completed generation
        avg_fitness = sim.get_average_fitness()
        print(
            f"Generation {gen + 1}/{num_generations} | Average Fitness: {avg_fitness:.2f}"
        )

        # Evolve the population to create the next generation
        sim.evolve()

    print("--- Simulation Complete ---")


if __name__ == "__main__":
    main()
