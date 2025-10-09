import numpy as np
import random
from Environment.environment import Environment
from Agent.agent import Agent
from LoraBrain.lora_brain import LoraBrain


class Simulation:
    """
    Orchestrates the evolutionary simulation.

    This class manages the population of agents, the environment, and the
    main evolutionary loop: running generations, evaluating fitness, and
    evolving the population through selection, reproduction, and mutation.
    """

    def __init__(
        self,
        population_size: int = 50,
        grid_size: int = 50,
        turns_per_generation: int = 100,
    ):
        """
        Initializes the simulation.

        Args:
            population_size (int): The number of agents in the population.
            grid_size (int): The size of the environment grid.
            turns_per_generation (int): The number of turns each agent gets per generation.
        """
        self.population_size = population_size
        self.grid_size = grid_size
        self.turns_per_generation = turns_per_generation
        self.population = self._create_initial_population()
        self.performance_history = []
        self.genetic_history = []
        self.behavioral_history = []

    def _create_initial_population(self) -> list[Agent]:
        """
        Creates the first generation of agents with random brains.
        """
        population = []
        for _ in range(self.population_size):
            brain = LoraBrain()  # Create a brain with random initial weights
            initial_position = (
                random.randint(0, self.grid_size - 1),
                random.randint(0, self.grid_size - 1),
            )
            agent = Agent(brain, initial_position, self.grid_size)
            population.append(agent)
        return population

    def run_generation(self):
        """
        Runs a single generation in a turn-based manner.

        In each turn, every agent makes one move. This process is repeated for
        the total number of turns, creating a shared, dynamic environment where
        agents can indirectly interact.
        """
        # A fresh environment is created for each generation to ensure a fair test
        self.environment = Environment(self.grid_size)

        # Reset agents' fitness and position for the new generation
        for agent in self.population:
            agent.fitness_score = 0
            agent.position = (
                random.randint(0, self.grid_size - 1),
                random.randint(0, self.grid_size - 1),
            )
            agent.path_history = [("START", agent.position)]
            agent.action_history = []

        # Run the turn-based simulation
        for _ in range(self.turns_per_generation):
            for agent in self.population:
                agent.decide_and_act(self.environment)

        # After the generation is complete, calculate and store performance metrics
        self._calculate_and_store_metrics()

    def evolve(self):
        """
        Evolves the population for the next generation.

        This involves three steps:
        1. Selection: The best-performing agents are selected to be parents.
        2. Reproduction: New child agents are created from pairs of parents.
        3. Mutation: Small random changes are introduced into the children's brains.
        """
        # 1. Selection (Truncation Selection)
        self.population.sort(key=lambda agent: agent.fitness_score, reverse=True)
        num_survivors = self.population_size // 2  # Keep the top 50%
        parents = self.population[:num_survivors]

        # 2. Reproduction & 3. Mutation
        next_generation = []
        while len(next_generation) < self.population_size:
            parent1, parent2 = random.choices(parents, k=2)

            # Crossover: Average the weights of the parents' brains
            child_weights = (
                parent1.brain.get_weights() + parent2.brain.get_weights()
            ) / 2.0

            # Mutation: Add a small amount of random noise
            mutation_strength = 0.05  # Controls the magnitude of mutation
            mutation = np.random.randn(*child_weights.shape) * mutation_strength
            child_weights += mutation

            # Create the new child agent
            child_brain = LoraBrain(weights=child_weights)
            initial_position = (
                random.randint(0, self.grid_size - 1),
                random.randint(0, self.grid_size - 1),
            )
            child = Agent(child_brain, initial_position, self.grid_size)
            next_generation.append(child)

        self.population = next_generation

    def get_average_fitness(self) -> float:
        """
        Calculates the average fitness score of the current population.
        """
        if not self.population:
            return 0.0
        total_fitness = sum(agent.fitness_score for agent in self.population)
        return total_fitness / len(self.population)

    def get_top_agents(self, n: int = 1) -> list['Agent']:
        """
        Finds the top n agents with the highest fitness scores in the current population.

        Args:
            n (int): The number of top agents to return.

        Returns:
            list[Agent]: A list of the top n agents.
        """
        return sorted(self.population, key=lambda agent: agent.fitness_score, reverse=True)[:n]

    def _calculate_genetic_diversity(self, sample_size=100) -> float:
        """
        Calculates the genetic diversity of the population.

        This is done by taking a sample of agent pairs and calculating the
        average Euclidean distance between their brain weights.

        Args:
            sample_size (int): The number of pairs to sample.

        Returns:
            float: The average genetic distance.
        """
        if len(self.population) < 2:
            return 0.0

        distances = []
        for _ in range(sample_size):
            agent1, agent2 = random.sample(self.population, 2)
            dist = np.linalg.norm(agent1.brain.get_weights() - agent2.brain.get_weights())
            distances.append(dist)
        
        return float(np.mean(distances))

    def _calculate_and_store_metrics(self):
        """
        Calculates performance metrics for the current population and stores them.
        """
        if not self.population:
            return

        fitness_scores = [agent.fitness_score for agent in self.population]

        metrics = {
            "average_fitness": float(np.mean(fitness_scores)),
            "max_fitness": int(np.max(fitness_scores)),
            "min_fitness": int(np.min(fitness_scores)),
            "median_fitness": float(np.median(fitness_scores)),
            "std_dev_fitness": float(np.std(fitness_scores)),
        }
        self.performance_history.append(metrics)

        # Also calculate and store genetic diversity
        genetic_diversity = self._calculate_genetic_diversity()
        self.genetic_history.append(genetic_diversity)

        # And finally, calculate and store behavioral metrics
        action_counts = {}
        for agent in self.population:
            for action in agent.action_history:
                action_counts[action] = action_counts.get(action, 0) + 1
        self.behavioral_history.append(action_counts)
