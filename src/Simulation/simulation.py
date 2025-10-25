import numpy as np
import random
import os
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, PeftModel
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
        grid_size: int = 30,
        turns_per_generation: int = 100,
        mutation_strength: float = 0.05,
        num_elites: int = 2,
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
        self.mutation_strength = mutation_strength
        self.num_elites = num_elites
        self.current_generation = 0  # Track generation number for seeded environments

        # Load environment variables and get the token
        load_dotenv()
        hf_token = os.getenv("HF_TOKEN")

        # Load the base LLM once for the entire simulation
        print("Loading base LLM (Qwen/Qwen2.5-0.5B-Instruct) and preparing for LoRA...")
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True, token=hf_token
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct", token=hf_token
        )
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Define LoRA configuration (must match fine-tuning config)
        peft_config = LoraConfig(
            r=16,  # Rank of the LoRA matrices (matches fine-tuning)
            lora_alpha=32,  # Scaling factor (matches fine-tuning)
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],  # All modules used in fine-tuning
            lora_dropout=0.05,
            bias="none",
        )

        # Wrap the base model with PEFT
        self.base_model = get_peft_model(model, peft_config)
        print("Base PeftModel and tokenizer loaded.")

        self.population = self._create_initial_population()
        self.performance_history = []
        self.genetic_history = []
        self.behavioral_history = []
        self.environment_history = {}
        self.initial_environment_history = {}

    def _create_initial_population(self) -> list[Agent]:
        """
        Creates the first generation of agents with brains initialized from fine-tuned weights.
        Each agent starts with the fine-tuned weights plus small random noise for diversity.
        """
        # Load fine-tuned weights from src/weights
        weights_path = os.path.join(os.path.dirname(__file__), "..", "weights")
        fine_tuned_weights = None

        if os.path.exists(weights_path):
            try:
                print(f"Loading fine-tuned weights from {weights_path}...")
                # Load the fine-tuned adapter into a temporary model
                temp_model = PeftModel.from_pretrained(self.base_model, weights_path)

                # Extract the LoRA weights
                fine_tuned_weights = {}
                for name, param in temp_model.named_parameters():
                    if "lora_A" in name or "lora_B" in name:
                        fine_tuned_weights[name] = param.data.clone()

                print(f"Loaded {len(fine_tuned_weights)} fine-tuned weight tensors.")
            except Exception as e:
                print(f"Warning: Could not load fine-tuned weights: {e}")
                print("Initializing with random weights instead.")
                fine_tuned_weights = None
        else:
            print(f"Warning: Fine-tuned weights not found at {weights_path}")
            print("Initializing with random weights instead.")

        # Create population
        population = []
        for i in range(self.population_size):
            brain = LoraBrain(self.base_model)

            # If we have fine-tuned weights, initialize with them + noise
            if fine_tuned_weights is not None:
                brain_weights = brain.get_weights()
                for name in brain_weights:
                    if name in fine_tuned_weights:
                        # Start with fine-tuned weights and add small random noise
                        noise = (
                            torch.randn_like(fine_tuned_weights[name]) * 0.01
                        )  # 1% noise (matches mutation strength)
                        brain_weights[name] = fine_tuned_weights[name].clone() + noise
                brain.set_weights(brain_weights)

            agent = Agent(
                brain=brain,
                initial_position=(
                    random.randint(0, self.grid_size - 1),
                    random.randint(0, self.grid_size - 1),
                ),
                grid_size=self.grid_size,
            )
            population.append(agent)

        if fine_tuned_weights is not None:
            print(
                f"Initialized {self.population_size} agents with fine-tuned weights + 1% noise (0.01)"
            )
        else:
            print(f"Initialized {self.population_size} agents with random weights")

        return population

    def run_generation(self):
        """
        Runs a single generation in a turn-based manner.

        In each turn, every agent makes one move. This process is repeated for
        the total number of turns, creating a shared, dynamic environment where
        agents can indirectly interact.
        """
        # Use generation number as seed for reproducible but varied environments
        # Each generation gets a different layout, but reruns are consistent
        np.random.seed(self.current_generation + 42)  # +42 for variety
        random.seed(self.current_generation + 42)
        
        # A fresh environment is created for each generation
        self.environment = Environment(self.grid_size)
        
        # Increment generation counter for next time
        self.current_generation += 1
        # Store the initial state of the environment for visualization
        self.initial_environment_history[len(self.performance_history)] = (
            self.environment.grid.copy().tolist()
        )

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
            # Get the model and tokenizer once per generation
            base_model, tokenizer = self.get_base_model_and_tokenizer()
            for agent in self.population:
                agent.decide_and_act(self.environment, base_model, tokenizer)

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
        elites = self.get_top_agents(n=self.num_elites)
        next_generation = elites[:]  # Make a copy

        # Generate new agents by crossover and mutation
        parents = self.get_top_agents(n=self.population_size // 2)
        while len(next_generation) < self.population_size:
            parent1, parent2 = random.choices(parents, k=2)

            # Create a new brain for the child
            child_brain = LoraBrain(self.base_model)
            child_weights = {}

            # Crossover: For each weight tensor, randomly pick from one parent
            p1_weights = parent1.brain.get_weights()
            p2_weights = parent2.brain.get_weights()
            for name in p1_weights:
                if random.random() < 0.5:
                    child_weights[name] = p1_weights[name].clone()
                else:
                    child_weights[name] = p2_weights[name].clone()

            # Mutation: Apply a small random change to the child's weights
            for name in child_weights:
                mutation = (
                    torch.randn_like(child_weights[name]) * self.mutation_strength
                )
                child_weights[name] += mutation

            child_brain.set_weights(child_weights)

            initial_position = (
                random.randint(0, self.grid_size - 1),
                random.randint(0, self.grid_size - 1),
            )
            next_generation.append(Agent(child_brain, initial_position, self.grid_size))

        self.population = next_generation

    def get_average_fitness(self) -> float:
        """
        Calculates the average fitness score of the current population.
        """
        if not self.population:
            return 0.0
        total_fitness = sum(agent.fitness_score for agent in self.population)
        return total_fitness / len(self.population)

    def get_top_agents(self, n: int = 1) -> list["Agent"]:
        """
        Finds the top n agents with the highest fitness scores in the current population.

        Args:
            n (int): The number of top agents to return.

        Returns:
            list[Agent]: A list of the top n agents.
        """
        return sorted(
            self.population, key=lambda agent: agent.fitness_score, reverse=True
        )[:n]

    def get_base_model_and_tokenizer(self):
        """Returns the shared base LLM and tokenizer."""
        return self.base_model, self.tokenizer

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
        # Ensure we have a population to sample from
        if len(self.population) < 2:
            return 0.0

        for _ in range(sample_size):
            agent1, agent2 = random.sample(self.population, 2)
            weights1 = agent1.brain.get_weights()
            weights2 = agent2.brain.get_weights()

            total_dist = 0
            num_weights = 0
            for name in weights1:
                dist = torch.norm(weights1[name].float() - weights2[name].float())
                total_dist += dist.item()
                num_weights += 1

            if num_weights > 0:
                distances.append(total_dist / num_weights)

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

        # --- Genetic Diversity ---
        genetic_diversity = self._calculate_genetic_diversity()
        metrics["genetic_diversity"] = genetic_diversity
        self.genetic_history.append(genetic_diversity)

        self.performance_history.append(metrics)

        # And finally, calculate and store behavioral metrics
        action_counts = {}
        for agent in self.population:
            for action in agent.action_history:
                action_counts[action] = action_counts.get(action, 0) + 1
        self.behavioral_history.append(action_counts)

        # Store the environment grid for this generation
        self.environment_history[len(self.performance_history) - 1] = (
            self.environment.grid.tolist()
        )
