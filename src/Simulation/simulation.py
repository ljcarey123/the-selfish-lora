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
from config_loader import GRID_SYMBOLS


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
        fixed_environment: bool = False,
        initial_noise: float = 0.01,
        sparse_mutation_rate: float = 0.1,
        weight_regularization: float = 0.001,
        random_init: bool = False,
        tournament_mode: bool = False,
        num_tournaments: int = 3,
        tournament_size: int = None,  # None = all agents compete
        all_compete: bool = True,  # True = all agents in each tournament, False = random subsets
        resource_regen_turns: int = 10,  # Number of turns before depleted resource regenerates
    ):
        """
        Initializes the simulation.

        Args:
            population_size (int): The number of agents in the population.
            grid_size (int): The size of the environment grid.
            turns_per_generation (int): The number of turns each agent gets per generation.
            mutation_strength (float): Strength of mutations during evolution.
            num_elites (int): Number of top agents to preserve unchanged (elitism).
            fixed_environment (bool): Whether to reuse the same environment each generation.
            initial_noise (float): Noise added to fine-tuned weights at initialization.
            sparse_mutation_rate (float): Fraction of weights selected for mutation.
            weight_regularization (float): Pull weights back toward initial values.
            random_init (bool): Use random initialization instead of fine-tuned weights.
            tournament_mode (bool): Use multi-agent competition for fitness evaluation.
            num_tournaments (int): Number of tournaments per generation (fitness averaged).
            tournament_size (int): Number of agents in each tournament (None = all agents).
            all_compete (bool): True = all agents in each tournament, False = random subsets.
            resource_regen_turns (int): Number of turns before depleted resource regenerates (0 = disabled).
        """
        self.population_size = population_size
        self.grid_size = grid_size
        self.turns_per_generation = turns_per_generation
        self.mutation_strength = mutation_strength
        self.num_elites = num_elites
        self.current_generation = 0  # Track generation number for seeded environments
        self.fixed_environment = fixed_environment
        self.fixed_env_grid = None  # Store fixed environment grid
        self.elite_positions = {}  # Store elite positions by agent ID
        self.initial_noise = initial_noise  # Noise added to fine-tuned weights at initialization
        self.sparse_mutation_rate = sparse_mutation_rate  # Fraction of weights to select for mutation
        self.weight_regularization = weight_regularization  # Pull weights back toward init
        self.initial_weights = None  # Store initial fine-tuned weights for regularization
        self.random_init = random_init  # Whether to use random initialization instead of fine-tuned weights
        self.tournament_mode = tournament_mode  # Whether to use tournament mode for fitness evaluation
        self.num_tournaments = num_tournaments  # Number of tournaments per generation
        # Tournament size: None means all agents, otherwise a specific number
        self.tournament_size = tournament_size if tournament_size is not None else population_size
        self.all_compete = all_compete  # Whether all agents compete in each tournament
        self.resource_regen_turns = resource_regen_turns  # Number of turns before depleted resource regenerates

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
        self.behavioural_history = []
        self.environment_history = {}
        self.initial_environment_history = {}

        # Tournament-specific history (only populated in tournament mode)
        self.tournament_history = {}  # {gen_idx: [{env, paths, energies}, ...] for each tournament}
        self.tournament_energy_variance = []  # Variance metrics per generation

    def _create_initial_population(self) -> list[Agent]:
        """
        Creates the first generation of agents with brains initialized from fine-tuned weights.
        Supports multiple parent models for increased genetic diversity.
        Each agent starts with weights from one of the parent models plus small random noise.

        If random_init=True, skips loading fine-tuned weights and uses small random initialization.
        """
        parent_weights_list = []  # List of different parent model weights

        # Skip loading weights if random initialization is requested
        if self.random_init:
            print("Skipping fine-tuned weight loading (random initialization mode)")
        else:
            # Load fine-tuned weights from src/weights
            weights_path = os.path.join(os.path.dirname(__file__), "..", "weights")

            if os.path.exists(weights_path):
                # Check if weights_path is a directory with multiple run subdirectories
                if os.path.isdir(weights_path):
                    # Look for run-* subdirectories (from hyperparameter search)
                    run_dirs = [d for d in os.listdir(weights_path) if d.startswith("run-") and os.path.isdir(os.path.join(weights_path, d))]

                    if run_dirs:
                        print(f"Found {len(run_dirs)} fine-tuned runs. Loading multiple parent models...")
                        for run_dir in sorted(run_dirs):
                            run_path = os.path.join(weights_path, run_dir)
                            try:
                                # Load the fine-tuned adapter
                                temp_model = PeftModel.from_pretrained(self.base_model, run_path)

                                # Extract the LoRA weights
                                weights = {}
                                for name, param in temp_model.named_parameters():
                                    if "lora_A" in name or "lora_B" in name:
                                        weights[name] = param.data.clone()

                                if weights:
                                    parent_weights_list.append({"name": run_dir, "weights": weights})
                                    print(f"  Loaded {run_dir}: {len(weights)} weight tensors")
                            except Exception as e:
                                print(f"  Warning: Could not load {run_dir}: {e}")
                    else:
                        # Single model in weights directory
                        try:
                            print(f"Loading single fine-tuned model from {weights_path}...")
                            temp_model = PeftModel.from_pretrained(self.base_model, weights_path)
                            weights = {}
                            for name, param in temp_model.named_parameters():
                                if "lora_A" in name or "lora_B" in name:
                                    weights[name] = param.data.clone()
                            if weights:
                                parent_weights_list.append({"name": "default", "weights": weights})
                                print(f"Loaded {len(weights)} fine-tuned weight tensors.")
                        except Exception as e:
                            print(f"Warning: Could not load fine-tuned weights: {e}")
        
        if not parent_weights_list:
            if not self.random_init:
                print(f"Warning: No fine-tuned weights found at {weights_path}")
            print("Initializing with random weights (very small noise to prevent gibberish)")
        else:
            # Store first parent's weights for regularization (or average of all parents)
            self.initial_weights = {k: v.clone() for k, v in parent_weights_list[0]["weights"].items()}
            print(f"Using {len(parent_weights_list)} parent model(s) for initialization")

        # Create population
        population = []
        for i in range(self.population_size):
            brain = LoraBrain(self.base_model)

            # If we have fine-tuned weights, initialize with them + noise
            if parent_weights_list and not self.random_init:
                # Cycle through parent models to distribute diversity
                parent = parent_weights_list[i % len(parent_weights_list)]
                parent_weights = parent["weights"]
                
                # Variable noise strategy: some agents very similar, some more different
                # Use exponential distribution: most agents close to parent, few far away
                # Range: 0.1x to 3x the base initial_noise
                noise_multiplier = np.random.exponential(scale=1.0)
                noise_multiplier = np.clip(noise_multiplier, 0.1, 3.0)
                agent_noise = self.initial_noise * noise_multiplier
                
                brain_weights = brain.get_weights()
                for name in brain_weights:
                    if name in parent_weights:
                        # Apply variable noise
                        noise = torch.randn_like(parent_weights[name]) * agent_noise
                        brain_weights[name] = parent_weights[name].clone() + noise
                brain.set_weights(brain_weights)
            else:
                # Random initialization with very small noise (0.001 = 0.1%)
                # This prevents complete gibberish while allowing evolution to shape behavior
                brain_weights = brain.get_weights()
                random_noise_scale = 0.001
                for name in brain_weights:
                    noise = torch.randn_like(brain_weights[name]) * random_noise_scale
                    brain_weights[name] = brain_weights[name] + noise
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

        if parent_weights_list and not self.random_init:
            if len(parent_weights_list) > 1:
                print(f"Initialized {self.population_size} agents from {len(parent_weights_list)} parent models")
                print(f"  Variable noise: {self.initial_noise*0.1*100:.2f}% to {self.initial_noise*3.0*100:.1f}% (exponential distribution)")
                # Show distribution
                for j, parent in enumerate(parent_weights_list):
                    count = sum(1 for i in range(self.population_size) if i % len(parent_weights_list) == j)
                    print(f"  {parent['name']}: {count} agents")
            else:
                print(f"Initialized {self.population_size} agents with fine-tuned weights")
                print(f"  Variable noise: {self.initial_noise*0.1*100:.2f}% to {self.initial_noise*3.0*100:.1f}% (exponential distribution)")
        else:
            print(f"Initialized {self.population_size} agents with random weights (0.1% noise)")

        return population

    def run_generation(self):
        """
        Runs a single generation in a turn-based manner.

        In each turn, every agent makes one move. This process is repeated for
        the total number of turns, creating a shared, dynamic environment where
        agents can indirectly interact.
        """
        print(f"\n{'='*50}")
        print(f"GENERATION {self.current_generation + 1} - Standard Mode")
        print(f"{'='*50}")

        # Use generation number as seed for reproducible but varied environments
        # Each generation gets a different layout, but reruns are consistent
        np.random.seed(self.current_generation + 42)  # +42 for variety
        random.seed(self.current_generation + 42)

        # Create or reuse environment based on fixed_environment flag
        if self.fixed_environment and self.fixed_env_grid is not None:
            # Reuse the same environment layout
            self.environment = Environment(self.grid_size, self.resource_regen_turns)
            self.environment.grid = self.fixed_env_grid.copy()
            print("Environment: FIXED (reusing saved layout)")
        else:
            # Create a fresh environment
            self.environment = Environment(self.grid_size, self.resource_regen_turns)
            if self.fixed_environment and self.fixed_env_grid is None:
                # First generation with fixed environment - save it
                self.fixed_env_grid = self.environment.grid.copy()
                print("Environment: FIXED (saving new layout)")
            else:
                print("Environment: RANDOM (new layout)")

        # Count resources
        resource_count = np.sum(self.environment.grid == GRID_SYMBOLS["resource"])
        print(f"Resources: {resource_count} available")

        # Increment generation counter for next time
        self.current_generation += 1

        # Store the initial state of the environment for visualization
        self.initial_environment_history[len(self.performance_history)] = (
            self.environment.grid.copy().tolist()
        )

        print(f"\nResetting {self.population_size} agents...")
        # Reset agents for the new generation
        for agent in self.population:
            # Use fixed position for elites if in fixed environment mode
            if self.fixed_environment and agent.id in self.elite_positions:
                agent.reset(self.elite_positions[agent.id])
            else:
                new_position = (
                    random.randint(0, self.grid_size - 1),
                    random.randint(0, self.grid_size - 1)
                )
                agent.reset(new_position)

        # Run the turn-based simulation
        print(f"Running {self.turns_per_generation} turns with {self.population_size} agents...\n")
        for _ in range(self.turns_per_generation):
            # Get the model and tokenizer once per generation
            base_model, tokenizer = self.get_base_model_and_tokenizer()
            for agent in self.population:
                agent.decide_and_act(self.environment, base_model, tokenizer)
            # Advance environment for resource regeneration
            self.environment.step()

        # After the generation is complete, check for dead agents and store metrics
        # No need to set fitness_score as we use energy directly
        print("\n--- Generation Complete ---")
        metrics = self._calculate_and_store_metrics()

        # Log final statistics
        living = metrics['living_agents']
        dead = metrics['energy_metrics']['dead_count']
        print(f"Final Status: {living} alive, {dead} dead")
        print(f"Energy: max={metrics['energy_metrics']['max']:.2f}, avg={metrics['energy_metrics']['avg']:.2f}, min={metrics['energy_metrics']['min']:.2f}")

        # Save elite positions for next generation (if using fixed environment)
        if self.fixed_environment:
            # Sort to get elites
            sorted_pop = sorted(self.population, key=lambda agent: agent.energy, reverse=True)
            for i in range(min(self.num_elites, len(sorted_pop))):
                elite = sorted_pop[i]
                self.elite_positions[elite.id] = elite.position
                print(f"  Elite {i+1} (ID: {elite.id}) position saved: {elite.position}")

    def run_tournament_generation(self):
        """
        Runs a generation using tournament mode with multi-agent competition.

        Key differences from standard mode:
        - Agents compete in shared environments for fitness evaluation
        - Turn order is randomized each timestep
        - Multiple tournaments run, energy is averaged across tournaments each agent participates in
        - Tournament composition controlled by tournament_size and all_compete
        - Uses truncation selection for parent selection (not tournament selection)

        Tournament Ranking:
        - Within each tournament, agents compete for resources in a shared environment
        - Agents that gather more resources accumulate more energy
        - Final fitness (energy) is averaged across all tournaments an agent participates in
        - Agents with higher average energy are ranked higher for selection
        """
        print(f"\n{'='*50}")
        print(f"GENERATION {self.current_generation + 1} - Tournament Mode")
        print(f"{'='*50}")
        print(f"Running {self.num_tournaments} tournaments...")
        if self.all_compete:
            print(f"Competition: All {self.population_size} agents compete in each tournament")
        else:
            print(f"Competition: {self.tournament_size} randomly selected agents per tournament")

        # Initialize cumulative tracking for all agents
        for agent in self.population:
            agent.cumulative_energy = 0.0
            agent.tournament_count = 0
            agent.tournament_energies = []  # Track energy from each tournament for variance

        # Store tournament data for visualisation
        gen_tournament_data = []

        # Run multiple tournaments
        for tournament_num in range(self.num_tournaments):
            print(f"\n--- Tournament {tournament_num + 1}/{self.num_tournaments} ---")

            # Select agents for this tournament
            if self.all_compete:
                # All agents compete in each tournament
                tournament_agents = self.population
                print(f"Participants: All {len(tournament_agents)} agents")
            else:
                # Random subset of agents compete
                tournament_agents = random.sample(self.population, min(self.tournament_size, len(self.population)))
                agent_ids = [agent.id for agent in tournament_agents]
                print(f"Participants: {len(tournament_agents)} agents (IDs: {', '.join(agent_ids)})")

            # Create fresh environment for this tournament
            np.random.seed(self.current_generation * 1000 + tournament_num + 42)
            random.seed(self.current_generation * 1000 + tournament_num + 42)

            self.environment = Environment(self.grid_size, self.resource_regen_turns)
            resource_count = np.sum(self.environment.grid == GRID_SYMBOLS["resource"])
            print(f"Environment: {resource_count} resources available")

            # Reset tournament agents with random positions
            for agent in tournament_agents:
                new_position = (
                    random.randint(0, self.grid_size - 1),
                    random.randint(0, self.grid_size - 1)
                )
                agent.reset(new_position)

            # Run tournament with randomized turn order
            print(f"Running {self.turns_per_generation} turns...\n")
            for _ in range(self.turns_per_generation):
                # Randomize agent order each timestep (fairness)
                shuffled_agents = random.sample(tournament_agents, len(tournament_agents))

                base_model, tokenizer = self.get_base_model_and_tokenizer()
                for agent in shuffled_agents:
                    # Only act if agent has energy
                    if agent.energy > 0:
                        agent.decide_and_act(self.environment, base_model, tokenizer)
                # Advance environment for resource regeneration
                self.environment.step()

            # Accumulate energy from this tournament (only for participating agents)
            for agent in tournament_agents:
                agent.cumulative_energy += agent.energy
                agent.tournament_count += 1
                agent.tournament_energies.append(agent.energy)

            # Store tournament data for visualisation
            tournament_data = {
                'environment': self.environment.grid.copy().tolist(),
                'champion_paths': [agent.path_history for agent in self.get_top_agents_from_list(tournament_agents, n=min(3, len(tournament_agents)))],
                'all_agent_paths': {agent.id: agent.path_history for agent in tournament_agents},
                'energies': {agent.id: agent.energy for agent in tournament_agents}
            }
            gen_tournament_data.append(tournament_data)

            # Tournament results
            living = sum(1 for agent in tournament_agents if agent.energy > 0)
            dead = len(tournament_agents) - living
            energies = [agent.energy for agent in tournament_agents]
            print(f"\nTournament {tournament_num + 1} Results:")
            print(f"  Status: {living} alive, {dead} dead")
            print(f"  Energy: max={max(energies):.2f}, avg={sum(energies)/len(energies):.2f}, min={min(energies):.2f}")

        # Average energy across all tournaments each agent participated in
        print(f"\n--- Averaging Fitness Across Tournaments ---")

        # Calculate energy variance across tournaments for each agent
        tournament_energy_stats = []
        for agent in self.population:
            if agent.tournament_count > 0:
                agent.energy = agent.cumulative_energy / agent.tournament_count
                # Calculate variance across tournaments
                if len(agent.tournament_energies) > 1:
                    variance = np.var(agent.tournament_energies)
                    std_dev = np.std(agent.tournament_energies)
                else:
                    variance = 0.0
                    std_dev = 0.0

                tournament_energy_stats.append({
                    'agent_id': agent.id,
                    'mean_energy': agent.energy,
                    'variance': variance,
                    'std_dev': std_dev,
                    'tournament_energies': agent.tournament_energies.copy()
                })
            else:
                # Agent didn't participate in any tournaments (only possible if all_compete=False)
                agent.energy = 0.0

            # Clean up temporary attributes
            del agent.cumulative_energy
            del agent.tournament_count
            del agent.tournament_energies

        # Store tournament variance data
        self.tournament_energy_variance.append(tournament_energy_stats)

        # Store tournament history for visualisation
        gen_idx = len(self.performance_history)
        self.tournament_history[gen_idx] = gen_tournament_data

        # Increment generation counter
        self.current_generation += 1

        # Store metrics (using averaged energy)
        print("\n--- Generation Complete ---")
        metrics = self._calculate_and_store_metrics()

        # Log final statistics
        living = metrics['living_agents']
        dead = metrics['energy_metrics']['dead_count']
        print(f"Final Averaged Status: {living} alive, {dead} dead")
        print(f"Averaged Energy: max={metrics['energy_metrics']['max']:.2f}, avg={metrics['energy_metrics']['avg']:.2f}, min={metrics['energy_metrics']['min']:.2f}")

        # Store final environment state (from last tournament)
        self.environment_history[len(self.performance_history) - 1] = (
            self.environment.grid.tolist()
        )
        self.initial_environment_history[len(self.performance_history) - 1] = (
            self.environment.grid.copy().tolist()
        )

    def evolve(self):
        """
        Evolves the population for the next generation using truncation selection.

        This involves three steps:
        1. Selection: The best-performing agents are selected to be parents (top 50%).
        2. Reproduction: New child agents are created from pairs of parents via crossover.
        3. Mutation: Small random changes are introduced into the children's brains.

        Always uses truncation selection for parent selection (not tournament selection).
        """
        print("\n=== Evolution Phase: Creating Next Generation ===")

        # 1. Selection - sort by energy in descending order
        # Dead agents (energy <= 0) will sort to the end
        self.population.sort(key=lambda agent: agent.energy, reverse=True)

        # Count living vs dead agents
        living_count = sum(1 for agent in self.population if agent.energy > 0)
        dead_count = self.population_size - living_count

        # Get energy statistics
        energies = [agent.energy for agent in self.population]
        max_energy = max(energies)
        min_energy = min(energies)
        avg_energy = sum(energies) / len(energies)

        print(f"Population Status: {living_count} alive, {dead_count} dead")
        print(f"Energy Range: {min_energy:.2f} (min) → {avg_energy:.2f} (avg) → {max_energy:.2f} (max)")

        # Get elites (top performers) for elitism
        elites = self.get_top_agents(n=self.num_elites)
        elite_energies = [e.energy for e in elites]
        print(f"Elitism: Preserving top {len(elites)} agents (energies: {[f'{e:.2f}' for e in elite_energies]})")

        # Create new agents with elite brains (preserve weights exactly)
        next_generation = []
        for elite in elites:
            elite_brain = LoraBrain(self.base_model)
            elite_weights = {name: weight.clone() for name, weight in elite.brain.get_weights().items()}
            elite_brain.set_weights(elite_weights)
            new_elite = Agent(
                brain=elite_brain,
                initial_position=(random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)),
                grid_size=self.grid_size
            )
            next_generation.append(new_elite)

        # Get mating pool: top 50% of population (truncation selection)
        parents = self.get_top_agents(n=self.population_size // 2)
        if len(parents) == 0:
            # If no living agents, use entire population
            parents = self.population
            print(f"WARNING: No living agents! Using entire population for mating pool")
        else:
            parent_energies = [p.energy for p in parents]
            print(f"Mating Pool: Top {len(parents)} agents (energy range: {min(parent_energies):.2f} - {max(parent_energies):.2f})")

        # Generate new agents by crossover and mutation
        print(f"Reproduction: Creating {self.population_size - len(elites)} offspring via crossover + mutation")
        while len(next_generation) < self.population_size:
            # Select two parents randomly from the mating pool
            parent1, parent2 = random.choices(parents, k=2)

            # Create a new brain for the child
            child_brain = LoraBrain(self.base_model)
            child_weights = {}

            # Crossover: Blend weights from both parents (averaging)
            p1_weights = parent1.brain.get_weights()
            p2_weights = parent2.brain.get_weights()
            for name in p1_weights:
                # Average the weights from both parents (alpha=0.5)
                child_weights[name] = 0.5 * p1_weights[name] + 0.5 * p2_weights[name]

            # Mutation: Apply sparse Gaussian-weighted mutations with regularization
            for name in child_weights:
                # Sparse selection: only mutate a fraction of weights
                sparse_mask = torch.rand_like(child_weights[name]) < self.sparse_mutation_rate

                # Gaussian weighting: selected weights get Gaussian-distributed mutation strength
                gaussian_weights = torch.randn_like(child_weights[name]).abs()  # Absolute Gaussian

                # Combine: sparse selection with Gaussian weighting
                mutation = torch.randn_like(child_weights[name]) * self.mutation_strength
                mutation = mutation * sparse_mask.float() * gaussian_weights

                # Weight regularization: pull weights back toward initial fine-tuned values
                # This prevents catastrophic genetic drift
                if self.initial_weights and name in self.initial_weights:
                    regularization = (self.initial_weights[name] - child_weights[name]) * self.weight_regularization
                    child_weights[name] += mutation + regularization
                else:
                    child_weights[name] += mutation

            child_brain.set_weights(child_weights)

            initial_position = (
                random.randint(0, self.grid_size - 1),
                random.randint(0, self.grid_size - 1),
            )
            next_generation.append(Agent(child_brain, initial_position, self.grid_size))

        self.population = next_generation
        print(f"Evolution Complete: New generation of {len(self.population)} agents created")
        print("=" * 50)

    def get_average_fitness(self) -> float:
        """
        Calculates the average fitness score of the current population.
        """
        if not self.population:
            return 0.0
        total_energy = sum(agent.energy for agent in self.population)
        return total_energy / len(self.population)

    def get_top_agents(self, n: int = 1) -> list["Agent"]:
        """
        Returns the top n living agents based on energy.
        If there aren't enough living agents, returns as many as available.

        Args:
            n: Maximum number of top agents to return.

        Returns:
            list[Agent]: A list of the top n living agents with energy > 0.
        """
        living_agents = [a for a in self.population if a.energy > 0]
        return sorted(living_agents, key=lambda agent: agent.energy, reverse=True)[:n]

    def get_top_agents_from_list(self, agent_list: list["Agent"], n: int = 1) -> list["Agent"]:
        """
        Returns the top n agents from a given list based on energy.
        Similar to get_top_agents but works on a custom list instead of self.population.

        Args:
            agent_list: List of agents to select from.
            n: Maximum number of top agents to return.

        Returns:
            list[Agent]: A list of the top n agents from the list.
        """
        return sorted(agent_list, key=lambda agent: agent.energy, reverse=True)[:n]

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
        
        Returns:
            dict: Dictionary containing various metrics about the population
                - generation: Current generation number
                - energy_metrics: Dictionary of energy statistics
                - action_metrics: Dictionary of action statistics
                - action_counts: Raw counts of each action
                - living_agents: Number of agents with energy > 0
        """
        if not self.population:
            return {
                "generation": self.current_generation,
                "energy_metrics": {
                    "max": 0.0,
                    "min": 0.0,
                    "avg": 0.0,
                    "median": 0.0,
                    "std": 0.0,
                    "total": 0.0,
                    "alive_count": 0,
                    "dead_count": 0,
                },
                "action_metrics": {},
                "action_counts": {},
                "living_agents": 0
            }

        # Energy metrics
        energy_levels = [agent.energy for agent in self.population]
        energy_array = np.array(energy_levels)
        
        # Action metrics
        action_energy = {}
        action_counts = {}
        
        for agent in self.population:
            # Track action counts and energy impact
            for action in agent.action_history:
                if action not in action_energy:
                    action_energy[action] = []
                action_energy[action].append(agent.energy)
                action_counts[action] = action_counts.get(action, 0) + 1
        
        # Calculate action energy statistics
        action_stats = {}
        for action, energies in action_energy.items():
            if energies:  # Only calculate if we have data
                action_stats[action] = {
                    "count": len(energies),
                    "avg_energy": float(np.mean(energies)),
                    "median_energy": float(np.median(energies)),
                    "min_energy": float(min(energies)),
                    "max_energy": float(max(energies)),
                }
        
        metrics = {
            "generation": self.current_generation,
            "energy_metrics": {
                "max": float(np.max(energy_array)),
                "min": float(np.min(energy_array)),
                "avg": float(np.mean(energy_array)),
                "median": float(np.median(energy_array)),
                "std": float(np.std(energy_array)),
                "total": float(np.sum(energy_array)),
                "alive_count": int(np.sum(energy_array > 0)),
                "dead_count": int(np.sum(energy_array <= 0)),
            },
            "action_metrics": action_stats,
            "action_counts": action_counts,
            "living_agents": int(np.sum(energy_array > 0))
        }
        
        # Store metrics in history
        self.performance_history.append(metrics)
        
        # Calculate and store genetic diversity
        genetic_diversity = self._calculate_genetic_diversity()
        self.genetic_history.append(genetic_diversity)
        
        # Store behavioral history
        self.behavioural_history.append(action_counts)
        
        # Store the environment grid for this generation
        self.environment_history[len(self.performance_history) - 1] = (
            self.environment.grid.tolist()
        )
        
        return metrics

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
