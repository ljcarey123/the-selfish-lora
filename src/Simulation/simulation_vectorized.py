"""
Vectorized Simulation for GPU-efficient evolution.

This simulation orchestrates:
- VectorizedEnvironment (NumPy-based physics)
- InferenceEngine (sequential LLM inference with per-agent LoRA weights)
- VectorizedAgent (lightweight state containers)

Key optimizations:
- Perception extraction is vectorized (no loops)
- Physics updates are vectorized (metabolism, combat damage)
- State updates are vectorized where possible
- LLM inference is sequential to preserve per-agent LoRA diversity
"""

import numpy as np
import random
import os
import torch
import json
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig

from Environment.environment_vectorized import VectorizedEnvironment
from Agent.agent_vectorized import VectorizedAgent
from LoraBrain.lora_brain import LoraBrain
from LoraBrain.inference_engine import InferenceEngine


class VectorizedSimulation:
    """
    GPU-optimized evolutionary simulation with vectorized operations.

    This simulation runs 32 agents per generation in 4 batches of 8,
    using vectorized NumPy operations and batch inference.
    """

    def __init__(
        self,
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
        Initialise vectorized simulation.

        Args:
            population_size: Total agents per generation
            batch_size: Agents per batch (8 for 16x16 grid)
            turns_per_batch: Max turns per batch
            mutation_strength: Mutation strength
            num_elites: Elite agents to preserve
            initial_noise: Initial weight noise
            sparse_mutation_rate: Fraction of weights to mutate
            weight_regularization: Weight regularization strength
            random_init: Use random initialization
            rounds_per_generation: Number of rounds each agent competes per generation
        """

        if population_size % batch_size != 0:
            raise ValueError(f"population_size ({population_size}) must be divisible by batch_size ({batch_size})")

        self.population_size = population_size
        self.batch_size = batch_size
        self.num_batches = population_size // batch_size
        self.turns_per_batch = turns_per_batch
        self.mutation_strength = mutation_strength
        self.num_elites = num_elites
        self.current_generation = 0
        self.initial_noise = initial_noise
        self.sparse_mutation_rate = sparse_mutation_rate
        self.weight_regularization = weight_regularization
        self.random_init = random_init
        self.rounds_per_generation = rounds_per_generation
        self.initial_weights = None

        # History tracking for visualization
        self.performance_history = []
        self.champion_paths_by_gen = {}  # {gen: {round: [paths]}} - top agents' paths per round
        self.environment_history = {}  # {gen: {round: grid}} - environment per round
        self.genetic_history = []  # genetic diversity over time
        self.behavioural_history = []  # action distribution over time
        self.generation_champions = []  # List of champion agents from each generation (for final tournament)

        # Load base model
        load_dotenv()
        hf_token = os.getenv("HF_TOKEN")

        print("Loading base LLM (Qwen/Qwen2.5-0.5B-Instruct) for LoRA...")
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True, token=hf_token
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct", token=hf_token
        )
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # LoRA config
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        self.base_model = get_peft_model(model, peft_config)

        # Move to GPU if available
        if torch.cuda.is_available():
            print(f"CUDA available: Moving model to GPU...")
            self.base_model = self.base_model.to("cuda")
            print(f"Model device: {next(self.base_model.parameters()).device}")
        else:
            print(f"WARNING: CUDA not available - using CPU (will be very slow!)")
            print(f"Model device: {next(self.base_model.parameters()).device}")

        self.base_model.eval()

        # Inference engine (will be updated with agent brains each generation)
        self.inference_engine = InferenceEngine(
            self.base_model,
            self.tokenizer
        )

        print("Initialising population...")
        self.population = self._initialise_population()
        print(f"Vectorized simulation initialised: {self.population_size} agents, {self.num_batches} batches of {self.batch_size}, {self.rounds_per_generation} rounds per generation")

    def _initialise_population(self) -> list:
        """
        Create initial population.

        Returns:
            List of VectorizedAgent instances
        """
        # Load fine-tuned weights if available
        parent_weights_list = []
        weights_path = "./output/weights"

        if not self.random_init and os.path.exists(weights_path):
            for filename in os.listdir(weights_path):
                if filename.endswith("_lora_weights.pth"):
                    try:
                        full_path = os.path.join(weights_path, filename)
                        weights = torch.load(full_path)
                        parent_weights_list.append({"name": filename, "weights": weights})
                    except Exception as e:
                        print(f"Warning: Could not load weights: {e}")

        if parent_weights_list:
            self.initial_weights = {k: v.clone() for k, v in parent_weights_list[0]["weights"].items()}
            print(f"Loaded {len(parent_weights_list)} parent model(s)")
        else:
            if not self.random_init:
                print("Warning: No fine-tuned weights found")
            print("Initializing with random weights")

        # Create population
        population = []
        for i in range(self.population_size):
            brain = LoraBrain(self.base_model)

            # Initialize weights
            if parent_weights_list and not self.random_init:
                parent = parent_weights_list[i % len(parent_weights_list)]
                parent_weights = parent["weights"]

                noise_multiplier = np.clip(np.random.exponential(scale=1.0), 0.1, 3.0)
                agent_noise = self.initial_noise * noise_multiplier

                brain_weights = brain.get_weights()
                for name in brain_weights:
                    if name in parent_weights:
                        noise = torch.randn_like(parent_weights[name]) * agent_noise
                        brain_weights[name] = parent_weights[name].clone() + noise
                brain.set_weights(brain_weights)
            else:
                brain_weights = brain.get_weights()
                for name in brain_weights:
                    noise = torch.randn_like(brain_weights[name]) * 0.001
                    brain_weights[name] = brain_weights[name] + noise
                brain.set_weights(brain_weights)

            agent = VectorizedAgent(brain=brain, grid_size=16)
            population.append(agent)

        return population

    def run_generation(self):
        """
        Run a generation with tournament format (multiple rounds per generation).
        Each agent competes in multiple rounds and fitness is averaged.
        """
        print(f"\n{'='*60}")
        print(f"GENERATION {self.current_generation + 1}")
        print(f"{'='*60}")

        # Initialize cumulative fitness tracking
        cumulative_energy = {agent.id: 0.0 for agent in self.population}
        cumulative_food = {agent.id: 0 for agent in self.population}
        cumulative_combat = {agent.id: 0 for agent in self.population}
        total_energy_available = 0.0  # Track total energy spawned across rounds

        # Track per-round scores for table display
        round_scores = {agent.id: [] for agent in self.population}  # {agent_id: [round1_energy, round2_energy, ...]}

        # Run multiple rounds
        for round_idx in range(self.rounds_per_generation):
            print(f"\n--- Round {round_idx + 1}/{self.rounds_per_generation} ---")

            # Reset agents for new round (keep cumulative food/combat stats)
            for agent in self.population:
                agent.reset_for_round()

            # Create fresh environment for this round
            env = VectorizedEnvironment(generation=self.current_generation)

            # Get generation info
            stats = env.get_stats()
            phase = env.generation_manager.get_phase_description()
            print(f"Phase: {phase} | Food: {stats['food_remaining']} items")

            # Split population into batches
            batches = [
                self.population[i:i + self.batch_size]
                for i in range(0, self.population_size, self.batch_size)
            ]

            print(f"Running {len(batches)} batches of {self.batch_size} agents each...")

            for batch_idx, batch in enumerate(batches):
                print(f"\n  Batch {batch_idx + 1}/{len(batches)}")

                # Reset environment
                env.reset()

                # Register all agents in batch (VECTORIZED)
                agent_ids = [agent.id for agent in batch]
                env.register_agents(agent_ids, initial_energy=100.0)

                # Build adapter mapping
                adapter_mapping = {agent.id: f"adapter_{agent.id}" for agent in batch}

                # Run batch
                self._run_batch_vectorized(env, batch, adapter_mapping, round_idx)

                # Sync agent states from environment and accumulate fitness
                agent_data = env.get_agent_data()
                for i, agent in enumerate(batch):
                    energy = agent_data["energies"][i]
                    alive = agent_data["alive"][i]
                    position = tuple(int(x) for x in agent_data["positions"][i])

                    agent.sync_from_environment(energy, alive, position)

                    # Accumulate fitness metrics
                    cumulative_energy[agent.id] += energy
                    cumulative_food[agent.id] += agent.food_collected
                    cumulative_combat[agent.id] += (agent.combat_attacks + agent.combat_victims)

                stats = env.get_stats()
                print(f"  Batch {batch_idx + 1} complete: {stats['agents_alive']}/{self.batch_size} alive, {stats['turn']} turns, {stats['food_remaining']} food left")

            # Record scores for this round (after all batches complete)
            for agent in self.population:
                round_scores[agent.id].append(agent.energy)

            # Track total energy spawned this round
            total_energy_available += env.total_energy_spawned

            # Save round data
            self._save_round_data(env, round_idx)

        # Calculate average fitness and assign to agents
        # Average energy available per round
        avg_energy_per_round = total_energy_available / self.rounds_per_generation

        # Fair share of energy per agent (accounting for environment difficulty)
        energy_per_agent = avg_energy_per_round / self.batch_size

        print(f"\nEnergy economics:")
        print(f"  Total energy spawned (avg/round): {avg_energy_per_round:.1f}")
        print(f"  Fair share per agent: {energy_per_agent:.1f}")

        for agent in self.population:
            avg_energy = cumulative_energy[agent.id] / self.rounds_per_generation
            agent.energy = avg_energy
            # Calculate normalized fitness percentage relative to available resources
            # This accounts for environment difficulty (food density, regrowth rate)
            # >100% means agent captured more than their fair share
            agent.fitness_percentage = (avg_energy / energy_per_agent) * 100 if energy_per_agent > 0 else 0
            # Note: food_collected and combat_events remain cumulative across rounds

        # Store metrics
        self._calculate_metrics()

        # Print tournament score table
        self._print_tournament_scores(round_scores)

        # Save generation champion (for final tournament)
        champion = self.get_top_agents(n=1)[0]
        # Create a copy of the champion's brain to preserve it
        champion_copy = VectorizedAgent(brain=LoraBrain(self.base_model), grid_size=16)
        champion_copy.brain.set_weights(champion.brain.get_weights().copy())
        champion_copy.id = f"Gen{self.current_generation}_Champion"
        champion_copy.energy = champion.energy
        champion_copy.fitness_percentage = champion.fitness_percentage
        champion_copy.food_collected = champion.food_collected
        champion_copy.combat_attacks = champion.combat_attacks
        champion_copy.combat_victims = champion.combat_victims
        self.generation_champions.append(champion_copy)

        # Save generation summary
        self._save_generation_summary()

        self.current_generation += 1

    def _run_batch_vectorized(
        self,
        env: VectorizedEnvironment,
        batch: list,
        adapter_mapping: dict,
        round_idx: int
    ):
        """
        Run a batch with vectorized operations.

        Args:
            env: VectorizedEnvironment
            batch: List of agents
            adapter_mapping: {agent_id -> adapter_id}
            round_idx: Current round index (for tracking)
        """
        # Update inference engine with current batch's agent brains
        agent_brains = {agent.id: agent.brain for agent in batch}
        self.inference_engine.set_agent_brains(agent_brains)

        # Track recent actions (last 3 moves per agent)
        recent_actions = {}  # {agent_id: [action1, action2, action3]} (most recent last)

        for turn in range(self.turns_per_batch):
            # Get alive agent indices (vectorized)
            alive_indices = np.where(env.agent_alive)[0]

            if len(alive_indices) == 0:
                print(f"    All agents died by turn {turn + 1}")
                break

            # Randomize turn order (fairness)
            np.random.shuffle(alive_indices)

            # Get batch perception (VECTORIZED - all agents at once)
            perception = env.get_batch_perception(alive_indices)

            # Prepare batch inference request (NO LLM call here)
            batch_request = env.prepare_batch_inference_request(
                perception,
                adapter_mapping,
                recent_actions
            )

            # Sequential inference with per-agent LoRA weights
            batch_response = self.inference_engine.process_batch(batch_request)

            # Parse results & apply actions
            response_dict = batch_response.to_dict()

            # Count invalid actions
            invalid_count = sum(1 for aid in perception.agent_ids if not response_dict[aid]["success"])

            # Apply actions (vectorized where possible)
            actions_list = [response_dict[aid]["action"] for aid in perception.agent_ids]
            results = env.apply_actions(alive_indices, actions_list)

            # Process results
            victims_to_damage = []
            combat_victims = {}  # {victim_id: attacker_position}
            combat_count = 0
            food_count = 0

            for agent_id, result in results.items():
                # Find agent in batch
                agent = next((a for a in batch if a.id == agent_id), None)
                if not agent:
                    continue

                # Get agent index to access position
                agent_idx = env.agent_ids.index(agent_id)
                position = tuple(int(x) for x in env.agent_positions[agent_idx])

                # Record action and track recent moves (keep last 3)
                action = response_dict[agent_id]["action"]
                agent.record_action(action, position)

                if agent_id not in recent_actions:
                    recent_actions[agent_id] = []
                recent_actions[agent_id].append(action)
                if len(recent_actions[agent_id]) > 3:
                    recent_actions[agent_id].pop(0)  # Keep only last 3

                # Handle combat
                if "combat_victim" in result:
                    victim_id = result["combat_victim"]
                    victims_to_damage.append(victim_id)
                    combat_victims[victim_id] = position  # Store attacker position for victim
                    agent.record_combat_event(position, is_attacker=True)
                    combat_count += 1

                # Handle food
                elif "food" in result["reason"].lower():
                    agent.record_food_collected(position)
                    food_count += 1

            # Record combat events for victims
            for victim_id, attacker_pos in combat_victims.items():
                victim_agent = next((a for a in batch if a.id == victim_id), None)
                if victim_agent:
                    # Get victim's position
                    victim_idx = env.agent_ids.index(victim_id)
                    victim_pos = tuple(int(x) for x in env.agent_positions[victim_idx])
                    victim_agent.record_combat_event(victim_pos, is_attacker=False)

            # Apply combat damage, metabolism, and environment updates (vectorized)
            if victims_to_damage:
                env.apply_combat_damage(victims_to_damage, damage=-20)

            # Apply metabolism (VECTORIZED - all agents at once)
            env.apply_metabolism()

            # Update alive status (vectorized)
            dead_ids = env.update_alive_status()

            # Environment step (food regrowth)
            env.step()

            # Turn summary - print after all agents have acted
            alive_count = int(np.sum(env.agent_alive))
            events = []
            if food_count > 0:
                events.append(f"{food_count} food")
            if combat_count > 0:
                events.append(f"{combat_count} combat")
            if dead_ids:
                events.append(f"{len(dead_ids)} died")
            if invalid_count > 0:
                events.append(f"{invalid_count} invalid")

            event_str = f" | {', '.join(events)}" if events else ""
            print(f"    Turn {turn + 1:3d}/{self.turns_per_batch}: {alive_count} alive{event_str}")

    def _print_tournament_scores(self, round_scores: dict):
        """
        Print a table showing each agent's score in each tournament round.

        Args:
            round_scores: {agent_id: [round1_energy, round2_energy, ...]}
        """
        if self.rounds_per_generation <= 1:
            return  # No need for table if only 1 round

        print(f"\n{'='*60}")
        print(f"TOURNAMENT SCORES - GENERATION {self.current_generation + 1}")
        print(f"{'='*60}")

        # Sort agents by final average energy
        sorted_agents = sorted(self.population, key=lambda a: a.energy, reverse=True)

        # Print header
        header = f"{'Rank':<6} {'Agent ID':<12}"
        for r in range(self.rounds_per_generation):
            header += f" {'Round ' + str(r+1):>10}"
        header += f" {'Average':>10} {'Food':>6} {'Attacks':>8} {'Victim':>7}"
        print(header)
        print("-" * len(header))

        # Print each agent's scores
        for rank, agent in enumerate(sorted_agents, 1):
            scores = round_scores[agent.id]
            row = f"{rank:<6} {agent.id:<12}"
            for score in scores:
                row += f" {score:>10.1f}"
            avg = sum(scores) / len(scores) if scores else 0
            row += f" {avg:>10.1f} {agent.food_collected:>6} {agent.combat_attacks:>8} {agent.combat_victims:>7}"
            print(row)

        print(f"{'='*60}\n")

    def _calculate_metrics(self):
        """Calculate and store generation metrics."""
        energies = [agent.energy for agent in self.population]
        fitness_percentages = [agent.fitness_percentage for agent in self.population]

        metrics = {
            "generation": self.current_generation,
            "living_agents": sum(1 for agent in self.population if agent.is_alive),
            "energy_metrics": {
                "max": max(energies),
                "avg": np.mean(energies),
                "min": min(energies),
            },
            "fitness_percentage_metrics": {
                "max": max(fitness_percentages),
                "avg": np.mean(fitness_percentages),
                "min": min(fitness_percentages),
            }
        }

        self.performance_history.append(metrics)

        print(f"\n--- Generation {self.current_generation + 1} Complete ---")
        print(f"Living agents: {metrics['living_agents']}/{self.population_size}")
        print(f"Energy: max={metrics['energy_metrics']['max']:.1f}, "
              f"avg={metrics['energy_metrics']['avg']:.1f}, "
              f"min={metrics['energy_metrics']['min']:.1f}")
        print(f"Fitness %: max={metrics['fitness_percentage_metrics']['max']:.1f}%, "
              f"avg={metrics['fitness_percentage_metrics']['avg']:.1f}%, "
              f"min={metrics['fitness_percentage_metrics']['min']:.1f}%")

    def evolve_population(self):
        """Evolve population using tournament fitness."""
        print(f"\n=== Evolution Phase ===")

        # Sort by average energy (already calculated)
        self.population.sort(key=lambda a: a.energy, reverse=True)

        # Elitism
        elites = self.population[:self.num_elites]
        print(f"Preserving top {self.num_elites} agents")

        # Mating pool
        mating_pool_size = max(2, self.population_size // 2)
        mating_pool = self.population[:mating_pool_size]

        # Create offspring
        offspring_needed = self.population_size - self.num_elites
        offspring = []

        for _ in range(offspring_needed):
            parent1, parent2 = random.sample(mating_pool, 2)
            child_brain = self._crossover_and_mutate(parent1.brain, parent2.brain)
            child_agent = VectorizedAgent(brain=child_brain, grid_size=16)
            offspring.append(child_agent)

        self.population = elites + offspring

        print(f"Evolution complete: New generation of {self.population_size} agents")

    def _crossover_and_mutate(self, brain1: LoraBrain, brain2: LoraBrain) -> LoraBrain:
        """Crossover and mutate (same as before)."""
        child_brain = LoraBrain(self.base_model)
        child_weights = {}

        parent1_weights = brain1.get_weights()
        parent2_weights = brain2.get_weights()

        for name in parent1_weights:
            # Crossover
            child_weights[name] = (parent1_weights[name] + parent2_weights[name]) / 2.0

            # Mutation
            if np.random.random() < self.sparse_mutation_rate:
                mutation = torch.randn_like(child_weights[name]) * self.mutation_strength
                child_weights[name] = child_weights[name] + mutation

            # Regularization
            if self.initial_weights and name in self.initial_weights:
                pull = (self.initial_weights[name] - child_weights[name]) * self.weight_regularization
                child_weights[name] = child_weights[name] + pull

        child_brain.set_weights(child_weights)
        return child_brain

    def get_top_agents(self, n: int = 3) -> list:
        """Get top N agents by energy."""
        sorted_pop = sorted(self.population, key=lambda a: a.energy, reverse=True)
        return sorted_pop[:n]

    def _save_round_data(self, env: VectorizedEnvironment, round_idx: int):
        """
        Save round data for visualization and analysis.

        Args:
            env: Environment used this round
            round_idx: Round index within generation
        """
        gen_str = str(self.current_generation)
        round_str = str(round_idx)

        # Initialize generation dict if needed
        if gen_str not in self.champion_paths_by_gen:
            self.champion_paths_by_gen[gen_str] = {}
        if gen_str not in self.environment_history:
            self.environment_history[gen_str] = {}

        # Save environment grid (convert to string format for visualization)
        grid_symbolic = self._convert_grid_to_symbolic(env.grid)
        self.environment_history[gen_str][round_str] = grid_symbolic.tolist()

        # Save top agent paths (sorted by energy)
        top_agents = self.get_top_agents(n=5)  # Save top 5 paths
        champion_paths = [agent.get_path_copy() for agent in top_agents]
        self.champion_paths_by_gen[gen_str][round_str] = champion_paths

    def _save_generation_summary(self):
        """Save generation-level summary data."""
        # Calculate and save genetic diversity
        diversity = self._calculate_genetic_diversity()
        self.genetic_history.append(diversity)

        # Calculate and save behavioural metrics (action distribution)
        action_dist = self._calculate_action_distribution()
        self.behavioural_history.append(action_dist)

        # Save weights (top 1 per generation)
        self._save_weights()

        # Save all data to JSON
        self._save_visualization_data()

    def _convert_grid_to_symbolic(self, grid: np.ndarray) -> np.ndarray:
        """
        Convert integer grid to symbolic grid for visualization.

        Args:
            grid: Integer grid

        Returns:
            Symbolic grid with 'F' for food, 'W' for wall, etc.
        """
        from config_loader import GRID_SYMBOLS

        symbolic = np.full(grid.shape, '.', dtype='<U1')

        # Map integers to symbols
        int_encoding = {
            0: '.',  # empty
            1: 'W',  # wall
            2: 'F',  # food (using F instead of R for consistency)
            3: 'A'   # agent
        }

        for int_val, symbol in int_encoding.items():
            symbolic[grid == int_val] = symbol

        return symbolic

    def _calculate_genetic_diversity(self) -> float:
        """
        Calculate average genetic distance between agents.

        Returns:
            Average pairwise distance
        """
        if len(self.population) < 2:
            return 0.0

        # Sample pairs to avoid O(n^2) for large populations
        num_samples = min(100, len(self.population) * (len(self.population) - 1) // 2)
        distances = []

        for _ in range(num_samples):
            agent1, agent2 = random.sample(self.population, 2)
            weights1 = agent1.brain.get_weights()
            weights2 = agent2.brain.get_weights()

            # Calculate L2 distance
            total_dist = 0.0
            for name in weights1:
                diff = weights1[name] - weights2[name]
                total_dist += torch.sum(diff ** 2).item()

            distances.append(np.sqrt(total_dist))

        return float(np.mean(distances)) if distances else 0.0

    def _calculate_action_distribution(self) -> dict:
        """
        Calculate distribution of actions taken by all agents.

        Returns:
            Dict of action -> count
        """
        action_counts = {
            "MOVE_NORTH": 0,
            "MOVE_SOUTH": 0,
            "MOVE_EAST": 0,
            "MOVE_WEST": 0,
            "COMBAT": 0,
            "FOOD_COLLECTED": 0
        }

        for agent in self.population:
            for action_or_event in agent.action_history:
                if action_or_event in action_counts:
                    action_counts[action_or_event] += 1

        return action_counts

    def _save_weights(self):
        """
        Save top agent weights.
        - Top 1 each generation
        - Top 3 on final generation
        """
        weights_dir = "./output/weights"
        os.makedirs(weights_dir, exist_ok=True)

        # Always save top 1
        top_agent = self.get_top_agents(n=1)[0]
        weight_path = os.path.join(
            weights_dir,
            f"gen{self.current_generation}_top1_lora_weights.pth"
        )
        torch.save(top_agent.brain.get_weights(), weight_path)
        print(f"  Saved top agent weights: {weight_path}")

    def _save_visualization_data(self):
        """Save all visualization data to JSON."""
        output_dir = "./output"
        os.makedirs(output_dir, exist_ok=True)

        viz_data = {
            "champion_paths_by_gen": self.champion_paths_by_gen,
            "environment_history": self.environment_history,
            "genetic_history": self.genetic_history,
            "behavioural_history": self.behavioural_history,
            "performance_history": self.performance_history,
            "tournament_mode": self.rounds_per_generation > 1,
            "rounds_per_generation": self.rounds_per_generation,
            "grid_size": 16
        }

        json_path = os.path.join(output_dir, "simulation_paths.json")
        with open(json_path, 'w') as f:
            json.dump(viz_data, f, indent=2)

    def run_champion_tournament(self):
        """
        Run a final tournament where champions from each generation compete.
        This tests whether later generations truly evolved better strategies.
        Uses the final generation's difficulty to test adaptation to scarcity.
        """
        if len(self.generation_champions) <= 1:
            print("Skipping champion tournament (only 1 generation)")
            return

        print(f"\n{'='*60}")
        print("CHAMPION TOURNAMENT")
        print(f"{'='*60}")
        print(f"Champions from {len(self.generation_champions)} generations will compete!")
        print()

        # Create environment at FINAL generation difficulty
        # This tests which generation best adapted to scarcity/challenge
        final_gen = self.current_generation - 1  # Last completed generation
        env = VectorizedEnvironment(generation=final_gen)

        # Show difficulty settings
        phase_desc = env.generation_manager.get_phase_description()
        print(f"Tournament Environment: Generation {final_gen} difficulty")
        print(f"  {phase_desc}")
        print()

        # Determine batch configuration
        num_champions = len(self.generation_champions)
        batch_size = min(8, num_champions)  # Use batch_size 8 or fewer if needed
        num_batches = (num_champions + batch_size - 1) // batch_size  # Ceiling division

        print(f"Tournament format: {num_champions} champions, {num_batches} batch(es) of up to {batch_size}")
        print()

        # Split champions into batches
        batches = []
        for i in range(0, num_champions, batch_size):
            batch = self.generation_champions[i:i + batch_size]
            batches.append(batch)

        # Reset champion stats before tournament (they contain data from original generations)
        for champion in self.generation_champions:
            champion.combat_attacks = 0
            champion.combat_victims = 0
            champion.food_collected = 0

        # Run each batch
        for batch_idx, batch in enumerate(batches):
            print(f"\n{'='*50}")
            print(f"TOURNAMENT BATCH {batch_idx + 1}/{num_batches}")
            print(f"{'='*50}")

            # Reset environment
            env.reset()

            # Register champions in batch
            agent_ids = [agent.id for agent in batch]
            env.register_agents(agent_ids, initial_energy=100.0)

            # Build adapter mapping
            adapter_mapping = {agent.id: f"adapter_{agent.id}" for agent in batch}

            # Run batch (single round)
            self._run_batch_vectorized(env, batch, adapter_mapping, round_idx=0)

            # Sync results
            agent_data = env.get_agent_data()
            for i, agent in enumerate(batch):
                energy = agent_data["energies"][i]
                alive = agent_data["alive"][i]
                position = tuple(int(x) for x in agent_data["positions"][i])
                agent.sync_from_environment(energy, alive, position)
                # Update fitness percentage
                agent.fitness_percentage = (energy / 100.0) * 100

        # Display results
        print(f"\n{'='*60}")
        print("CHAMPION TOURNAMENT RESULTS")
        print(f"{'='*60}")

        # Sort champions by final energy
        sorted_champions = sorted(self.generation_champions, key=lambda a: a.energy, reverse=True)

        print("\nFinal Rankings:")
        for rank, champion in enumerate(sorted_champions, 1):
            status = "✓ ALIVE" if champion.is_alive else "✗ DEAD"
            gen_num = champion.id.split("_")[0].replace("Gen", "")
            print(f"  #{rank}: {champion.id:20s} | "
                  f"Energy: {champion.energy:6.1f} ({champion.fitness_percentage:5.1f}%) | "
                  f"Food: {champion.food_collected} | "
                  f"Attacks: {champion.combat_attacks} | "
                  f"Victim: {champion.combat_victims} | "
                  f"{status}")

        # Determine winner
        winner = sorted_champions[0]
        winner_gen = winner.id.split("_")[0].replace("Gen", "")
        print(f"\n🏆 TOURNAMENT WINNER: {winner.id} (Generation {winner_gen})")
        print(f"   Final Energy: {winner.energy:.1f} ({winner.fitness_percentage:.1f}%)")

    def save_final_weights(self):
        """Save top 3 weights at end of training."""
        weights_dir = "./output/weights"
        os.makedirs(weights_dir, exist_ok=True)

        top_3 = self.get_top_agents(n=3)
        for rank, agent in enumerate(top_3, 1):
            weight_path = os.path.join(
                weights_dir,
                f"final_top{rank}_lora_weights.pth"
            )
            torch.save(agent.brain.get_weights(), weight_path)
            print(f"Saved final top-{rank} weights: {weight_path}")
