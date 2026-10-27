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
        Initialise vectorized simulation.

        Args:
            num_generations: Total number of generations to run
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

        self.num_generations = num_generations
        self.population_size = population_size
        self.batch_size = batch_size
        self.num_batches = population_size // batch_size
        self.turns_per_batch = turns_per_batch
        self.base_mutation_strength = mutation_strength  # Starting mutation strength
        self.mutation_strength = mutation_strength  # Current mutation strength (will decay)
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
        self.turn_by_turn_history = {}  # {gen: {round: {batch: [turn_states]}}} - full playthrough data
        self.genetic_history = []  # genetic diversity over time
        self.behavioural_history = []  # action distribution over time
        self.analytics_history = []  # Extended analytics per generation
        self.percentile_behavioural_history = []  # Percentile-specific behavioral metrics per generation
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

        # Initialize fitness tracking
        cumulative_energy = {agent.id: 0.0 for agent in self.population}
        total_energy_available = 0.0  # Track total energy spawned across rounds

        # Track per-round statistics for table display
        round_scores = {agent.id: [] for agent in self.population}  # {agent_id: [round1_energy, round2_energy, ...]}
        round_food = {agent.id: [] for agent in self.population}  # {agent_id: [round1_food, round2_food, ...]}
        round_attacks = {agent.id: [] for agent in self.population}  # {agent_id: [round1_attacks, round2_attacks, ...]}
        round_victims = {agent.id: [] for agent in self.population}  # {agent_id: [round1_victims, round2_victims, ...]}
        round_deaths = {agent.id: [] for agent in self.population}  # {agent_id: [round1_died, round2_died, ...]}

        # Run multiple rounds
        for round_idx in range(self.rounds_per_generation):
            print(f"\n--- Round {round_idx + 1}/{self.rounds_per_generation} ---")

            # Track stats at start of round to calculate per-round deltas
            round_start_food = {agent.id: agent.food_collected for agent in self.population}
            round_start_attacks = {agent.id: agent.combat_attacks for agent in self.population}
            round_start_victims = {agent.id: agent.combat_victims for agent in self.population}

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

                # Determine if we should record playthrough (only for final generation)
                is_final_gen = (self.current_generation == self.num_generations - 1)
                record = is_final_gen

                # Run batch
                batch_turn_states = self._run_batch_vectorized(
                    env, batch, adapter_mapping, round_idx,
                    batch_idx=batch_idx,
                    record_playthrough=record
                )

                # Save turn-by-turn data if recorded
                if batch_turn_states is not None:
                    gen_str = str(self.current_generation)
                    round_str = str(round_idx)
                    batch_str = str(batch_idx)

                    if gen_str not in self.turn_by_turn_history:
                        self.turn_by_turn_history[gen_str] = {}
                    if round_str not in self.turn_by_turn_history[gen_str]:
                        self.turn_by_turn_history[gen_str][round_str] = {}

                    self.turn_by_turn_history[gen_str][round_str][batch_str] = batch_turn_states

                # Sync agent states from environment and accumulate fitness
                agent_data = env.get_agent_data()
                for i, agent in enumerate(batch):
                    energy = agent_data["energies"][i]
                    alive = agent_data["alive"][i]
                    position = tuple(int(x) for x in agent_data["positions"][i])

                    agent.sync_from_environment(energy, alive, position)

                    # Accumulate energy for fitness calculation
                    cumulative_energy[agent.id] += energy
                    # Note: food/combat stats accumulate on agent object across rounds

                stats = env.get_stats()
                print(f"  Batch {batch_idx + 1} complete: {stats['agents_alive']}/{self.batch_size} alive, {stats['turn']} turns, {stats['food_remaining']} food left")

            # Record per-round statistics (after all batches complete)
            for agent in self.population:
                # Energy at end of round
                round_scores[agent.id].append(agent.energy)

                # Food/combat collected THIS round (delta from start)
                food_this_round = agent.food_collected - round_start_food[agent.id]
                attacks_this_round = agent.combat_attacks - round_start_attacks[agent.id]
                victims_this_round = agent.combat_victims - round_start_victims[agent.id]

                round_food[agent.id].append(food_this_round)
                round_attacks[agent.id].append(attacks_this_round)
                round_victims[agent.id].append(victims_this_round)

                # Death status (died if not alive)
                round_deaths[agent.id].append(not agent.is_alive)

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
        self._print_tournament_scores(round_scores, round_food, round_attacks, round_victims, round_deaths)

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

        # Adaptive mutation: decrease mutation strength over generations
        # Start high for exploration, decrease for exploitation
        # Decay formula: mutation_strength = base_strength * exp(-decay_rate * generation)
        decay_rate = 0.02  # Slower decay to maintain exploration longer
        self.mutation_strength = self.base_mutation_strength * np.exp(-decay_rate * self.current_generation)
        print(f"Mutation strength for next generation: {self.mutation_strength:.6f} (decay rate: {decay_rate})")

    def _run_batch_vectorized(
        self,
        env: VectorizedEnvironment,
        batch: list,
        adapter_mapping: dict,
        round_idx: int,
        batch_idx: int = 0,
        record_playthrough: bool = False
    ):
        """
        Run a batch with vectorized operations.

        Args:
            env: VectorizedEnvironment
            batch: List of agents
            adapter_mapping: {agent_id -> adapter_id}
            round_idx: Current round index (for tracking)
            batch_idx: Batch index within round
            record_playthrough: If True, record full turn-by-turn state for animation
        """
        # Update inference engine with current batch's agent brains
        agent_brains = {agent.id: agent.brain for agent in batch}
        self.inference_engine.set_agent_brains(agent_brains)

        # Track recent actions (last 3 moves per agent)
        recent_actions = {}  # {agent_id: [action1, action2, action3]} (most recent last)

        # Initialize playthrough recording if requested
        turn_states = [] if record_playthrough else None

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
                parse_success = response_dict[agent_id]["success"]

                # Record "INVALID" for failed parses, otherwise record the actual action
                recorded_action = action if parse_success else "INVALID"
                agent.record_action(recorded_action, position)

                # Update position metrics (track starting position, distance, unique cells)
                old_pos = agent.position  # Previous position before this action
                agent.update_position_metrics(position, old_pos, turn)

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
                    agent.record_first_combat(turn)  # Track first combat turn
                    combat_count += 1

                # Handle food
                elif "food" in result["reason"].lower():
                    agent.record_food_collected(position)
                    agent.record_first_food(turn)  # Track first food collection turn
                    food_count += 1

            # Record combat events for victims
            for victim_id, attacker_pos in combat_victims.items():
                victim_agent = next((a for a in batch if a.id == victim_id), None)
                if victim_agent:
                    # Get victim's position
                    victim_idx = env.agent_ids.index(victim_id)
                    victim_pos = tuple(int(x) for x in env.agent_positions[victim_idx])
                    victim_agent.record_combat_event(victim_pos, is_attacker=False)
                    victim_agent.record_first_combat(turn)  # Track first combat turn for victims too

            # Apply combat damage, metabolism, and environment updates (vectorized)
            if victims_to_damage:
                env.apply_combat_damage(victims_to_damage)

            # Apply metabolism (VECTORIZED - all agents at once)
            env.apply_metabolism()

            # Update alive status (vectorized)
            dead_ids = env.update_alive_status()

            # Record death turns for agents that just died
            if dead_ids:
                for dead_id in dead_ids:
                    dead_agent = next((a for a in batch if a.id == dead_id), None)
                    if dead_agent:
                        dead_agent.record_death(turn)

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

            # Record turn state for playthrough animation (if requested)
            if record_playthrough:
                turn_state = {
                    "turn": turn,
                    "grid": self._convert_grid_to_symbolic(env.grid.copy()).tolist(),
                    "agents": [{
                        "id": agent.id,
                        "position": [int(env.agent_positions[i][0]), int(env.agent_positions[i][1])],
                        "energy": float(env.agent_energies[i]),
                        "alive": bool(env.agent_alive[i]),
                        "last_action": agent.get_last_action()
                    } for i, agent in enumerate(batch)],
                    "food_collected_this_turn": food_count,
                    "combat_events_this_turn": combat_count
                }
                turn_states.append(turn_state)

        # Return turn states if recording
        return turn_states if record_playthrough else None

    def _print_tournament_scores(self, round_scores: dict, round_food: dict,
                                  round_attacks: dict, round_victims: dict, round_deaths: dict):
        """
        Print a table showing each agent's per-round statistics.

        Args:
            round_scores: {agent_id: [round1_energy, round2_energy, ...]}
            round_food: {agent_id: [round1_food, round2_food, ...]}
            round_attacks: {agent_id: [round1_attacks, round2_attacks, ...]}
            round_victims: {agent_id: [round1_victims, round2_victims, ...]}
            round_deaths: {agent_id: [round1_died, round2_died, ...]}
        """
        if self.rounds_per_generation <= 1:
            return  # No need for table if only 1 round

        print(f"\n{'='*100}")
        print(f"TOURNAMENT SCORES - GENERATION {self.current_generation + 1}")
        print(f"{'='*100}")

        # Sort agents by final average energy
        sorted_agents = sorted(self.population, key=lambda a: a.energy, reverse=True)

        # Print header - now with per-round breakdown
        header = f"{'Rank':<6} {'Agent ID':<12}"
        for r in range(self.rounds_per_generation):
            header += f" {'R' + str(r+1) + ' Enrg':>9}"
        for r in range(self.rounds_per_generation):
            header += f" {'R' + str(r+1) + ' Food':>9}"
        for r in range(self.rounds_per_generation):
            header += f" {'R' + str(r+1) + ' Atk':>8}"
        for r in range(self.rounds_per_generation):
            header += f" {'R' + str(r+1) + ' Vict':>9}"
        header += f" {'Avg Enrg':>10}"
        print(header)
        print("-" * len(header))

        # Print each agent's per-round stats
        for rank, agent in enumerate(sorted_agents, 1):
            row = f"{rank:<6} {agent.id:<12}"

            # Energy per round (with death indicator)
            for r in range(self.rounds_per_generation):
                energy = round_scores[agent.id][r]
                died = round_deaths[agent.id][r]
                if died:
                    row += f" {energy:>7.1f}†"
                else:
                    row += f" {energy:>9.1f}"

            # Food per round
            for r in range(self.rounds_per_generation):
                food = round_food[agent.id][r]
                row += f" {food:>9}"

            # Attacks per round
            for r in range(self.rounds_per_generation):
                attacks = round_attacks[agent.id][r]
                row += f" {attacks:>8}"

            # Victims per round
            for r in range(self.rounds_per_generation):
                victims = round_victims[agent.id][r]
                row += f" {victims:>9}"

            # Average energy
            avg = sum(round_scores[agent.id]) / len(round_scores[agent.id]) if round_scores[agent.id] else 0
            row += f" {avg:>10.1f}"

            print(row)

        print(f"{'='*100}")
        print(f"† = Agent died this round | Enrg = Energy | Atk = Attacks | Vict = Times Attacked")
        print(f"{'='*100}\n")

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

        # Mating pool - only top 25% can breed (increased selection pressure)
        mating_pool_size = max(2, self.population_size // 4)
        mating_pool = self.population[:mating_pool_size]
        print(f"Mating pool: top {mating_pool_size} agents ({mating_pool_size/self.population_size*100:.0f}%)")

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

        # Calculate and save extended analytics
        analytics = self._calculate_extended_analytics()
        self.analytics_history.append(analytics)

        # Calculate and save percentile behavioral metrics
        percentile_metrics = self._calculate_percentile_behavioural_metrics()
        self.percentile_behavioural_history.append(percentile_metrics)

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
        Calculate distribution of actions chosen by agents.
        Only counts deliberate movement actions, not events (food/combat).

        Returns:
            Dict of action -> count
        """
        # Only track deliberate actions (movements chosen by agents)
        action_counts = {
            "MOVE_NORTH": 0,
            "MOVE_SOUTH": 0,
            "MOVE_EAST": 0,
            "MOVE_WEST": 0,
            "INVALID": 0  # Failed to parse LLM output
        }

        for agent in self.population:
            for action_or_event in agent.action_history:
                # Only count movement actions and invalid parses
                # Ignore events like FOOD_COLLECTED, COMBAT_ATTACK, COMBAT_VICTIM
                if action_or_event in action_counts:
                    action_counts[action_or_event] += 1

        return action_counts

    def _calculate_extended_analytics(self) -> dict:
        """
        Calculate extended analytics for understanding agent success patterns.

        Returns:
            Dict with various analytics metrics
        """
        # Divide agents into quartiles by fitness
        sorted_agents = sorted(self.population, key=lambda a: a.energy, reverse=True)
        q1_size = len(sorted_agents) // 4
        top_quartile = sorted_agents[:q1_size]
        bottom_quartile = sorted_agents[-q1_size:]

        def calc_action_ratios(agents):
            """Calculate action ratios for a group of agents."""
            total_actions = sum(len(a.action_history) for a in agents)
            if total_actions == 0:
                return {}

            action_counts = {"MOVE_NORTH": 0, "MOVE_SOUTH": 0, "MOVE_EAST": 0, "MOVE_WEST": 0, "INVALID": 0}
            for agent in agents:
                for action in agent.action_history:
                    if action in action_counts:
                        action_counts[action] += 1

            return {k: v / total_actions for k, v in action_counts.items()}

        # Calculate metrics
        analytics = {
            "generation": self.current_generation,

            # Starting position analysis
            "top_quartile_starting_positions": [
                list(a.starting_position) if a.starting_position else None
                for a in top_quartile
            ],
            "bottom_quartile_starting_positions": [
                list(a.starting_position) if a.starting_position else None
                for a in bottom_quartile
            ],

            # Action patterns by performance
            "top_quartile_action_ratios": calc_action_ratios(top_quartile),
            "bottom_quartile_action_ratios": calc_action_ratios(bottom_quartile),

            # Spatial metrics
            "top_quartile_avg_distance": np.mean([a.total_distance_traveled for a in top_quartile]),
            "bottom_quartile_avg_distance": np.mean([a.total_distance_traveled for a in bottom_quartile]),
            "top_quartile_avg_unique_cells": np.mean([len(a.unique_positions_visited) for a in top_quartile]),
            "bottom_quartile_avg_unique_cells": np.mean([len(a.unique_positions_visited) for a in bottom_quartile]),

            # Temporal metrics
            "top_quartile_avg_first_food_turn": np.mean([a.first_food_turn for a in top_quartile if a.first_food_turn is not None]) if any(a.first_food_turn is not None for a in top_quartile) else None,
            "bottom_quartile_avg_first_food_turn": np.mean([a.first_food_turn for a in bottom_quartile if a.first_food_turn is not None]) if any(a.first_food_turn is not None for a in bottom_quartile) else None,
            "top_quartile_avg_first_combat_turn": np.mean([a.first_combat_turn for a in top_quartile if a.first_combat_turn is not None]) if any(a.first_combat_turn is not None for a in top_quartile) else None,
            "bottom_quartile_avg_first_combat_turn": np.mean([a.first_combat_turn for a in bottom_quartile if a.first_combat_turn is not None]) if any(a.first_combat_turn is not None for a in bottom_quartile) else None,

            # Survival metrics
            "survival_rate": sum(1 for a in self.population if a.is_alive) / len(self.population),
            "top_quartile_survival_rate": sum(1 for a in top_quartile if a.is_alive) / len(top_quartile),
            "bottom_quartile_survival_rate": sum(1 for a in bottom_quartile if a.is_alive) / len(bottom_quartile),
            "avg_death_turn": np.mean([a.death_turn for a in self.population if a.death_turn is not None]) if any(a.death_turn is not None for a in self.population) else None,

            # Combat & food efficiency
            "top_quartile_combat_ratio": np.mean([(a.combat_attacks + a.combat_victims) / max(len(a.action_history), 1) for a in top_quartile]),
            "bottom_quartile_combat_ratio": np.mean([(a.combat_attacks + a.combat_victims) / max(len(a.action_history), 1) for a in bottom_quartile]),
            "top_quartile_food_efficiency": np.mean([a.food_collected / max(a.total_distance_traveled, 1) for a in top_quartile]),
            "bottom_quartile_food_efficiency": np.mean([a.food_collected / max(a.total_distance_traveled, 1) for a in bottom_quartile]),
        }

        return analytics

    def _calculate_percentile_behavioural_metrics(self) -> dict:
        """
        Calculate comprehensive behavioral metrics for different performance percentiles.
        Tracks how different performance tiers behave across generations.

        Returns:
            Dict with metrics for top 5%, top 10%, quartiles, and bottom tiers
        """
        # Divide agents into performance tiers by fitness
        sorted_agents = sorted(self.population, key=lambda a: a.energy, reverse=True)
        n = len(sorted_agents)

        # Calculate tier sizes
        p5_size = max(1, n // 20)   # Top 5%
        p10_size = max(1, n // 10)  # Top 10%
        q_size = max(1, n // 4)     # Quartile (25%)

        tiers = {
            "top_5pct": sorted_agents[:p5_size],
            "top_10pct": sorted_agents[:p10_size],
            "Q1_top_25pct": sorted_agents[:q_size],
            "Q2_upper_mid": sorted_agents[q_size:2*q_size],
            "Q3_lower_mid": sorted_agents[2*q_size:3*q_size],
            "Q4_bottom_25pct": sorted_agents[3*q_size:],
            "bottom_10pct": sorted_agents[-p10_size:],
            "bottom_5pct": sorted_agents[-p5_size:],
        }

        def analyze_tier(agents, tier_name):
            """Analyze behavior for a performance tier."""
            if not agents:
                return None

            # Action distribution
            total_actions = sum(len(a.action_history) for a in agents)
            action_counts = {"MOVE_NORTH": 0, "MOVE_SOUTH": 0, "MOVE_EAST": 0, "MOVE_WEST": 0, "INVALID": 0}
            for agent in agents:
                for action in agent.action_history:
                    if action in action_counts:
                        action_counts[action] += 1

            action_percentages = {k: (v / total_actions * 100) if total_actions > 0 else 0
                                 for k, v in action_counts.items()}

            # Performance metrics
            avg_energy = np.mean([a.energy for a in agents])
            avg_food = np.mean([a.food_collected for a in agents])
            avg_attacks = np.mean([a.combat_attacks for a in agents])
            avg_victims = np.mean([a.combat_victims for a in agents])

            # Behavioral metrics
            avg_distance = np.mean([a.total_distance_traveled for a in agents])
            avg_unique_cells = np.mean([len(a.unique_positions_visited) for a in agents])
            survival_rate = sum(1 for a in agents if a.is_alive) / len(agents) * 100

            # Efficiency metrics
            food_per_distance = np.mean([a.food_collected / max(a.total_distance_traveled, 1) for a in agents])
            energy_per_action = avg_energy / max(total_actions / len(agents), 1)

            # Combat behavior
            combat_engagement_rate = sum(1 for a in agents if (a.combat_attacks + a.combat_victims) > 0) / len(agents) * 100
            avg_combat_ratio = np.mean([(a.combat_attacks + a.combat_victims) / max(len(a.action_history), 1) for a in agents])

            # Aggression vs defensive
            total_attacks = sum(a.combat_attacks for a in agents)
            total_victimizations = sum(a.combat_victims for a in agents)
            aggression_score = (total_attacks - total_victimizations) / max(total_attacks + total_victimizations, 1)

            return {
                "tier": tier_name,
                "size": len(agents),
                "percentile_range": tier_name,  # e.g., "top_5pct", "Q1_top_25pct"

                # Action preferences (%)
                "move_north_pct": action_percentages["MOVE_NORTH"],
                "move_south_pct": action_percentages["MOVE_SOUTH"],
                "move_east_pct": action_percentages["MOVE_EAST"],
                "move_west_pct": action_percentages["MOVE_WEST"],
                "invalid_pct": action_percentages["INVALID"],

                # Performance
                "avg_energy": float(avg_energy),
                "avg_food": float(avg_food),
                "avg_attacks": float(avg_attacks),
                "avg_victims": float(avg_victims),

                # Behavior
                "avg_distance_traveled": float(avg_distance),
                "avg_unique_cells_visited": float(avg_unique_cells),
                "survival_rate_pct": float(survival_rate),

                # Efficiency
                "food_per_distance": float(food_per_distance),
                "energy_per_action": float(energy_per_action),

                # Combat style
                "combat_engagement_rate_pct": float(combat_engagement_rate),
                "combat_ratio": float(avg_combat_ratio),
                "aggression_score": float(aggression_score),  # -1 (defensive) to +1 (aggressive)
            }

        metrics = {
            "generation": self.current_generation,
            "population_size": n,
            "tiers": {
                name: analyze_tier(agents, name)
                for name, agents in tiers.items()
            }
        }

        return metrics

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
            "turn_by_turn_history": self.turn_by_turn_history,  # Full playthrough data for final generation
            "genetic_history": self.genetic_history,
            "behavioural_history": self.behavioural_history,
            "analytics_history": self.analytics_history,  # Extended analytics
            "percentile_behavioural_history": self.percentile_behavioural_history,  # Percentile-specific behavioral metrics
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
