"""
Vectorized Environment for GPU-efficient batch processing.

This refactored environment prioritizes:
- NumPy vectorized operations (no Python loops for physics/vision)
- Batch perception extraction (all agents at once)
- vLLM-ready inference structure (returns BatchInferenceRequest)
- Integer grid encoding for fast array operations
"""

import json
import os
import numpy as np
from typing import Dict, List, Tuple, Optional
from config_loader import GRID_SYMBOLS
from Environment.map_generator import MapGenerator
from Environment.generation_manager import GenerationManager
from Environment.batch_types import BatchInferenceRequest, BatchPerception


# Integer encoding for grid symbols (for fast NumPy operations)
GRID_INT_ENCODING = {
    GRID_SYMBOLS["empty"]: 0,
    GRID_SYMBOLS["wall"]: 1,
    GRID_SYMBOLS["food"]: 2,
    GRID_SYMBOLS["agent"]: 3,
}

# Reverse mapping
INT_TO_SYMBOL = {v: k for k, v in GRID_INT_ENCODING.items()}


class VectorizedEnvironment:
    """
    GPU-optimized environment using NumPy vectorization.

    Key optimizations:
    - Grid stored as int array (faster than string symbols)
    - Batch perception extraction with array slicing (no loops)
    - Returns BatchInferenceRequest for vLLM compatibility
    - Vectorized physics updates where possible
    """

    def __init__(self, generation: int = 0):
        """
        Initialise vectorized environment.

        Args:
            generation: Current generation (affects difficulty curriculum)
        """
        self.grid_size = 16
        self.generation = generation

        # Load prompt configuration
        config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "prompts.json")
        with open(config_path, 'r') as f:
            self.prompt_config = json.load(f)

        # Managers
        self.generation_manager = GenerationManager()
        self.generation_manager.set_generation(generation)
        self.map_generator = MapGenerator(
            grid_size=self.grid_size,
            wall_symbol=GRID_SYMBOLS["wall"],
            empty_symbol=GRID_SYMBOLS["empty"]
        )

        # Grid state (INTEGER encoding for fast operations)
        self.grid: np.ndarray = None  # Shape: (16, 16), dtype=int

        # Agent tracking (vectorized where possible)
        self.agent_ids: List[str] = []
        self.agent_positions: np.ndarray = None  # Shape: (N, 2), dtype=int
        self.agent_energies: np.ndarray = None  # Shape: (N,), dtype=float
        self.agent_alive: np.ndarray = None  # Shape: (N,), dtype=bool

        # Position lookup (for O(1) collision detection)
        self.position_to_agent_idx: Dict[Tuple[int, int], int] = {}

        # Energy tracking
        self.total_energy_spawned = 0
        self.agent_energy_collected: np.ndarray = None  # Shape: (N,), dtype=float
        self.combat_energy_transferred: np.ndarray = None  # Shape: (N,), dtype=float

        # Turn counter
        self.current_turn = 0

        # Initialise
        self.reset()

    def reset(self):
        """Reset environment for new batch."""
        # Generate map (returns string grid)
        wall_density = np.random.uniform(0.10, 0.15)
        string_grid = self.map_generator.generate_map(wall_density)

        # Convert to integer grid
        self.grid = self._string_grid_to_int(string_grid)

        # Clear agent tracking
        self.agent_ids.clear()
        self.agent_positions = np.empty((0, 2), dtype=int)
        self.agent_energies = np.empty(0, dtype=float)
        self.agent_alive = np.empty(0, dtype=bool)
        self.agent_energy_collected = np.empty(0, dtype=float)
        self.combat_energy_transferred = np.empty(0, dtype=float)
        self.position_to_agent_idx.clear()

        # Reset turn
        self.current_turn = 0

        # Spawn food
        self._spawn_initial_food()

    def _string_grid_to_int(self, string_grid: np.ndarray) -> np.ndarray:
        """
        Convert string grid to integer encoding.

        Args:
            string_grid: (16, 16) array of string symbols

        Returns:
            (16, 16) array of integers
        """
        int_grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        for symbol, int_val in GRID_INT_ENCODING.items():
            int_grid[string_grid == symbol] = int_val
        return int_grid

    def _int_grid_to_string(self, int_grid: np.ndarray) -> np.ndarray:
        """Convert integer grid back to string symbols (for compatibility)."""
        string_grid = np.full((self.grid_size, self.grid_size), GRID_SYMBOLS["empty"], dtype='<U1')
        for int_val, symbol in INT_TO_SYMBOL.items():
            string_grid[int_grid == int_val] = symbol
        return string_grid

    def _spawn_initial_food(self):
        """Spawn initial food based on curriculum."""
        num_food = self.generation_manager.get_initial_food_count(self.grid_size)

        # Find valid positions (empty cells)
        empty_mask = (self.grid == GRID_INT_ENCODING[GRID_SYMBOLS["empty"]])
        empty_positions = np.argwhere(empty_mask)

        if len(empty_positions) < num_food:
            num_food = len(empty_positions)

        # Randomly select positions
        selected_indices = np.random.choice(len(empty_positions), size=num_food, replace=False)
        food_positions = empty_positions[selected_indices]

        # Place food (vectorized)
        self.grid[food_positions[:, 0], food_positions[:, 1]] = GRID_INT_ENCODING[GRID_SYMBOLS["food"]]

        # Track energy
        self.total_energy_spawned = num_food * 30

        # Suppress verbose output (called for each batch)

    def register_agents(self, agent_ids: List[str], initial_energy: float = 100.0):
        """
        Register multiple agents at once (vectorized).

        Args:
            agent_ids: List of agent identifiers
            initial_energy: Starting energy for all agents
        """
        n = len(agent_ids)

        # Find valid spawn positions
        empty_mask = (self.grid == GRID_INT_ENCODING[GRID_SYMBOLS["empty"]])
        empty_positions = np.argwhere(empty_mask)

        if len(empty_positions) < n:
            raise ValueError(f"Not enough empty positions for {n} agents")

        # Randomly select spawn positions
        selected_indices = np.random.choice(len(empty_positions), size=n, replace=False)
        spawn_positions = empty_positions[selected_indices]

        # Store agent data
        self.agent_ids = agent_ids.copy()
        self.agent_positions = spawn_positions.astype(int)
        self.agent_energies = np.full(n, initial_energy, dtype=float)
        self.agent_alive = np.ones(n, dtype=bool)
        self.agent_energy_collected = np.zeros(n, dtype=float)
        self.combat_energy_transferred = np.zeros(n, dtype=float)

        # Build position lookup
        self.position_to_agent_idx = {
            tuple(pos): idx for idx, pos in enumerate(self.agent_positions)
        }

        # Mark positions on grid
        self.grid[spawn_positions[:, 0], spawn_positions[:, 1]] = GRID_INT_ENCODING[GRID_SYMBOLS["agent"]]

    def get_batch_perception(self, active_agent_indices: Optional[np.ndarray] = None) -> BatchPerception:
        """
        Extract 5x5 perception grids for multiple agents at once (VECTORIZED).

        Args:
            active_agent_indices: Indices of agents to get perception for (None = all alive)

        Returns:
            BatchPerception object with vectorized data
        """
        if active_agent_indices is None:
            active_agent_indices = np.where(self.agent_alive)[0]

        n = len(active_agent_indices)
        perception_grids = np.zeros((n, 5, 5), dtype=int)

        # Pad grid for boundary handling
        padded_grid = np.pad(self.grid, pad_width=2, mode='constant',
                            constant_values=GRID_INT_ENCODING[GRID_SYMBOLS["wall"]])

        # Extract 5x5 views (vectorized with fancy indexing)
        for i, agent_idx in enumerate(active_agent_indices):
            x, y = self.agent_positions[agent_idx]
            # Account for padding offset (+2)
            perception_grids[i] = padded_grid[x:x+5, y:y+5]

        # Get agent data
        agent_ids = [self.agent_ids[idx] for idx in active_agent_indices]
        positions = self.agent_positions[active_agent_indices]
        energies = self.agent_energies[active_agent_indices]

        # Check if agents are on food
        on_food = np.array([
            self.grid[tuple(pos)] == GRID_INT_ENCODING[GRID_SYMBOLS["food"]]
            for pos in positions
        ], dtype=bool)

        return BatchPerception(
            grids=perception_grids,
            agent_ids=agent_ids,
            positions=positions,
            energies=energies,
            on_food=on_food
        )

    def prepare_batch_inference_request(
        self,
        perception: BatchPerception,
        adapter_mapping: Dict[str, str],
        recent_actions: Optional[Dict[str, list]] = None
    ) -> BatchInferenceRequest:
        """
        Prepare batch inference request for inference engine.

        This method does NOT call the LLM directly. Instead, it returns
        a structured request that can be passed to a batch inference engine.

        Args:
            perception: Batch perception data
            adapter_mapping: {agent_id -> adapter_path/id}
            recent_actions: Optional dict of recent action lists (up to 3 moves)

        Returns:
            BatchInferenceRequest ready for inference processing
        """
        prompts = []
        adapter_ids = []

        for i, agent_id in enumerate(perception.agent_ids):
            # Build prompt (using cached static parts)
            prompt = self._build_prompt(
                perception.grids[i],
                perception.energies[i],
                perception.on_food[i],
                recent_actions.get(agent_id, []) if recent_actions else []
            )
            prompts.append(prompt)
            adapter_ids.append(adapter_mapping[agent_id])

        return BatchInferenceRequest(
            prompts=prompts,
            adapter_ids=adapter_ids,
            agent_ids=perception.agent_ids,
            metadata={
                "positions": perception.positions.tolist(),
                "energies": perception.energies.tolist(),
                "turn": self.current_turn
            }
        )

    def _build_prompt(
        self,
        perception_grid: np.ndarray,
        energy: float,
        on_food: bool,
        recent_moves: list
    ) -> str:
        """
        Build prompt for a single agent (optimized with caching).

        Args:
            perception_grid: (5, 5) integer grid
            energy: Current energy
            on_food: Whether agent is on food
            recent_moves: List of recent actions (up to 3, most recent last)

        Returns:
            Formatted prompt string
        """
        # Get system prompt from config
        system_prompt = self.prompt_config["SYSTEM_PROMPT"]

        # Format perception (convert int grid to directions)
        perception_str = self._format_perception_directions(perception_grid)

        # Energy status from config
        energy_thresholds = self.prompt_config["ENERGY_THRESHOLDS"]
        energy_status = (
            energy_thresholds["low_label"]
            if energy <= energy_thresholds["low_threshold"]
            else energy_thresholds["ok_label"]
        )

        # Build prompt from config templates
        prompt = system_prompt
        prompt += self.prompt_config["PROMPT_FORMAT"]["energy_template"].format(
            energy=int(energy), status=energy_status
        )
        prompt += self.prompt_config["PROMPT_FORMAT"]["vision_template"].format(
            vision=perception_str
        )

        if recent_moves:
            # Format recent moves: "MOVE_NORTH, MOVE_EAST, MOVE_SOUTH" (oldest to newest)
            moves_str = ", ".join(recent_moves)
            prompt += self.prompt_config["PROMPT_FORMAT"]["recent_moves_template"].format(
                moves=moves_str
            )

        prompt += self.prompt_config["PROMPT_FORMAT"]["action_prompt"]

        return prompt

    def _format_perception_directions(self, grid: np.ndarray) -> str:
        """
        Convert 5x5 integer grid to directional descriptions with line-of-sight.

        Line of sight: Only show the first object in each direction (walls/agents/food block vision).
        Shows 8 directions: N, NE, E, SE, S, SW, W, NW

        Args:
            grid: (5, 5) integer array

        Returns:
            Formatted string with visible objects in each direction
        """
        center = 2

        # Define 8 direction rays (ordered by distance from center)
        # Each ray: list of (row, col) positions to check in order
        rays = {
            "NORTH": [(center - 1, center), (center - 2, center)],
            "SOUTH": [(center + 1, center), (center + 2, center)],
            "EAST": [(center, center + 1), (center, center + 2)],
            "WEST": [(center, center - 1), (center, center - 2)],
            "NORTH-EAST": [(center - 1, center + 1), (center - 2, center + 2)],
            "NORTH-WEST": [(center - 1, center - 1), (center - 2, center - 2)],
            "SOUTH-EAST": [(center + 1, center + 1), (center + 2, center + 2)],
            "SOUTH-WEST": [(center + 1, center - 1), (center + 2, center - 2)]
        }

        # Object names
        obj_names = {
            GRID_INT_ENCODING[GRID_SYMBOLS["food"]]: "FOOD",
            GRID_INT_ENCODING[GRID_SYMBOLS["wall"]]: "WALL",
            GRID_INT_ENCODING[GRID_SYMBOLS["agent"]]: "AGENT",
            GRID_INT_ENCODING[GRID_SYMBOLS["empty"]]: None  # Empty = no object
        }

        # Cast rays in each direction (line of sight)
        visible = []
        for direction, ray_positions in rays.items():
            for row, col in ray_positions:
                # Check bounds
                if not (0 <= row < 5 and 0 <= col < 5):
                    break

                cell = grid[row, col]
                obj_name = obj_names.get(cell)

                # If we hit an object (not empty), report it and stop looking further
                if obj_name is not None:
                    visible.append(f"{obj_name} {direction}")
                    break  # Line of sight blocked

        # Format output
        if not visible:
            return "All clear"

        return ", ".join(visible)

    def apply_actions(
        self,
        agent_indices: np.ndarray,
        actions: List[str]
    ) -> Dict[str, Dict]:
        """
        Apply actions for multiple agents (vectorized where possible).

        Args:
            agent_indices: Array of agent indices
            actions: List of action strings

        Returns:
            Dict mapping agent_id -> result dict
        """
        results = {}

        # Direction vectors
        direction_map = {
            "MOVE_NORTH": (-1, 0),
            "MOVE_SOUTH": (1, 0),
            "MOVE_EAST": (0, 1),
            "MOVE_WEST": (0, -1)
        }

        for agent_idx, action in zip(agent_indices, actions):
            agent_id = self.agent_ids[agent_idx]

            if not self.agent_alive[agent_idx]:
                results[agent_id] = {"success": False, "energy_change": 0, "reason": "Agent is dead"}
                continue

            # Parse action
            action_upper = action.upper()
            if action_upper not in direction_map:
                results[agent_id] = {"success": False, "energy_change": -2, "reason": "Invalid action"}
                self.agent_energies[agent_idx] -= 2
                continue

            # Calculate new position
            dx, dy = direction_map[action_upper]
            old_pos = self.agent_positions[agent_idx]
            new_x, new_y = old_pos[0] + dx, old_pos[1] + dy

            # Bounds check
            if not (0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size):
                results[agent_id] = {"success": False, "energy_change": -2, "reason": "Hit wall (boundary)"}
                self.agent_energies[agent_idx] -= 2
                continue

            new_pos_tuple = (new_x, new_y)
            target_cell = self.grid[new_x, new_y]

            # Check for wall
            if target_cell == GRID_INT_ENCODING[GRID_SYMBOLS["wall"]]:
                results[agent_id] = {"success": False, "energy_change": -2, "reason": "Hit wall"}
                self.agent_energies[agent_idx] -= 2
                continue

            # Check for agent collision (COMBAT)
            if new_pos_tuple in self.position_to_agent_idx:
                victim_idx = self.position_to_agent_idx[new_pos_tuple]
                victim_id = self.agent_ids[victim_idx]

                # Combat: attacker gains +5, victim loses -10
                self.agent_energies[agent_idx] += 5
                self.agent_energy_collected[agent_idx] += 5
                self.combat_energy_transferred[agent_idx] += 5

                results[agent_id] = {
                    "success": False,
                    "energy_change": 5,
                    "reason": f"Attacked agent {victim_id}",
                    "combat_victim": victim_id,
                    "victim_damage": -10
                }
                continue

            # Move is valid - update position
            # Clear old position on grid
            self.grid[old_pos[0], old_pos[1]] = GRID_INT_ENCODING[GRID_SYMBOLS["empty"]]
            del self.position_to_agent_idx[tuple(old_pos)]

            # Check for food
            energy_change = 0
            reason = "Moved successfully"
            if target_cell == GRID_INT_ENCODING[GRID_SYMBOLS["food"]]:
                energy_change = 30
                reason = "Collected food (+30 energy)"
                self.agent_energy_collected[agent_idx] += 30

            # Update position
            self.agent_positions[agent_idx] = [new_x, new_y]
            self.position_to_agent_idx[new_pos_tuple] = agent_idx

            # Mark new position on grid
            self.grid[new_x, new_y] = GRID_INT_ENCODING[GRID_SYMBOLS["agent"]]

            # Apply energy change
            self.agent_energies[agent_idx] += energy_change

            results[agent_id] = {
                "success": True,
                "energy_change": energy_change,
                "reason": reason
            }

        return results

    def apply_combat_damage(self, victim_ids: List[str], damage: int = -10):
        """
        Apply combat damage to victims (vectorized).

        Args:
            victim_ids: List of victim agent IDs
            damage: Damage amount (negative)
        """
        for victim_id in victim_ids:
            try:
                victim_idx = self.agent_ids.index(victim_id)
                self.agent_energies[victim_idx] += damage
            except ValueError:
                continue

    def apply_metabolism(self):
        """Apply -2 energy to all alive agents (vectorized)."""
        self.agent_energies[self.agent_alive] -= 2

    def update_alive_status(self) -> List[str]:
        """
        Update alive status based on energy (vectorized).

        Returns:
            List of agent IDs that died this turn
        """
        # Find newly dead agents
        newly_dead = self.agent_alive & (self.agent_energies <= 0)
        dead_indices = np.where(newly_dead)[0]

        dead_ids = []
        for idx in dead_indices:
            agent_id = self.agent_ids[idx]
            dead_ids.append(agent_id)

            # Remove from grid
            pos = tuple(self.agent_positions[idx])
            if pos in self.position_to_agent_idx:
                self.grid[pos[0], pos[1]] = GRID_INT_ENCODING[GRID_SYMBOLS["empty"]]
                del self.position_to_agent_idx[pos]

        # Update alive status
        self.agent_alive = self.agent_energies > 0

        return dead_ids

    def step(self):
        """Environment step (food regrowth, turn increment)."""
        self.current_turn += 1

        # Food regrowth based on curriculum
        should_regrow, amount = self.generation_manager.should_regrow_food(self.current_turn)

        if should_regrow:
            if isinstance(amount, float):  # Percentage regrowth
                # Count current food
                food_count = np.sum(self.grid == GRID_INT_ENCODING[GRID_SYMBOLS["food"]])
                max_food = self.generation_manager.get_initial_food_count(self.grid_size)
                new_food = int((max_food - food_count) * amount)
            else:  # Absolute count
                new_food = amount

            if new_food > 0:
                # Find empty cells (not walls, agents, or existing food)
                empty_mask = (self.grid == GRID_INT_ENCODING[GRID_SYMBOLS["empty"]])
                empty_positions = np.argwhere(empty_mask)

                if len(empty_positions) > 0:
                    num_spawn = min(new_food, len(empty_positions))
                    selected = np.random.choice(len(empty_positions), size=num_spawn, replace=False)
                    spawn_positions = empty_positions[selected]

                    self.grid[spawn_positions[:, 0], spawn_positions[:, 1]] = GRID_INT_ENCODING[GRID_SYMBOLS["food"]]
                    self.total_energy_spawned += num_spawn * 15

    def get_stats(self) -> Dict:
        """Get environment statistics."""
        return {
            "generation": self.generation,
            "turn": self.current_turn,
            "food_remaining": int(np.sum(self.grid == GRID_INT_ENCODING[GRID_SYMBOLS["food"]])),
            "agents_alive": int(np.sum(self.agent_alive)),
            "total_agents": len(self.agent_ids),
            "total_energy_spawned": self.total_energy_spawned
        }

    def calculate_fitness(self, agent_id: str) -> float:
        """
        Calculate normalized fitness for an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            Fitness percentage (0-100+)
        """
        try:
            agent_idx = self.agent_ids.index(agent_id)
            energy_collected = self.agent_energy_collected[agent_idx]

            if self.total_energy_spawned == 0:
                return 0.0

            return (energy_collected / self.total_energy_spawned) * 100.0
        except ValueError:
            return 0.0

    def get_agent_data(self) -> Dict:
        """
        Get all agent data for external processing.

        Returns:
            Dict with agent IDs, positions, energies, alive status
        """
        return {
            "ids": self.agent_ids.copy(),
            "positions": self.agent_positions.copy(),
            "energies": self.agent_energies.copy(),
            "alive": self.agent_alive.copy(),
            "energy_collected": self.agent_energy_collected.copy()
        }
