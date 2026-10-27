"""
Lightweight Agent for vectorized batch processing.

This agent is designed to work with VectorizedEnvironment and BatchInferenceEngine.
It stores minimal state and relies on the environment for physics/perception.
"""

import uuid
from LoraBrain.lora_brain import LoraBrain


class VectorizedAgent:
    """
    Lightweight agent for batch processing.

    This agent:
    - Stores only brain (LoRA weights) and metadata
    - Does NOT handle physics or perception directly
    - Works with vectorized environment for batch operations
    - Compatible with batch inference architecture
    """

    def __init__(self, brain: LoraBrain, grid_size: int = 16):
        """
        Initialise agent.

        Args:
            brain: LoraBrain with LoRA weights
            grid_size: Grid size (for compatibility)
        """
        self.id = str(uuid.uuid4())[:8]
        self.brain = brain
        self.grid_size = grid_size

        # Lightweight tracking (main state in environment)
        self.action_history: list = []
        self.combat_attacks: int = 0  # Times this agent attacked others
        self.combat_victims: int = 0  # Times this agent was attacked
        self.food_collected: int = 0

        # Path tracking for visualization
        self.path: list = []  # [(action, position), ...]

        # These will be synced from environment
        self.energy = 100.0
        self.is_alive = True
        self.position = None

        # Fitness tracking (normalized percentage for cross-generation comparison)
        self.fitness_percentage = 100.0

        # Extended metrics for analysis
        self.starting_position = None  # (x, y) where agent spawned
        self.unique_positions_visited = set()  # Set of (x, y) positions
        self.first_food_turn = None  # Turn number when first food collected
        self.first_combat_turn = None  # Turn number when first combat occurred
        self.death_turn = None  # Turn number when agent died
        self.total_distance_traveled = 0.0  # Cumulative movement distance

    def sync_from_environment(self, energy: float, alive: bool, position: tuple):
        """
        Sync state from vectorized environment.

        Args:
            energy: Current energy
            alive: Alive status
            position: Current position
        """
        self.energy = energy
        self.is_alive = alive
        self.position = position

    def record_action(self, action: str, position: tuple = None):
        """
        Record an action in history and path.

        Args:
            action: Action taken
            position: Position where action was taken (for visualization)
        """
        self.action_history.append(action)
        if position is not None:
            self.path.append((action, position))

    def record_food_collected(self, position: tuple = None):
        """
        Increment food collection counter and mark in path.

        Args:
            position: Position where food was collected (for visualization)
        """
        self.food_collected += 1
        if position is not None:
            self.path.append(("FOOD_COLLECTED", position))

    def record_combat_event(self, position: tuple = None, is_attacker: bool = True):
        """
        Increment combat event counter and mark in path.

        Args:
            position: Position where combat occurred (for visualization)
            is_attacker: True if this agent attacked, False if this agent was attacked
        """
        if is_attacker:
            self.combat_attacks += 1
        else:
            self.combat_victims += 1

        if position is not None:
            event_type = "COMBAT_ATTACK" if is_attacker else "COMBAT_VICTIM"
            self.path.append((event_type, position))

    def get_last_action(self) -> str:
        """Get last action (for context in next turn)."""
        return self.action_history[-1] if self.action_history else None

    def get_fitness_metrics(self) -> dict:
        """
        Get fitness-related metrics.

        Returns:
            Dict with energy, food, combat stats
        """
        return {
            "energy": self.energy,
            "food_collected": self.food_collected,
            "combat_attacks": self.combat_attacks,
            "combat_victims": self.combat_victims,
            "alive": self.is_alive
        }

    def reset_for_generation(self):
        """Reset agent for new generation."""
        self.action_history.clear()
        self.path.clear()
        self.combat_attacks = 0
        self.combat_victims = 0
        self.food_collected = 0
        self.energy = 100.0
        self.is_alive = True
        self.position = None
        # Reset extended metrics
        self.starting_position = None
        self.unique_positions_visited = set()
        self.first_food_turn = None
        self.first_combat_turn = None
        self.death_turn = None
        self.total_distance_traveled = 0.0

    def reset_for_round(self):
        """
        Reset agent for new round within generation.
        Keeps cumulative food/combat stats across rounds.
        """
        self.action_history.clear()
        self.path.clear()
        self.energy = 100.0
        self.is_alive = True
        self.position = None
        # Note: Do NOT reset food_collected, combat_attacks, combat_victims
        # These accumulate across rounds within a generation for tournament scoring
        # Reset per-round metrics
        self.starting_position = None
        self.unique_positions_visited = set()
        self.first_food_turn = None
        self.first_combat_turn = None
        self.death_turn = None
        self.total_distance_traveled = 0.0

    def get_path_copy(self) -> list:
        """Get a copy of the path for visualization."""
        return self.path.copy()

    def update_position_metrics(self, new_position: tuple, old_position: tuple = None, turn: int = 0):
        """
        Update position-related metrics.

        Args:
            new_position: New (x, y) position
            old_position: Previous (x, y) position (for distance calculation)
            turn: Current turn number
        """
        # Set starting position on first call
        if self.starting_position is None:
            self.starting_position = new_position

        # Track unique positions
        self.unique_positions_visited.add(new_position)

        # Calculate distance traveled
        if old_position is not None and old_position != new_position:
            # Manhattan distance
            dx = abs(new_position[0] - old_position[0])
            dy = abs(new_position[1] - old_position[1])
            self.total_distance_traveled += (dx + dy)

    def record_first_food(self, turn: int):
        """Record the turn when first food was collected."""
        if self.first_food_turn is None:
            self.first_food_turn = turn

    def record_first_combat(self, turn: int):
        """Record the turn when first combat occurred."""
        if self.first_combat_turn is None:
            self.first_combat_turn = turn

    def record_death(self, turn: int):
        """Record the turn when agent died."""
        if self.death_turn is None:
            self.death_turn = turn

    def __repr__(self):
        """String representation."""
        status = "ALIVE" if self.is_alive else "DEAD"
        return f"Agent({self.id}, energy={self.energy:.1f}, status={status})"
