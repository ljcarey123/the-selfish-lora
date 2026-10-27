"""
Manages difficulty curriculum across generations.

This class calculates generation-dependent parameters like food density
and regrowth rates to create a progressive difficulty curve.
"""

import numpy as np


class GenerationManager:
    """
    Handles generation-based difficulty progression through curriculum learning.

    The difficulty increases across three phases:
    - Phase 1 (Gen 0-10): High food density, generous regrowth
    - Phase 2 (Gen 11-30): Medium density, trickle regrowth
    - Phase 3 (Gen 31+): Low density, scarcity regrowth
    """

    def __init__(self):
        """Initialise the generation manager."""
        self.current_generation = 0

    def set_generation(self, generation: int):
        """
        Set the current generation number.

        Args:
            generation: The generation number (0-indexed)
        """
        self.current_generation = generation

    def get_food_density(self) -> float:
        """
        Calculate food density for the current generation.

        Formula: Density = max(0.10, 0.50 - (Generation * 0.01))
        - Gen 0: 50% of tiles are food
        - Gen 40+: 10% of tiles are food (minimum)

        Returns:
            float: Food density as a percentage (0.10 to 0.50)
        """
        density = 0.50 - (self.current_generation * 0.01)
        return max(0.10, density)

    def get_regrowth_phase(self) -> int:
        """
        Determine which regrowth phase the current generation is in.

        Returns:
            int: Phase number (1, 2, or 3)
        """
        if self.current_generation <= 10:
            return 1  # High regrowth
        elif self.current_generation <= 30:
            return 2  # Trickle regrowth
        else:
            return 3  # Scarcity

    def should_regrow_food(self, turn: int) -> tuple[bool, int]:
        """
        Determine if food should regrow on this turn and how much.

        Phase 1 (Gen 0-10): Every 3 turns, respawn 15% of total food
        Phase 2 (Gen 11-30): Every turn, 10% chance to spawn 1 food
        Phase 3 (Gen 31+): Every turn, 5% chance to spawn 1 food

        Args:
            turn: Current turn number in the generation

        Returns:
            tuple: (should_regrow, amount_to_regrow)
                - should_regrow: Boolean indicating if regrowth should occur
                - amount_to_regrow: Number of food items to spawn (or percentage)
        """
        phase = self.get_regrowth_phase()

        if phase == 1:
            # Every 3 turns, regrow 15% of total food
            if turn > 0 and turn % 3 == 0:
                return (True, 0.15)  # Return percentage for phase 1
            return (False, 0)

        elif phase == 2:
            # 10% chance per turn to spawn 1 food
            if np.random.random() < 0.10:
                return (True, 1)
            return (False, 0)

        else:  # phase == 3
            # 5% chance per turn to spawn 1 food
            if np.random.random() < 0.05:
                return (True, 1)
            return (False, 0)

    def get_initial_food_count(self, grid_size: int) -> int:
        """
        Calculate the initial number of food items for the current generation.

        Args:
            grid_size: Size of the grid (grid_size x grid_size)

        Returns:
            int: Number of food items to place initially
        """
        total_tiles = grid_size * grid_size
        density = self.get_food_density()
        return int(total_tiles * density)

    def get_phase_description(self) -> str:
        """
        Get a human-readable description of the current phase.

        Returns:
            str: Description of the current difficulty phase
        """
        phase = self.get_regrowth_phase()
        density = self.get_food_density()

        if phase == 1:
            return f"Phase 1 (HIGH REGROWTH) - {density*100:.1f}% food density, regrows 15% every 3 turns"
        elif phase == 2:
            return f"Phase 2 (TRICKLE REGROWTH) - {density*100:.1f}% food density, 10% chance per turn"
        else:
            return f"Phase 3 (SCARCITY) - {density*100:.1f}% food density, 5% chance per turn"