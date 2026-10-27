"""
Generates validated grid maps with walls and food placement.

This module handles procedural map generation with connectivity validation
to ensure all areas of the map are reachable.
"""

import numpy as np
from collections import deque


class MapGenerator:
    """
    Generates random maps with walls, ensuring connectivity via flood-fill.

    The generator places walls randomly but validates that no areas become
    sealed off, ensuring agents can theoretically reach all parts of the map.
    """

    def __init__(self, grid_size: int = 16, wall_symbol: str = "W", empty_symbol: str = "."):
        """
        Initialise the map generator.

        Args:
            grid_size: Size of the square grid (default 16x16)
            wall_symbol: Symbol representing walls
            empty_symbol: Symbol representing empty tiles
        """
        self.grid_size = grid_size
        self.wall_symbol = wall_symbol
        self.empty_symbol = empty_symbol

    def generate_map(self, wall_density: float = 0.125) -> np.ndarray:
        """
        Generate a random map with walls, validated for connectivity.

        The method will retry map generation up to 50 times if connectivity
        validation fails, ensuring a valid map is always returned.

        Args:
            wall_density: Percentage of tiles that should be walls (0.10-0.15 recommended)

        Returns:
            np.ndarray: A grid_size x grid_size array with walls and empty spaces
        """
        max_attempts = 50

        for attempt in range(max_attempts):
            grid = self._generate_random_walls(wall_density)

            if self._is_connected(grid):
                return grid

        # Fallback: return empty grid if we can't generate a valid one
        print(f"Warning: Could not generate connected map after {max_attempts} attempts. Returning empty grid.")
        return np.full((self.grid_size, self.grid_size), self.empty_symbol, dtype=object)

    def _generate_random_walls(self, wall_density: float) -> np.ndarray:
        """
        Generate a grid with randomly placed walls.

        Args:
            wall_density: Percentage of tiles that should be walls

        Returns:
            np.ndarray: Grid with walls randomly placed
        """
        grid = np.full((self.grid_size, self.grid_size), self.empty_symbol, dtype=object)

        num_walls = int(self.grid_size * self.grid_size * wall_density)

        for _ in range(num_walls):
            x, y = np.random.randint(0, self.grid_size, 2)
            grid[x, y] = self.wall_symbol

        return grid

    def _is_connected(self, grid: np.ndarray) -> bool:
        """
        Validate that all empty tiles are connected via flood-fill.

        Uses BFS (breadth-first search) to check if all empty tiles
        can be reached from a starting empty tile.

        Args:
            grid: The grid to validate

        Returns:
            bool: True if all empty tiles are reachable, False otherwise
        """
        # Find the first empty tile as starting point
        start_pos = None
        empty_count = 0

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if grid[i, j] != self.wall_symbol:
                    empty_count += 1
                    if start_pos is None:
                        start_pos = (i, j)

        # If no empty tiles or all tiles are empty, consider it connected
        if empty_count == 0 or empty_count == self.grid_size * self.grid_size:
            return True

        # Flood-fill from starting position
        visited = set()
        queue = deque([start_pos])
        visited.add(start_pos)

        while queue:
            x, y = queue.popleft()

            # Check all 4 cardinal directions
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy

                # Check bounds
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    # Check if not wall and not visited
                    if grid[nx, ny] != self.wall_symbol and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append((nx, ny))

        # All empty tiles should be reachable
        return len(visited) == empty_count

    def place_food(self, grid: np.ndarray, num_food: int, food_symbol: str = "F",
                   avoid_positions: list[tuple[int, int]] = None) -> np.ndarray:
        """
        Place food items on empty tiles in the grid.

        Args:
            grid: The grid to place food on
            num_food: Number of food items to place
            food_symbol: Symbol representing food
            avoid_positions: List of (x, y) positions to avoid (e.g., agent positions)

        Returns:
            np.ndarray: Updated grid with food placed
        """
        if avoid_positions is None:
            avoid_positions = []

        avoid_set = set(avoid_positions)

        # Find all empty positions
        empty_positions = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if grid[i, j] == self.empty_symbol and (i, j) not in avoid_set:
                    empty_positions.append((i, j))

        # Randomly select positions for food
        if len(empty_positions) < num_food:
            print(f"Warning: Only {len(empty_positions)} empty positions available for {num_food} food items")
            num_food = len(empty_positions)

        food_positions = np.random.choice(len(empty_positions), size=num_food, replace=False)

        for idx in food_positions:
            x, y = empty_positions[idx]
            grid[x, y] = food_symbol

        return grid

    def get_random_empty_position(self, grid: np.ndarray,
                                   avoid_positions: list[tuple[int, int]] = None) -> tuple[int, int]:
        """
        Get a random empty position from the grid.

        Args:
            grid: The grid to search
            avoid_positions: List of (x, y) positions to avoid

        Returns:
            tuple: (x, y) position of an empty tile, or None if none available
        """
        if avoid_positions is None:
            avoid_positions = []

        avoid_set = set(avoid_positions)

        empty_positions = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if grid[i, j] == self.empty_symbol and (i, j) not in avoid_set:
                    empty_positions.append((i, j))

        if not empty_positions:
            return None

        return empty_positions[np.random.randint(len(empty_positions))]