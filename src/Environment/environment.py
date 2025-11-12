import numpy as np
from config_loader import GRID_SYMBOLS


class Environment:
    """
    Manages the simulation world, including the grid and resources.

    For the MVP, the environment is a fixed, static grid with a
    predetermined layout of resources. This provides a consistent
    benchmark for evaluating agent fitness.
    """

    def __init__(self, grid_size: int = 50, resource_regen_turns: int = 10):
        """
        Initializes the environment.

        Args:
            grid_size (int): The width and height of the square grid.
            resource_regen_turns (int): Number of turns before a depleted resource regenerates (0 = no regeneration).
        """
        self.grid_size = grid_size
        self.resource_regen_turns = resource_regen_turns

        # Use string array to store symbolic characters (., R, W, etc.)
        self.grid = np.full((grid_size, grid_size), GRID_SYMBOLS["empty"], dtype=object)

        # Track depleted resources for regeneration: {(x, y): turns_until_regen}
        self.depleted_resources = {}

        self.reset()

    def reset(self):
        """Clears the grid and places resources randomly."""
        self.grid.fill(GRID_SYMBOLS["empty"])
        self.depleted_resources.clear()  # Clear regeneration tracking
        num_resources = int(self.grid_size * self.grid_size * 0.20)  # 20% of the grid has resources randomly.
        self._place_resources(num_resources)

    def _place_resources(self, num_resources):
        """
        Places a specified number of resources at random locations.
        """
        for _ in range(num_resources):
            x, y = np.random.randint(0, self.grid_size, 2)
            self.grid[x, y] = GRID_SYMBOLS["resource"]

    def get_state(self) -> np.ndarray:
        """
        Provides a complete view of the current grid state.

        Returns:
            np.ndarray: A 2D array representing the grid.
        """
        return self.grid

    def remove_resource_at(self, position: tuple[int, int]) -> int:
        """
        Removes a resource from a given position if one exists.

        Args:
            position (tuple[int, int]): The (x, y) coordinates to check.

        Returns:
            int: 1 if a resource was collected, 0 otherwise.
        """
        x, y = position
        if self.grid[x, y] == GRID_SYMBOLS["resource"]:
            self.grid[x, y] = GRID_SYMBOLS["empty"]
            # Track depleted resource for regeneration if enabled
            if self.resource_regen_turns > 0:
                self.depleted_resources[position] = self.resource_regen_turns
            return 1
        return 0

    def step(self):
        """
        Advances one turn and handles resource regeneration.

        Decrements the regeneration counter for each depleted resource.
        When a counter reaches 0, the resource regenerates at that position.
        """
        if self.resource_regen_turns <= 0:
            return  # Regeneration disabled

        positions_to_regenerate = []

        for pos, turns_left in list(self.depleted_resources.items()):
            turns_left -= 1
            if turns_left <= 0:
                # Resource ready to regenerate
                x, y = pos
                self.grid[x, y] = GRID_SYMBOLS["resource"]
                positions_to_regenerate.append(pos)
            else:
                # Update countdown
                self.depleted_resources[pos] = turns_left

        # Remove regenerated resources from tracking
        for pos in positions_to_regenerate:
            del self.depleted_resources[pos]

    def get_perception(
        self, position: tuple[int, int], view_range: int = 2
    ) -> np.ndarray:
        """
        Returns a local 5x5 view for the agent (agent at center).

        Args:
            position (tuple[int, int]): The (x, y) center of the view.
            view_range (int): The distance the agent can see in each direction (default 2 = 5x5 grid).

        Returns:
            np.ndarray: A 5x5 grid with symbolic characters (., R, W, A, @).
        """
        x, y = position
        view_size = view_range * 2 + 1  # 2*2+1 = 5x5
        perception = np.full((view_size, view_size), GRID_SYMBOLS["empty"], dtype=object)

        for i in range(-view_range, view_range + 1):
            for j in range(-view_range, view_range + 1):
                grid_x, grid_y = x + i, y + j

                # Check bounds
                if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                    perception[i + view_range, j + view_range] = self.grid[grid_x, grid_y]
                else:
                    # Use wall symbol for out-of-bounds
                    perception[i + view_range, j + view_range] = GRID_SYMBOLS["wall"]

        return perception
