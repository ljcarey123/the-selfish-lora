import numpy as np


class Environment:
    """
    Manages the simulation world, including the grid and resources.

    For the MVP, the environment is a fixed, static grid with a
    predetermined layout of resources. This provides a consistent
    benchmark for evaluating agent fitness.
    """

    def __init__(self, grid_size: int = 50):
        """
        Initializes the environment.

        Args:
            grid_size (int): The width and height of the square grid.
        """
        self.grid_size = grid_size
        self.grid = np.zeros((grid_size, grid_size), dtype=int)
        self.reset()

    def reset(self):
        """Clears the grid and places resources randomly."""
        self.grid.fill(0)
        num_resources = int(self.grid_size * self.grid_size * 0.20)  # 20% of the grid has resources randomly.
        self._place_resources(num_resources)

    def _place_resources(self, num_resources):
        """
        Places a specified number of resources at random locations.
        """
        for _ in range(num_resources):
            x, y = np.random.randint(0, self.grid_size, 2)
            self.grid[x, y] = 1

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
        if self.grid[x, y] == 1:
            self.grid[x, y] = 0
            return 1
        return 0

    def get_perception(
        self, position: tuple[int, int], view_range: int = 1
    ) -> np.ndarray:
        """
        Returns a local view for the agent.

        Args:
            position (tuple[int, int]): The (x, y) center of the view.
            view_range (int): The distance the agent can see in each direction.

        Returns:
            np.ndarray: A (view_range*2+1) x (view_range*2+1) grid.
        """
        x, y = position
        view_size = view_range * 2 + 1
        perception = np.zeros((view_size, view_size), dtype=int)

        for i in range(-view_range, view_range + 1):
            for j in range(-view_range, view_range + 1):
                grid_x, grid_y = x + i, y + j

                # Check bounds
                if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                    perception[i + view_range, j + view_range] = self.grid[
                        grid_x, grid_y
                    ]
                else:
                    # Use a value (e.g., -1) to indicate an out-of-bounds wall
                    perception[i + view_range, j + view_range] = -1

        return perception
