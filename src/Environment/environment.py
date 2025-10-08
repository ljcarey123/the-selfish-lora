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
        self._place_resources()

    def _place_resources(self):
        """
        Places resources on the grid in a fixed, hardcoded pattern.

        In this MVP, resources are represented by '1'. This simple,
        reproducible layout ensures that every simulation run starts
        with the same conditions.
        """
        # Place a cluster of resources in the center
        center = self.grid_size // 2
        self.grid[center-5:center+5, center-5:center+5] = 1
        
        # Place some resources in the corners
        self.grid[2:7, 2:7] = 1
        self.grid[-7:-2, -7:-2] = 1
        self.grid[2:7, -7:-2] = 1
        self.grid[-7:-2, 2:7] = 1


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
