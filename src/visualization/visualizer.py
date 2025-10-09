import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Visualizer:
    """
    Handles the visualization of the simulation results.
    """

    def __init__(self, data_path, grid_size=50):
        """
        Initializes the Visualizer.

        Args:
            data_path (str): Path to the simulation data JSON file.
            grid_size (int): The size of the simulation grid.
        """
        self.grid_size = grid_size
        with open(data_path, "r") as f:
            self.data = json.load(f)

        # The original environment places resources in a fixed pattern.
        # We can recreate it here for the background.
        self.resource_grid = np.zeros((grid_size, grid_size), dtype=int)
        center = self.grid_size // 2
        self.resource_grid[center - 5 : center + 5, center - 5 : center + 5] = 1
        self.resource_grid[2:7, 2:7] = 1
        self.resource_grid[-7:-2, -7:-2] = 1
        self.resource_grid[2:7, -7:-2] = 1
        self.resource_grid[-7:-2, 2:7] = 1

    def animate_champion_path(self, generation, champion_rank=0, save_as_gif=True):
        """
        Creates and displays an animation of a champion's path for a given generation.

        Args:
            generation (int): The generation number to animate.
            champion_rank (int): The rank of the champion to display (0 for top champion).
            save_as_gif (bool): If True, saves the animation as a GIF.
        """
        paths = self.data["champion_paths_by_gen"][str(generation)]
        if champion_rank >= len(paths):
            print(f"Error: Champion rank {champion_rank} is out of bounds.")
            return
        path = paths[champion_rank]

        # If the path has 1 or fewer steps, animation is not possible/meaningful.
        if len(path) <= 1:
            print(f"Skipping animation for Generation {generation}, Champion Rank {champion_rank}: Not enough path data.")
            return

        fig, ax = plt.subplots()
        ax.set_title(f"Champion Path - Generation {generation}")
        ax.set_xticks([]), ax.set_yticks([])

        # Draw the resource grid background
        ax.imshow(self.resource_grid, cmap="Greens", alpha=0.3)

        # Initialize plot elements
        (agent_marker,) = ax.plot([], [], "bo", markersize=10, label="Agent")
        (path_line,) = ax.plot([], [], "b-", alpha=0.5, label="Path")
        (success_markers,) = ax.plot(
            [], [], "gx", markersize=8, label="Gather Success", linestyle='None'
        )
        (fail_markers,) = ax.plot(
            [], [], "rx", markersize=8, label="Gather Fail", linestyle='None'
        )
        successful_gather_coords = set()
        unsuccessful_gather_coords = set()

        def init():
            agent_marker.set_data([], [])
            path_line.set_data([], [])
            success_markers.set_data([], [])
            fail_markers.set_data([], [])
            return agent_marker, path_line, success_markers, fail_markers

        def update(frame):
            action, pos = path[frame]
            x, y = pos
            agent_marker.set_data(y, x)

            # Check for GATHER actions
            if action == "GATHER_SUCCESS":
                successful_gather_coords.add(tuple(pos))
            elif action == "GATHER_FAIL":
                unsuccessful_gather_coords.add(tuple(pos))

            # Update success markers
            if successful_gather_coords:
                success_y, success_x = zip(*successful_gather_coords)
                success_markers.set_data(success_x, success_y)

            # Update fail markers
            if unsuccessful_gather_coords:
                fail_y, fail_x = zip(*unsuccessful_gather_coords)
                fail_markers.set_data(fail_x, fail_y)

            # Draw the path up to the current frame
            path_positions = [p[1] for p in path[: frame + 1]]
            path_y_coords = [p[0] for p in path_positions]
            path_x_coords = [p[1] for p in path_positions]
            path_line.set_data(path_x_coords, path_y_coords)

            return agent_marker, path_line, success_markers, fail_markers

        ani = animation.FuncAnimation(
            fig,
            update,
            frames=len(path),
            init_func=init,
            blit=True,
            repeat=False,
            interval=100,
        )

        if save_as_gif:
            output_dir = "output"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            gif_path = os.path.join(output_dir, f"champion_gen_{generation}.gif")
            ani.save(gif_path, writer="pillow")
            print(f"Animation saved to {gif_path}")
        else:
            plt.legend()
            plt.show()


def main():
    """
    Main function to run the visualizer.
    """
    # Assuming the script is run from the `src` directory
    data_path = os.path.join("output", "simulation_paths.json")

    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        print("Please run the simulation first (`python main.py`)")
        return

    visualizer = Visualizer(data_path)

    # Animate the top champion of the first generation as a test
    print("Visualizing top champion path for generation 0...")
    visualizer.animate_champion_path(generation=0, champion_rank=0)


if __name__ == "__main__":
    main()
