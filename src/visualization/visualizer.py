import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Visualizer:
    """
    Handles the visualization of the simulation results.
    """

    def __init__(self, data_path=None, data=None, grid_size=50):
        """
        Initializes the Visualizer.

        Can be initialized either from a file path or a direct data dictionary.

        Args:
            data_path (str, optional): Path to the simulation data JSON file.
            data (dict, optional): A dictionary containing simulation data.
            grid_size (int): The size of the simulation grid.
        """
        self.grid_size = grid_size

        if data_path:
            with open(data_path, "r") as f:
                self.data = json.load(f)
        elif data:
            self.data = data
        else:
            raise ValueError("Visualizer must be initialized with either 'data_path' or 'data'.")



    def animate_top_n_champion_paths(
        self, generation, n=3, save_as_gif=True
    ):
        """
        Creates an animation of the top n champions' paths for a given generation.

        Args:
            generation (int): The generation number to animate.
            n (int): The number of top champions to display.
            save_as_gif (bool): If True, saves the animation as a GIF.
        """
        paths = self.data["champion_paths_by_gen"].get(str(generation))
        if paths is None:
            print(f"Error: No champion paths found for generation {generation}.")
            return
        if len(paths) < n:
            print(f"Warning: Only {len(paths)} champion paths for generation {generation}, showing all.")
            n = len(paths)  # Show all available paths

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_title(f"Top {n} Champion Paths - Generation {generation}")
        ax.set_xticks([]), ax.set_yticks([])

        # Load the initial environment for the specified generation to show all original resources
        history_key = "initial_environment_history"
        if history_key not in self.data:
            print(f"Warning: '{history_key}' not found. Falling back to 'environment_history'.")
            history_key = "environment_history"
        resource_grid = np.array(self.data[history_key][str(generation)])
        ax.imshow(resource_grid, cmap="Greens", alpha=0.3)

        colors = plt.cm.jet(np.linspace(0, 1, n))
        agent_markers = []
        path_lines = []
        for i in range(n):
            (agent_marker,) = ax.plot([], [], 'o', color=colors[i], markersize=10, label=f"Agent {i+1}")
            (path_line,) = ax.plot([], [], '-', color=colors[i], alpha=0.6)
            agent_markers.append(agent_marker)
            path_lines.append(path_line)
        
        # Add gather markers (shared across all agents)
        (success_markers,) = ax.plot([], [], 'g*', markersize=12, label="Gather Success", linestyle='None')
        (fail_markers,) = ax.plot([], [], 'rx', markersize=10, label="Gather Fail", linestyle='None')
        (unparseable_markers,) = ax.plot([], [], 'o', color='orange', markersize=8, label="Unparseable", linestyle='None', alpha=0.7)
        successful_gather_coords = set()
        unsuccessful_gather_coords = set()
        unparseable_coords = set()

        ax.legend()

        # Determine the maximum path length for uniform animation
        max_frames = max(len(p) for p in paths[:n])

        def init():
            for i in range(n):
                agent_markers[i].set_data([], [])
                path_lines[i].set_data([], [])
            success_markers.set_data([], [])
            fail_markers.set_data([], [])
            unparseable_markers.set_data([], [])
            return agent_markers + path_lines + [success_markers, fail_markers, unparseable_markers]

        def update(frame):
            for i in range(n):
                path = paths[i]
                if frame < len(path):
                    action, pos = path[frame]
                    x, y = pos
                    agent_markers[i].set_data([y], [x])
                    
                    # Check for special actions
                    if action == "GATHER_SUCCESS":
                        successful_gather_coords.add(tuple(pos))
                    elif action == "GATHER_FAIL":
                        unsuccessful_gather_coords.add(tuple(pos))
                    elif action == "UNPARSEABLE":
                        unparseable_coords.add(tuple(pos))

                    # Draw path up to current frame
                    path_positions = [p[1] for p in path[: frame + 1]]
                    path_y_coords = [p[0] for p in path_positions]
                    path_x_coords = [p[1] for p in path_positions]
                    path_lines[i].set_data(path_x_coords, path_y_coords)
            
            # Update gather markers
            if successful_gather_coords:
                success_y, success_x = zip(*successful_gather_coords)
                success_markers.set_data(success_x, success_y)
            if unsuccessful_gather_coords:
                fail_y, fail_x = zip(*unsuccessful_gather_coords)
                fail_markers.set_data(fail_x, fail_y)
            if unparseable_coords:
                unparseable_y, unparseable_x = zip(*unparseable_coords)
                unparseable_markers.set_data(unparseable_x, unparseable_y)
            
            return agent_markers + path_lines + [success_markers, fail_markers, unparseable_markers]

        ani = animation.FuncAnimation(
            fig, update, frames=max_frames, init_func=init, blit=True, interval=150
        )

        if save_as_gif:
            gifs_dir = os.path.join("output", "gifs")
            if not os.path.exists(gifs_dir):
                os.makedirs(gifs_dir)
            gif_path = os.path.join(gifs_dir, f"champion_paths_gen_{generation}_top_{n}.gif")
            ani.save(gif_path, writer="pillow")
            print(f"Animation for generation {generation} saved to {gif_path}")
        else:
            plt.show()
        plt.close(fig)

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

        # Load the initial environment for the specified generation to show all original resources
        generation_str = str(generation)
        history_key = "initial_environment_history"
        if history_key not in self.data or generation_str not in self.data[history_key]:
            print(f"Warning: Initial environment for generation {generation} not found. Falling back to 'environment_history'.")
            history_key = "environment_history"
        
        if generation_str not in self.data[history_key]:
            print(f"Error: Environment for generation {generation} not found in data.")
            return
        resource_grid = np.array(self.data[history_key][generation_str])

        # Draw the resource grid background
        ax.imshow(resource_grid, cmap="Greens", alpha=0.3)

        # Initialize plot elements
        (agent_marker,) = ax.plot([], [], "bo", markersize=10, label="Agent")
        (path_line,) = ax.plot([], [], "b-", alpha=0.5, label="Path")
        (success_markers,) = ax.plot(
            [], [], "gx", markersize=8, label="Gather Success", linestyle='None'
        )
        (fail_markers,) = ax.plot(
            [], [], "rx", markersize=8, label="Gather Fail", linestyle='None'
        )
        (unparseable_markers,) = ax.plot(
            [], [], "o", color='orange', markersize=6, label="Unparseable", linestyle='None', alpha=0.7
        )
        successful_gather_coords = set()
        unsuccessful_gather_coords = set()
        unparseable_coords = set()

        def init():
            agent_marker.set_data([], [])
            path_line.set_data([], [])
            success_markers.set_data([], [])
            fail_markers.set_data([], [])
            unparseable_markers.set_data([], [])
            return agent_marker, path_line, success_markers, fail_markers, unparseable_markers

        def update(frame):
            action, pos = path[frame]
            x, y = pos
            agent_marker.set_data([y], [x])

            # Check for special actions
            if action == "GATHER_SUCCESS":
                successful_gather_coords.add(tuple(pos))
            elif action == "GATHER_FAIL":
                unsuccessful_gather_coords.add(tuple(pos))
            elif action == "UNPARSEABLE":
                unparseable_coords.add(tuple(pos))

            # Update success markers
            if successful_gather_coords:
                success_y, success_x = zip(*successful_gather_coords)
                success_markers.set_data(success_x, success_y)

            # Update fail markers
            if unsuccessful_gather_coords:
                fail_y, fail_x = zip(*unsuccessful_gather_coords)
                fail_markers.set_data(fail_x, fail_y)
            
            # Update unparseable markers
            if unparseable_coords:
                unparseable_y, unparseable_x = zip(*unparseable_coords)
                unparseable_markers.set_data(unparseable_x, unparseable_y)

            # Draw the path up to the current frame
            path_positions = [p[1] for p in path[: frame + 1]]
            path_y_coords = [p[0] for p in path_positions]
            path_x_coords = [p[1] for p in path_positions]
            path_line.set_data(path_x_coords, path_y_coords)

            return agent_marker, path_line, success_markers, fail_markers, unparseable_markers

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
            gifs_dir = os.path.join("output", "gifs")
            if not os.path.exists(gifs_dir):
                os.makedirs(gifs_dir)
            gif_path = os.path.join(
                gifs_dir, f"champion_path_gen_{generation}_rank_{champion_rank}.gif"
            )
            ani.save(gif_path, writer="pillow")
            print(f"Animation saved to {gif_path}")
        else:
            plt.legend()
            plt.show()

    def plot_performance_metrics(self, save_as_png=True):
        """
        Plots the collected performance metrics over generations.

        Args:
            save_as_png (bool): If True, saves the plot as a PNG image.
        """
        history = self.data['performance_history']
        if not history:
            print("No performance history to plot.")
            return

        generations = range(len(history))
        avg_fitness = [g['average_fitness'] for g in history]
        max_fitness = [g['max_fitness'] for g in history]
        min_fitness = [g['min_fitness'] for g in history]
        std_dev = [g['std_dev_fitness'] for g in history]

        avg_fitness = np.array(avg_fitness)
        std_dev = np.array(std_dev)

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(generations, avg_fitness, 'b-', label='Average Fitness')
        ax.plot(generations, max_fitness, 'g--', label='Maximum Fitness')
        ax.plot(generations, min_fitness, 'r:', label='Minimum Fitness')

        # Add a shaded region for the standard deviation, ensuring it doesn't go below zero
        lower_bound = np.maximum(0, avg_fitness - std_dev)
        ax.fill_between(generations, lower_bound, avg_fitness + std_dev,
                        color='blue', alpha=0.1, label='Std. Deviation')

        ax.set_title('Population Fitness Over Generations')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Fitness Score')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

        if save_as_png:
            output_dir = 'output'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            png_path = os.path.join(output_dir, 'performance_metrics.png')
            plt.savefig(png_path)
            print(f'Performance plot saved to {png_path}')
        else:
            plt.show()

    def plot_genetic_diversity(self, save_as_png=True):
        """
        Plots the genetic diversity of the population over generations.

        Args:
            save_as_png (bool): If True, saves the plot as a PNG image.
        """
        history = self.data['genetic_history']
        if not history:
            print("No genetic history to plot.")
            return

        generations = range(len(history))

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(generations, history, 'm-', label='Genetic Diversity')

        ax.set_title('Genetic Diversity Over Generations')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Average Genetic Distance')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

        if save_as_png:
            output_dir = 'output'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            png_path = os.path.join(output_dir, 'genetic_diversity.png')
            plt.savefig(png_path)
            print(f'Genetic diversity plot saved to {png_path}')
        else:
            plt.show()

    def plot_behavioral_metrics(self, save_as_png=True):
        """
        Plots the distribution of agent actions over generations.

        Args:
            save_as_png (bool): If True, saves the plot as a PNG image.
        """
        history = self.data['behavioral_history']
        if not history:
            print("No behavioral history to plot.")
            return

        # Get all unique action types across the entire simulation
        all_actions = sorted(list(set(action for gen_actions in history for action in gen_actions.keys())))
        
        # Prepare data for stacked area chart
        y_data = {action: [] for action in all_actions}
        for gen_actions in history:
            total_actions = sum(gen_actions.values())
            for action in all_actions:
                count = gen_actions.get(action, 0)
                percentage = (count / total_actions) * 100 if total_actions > 0 else 0
                y_data[action].append(percentage)

        generations = range(len(history))
        labels = list(y_data.keys())
        data_to_stack = list(y_data.values())

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.stackplot(generations, data_to_stack, labels=labels)

        ax.set_title('Action Distribution Over Generations')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Percentage of Actions (%)')
        ax.set_ylim(0, 100)
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.6)

        if save_as_png:
            output_dir = 'output'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            png_path = os.path.join(output_dir, 'behavioral_metrics.png')
            plt.savefig(png_path)
            print(f'Behavioral metrics plot saved to {png_path}')
        else:
            plt.show()

    def plot_agent_heatmap(self, generation, save_as_png=True):
        """
        Creates a heatmap showing where agents spent time during a generation.
        
        Args:
            generation (int): The generation to visualize.
            save_as_png (bool): If True, saves the plot as a PNG image.
        """
        generation_str = str(generation)
        
        # Get all agent paths for this generation
        if 'champion_paths_by_gen' in self.data and generation_str in self.data['champion_paths_by_gen']:
            paths = self.data['champion_paths_by_gen'][generation_str]
        else:
            print(f"No path data found for generation {generation}")
            return
        
        # Create a heatmap grid
        heatmap = np.zeros((self.grid_size, self.grid_size))
        
        # Count visits to each cell
        for path in paths:
            for action, pos in path:
                if action != "START":  # Skip the start marker
                    x, y = pos
                    if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                        heatmap[x, y] += 1
        
        # Get the environment to show resources
        history_key = "initial_environment_history"
        if history_key not in self.data:
            history_key = "environment_history"
        
        if generation_str in self.data.get(history_key, {}):
            resource_grid = np.array(self.data[history_key][generation_str])
        else:
            resource_grid = None
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot the heatmap
        im = ax.imshow(heatmap, cmap='YlOrRd', alpha=0.8)
        
        # Overlay resource locations if available
        if resource_grid is not None:
            # Show resources as green contours
            resource_mask = (resource_grid == 1)
            ax.contour(resource_mask, colors='green', linewidths=2, levels=[0.5])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Number of Agent Visits', rotation=270, labelpad=20)
        
        ax.set_title(f'Agent Movement Heatmap - Generation {generation}\n(Green outlines = Resources)')
        ax.set_xlabel('Y Position')
        ax.set_ylabel('X Position')
        
        if save_as_png:
            output_dir = 'output'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            png_path = os.path.join(output_dir, f'agent_heatmap_gen_{generation}.png')
            plt.savefig(png_path, dpi=150, bbox_inches='tight')
            print(f'Agent heatmap saved to {png_path}')
            plt.close()
        else:
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

    # Plot the performance metrics
    print("\nPlotting performance metrics...")
    visualizer.plot_performance_metrics()

    # Plot the genetic diversity
    print("\nPlotting genetic diversity...")
    visualizer.plot_genetic_diversity()

    # Plot the behavioral metrics
    print("\nPlotting behavioral metrics...")
    visualizer.plot_behavioral_metrics()


if __name__ == "__main__":
    main()
