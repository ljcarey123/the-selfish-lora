import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Visualiser:
    """
    Handles the visualisation of the simulation results.
    """

    def __init__(self, data_path=None, data=None, grid_size=None):
        """
        Initialises the Visualiser.

        Can be initialised either from a file path or a direct data dictionary.

        Args:
            data_path (str, optional): Path to the simulation data JSON file.
            data (dict, optional): A dictionary containing simulation data.
            grid_size (int, optional): The size of the simulation grid. If not provided, will be read from data.
        """
        if data_path:
            with open(data_path, "r") as f:
                self.data = json.load(f)
        elif data:
            self.data = data
        else:
            raise ValueError("Visualiser must be initialised with either 'data_path' or 'data'.")

        # Get grid size from data if not explicitly provided
        if grid_size is None:
            self.grid_size = self.data.get('grid_size', 50)  # Default to 50 for backwards compatibility
        else:
            self.grid_size = grid_size

        # Load game mechanics config
        config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "game_mechanics.json")
        with open(config_path, 'r') as f:
            self.game_mechanics = json.load(f)

    def _convert_to_numeric_grid(self, symbolic_grid):
        """
        Converts a symbolic grid (with 'F', 'W', '.', etc.) to a numeric grid for visualization.

        Args:
            symbolic_grid: numpy array with string symbols

        Returns:
            numpy array with numeric values (1 for food, 0 for empty)
        """
        numeric_grid = np.zeros(symbolic_grid.shape)
        # Mark food as 1, everything else as 0
        numeric_grid[symbolic_grid == 'F'] = 1
        numeric_grid[symbolic_grid == 'R'] = 1  # Support old format too
        return numeric_grid

    def animate_top_n_champion_paths(
        self, generation, n=3, save_as_gif=True
    ):
        """
        Creates an animation of the top n champions' paths for a given generation.
        In tournament mode (multiple rounds), creates separate animations for each round.

        Args:
            generation (int): The generation number to animate.
            n (int): The number of top champions to display.
            save_as_gif (bool): If True, saves the animation as a GIF.
        """
        is_tournament_mode = self.data.get('tournament_mode', False)
        gen_str = str(generation)

        # Get paths for this generation
        gen_data = self.data["champion_paths_by_gen"].get(gen_str)
        if gen_data is None:
            print(f"Error: No champion paths found for generation {generation}.")
            return

        # Check if this is tournament mode (dict of rounds) or standard mode (list of paths)
        if is_tournament_mode and isinstance(gen_data, dict):
            # Tournament mode: gen_data is {round_idx: [paths]}
            rounds = sorted(gen_data.keys(), key=int)
            print(f"Tournament mode detected: creating {len(rounds)} separate animations (one per round)")

            for round_idx in rounds:
                round_paths = gen_data[round_idx]

                # Get environment for this round
                env_data = self.data["environment_history"][gen_str][round_idx]
                resource_grid = np.array(env_data)

                if len(round_paths) < n:
                    print(f"Warning: Only {len(round_paths)} champion paths for generation {generation} round {round_idx}, showing all.")
                    n_paths = len(round_paths)
                else:
                    n_paths = n

                self._create_path_animation(
                    round_paths[:n_paths], resource_grid, generation, n_paths, save_as_gif,
                    title_suffix=f" - Round {int(round_idx) + 1}",
                    filename_suffix=f"_round_{int(round_idx) + 1}"
                )
        else:
            # Standard mode: gen_data is [paths] directly
            paths = gen_data if isinstance(gen_data, list) else []
            if not paths:
                print(f"Error: No champion paths found for generation {generation}.")
                return

            if len(paths) < n:
                print(f"Warning: Only {len(paths)} champion paths for generation {generation}, showing all.")
                n = len(paths)

            # Get environment (handle both old and new formats)
            env_data = self.data["environment_history"].get(gen_str)
            if isinstance(env_data, dict):
                # Tournament format but only 1 round - get round 0
                env_data = env_data.get("0", env_data)
            resource_grid = np.array(env_data)

            # Create the animation
            self._create_path_animation(
                paths[:n], resource_grid, generation, n, save_as_gif,
                title_suffix=""
            )


    def _create_path_animation(self, paths, resource_grid, generation, n, save_as_gif,
                               title_suffix="", filename_suffix=""):
        """Helper method to create a single path animation."""
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_title(f"Top {n} Champion Paths - Generation {generation}{title_suffix}")
        ax.set_xticks([]), ax.set_yticks([])

        # Convert symbolic grid to numeric for visualization
        numeric_grid = self._convert_to_numeric_grid(resource_grid)
        ax.imshow(numeric_grid, cmap="Greens", alpha=0.3)

        colors = plt.cm.jet(np.linspace(0, 1, n))
        agent_markers = []
        path_lines = []
        for i in range(n):
            (agent_marker,) = ax.plot([], [], 'o', color=colors[i], markersize=10, label=f"Agent {i+1}")
            (path_line,) = ax.plot([], [], '-', color=colors[i], alpha=0.6)
            agent_markers.append(agent_marker)
            path_lines.append(path_line)
        
        # Add event markers (shared across all agents)
        (food_markers,) = ax.plot([], [], 'g*', markersize=12, label="Food Collected", linestyle='None')
        (attack_markers,) = ax.plot([], [], 'r^', markersize=10, label="Attack (aggressor)", linestyle='None')
        (victim_markers,) = ax.plot([], [], 'mv', markersize=10, label="Attacked (victim)", linestyle='None')
        (unparseable_markers,) = ax.plot([], [], 'o', color='orange', markersize=8, label="Unparseable", linestyle='None', alpha=0.7)
        food_coords = set()
        attack_coords = set()
        victim_coords = set()
        unparseable_coords = set()

        ax.legend()

        # Determine the maximum path length for uniform animation
        max_frames = max(len(p) for p in paths[:n])

        def init():
            for i in range(n):
                agent_markers[i].set_data([], [])
                path_lines[i].set_data([], [])
            food_markers.set_data([], [])
            attack_markers.set_data([], [])
            victim_markers.set_data([], [])
            unparseable_markers.set_data([], [])
            return agent_markers + path_lines + [food_markers, attack_markers, victim_markers, unparseable_markers]

        def update(frame):
            for i in range(n):
                path = paths[i]
                if frame < len(path):
                    action, pos = path[frame]
                    x, y = pos
                    agent_markers[i].set_data([y], [x])

                    # Check for special actions
                    if action == "FOOD_COLLECTED" or action == "GATHER_SUCCESS":
                        food_coords.add(tuple(pos))
                    elif action == "COMBAT_ATTACK":
                        attack_coords.add(tuple(pos))
                    elif action == "COMBAT_VICTIM":
                        victim_coords.add(tuple(pos))
                    elif action == "COMBAT" or action == "GATHER_FAIL":
                        # Legacy support for old format
                        attack_coords.add(tuple(pos))
                    elif action == "UNPARSEABLE":
                        unparseable_coords.add(tuple(pos))

                    # Draw path up to current frame
                    path_positions = [p[1] for p in path[: frame + 1]]
                    path_y_coords = [p[0] for p in path_positions]
                    path_x_coords = [p[1] for p in path_positions]
                    path_lines[i].set_data(path_x_coords, path_y_coords)

            # Update event markers
            if food_coords:
                food_y, food_x = zip(*food_coords)
                food_markers.set_data(food_x, food_y)
            if attack_coords:
                attack_y, attack_x = zip(*attack_coords)
                attack_markers.set_data(attack_x, attack_y)
            if victim_coords:
                victim_y, victim_x = zip(*victim_coords)
                victim_markers.set_data(victim_x, victim_y)
            if unparseable_coords:
                unparseable_y, unparseable_x = zip(*unparseable_coords)
                unparseable_markers.set_data(unparseable_x, unparseable_y)

            return agent_markers + path_lines + [food_markers, attack_markers, victim_markers, unparseable_markers]

        ani = animation.FuncAnimation(
            fig, update, frames=max_frames, init_func=init, blit=True, interval=150
        )

        if save_as_gif:
            gifs_dir = os.path.join("output", "gifs")
            if not os.path.exists(gifs_dir):
                os.makedirs(gifs_dir)
            gif_path = os.path.join(gifs_dir, f"champion_paths_gen_{generation}_top_{n}{filename_suffix}.gif")
            ani.save(gif_path, writer="pillow")
            print(f"Animation saved to {gif_path}")
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
        (food_markers,) = ax.plot(
            [], [], "g*", markersize=12, label="Food Collected", linestyle='None'
        )
        (combat_markers,) = ax.plot(
            [], [], "rx", markersize=10, label="Combat", linestyle='None'
        )
        (unparseable_markers,) = ax.plot(
            [], [], "o", color='orange', markersize=6, label="Unparseable", linestyle='None', alpha=0.7
        )
        food_coords = set()
        combat_coords = set()
        unparseable_coords = set()

        def init():
            agent_marker.set_data([], [])
            path_line.set_data([], [])
            food_markers.set_data([], [])
            combat_markers.set_data([], [])
            unparseable_markers.set_data([], [])
            return agent_marker, path_line, food_markers, combat_markers, unparseable_markers

        def update(frame):
            action, pos = path[frame]
            x, y = pos
            agent_marker.set_data([y], [x])

            # Check for special actions
            if action == "FOOD_COLLECTED" or action == "GATHER_SUCCESS":
                food_coords.add(tuple(pos))
            elif action == "COMBAT" or action == "GATHER_FAIL":
                combat_coords.add(tuple(pos))
            elif action == "UNPARSEABLE":
                unparseable_coords.add(tuple(pos))

            # Update food markers
            if food_coords:
                food_y, food_x = zip(*food_coords)
                food_markers.set_data(food_x, food_y)

            # Update combat markers
            if combat_coords:
                combat_y, combat_x = zip(*combat_coords)
                combat_markers.set_data(combat_x, combat_y)

            # Update unparseable markers
            if unparseable_coords:
                unparseable_y, unparseable_x = zip(*unparseable_coords)
                unparseable_markers.set_data(unparseable_x, unparseable_y)

            # Draw the path up to the current frame
            path_positions = [p[1] for p in path[: frame + 1]]
            path_y_coords = [p[0] for p in path_positions]
            path_x_coords = [p[1] for p in path_positions]
            path_line.set_data(path_x_coords, path_y_coords)

            return agent_marker, path_line, food_markers, combat_markers, unparseable_markers

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

    def plot_behavioural_metrics(self, save_as_png=True):
        """
        Plots the distribution of agent actions over generations.

        Args:
            save_as_png (bool): If True, saves the plot as a PNG image.
        """
        history = self.data['behavioural_history']
        if not history:
            print("No behavioural history to plot.")
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
            png_path = os.path.join(output_dir, 'behavioural_metrics.png')
            plt.savefig(png_path)
            print(f'Behavioural metrics plot saved to {png_path}')
        else:
            plt.show()

    def plot_agent_heatmap(self, generation, save_as_png=True):
        """
        Creates a heatmap showing where agents spent time during a generation.
        In tournament mode (multiple rounds), creates subplots for each round side-by-side.

        Args:
            generation (int): The generation to visualise.
            save_as_png (bool): If True, saves the plot as a PNG image.
        """
        generation_str = str(generation)
        is_tournament_mode = self.data.get('tournament_mode', False)

        # Get generation data
        gen_paths = self.data['champion_paths_by_gen'].get(generation_str)
        gen_env = self.data['environment_history'].get(generation_str)

        if gen_paths is None or gen_env is None:
            print(f"No data found for generation {generation}")
            return

        # Check if this is tournament mode (dict of rounds) or standard mode (list of paths)
        if is_tournament_mode and isinstance(gen_paths, dict):
            # Tournament mode: create subplots for each round
            rounds = sorted(gen_paths.keys(), key=int)
            n_rounds = len(rounds)

            # Create subplots (1 row, n_rounds columns)
            fig, axes = plt.subplots(1, n_rounds, figsize=(8 * n_rounds, 8))
            if n_rounds == 1:
                axes = [axes]  # Make it iterable

            print(f"Creating heatmap with {n_rounds} round subplots")

            for round_idx_str, ax in zip(rounds, axes):
                # Get paths for this round (top 5 champions)
                round_paths = gen_paths[round_idx_str]
                resource_grid = np.array(gen_env[round_idx_str])

                # Create heatmap for this round
                heatmap = np.zeros((self.grid_size, self.grid_size))
                for path in round_paths:
                    for action, pos in path:
                        if action != "START":
                            row, col = pos
                            if 0 <= row < self.grid_size and 0 <= col < self.grid_size:
                                heatmap[row, col] += 1

                # Plot the heatmap
                im = ax.imshow(heatmap, cmap='YlOrRd', alpha=0.8)

                # Overlay resource locations (convert symbolic grid to mask)
                resource_mask = ((resource_grid == 'R') | (resource_grid == 'F'))
                ax.contour(resource_mask, colors='green', linewidths=2, levels=[0.5])

                # Add colorbar
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label('Visits', rotation=270, labelpad=15, fontsize=9)

                ax.set_title(f'Round {int(round_idx_str) + 1}', fontsize=12)
                ax.set_xlabel('Y Position')
                ax.set_ylabel('X Position')

            fig.suptitle(f'Agent Movement Heatmaps - Generation {generation}\n(Green outlines = Resources)',
                        fontsize=14, y=0.98)
            plt.tight_layout(rect=[0, 0, 1, 0.96])

        else:
            # Standard mode: single heatmap
            paths = gen_paths if isinstance(gen_paths, list) else []
            if not paths:
                print(f"No path data found for generation {generation}")
                return

            # Create a heatmap grid
            heatmap = np.zeros((self.grid_size, self.grid_size))

            # Count visits to each cell
            for path in paths:
                for action, pos in path:
                    if action != "START":
                        row, col = pos
                        if 0 <= row < self.grid_size and 0 <= col < self.grid_size:
                            heatmap[row, col] += 1

            # Get the environment to show resources
            env_data = gen_env
            if isinstance(env_data, dict):
                # Tournament format but only 1 round - get round 0
                env_data = env_data.get("0", next(iter(env_data.values())))

            resource_grid = np.array(env_data) if env_data is not None else None

            # Create the plot
            fig, ax = plt.subplots(figsize=(12, 10))

            # Plot the heatmap
            im = ax.imshow(heatmap, cmap='YlOrRd', alpha=0.8)

            # Overlay resource locations if available (symbolic grid)
            if resource_grid is not None:
                resource_mask = ((resource_grid == 'R') | (resource_grid == 'F'))
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

    def plot_champion_action_timeline(self, generation, n=3, save_as_png=True):
        """
        Plot detailed move-by-move timeline for champion agents showing:
        - Action taken
        - Whether attacked
        - Running fitness (energy)

        Args:
            generation: Which generation to visualize
            n: Number of top agents to show (default 3)
            save_as_png: Whether to save as PNG file
        """
        gen_str = str(generation)
        is_tournament_mode = self.data.get('tournament_mode', False)

        # Get champion paths
        if gen_str not in self.data["champion_paths_by_gen"]:
            print(f"No data found for generation {generation}")
            return

        gen_paths = self.data["champion_paths_by_gen"][gen_str]

        # Handle tournament vs standard format
        if is_tournament_mode and isinstance(gen_paths, dict):
            # Tournament mode: create timeline for each round
            rounds = sorted(gen_paths.keys(), key=int)

            for round_idx_str in rounds:
                round_paths = gen_paths[round_idx_str][:n]
                self._create_action_timeline(
                    round_paths, generation,
                    title_suffix=f" - Round {int(round_idx_str) + 1}",
                    filename_suffix=f"_round_{int(round_idx_str) + 1}",
                    save_as_png=save_as_png
                )
        else:
            # Standard mode: single timeline
            paths = gen_paths if isinstance(gen_paths, list) else []
            paths = paths[:n]
            self._create_action_timeline(paths, generation, save_as_png=save_as_png)

    def _create_action_timeline(self, paths, generation, title_suffix="", filename_suffix="", save_as_png=True):
        """
        Create the actual action timeline plot.

        Args:
            paths: List of agent paths to visualize
            generation: Generation number
            title_suffix: Optional suffix for the title
            filename_suffix: Optional suffix for the filename
            save_as_png: Whether to save as PNG
        """
        if not paths:
            print(f"No paths to visualize")
            return

        n_agents = len(paths)

        # Create figure with subplots - one per agent
        fig, axes = plt.subplots(n_agents, 1, figsize=(14, 4 * n_agents))
        if n_agents == 1:
            axes = [axes]

        # Process each agent
        for agent_idx, (path, ax) in enumerate(zip(paths, axes)):
            if not path:
                continue

            # Extract timeline data
            turns = []
            energies = []
            actions = []
            events = []  # 'FOOD', 'ATTACK', 'VICTIM', 'MOVE', 'START'

            # Get game mechanics from config
            initial_energy = self.game_mechanics["energy"]["initial"]
            metabolism_cost = self.game_mechanics["energy"]["metabolism_cost"]
            food_reward = self.game_mechanics["energy"]["food_reward"]
            combat_attacker_gain = self.game_mechanics["energy"]["combat_attacker_gain"]
            combat_victim_loss = self.game_mechanics["energy"]["combat_victim_loss"]

            current_energy = float(initial_energy)

            for turn_idx, (action, pos) in enumerate(path):
                turns.append(turn_idx)

                # Categorize action and calculate energy change
                if action == "START":
                    events.append("START")
                    actions.append("START")
                    energies.append(current_energy)

                elif action == "FOOD_COLLECTED":
                    events.append("FOOD")
                    actions.append("FOOD")
                    current_energy += food_reward        # Food reward
                    current_energy -= metabolism_cost    # Metabolism cost
                    energies.append(current_energy)

                elif action == "COMBAT_ATTACK":
                    events.append("ATTACK")
                    actions.append("ATTACK")
                    current_energy += combat_attacker_gain  # Energy stolen from victim
                    current_energy -= metabolism_cost       # Metabolism cost
                    energies.append(current_energy)

                elif action == "COMBAT_VICTIM":
                    events.append("VICTIM")
                    actions.append("VICTIM")
                    current_energy -= combat_victim_loss  # Combat damage
                    current_energy -= metabolism_cost     # Metabolism cost
                    energies.append(current_energy)

                elif action == "COMBAT":
                    # Legacy support for old format (assume attacker)
                    events.append("ATTACK")
                    actions.append("ATTACK")
                    current_energy += combat_attacker_gain  # Energy stolen
                    current_energy -= metabolism_cost       # Metabolism cost
                    energies.append(current_energy)

                else:
                    # Regular move action
                    events.append("MOVE")
                    actions.append(action)
                    current_energy -= metabolism_cost  # Metabolism cost only
                    energies.append(current_energy)

            # Plot energy timeline
            ax.plot(turns, energies, 'b-', linewidth=2, label='Energy')
            ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Death Threshold')

            # Mark special events
            food_turns = [t for t, e in zip(turns, events) if e == "FOOD"]
            food_energies = [energies[t] for t in food_turns]
            attack_turns = [t for t, e in zip(turns, events) if e == "ATTACK"]
            attack_energies = [energies[t] for t in attack_turns]
            victim_turns = [t for t, e in zip(turns, events) if e == "VICTIM"]
            victim_energies = [energies[t] for t in victim_turns]

            if food_turns:
                ax.scatter(food_turns, food_energies, color='green', s=100, marker='o',
                          label='Food Collected (+30)', zorder=5, edgecolor='darkgreen', linewidth=2)
            if attack_turns:
                ax.scatter(attack_turns, attack_energies, color='red', s=100, marker='^',
                          label='Attack (aggressor)', zorder=5, edgecolor='darkred', linewidth=2)
            if victim_turns:
                ax.scatter(victim_turns, victim_energies, color='magenta', s=100, marker='v',
                          label='Attacked (-20 victim)', zorder=5, edgecolor='darkmagenta', linewidth=2)

            # Styling
            ax.set_xlabel('Turn', fontsize=11)
            ax.set_ylabel('Energy', fontsize=11)
            ax.set_title(f'Agent #{agent_idx + 1} - Action Timeline', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=9)

            # Add summary statistics in text box
            total_food = len(food_turns)
            total_attacks = len(attack_turns)
            total_attacked = len(victim_turns)
            final_energy = energies[-1] if energies else 0
            status = "ALIVE" if final_energy > 0 else "DEAD"

            summary_text = f'Food: {total_food} | Attacks: {total_attacks} | Attacked: {total_attacked} | Final Energy: {final_energy:.1f} | Status: {status}'
            ax.text(0.02, 0.98, summary_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Overall title
        fig.suptitle(f'Champion Agent Timelines - Generation {generation}{title_suffix}',
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])

        if save_as_png:
            output_dir = 'output'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            png_path = os.path.join(output_dir, f'action_timeline_gen_{generation}_top_{n_agents}{filename_suffix}.png')
            plt.savefig(png_path, dpi=150, bbox_inches='tight')
            print(f'Action timeline saved to {png_path}')
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

    visualiser = Visualiser(data_path)

    # Determine the latest generation
    champion_paths = visualiser.data.get("champion_paths_by_gen", {})
    if not champion_paths:
        print("Error: No generation data found in simulation results")
        return

    latest_gen = max(int(gen) for gen in champion_paths.keys())
    print(f"Found {len(champion_paths)} generation(s), visualizing latest generation: {latest_gen}")
    print()

    # Generate heatmap for latest generation
    print(f"Generating agent heatmap for generation {latest_gen}...")
    visualiser.plot_agent_heatmap(generation=latest_gen, save_as_png=True)

    # Generate action timeline for top 3 agents
    print(f"\nGenerating action timeline for top 3 agents...")
    visualiser.plot_champion_action_timeline(generation=latest_gen, n=3, save_as_png=True)

    # Animate the top 3 champion paths
    print(f"\nVisualising top 3 champion paths for generation {latest_gen}...")
    visualiser.animate_top_n_champion_paths(generation=latest_gen, n=3, save_as_gif=True)

    # Plot the genetic diversity (if multiple generations)
    if len(champion_paths) > 1:
        print("\nPlotting genetic diversity...")
        visualiser.plot_genetic_diversity(save_as_png=True)

        # Plot the behavioural metrics
        print("Plotting behavioural metrics...")
        visualiser.plot_behavioural_metrics(save_as_png=True)

    print("\n✓ All visualizations generated successfully!")
    print(f"  - Heatmap: ./output/agent_heatmap_gen_{latest_gen}.png")
    print(f"  - Timeline: ./output/action_timeline_gen_{latest_gen}_top_3*.png")
    print(f"  - Animation: ./output/gifs/champion_paths_gen_{latest_gen}_top_3*.gif")
    if len(champion_paths) > 1:
        print("  - Genetic diversity: ./output/genetic_diversity.png")
        print("  - Behavioral metrics: ./output/behavioural_metrics.png")


if __name__ == "__main__":
    main()
