import numpy as np
from LoraBrain.lora_brain import LoraBrain
from Environment.environment import Environment


class Agent:
    """
    Represents an individual entity in the simulation.

    An agent has a position, a brain for decision-making, and a fitness score
    that reflects its performance. It interacts with the environment based on
    the actions chosen by its brain.
    """

    def __init__(
        self, brain: LoraBrain, initial_position: tuple[int, int], grid_size: int
    ):
        """
        Initializes an agent.

        Args:
            brain (LoraBrain): The decision-making brain for the agent.
            initial_position (tuple[int, int]): The starting (x, y) coordinates.
            grid_size (int): The size of the grid, used for boundary checks.
        """
        self.brain = brain
        self.position = initial_position
        self.grid_size = grid_size
        self.fitness_score = 0

    def decide_and_act(self, environment: Environment):
        """
        Orchestrates the agent's turn: perceive, decide, and act.

        Args:
            environment (Environment): The world in which the agent exists.
        """
        # 1. Perceive the environment
        perception = environment.get_state()

        # 2. Decide on an action using the brain
        action_string = self.brain.choose_action(perception)

        # 3. Act based on the decision
        self._execute_action(action_string, environment)

    def _execute_action(self, action: str, environment: Environment):
        """
        Parses the action string and updates the agent's state accordingly.

        Args:
            action (str): The action command from the brain.
            environment (Environment): The simulation environment.
        """
        x, y = self.position

        if action == "MOVE_N":
            self.position = (max(0, x - 1), y)
        elif action == "MOVE_S":
            self.position = (min(self.grid_size - 1, x + 1), y)
        elif action == "MOVE_E":
            self.position = (x, min(self.grid_size - 1, y + 1))
        elif action == "MOVE_W":
            self.position = (x, max(0, y - 1))
        elif action == "GATHER":
            resources_collected = environment.remove_resource_at(self.position)
            self.fitness_score += resources_collected
