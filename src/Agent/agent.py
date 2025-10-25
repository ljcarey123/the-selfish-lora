import numpy as np
import random
from LoraBrain.lora_brain import LoraBrain
from Environment.environment import Environment
import uuid


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
        self.id = str(uuid.uuid4())[:8]  # Short, unique ID for logging
        self.brain = brain
        self.position = initial_position
        self.grid_size = grid_size
        self.fitness_score = 0
        self.path_history = [("START", initial_position)]
        self.action_history = []

    def decide_and_act(self, environment: Environment, base_model, tokenizer):
        """
        The main loop for an agent's turn. It perceives, decides, and acts.

        Args:
            environment (Environment): The simulation environment.
            base_model: The shared base LLM.
            tokenizer: The shared tokenizer.
        """
        # 1. Perceive the environment
        perception = environment.get_perception(self.position)

        # 2. Decide on an action using the brain
        # Get the last 3 actions
        recent_history = self.action_history[-3:]
        action_string, raw_response = self.brain.choose_action(
            perception, base_model, tokenizer, self.id, recent_history
        )

        # --- Unparsable Response Penalty ---
        if action_string is None:
            print(
                f"[WARNING] Agent {self.id} produced unparsable response: '{raw_response}'. Falling back to random action."
            )
            self.fitness_score -= 2  # Harsh penalty for failing to communicate
            action_string = random.choice(LoraBrain.ACTIONS)

        self.action_history.append(action_string)

        print(
            f"Agent {self.id}: Chose action '{action_string}' at position {self.position}"
        )

        # 3. Act based on the decision
        self._execute_action(action_string, environment)

    def _execute_action(self, action: str, environment: Environment):
        """
        Executes the given action and updates the agent's state.
        Applies fitness rewards and penalties based on action outcomes.
        """

        x, y = self.position

        if action.upper() in ["NORTH", "SOUTH", "EAST", "WEST"]:
            # Calculate intended new position
            if action.upper() == "NORTH":
                new_x, new_y = x - 1, y
            elif action.upper() == "SOUTH":
                new_x, new_y = x + 1, y
            elif action.upper() == "EAST":
                new_x, new_y = x, y + 1
            elif action.upper() == "WEST":
                new_x, new_y = x, y - 1
            
            # Check if move is valid (within bounds)
            if 0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size:
                # Valid move
                self.position = (new_x, new_y)
            else:
                # Hit a wall - small penalty
                self.fitness_score -= 0.1
                # Position stays the same
            
            self.path_history.append(("MOVE", self.position))

        elif action.upper() == "GATHER":
            resources_collected = environment.remove_resource_at(self.position)
            if resources_collected > 0:
                # Successful gather - significant reward
                self.fitness_score += 5.0
                self.path_history.append(("GATHER_SUCCESS", self.position))
            else:
                # Failed gather attempt - small penalty
                self.fitness_score -= 0.1
                self.path_history.append(("GATHER_FAIL", self.position))
