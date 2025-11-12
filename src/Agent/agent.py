import numpy as np
import random
from LoraBrain.lora_brain import LoraBrain
from Environment.environment import Environment
from config_loader import GRID_SYMBOLS
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
        # Energy tracking (starts at 100.0, no upper limit)
        self.energy = 100.0  # Start with 100 energy
        self.initial_energy = 100.0
        self.consecutive_invalid_moves = 0  # Track consecutive invalid moves
        # We use energy directly for selection
        self.path_history = [("START", initial_position)]
        self.action_history = []
        self.consecutive_invalid_moves = 0  # Initialize invalid move counter
        
    def reset(self, position: tuple[int, int]):
        """
        Reset the agent's state for a new generation or tournament.
        
        Args:
            position: The new position to place the agent at
        """
        self.energy = 100.0
        self.position = position
        self.path_history = [("START", position)]
        self.action_history = []
        self.consecutive_invalid_moves = 0

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

        # Check if agent is on a resource
        x, y = self.position
        agent_on_resource = (environment.grid[x, y] == GRID_SYMBOLS["resource"])

        # 2. Decide on an action using the brain
        # Get the last action (or None if first turn)
        last_action = self.action_history[-1] if self.action_history else None

        action_string, raw_response = self.brain.choose_action(
            perception, base_model, tokenizer, self.id, self.energy, last_action, agent_on_resource
        )

        # --- Unparsable Response Penalty ---
        if action_string is None:
            print(
                f"[WARNING] Agent {self.id} produced unparsable response: '{raw_response}'. No action taken."
            )
            self.energy = max(0, self.energy - 2.0)  # Harsh penalty for failing to communicate
            self.consecutive_invalid_moves += 1  # Count as invalid move
            # Store as UNPARSEABLE in history (not a valid action)
            self.action_history.append("UNPARSEABLE")
            # Add to path history but don't move
            self.path_history.append(("UNPARSEABLE", self.position))
            
            # Check for disqualification (5+ consecutive invalid moves)
            if self.consecutive_invalid_moves >= 5:
                self.energy = 0
                self.path_history.append(("DISQUALIFIED", self.position))
                
            return  # Don't execute any action
        
        # --- Fitness Shaping: Reward Valid Actions ---
        # Small reward for producing a valid action (prevents genetic drift)
        self.energy += 0.05

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
        Also manages energy consumption and restoration.
        """

        x, y = self.position
        action_lower = action.lower()

        # Movement actions
        if action_lower in ["move_north", "move_south", "move_east", "move_west"]:
            # Calculate intended new position
            if action_lower == "move_north":
                new_x, new_y = x - 1, y
            elif action_lower == "move_south":
                new_x, new_y = x + 1, y
            elif action_lower == "move_east":
                new_x, new_y = x, y + 1
            elif action_lower == "move_west":
                new_x, new_y = x, y - 1

            # Check if move is valid (within bounds)
            if 0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size:
                # Valid move - costs 1.0 energy
                self.position = (new_x, new_y)
                self.energy = max(0, self.energy - 1.0)
                self.consecutive_invalid_moves = 0  # Reset counter on valid move
            else:
                # Hit a wall - still costs 1.0 energy and counts as invalid
                self.energy = max(0, self.energy - 1.0)
                self.consecutive_invalid_moves += 1
                
                # Check for disqualification (5+ consecutive invalid moves)
                if self.consecutive_invalid_moves >= 5:
                    self.energy = 0
                    self.path_history.append(("DISQUALIFIED", self.position))
            
            # Cap energy at 0
            self.energy = max(0, self.energy)
            self.path_history.append(("MOVE", self.position))

        elif action_lower == "gather":
            resources_collected = environment.remove_resource_at(self.position)
            if resources_collected > 0:
                # Successful gather - gain 50 energy (no upper limit)
                self.energy += 50.0
                self.path_history.append(("GATHER_SUCCESS", self.position))
                self.consecutive_invalid_moves = 0  # Reset counter on successful action
            else:
                # Failed gather attempt - costs 0.1 energy (not counted as invalid move)
                self.energy = max(0, self.energy - 0.1)
                self.path_history.append(("GATHER_FAIL", self.position))

        elif action_lower == "stand_still":
            # Standing still is a valid action with an energy cost (metabolism)
            self.energy = max(0, self.energy - 0.1)
            self.path_history.append(("STAND_STILL", self.position))
            self.consecutive_invalid_moves = 0  # Reset counter on valid action
