import numpy as np
import random


class LoraBrain:
    """
    A mock "brain" for an agent, simulating the behavior of an LLM-LoRA model.

    For the MVP, this class does not interact with a real LLM. Instead, it
    holds a set of weights (representing the LoRA adapter) and makes decisions
    by returning a random action. This allows the full simulation loop to be
    built and tested without the complexity of a real model.
    """

    ACTIONS = ["MOVE_N", "MOVE_S", "MOVE_E", "MOVE_W", "GATHER"]

    def __init__(
        self,
        weights: np.ndarray | None = None,
        weight_shape: tuple[int, int] = (10, 10),
    ):
        """
        Initializes the brain.

        Args:
            weights (np.ndarray | None): Pre-existing weights for this brain.
                                         If None, new random weights are generated.
            weight_shape (tuple[int, int]): The shape of the weight matrix if
                                            new weights are to be created.
        """
        if weights is not None:
            self.weights = weights
        else:
            # Initialize with small random values to represent a diverse, unbiased start.
            self.weights = np.random.randn(*weight_shape) * 0.1

    def choose_action(self, perception: np.ndarray) -> str:
        """
        Selects an action for the agent to perform.

        In this mock implementation, the perception is ignored, and a random
        action is chosen. This serves as a placeholder for the future LLM query.

        Args:
            perception (np.ndarray): The agent's view of the world (ignored in MVP).

        Returns:
            str: A string representing the chosen action.
        """
        return random.choice(self.ACTIONS)

    def get_weights(self) -> np.ndarray:
        """
        Provides access to the brain's internal weights (its "genes").

        Returns:
            np.ndarray: The array of weights.
        """
        return self.weights
