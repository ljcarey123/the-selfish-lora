"""
Contains core movement rules and position validation logic.
"""

from typing import List, Tuple, Optional
from config_loader import GRID_SYMBOLS, GRID_SIZE, AGENT_POS
import random


# Movement directions and their opposites
CARDINAL_DIRECTIONS = ["north", "south", "east", "west"]
OPPOSITE_DIRECTIONS = {
    "north": "south",
    "south": "north",
    "east": "west",
    "west": "east",
}


def get_possible_moves(
    grid: List[List[str]],
    pos: Tuple[int, int],
    last_action: Optional[str] = None,
    avoid_reverse: bool = True,
) -> List[Tuple[str, Tuple[int, int]]]:
    """Get all possible valid moves from the current position.

    Args:
        grid: The current grid state
        pos: Current (row, col) position
        last_action: The last action taken (to avoid reversals)
        avoid_reverse: Whether to avoid reversing the last action

    Returns:
        List of (action, (new_row, new_col)) tuples
    """
    row, col = pos
    moves = []

    # Only consider last direction if it was a move and we want to avoid reversals
    last_direction = None
    if avoid_reverse and last_action and last_action.startswith("move_"):
        last_direction = last_action[5:]

    # Check each possible move
    for action in [f"move_{d}" for d in CARDINAL_DIRECTIONS]:
        direction = action[5:]

        # Skip if this is the reverse of the last move
        if last_direction and direction == OPPOSITE_DIRECTIONS.get(last_direction):
            continue

        # Calculate new position
        if direction == "north":
            new_row, new_col = row - 1, col
        elif direction == "south":
            new_row, new_col = row + 1, col
        elif direction == "east":
            new_row, new_col = row, col + 1
        else:  # west
            new_row, new_col = row, col - 1

        # Check if new position is valid (within bounds and not a wall)
        if (
            0 <= new_row < GRID_SIZE
            and 0 <= new_col < GRID_SIZE
            and grid[new_row][new_col] != GRID_SYMBOLS["wall"]
        ):
            moves.append((action, (new_row, new_col)))

    return moves


def is_valid_move(grid: List[List[str]], pos: Tuple[int, int]) -> bool:
    """Check if a position is a valid move (within bounds and not a wall)."""
    row, col = pos
    return (
        0 <= row < GRID_SIZE
        and 0 <= col < GRID_SIZE
        and grid[row][col] != GRID_SYMBOLS["wall"]
    )


def get_valid_last_action(grid: List[List[str]]) -> Optional[str]:
    """Get a random valid last action based on the grid state.

    Args:
        grid: The current grid state

    Returns:
        A valid action that could have been taken in the previous state,
        or None to indicate no previous action.
    """
    agent_r, agent_c = AGENT_POS

    # Define all possible move directions with their resulting positions
    possible_moves = {
        "move_north": (agent_r - 1, agent_c),
        "move_south": (agent_r + 1, agent_c),
        "move_east": (agent_r, agent_c + 1),
        "move_west": (agent_r, agent_c - 1),
    }

    # Filter to only include valid moves (not blocked by walls or out of bounds)
    valid_actions = [
        action
        for action, (r, c) in possible_moves.items()
        if is_valid_move(grid, (r, c))
    ]

    # 20% chance to return None (no last action)
    if not valid_actions or random.random() < 0.2:
        return None

    return random.choice(valid_actions)


def get_adjacent_positions(pos: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Get adjacent positions (N, S, E, W) to the given position."""
    row, col = pos
    return [
        (row - 1, col),  # North
        (row + 1, col),  # South
        (row, col + 1),  # East
        (row, col - 1),  # West
    ]
