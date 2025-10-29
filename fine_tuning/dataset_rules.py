"""
This module contains the rules and heuristics for generating the training dataset.
It includes movement logic, action selection, and other decision-making rules.
"""

from typing import List, Tuple, Optional, Dict, Any
import random
import math
from config_loader import GRID_SYMBOLS, GRID_SIZE, AGENT_POS, VALID_ACTIONS

# Type aliases for better code readability
Position = Tuple[int, int]
Grid = List[List[str]]
Action = str

# Movement directions and their opposites
CARDINAL_DIRECTIONS = ["north", "south", "east", "west"]
OPPOSITE_DIRECTIONS = {
    "north": "south",
    "south": "north",
    "east": "west",
    "west": "east",
}


def get_energy_level(energy: float) -> str:
    """Convert energy value to level (low/medium/high)."""
    if energy < 0.3:
        return "low"
    elif energy < 0.7:
        return "medium"
    return "high"


def get_possible_moves(
    grid: Grid,
    pos: Position,
    last_action: Optional[Action] = None,
    avoid_reverse: bool = True,
) -> List[Tuple[Action, Position]]:
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


def is_valid_move(grid: Grid, pos: Position) -> bool:
    """Check if a position is a valid move (within bounds and not a wall)."""
    row, col = pos
    return (
        0 <= row < GRID_SIZE
        and 0 <= col < GRID_SIZE
        and grid[row][col] != GRID_SYMBOLS["wall"]
    )


def get_valid_last_action(grid: Grid) -> Optional[Action]:
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
        action for action, (r, c) in possible_moves.items() 
        if is_valid_move(grid, (r, c))
    ]
    
    # 20% chance to return None (no last action)
    if not valid_actions or random.random() < 0.2:
        return None
        
    return random.choice(valid_actions)


def find_nearest_resource(grid: Grid, start_pos: Position) -> Optional[Position]:
    """Find the position of the nearest resource using BFS."""
    from collections import deque

    if grid[start_pos[0]][start_pos[1]] == GRID_SYMBOLS["resource"]:
        return start_pos

    visited = set([start_pos])
    queue = deque([start_pos])

    while queue:
        r, c = queue.popleft()

        # Check all four directions
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc

            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and (nr, nc) not in visited:

                if grid[nr][nc] == GRID_SYMBOLS["resource"]:
                    return (nr, nc)

                if grid[nr][nc] != GRID_SYMBOLS["wall"]:
                    visited.add((nr, nc))
                    queue.append((nr, nc))

    return None


def score_move_towards(
    current_pos: Position,
    target_pos: Position,
    possible_moves: List[Tuple[Action, Position]],
) -> Optional[Action]:
    """Score moves based on how well they move toward a target position."""
    if not possible_moves:
        return None

    best_move = None
    best_score = -float("inf")

    # Calculate current distance to target
    r1, c1 = current_pos
    r2, c2 = target_pos
    current_dist = abs(r1 - r2) + abs(c1 - c2)

    for action, (new_r, new_c) in possible_moves:
        # Calculate new distance to target
        new_dist = abs(new_r - r2) + abs(new_c - c2)

        # Score based on distance improvement (closer is better)
        score = current_dist - new_dist

        # Slight preference for cardinal directions
        if action in [f"move_{d}" for d in CARDINAL_DIRECTIONS]:
            score += 0.1

        if score > best_score:
            best_score = score
            best_move = action

    return best_move if best_score > 0 else None


def score_move_away_from(
    current_pos: Position,
    avoid_pos: Position,
    possible_moves: List[Tuple[Action, Position]],
) -> Optional[Action]:
    """Score moves based on how well they move away from a position."""
    if not possible_moves:
        return None

    best_move = None
    best_score = -float("inf")

    # Calculate current distance from the position to avoid
    r1, c1 = current_pos
    r2, c2 = avoid_pos
    current_dist = abs(r1 - r2) + abs(c1 - c2)

    for action, (new_r, new_c) in possible_moves:
        # Calculate new distance from the position to avoid
        new_dist = abs(new_r - r2) + abs(new_c - c2)

        # Score based on distance increase (farther is better)
        score = new_dist - current_dist

        # Slight preference for cardinal directions
        if action in [f"move_{d}" for d in CARDINAL_DIRECTIONS]:
            score += 0.1

        if score > best_score:
            best_score = score
            best_move = action

    return best_move if best_score > 0 else None


def get_center_score(pos: Position) -> float:
    """Calculate how close a position is to the center of the grid."""
    center = (GRID_SIZE // 2, GRID_SIZE // 2)
    distance = abs(pos[0] - center[0]) + abs(pos[1] - center[1])
    return (GRID_SIZE - distance) / GRID_SIZE  # Higher is better


def get_anti_stuck_move(
    grid: Grid, current_pos: Position, last_action: Optional[Action]
) -> Optional[Action]:
    """Get a safe move if the last action would lead to a wall or out of bounds.
    
    Args:
        grid: The current grid state
        current_pos: Current (row, col) position
        last_action: The last action taken by the agent
        
    Returns:
        A safe action to take, or None if no anti-stuck move is needed
    """
    if not last_action or not last_action.startswith("move_"):
        return None
        
    direction = last_action[5:]
    row, col = current_pos
    
    # Calculate next position based on last action
    next_pos = {
        "north": (row - 1, col),
        "south": (row + 1, col),
        "east": (row, col + 1),
        "west": (row, col - 1),
    }.get(direction)
    
    if not next_pos:
        return None
        
    n_row, n_col = next_pos
    
    # Check if next position is valid
    is_valid = (
        0 <= n_row < GRID_SIZE
        and 0 <= n_col < GRID_SIZE
        and grid[n_row][n_col] != GRID_SYMBOLS["wall"]
    )
    
    # If the move is valid, no need for anti-stuck behavior
    if is_valid:
        return None
        
    # Get all possible safe moves (excluding the one that would cause a collision)
    possible_moves = get_possible_moves(
        grid, 
        current_pos, 
        last_action=last_action,
        avoid_reverse=True  # Don't reverse direction to avoid oscillation
    )
    
    if not possible_moves:
        return None
        
    # Choose the move that goes towards the center of the grid
    center = (GRID_SIZE // 2, GRID_SIZE // 2)
    return score_move_towards(current_pos, center, possible_moves)


def get_open_space_score(grid: Grid, pos: Position) -> float:
    """Calculate how much open space is around a position."""
    open_space = 0
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        r, c = pos[0] + dr, pos[1] + dc
        if (
            0 <= r < GRID_SIZE
            and 0 <= c < GRID_SIZE
            and grid[r][c] != GRID_SYMBOLS["wall"]
            and (r, c) != pos
        ):
            open_space += 1
    return open_space / 4.0  # Normalize to 0-1
