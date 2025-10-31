"""
Contains pathfinding and move scoring rules for agent navigation.
"""

from typing import List, Tuple, Optional, List
from collections import deque
from .movement_rules import get_adjacent_positions
from config_loader import GRID_SYMBOLS, GRID_SIZE


def find_nearest_resource(
    grid: List[List[str]], start_pos: Tuple[int, int]
) -> Optional[Tuple[int, int]]:
    """Find the position of the nearest resource using BFS."""
    visited = set()
    queue = deque([(start_pos, [])])

    while queue:
        pos, path = queue.popleft()
        if pos in visited:
            continue

        row, col = pos
        if not (0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE):
            continue

        if grid[row][col] == GRID_SYMBOLS["resource"]:
            return pos

        visited.add(pos)

        # Add adjacent positions to queue
        for neighbor in get_adjacent_positions(pos):
            n_row, n_col = neighbor
            if (
                0 <= n_row < GRID_SIZE
                and 0 <= n_col < GRID_SIZE
                and grid[n_row][n_col] != GRID_SYMBOLS["wall"]
            ):
                queue.append((neighbor, path + [neighbor]))

    return None


def score_move_towards(
    current_pos: Tuple[int, int],
    target_pos: Tuple[int, int],
    possible_moves: List[Tuple[str, Tuple[int, int]]],
) -> Optional[str]:
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
        if action in [f"move_{d}" for d in ["north", "south", "east", "west"]]:
            score += 0.1

        if score > best_score:
            best_score = score
            best_move = action

    return best_move if best_score > 0 else None


def score_move_away_from(
    current_pos: Tuple[int, int],
    avoid_pos: Tuple[int, int],
    possible_moves: List[Tuple[str, Tuple[int, int]]],
) -> Optional[str]:
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
        if action in [f"move_{d}" for d in ["north", "south", "east", "west"]]:
            score += 0.1

        if score > best_score:
            best_score = score
            best_move = action

    return best_move if best_score > 0 else None
