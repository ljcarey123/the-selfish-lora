"""
Contains the heuristic decision-making rules for the agent.
"""

from .movement_rules import get_possible_moves
from .pathfinding_rules import (
    find_nearest_resource,
    score_move_towards,
    score_move_away_from,
)
import random
from config_loader import GRID_SYMBOLS, GRID_SIZE, AGENT_POS


def get_energy_level(energy: float) -> str:
    """Convert energy value to level (low/medium/high)."""
    if energy < 0.3:
        return "low"
    elif energy < 0.7:
        return "medium"
    return "high"


def get_heuristic_action(
    grid: list[list[str]], energy: float, last_action: str | None = None
) -> tuple[str, str]:
    """
    Implements the heuristic decision making for the agent.
    Priority order:
    1. GATHER if on resource
    2. LOW-ENERGY survival if energy is critically low
    3. SEEK/AVOID strategy (resources/rivals)
    4. DEFAULT IDLE rules based on energy

    Returns:
        A tuple of (action, rule_name) where rule_name describes which heuristic rule was applied
    """
    energy_level = get_energy_level(energy)
    agent_r, agent_c = AGENT_POS

    # Priority 1: GATHER if on resource
    if grid[agent_r][agent_c] == GRID_SYMBOLS["resource"]:
        return "gather", "on_resource"

    # Get all possible valid moves (avoids walls and direction reversals)
    possible_moves = get_possible_moves(grid, (agent_r, agent_c), last_action)

    # If no moves are possible, stand still
    if not possible_moves:
        return "stand_still", "no_valid_moves"

    # Priority 2: LOW-ENERGY survival - if energy is critically low, find food or conserve energy
    if energy_level == "low":
        # If we're very low on energy, standing still conserves energy
        if energy < 0.1:
            return "stand_still", "low_energy_conserve"

        # Otherwise, look for nearby resources
        resource_pos = find_nearest_resource(grid, (agent_r, agent_c))
        if resource_pos:
            # Find best move towards resource
            best_move = score_move_towards(
                (agent_r, agent_c), resource_pos, possible_moves
            )
            if best_move:
                return best_move, "low_energy_seek_food"

    # Priority 3: SEEK/AVOID strategy - avoid rivals first
    rival_positions = [
        (r, c)
        for r in range(GRID_SIZE)
        for c in range(GRID_SIZE)
        if grid[r][c] == GRID_SYMBOLS["rival"]
    ]

    if rival_positions and energy_level != "low":
        # Find closest rival
        closest_rival = min(
            rival_positions,
            key=lambda p: abs(p[0] - agent_r) + abs(p[1] - agent_c),
        )

        # Find best move away from rival
        best_move = score_move_away_from(
            (agent_r, agent_c), closest_rival, possible_moves
        )
        if best_move:
            return best_move, "avoid_rival"

    # Priority 4: SEEK RESOURCES (if no rivals and not low energy)
    if energy_level != "low":
        resource_pos = find_nearest_resource(grid, (agent_r, agent_c))
        if resource_pos:
            best_move = score_move_towards(
                (agent_r, agent_c), resource_pos, possible_moves
            )
            if best_move:
                return best_move, "seek_resource"

    # Priority 5: DEFAULT IDLE rules
    if energy_level == "high":
        # Explore (pick a random valid move, but not stand_still)
        explore_moves = [a for a, _ in possible_moves if a != "stand_still"]
        if explore_moves:
            return random.choice(explore_moves), "explore_high_energy"

    # Default: stand still if no better move is found
    return "stand_still", "default_idle"
