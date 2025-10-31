from typing import List, Optional

from config_loader import GRID_SYMBOLS, AGENT_POS
from .decision_rules import get_energy_level

# Type aliases
Grid = List[List[str]]


def format_example(grid: Grid, energy: float, last_action: Optional[str] = None) -> str:
    """Format a single training example."""
    # Create display grid
    display_grid = [row[:] for row in grid]
    agent_r, agent_c = AGENT_POS
    if grid[agent_r][agent_c] == GRID_SYMBOLS["resource"]:
        display_grid[agent_r][agent_c] = GRID_SYMBOLS["on_resource"]

    # Convert grid to string
    grid_str = "\n".join(" ".join(row) for row in display_grid)

    # Format energy level
    energy_str = f"Energy: {energy:.2f} ({get_energy_level(energy)})"

    # Format history
    history_str = f"Last action: {last_action}" if last_action else "No previous action"

    return f"PERCEPTION:\n{grid_str}\n\n{energy_str}\n{history_str}\n\nACTION:"
