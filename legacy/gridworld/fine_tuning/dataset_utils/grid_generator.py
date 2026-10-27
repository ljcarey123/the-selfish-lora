import random
from typing import List

from config_loader import GRID_SIZE, AGENT_POS, GRID_SYMBOLS

# Type alias
Grid = List[List[str]]


def generate_random_grid() -> Grid:
    """Generate a random 5x5 grid with walls, resources, and rivals.

    Returns:
        A 5x5 grid with the agent at (2,2), walls, and random resources and rivals.
        The agent will be placed on a resource in 20% of cases.
    """
    grid = [[GRID_SYMBOLS["empty"] for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    agent_r, agent_c = AGENT_POS

    # 20% chance to place agent on a resource
    agent_on_resource = random.random() < 0.2

    # Place agent (will be overwritten with resource if needed)
    grid[agent_r][agent_c] = GRID_SYMBOLS["agent"]

    # If agent is on resource, mark that position
    if agent_on_resource:
        grid[agent_r][agent_c] = GRID_SYMBOLS["resource"]

    # Place walls (cardinal or corner, full or partial)
    wall_type = random.choice(["cardinal", "corner"])
    wall_coverage = random.choice([1, 2])  # 1 or 2 rows/columns

    if wall_type == "cardinal":
        direction = random.choice(["north", "south", "east", "west"])

        if direction == "north":
            # Place wall at top (1 or 2 rows)
            for r in range(wall_coverage):
                for c in range(GRID_SIZE):
                    if (r, c) != AGENT_POS:  # Don't overwrite agent
                        grid[r][c] = GRID_SYMBOLS["wall"]

        elif direction == "south":
            # Place wall at bottom (1 or 2 rows)
            for r in range(GRID_SIZE - wall_coverage, GRID_SIZE):
                for c in range(GRID_SIZE):
                    if (r, c) != AGENT_POS:
                        grid[r][c] = GRID_SYMBOLS["wall"]

        elif direction == "west":
            # Place wall at left (1 or 2 columns)
            for c in range(wall_coverage):
                for r in range(GRID_SIZE):
                    if (r, c) != AGENT_POS:
                        grid[r][c] = GRID_SYMBOLS["wall"]

        else:  # east
            # Place wall at right (1 or 2 columns)
            for c in range(GRID_SIZE - wall_coverage, GRID_SIZE):
                for r in range(GRID_SIZE):
                    if (r, c) != AGENT_POS:
                        grid[r][c] = GRID_SYMBOLS["wall"]

    else:  # corner wall
        corner = random.choice(["nw", "ne", "sw", "se"])

        if corner == "nw":
            # Top row and left column
            for c in range(GRID_SIZE):  # Top row
                if (0, c) != AGENT_POS:
                    grid[0][c] = GRID_SYMBOLS["wall"]
            for r in range(GRID_SIZE):  # Left column
                if (r, 0) != AGENT_POS:
                    grid[r][0] = GRID_SYMBOLS["wall"]

        elif corner == "ne":
            # Top row and right column
            for c in range(GRID_SIZE):  # Top row
                if (0, c) != AGENT_POS:
                    grid[0][c] = GRID_SYMBOLS["wall"]
            for r in range(GRID_SIZE):  # Right column
                if (r, GRID_SIZE - 1) != AGENT_POS:
                    grid[r][GRID_SIZE - 1] = GRID_SYMBOLS["wall"]

        elif corner == "sw":
            # Bottom row and left column
            for c in range(GRID_SIZE):  # Bottom row
                if (GRID_SIZE - 1, c) != AGENT_POS:
                    grid[GRID_SIZE - 1][c] = GRID_SYMBOLS["wall"]
            for r in range(GRID_SIZE):  # Left column
                if (r, 0) != AGENT_POS:
                    grid[r][0] = GRID_SYMBOLS["wall"]

        else:  # se
            # Bottom row and right column
            for c in range(GRID_SIZE):  # Bottom row
                if (GRID_SIZE - 1, c) != AGENT_POS:
                    grid[GRID_SIZE - 1][c] = GRID_SYMBOLS["wall"]
            for r in range(GRID_SIZE):  # Right column
                if (r, GRID_SIZE - 1) != AGENT_POS:
                    grid[r][GRID_SIZE - 1] = GRID_SYMBOLS["wall"]

    # Update available positions (excluding walls and agent)
    available_positions = [
        (r, c)
        for r in range(GRID_SIZE)
        for c in range(GRID_SIZE)
        if grid[r][c] == GRID_SYMBOLS["empty"] and (r, c) != AGENT_POS
    ]

    # Place resources (30% chance of 0 additional resources if agent is not on one)
    if agent_on_resource:
        # If agent is already on a resource, place 0-2 additional resources
        num_resources = random.choices([0, 1, 2], weights=[0.4, 0.4, 0.2], k=1)[0]
    else:
        # Otherwise, place 0-3 resources (30% chance of 0)
        num_resources = random.choices([0, 1, 2, 3], weights=[0.3, 0.3, 0.2, 0.2], k=1)[
            0
        ]

    if num_resources > 0 and available_positions:
        resource_positions = random.sample(
            available_positions, min(num_resources, len(available_positions))
        )
        for r, c in resource_positions:
            grid[r][c] = GRID_SYMBOLS["resource"]
            available_positions.remove((r, c))  # Update available positions

    # Place rivals
    for _ in range(1):  # Only one rival
        if random.random() < 0.2 and available_positions:
            r, c = random.choice(available_positions)
            grid[r][c] = GRID_SYMBOLS["rival"]

    return grid
