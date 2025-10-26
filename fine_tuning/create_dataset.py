import json
import random
import os
import argparse
from itertools import combinations
from collections import defaultdict
import sys

# Add parent directory to path to import config_loader
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config_loader import SYSTEM_PROMPT, VALID_ACTIONS

GRID_SIZE = 3
OUTPUT_DIR = "data"


def _generate_all_distractor_combinations(safe_spots, min_walls=1, max_walls=3):
    """
    Generates all possible combinations of placing 1-3 walls in safe spots.
    Returns a list of distractor position combinations.
    """
    all_combinations = []

    for num_walls in range(min_walls, min(max_walls + 1, len(safe_spots) + 1)):
        for combo in combinations(safe_spots, num_walls):
            all_combinations.append(list(combo))

    return all_combinations


def _add_distractors(grid, unsafe_coords, distractor_positions):
    """
    Adds specific 'W' (wall) distractors to the grid at given positions.
    """
    grid_copy = [row[:] for row in grid]  # Deep copy

    for pos in distractor_positions:
        grid_copy[pos[0]][pos[1]] = "W"

    return grid_copy


def _get_all_wall_patterns():
    """
    Returns 9 wall patterns:
    - No walls
    - 4 single sides (N, S, E, W)
    - 4 corner patterns (NW, NE, SW, SE)
    """
    return [
        [],  # No walls
        [(0, 0), (0, 1), (0, 2)],  # North wall
        [(2, 0), (2, 1), (2, 2)],  # South wall
        [(0, 2), (1, 2), (2, 2)],  # East wall
        [(0, 0), (1, 0), (2, 0)],  # West wall
        [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0)],  # North + West (NW corner)
        [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)],  # North + East (NE corner)
        [(2, 0), (2, 1), (2, 2), (1, 0), (0, 0)],  # South + West (SW corner)
        [(2, 0), (2, 1), (2, 2), (1, 2), (0, 2)],  # South + East (SE corner)
    ]


def _is_wall_pattern_valid(wall_positions, unsafe_coords):
    """
    Check if a wall pattern is valid (doesn't block unsafe coordinates).
    
    Args:
        wall_positions: List of (row, col) tuples for wall positions
        unsafe_coords: List of (row, col) tuples that must remain clear
    
    Returns:
        True if pattern is valid, False otherwise
    """
    for wall_pos in wall_positions:
        if wall_pos in unsafe_coords:
            return False
    return True


def _get_valid_history_actions(wall_positions):
    """
    Get valid single-action history based on wall positions.
    Agent couldn't have come from a direction that has walls.
    
    Args:
        wall_positions: List of (row, col) tuples for wall positions
    
    Returns:
        List of valid movement actions (N, S, E, W) - no Gather
    """
    # Check which directions have walls
    blocked_directions = []
    
    # North wall (row 0)
    if any(pos[0] == 0 for pos in wall_positions):
        blocked_directions.append("North")
    
    # South wall (row 2)
    if any(pos[0] == 2 for pos in wall_positions):
        blocked_directions.append("South")
    
    # East wall (col 2)
    if any(pos[1] == 2 for pos in wall_positions):
        blocked_directions.append("East")
    
    # West wall (col 0)
    if any(pos[1] == 0 for pos in wall_positions):
        blocked_directions.append("West")
    
    # Return valid movement actions (no Gather)
    movement_actions = ["North", "South", "East", "West"]
    valid_actions = [action for action in movement_actions if action not in blocked_directions]
    
    return valid_actions if valid_actions else movement_actions  # Fallback to all if none valid


def _generate_forced_scenario(scenario_type, orientation, include_history=True):
    """
    Constructs a grid where a specific action is the only logical choice,
    based on the scenario type and a given orientation.
    """
    grid = [["." for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    unsafe_coords = []

    if scenario_type == "GATHER":
        grid[1][1] = "R"
        correct_action = "Gather"
        unsafe_coords.append((1, 1))
    else:
        grid[1][1] = "A"
        resource_pos = orientation
        grid[resource_pos[0]][resource_pos[1]] = "R"
        unsafe_coords.append(resource_pos)

        if scenario_type == "ADJACENT_MOVE":
            if resource_pos == (0, 1):
                correct_action = "North"
            elif resource_pos == (2, 1):
                correct_action = "South"
            elif resource_pos == (1, 2):
                correct_action = "East"
            elif resource_pos == (1, 0):
                correct_action = "West"

        elif scenario_type == "DIAGONAL_MOVE":
            # NEW BALANCED TIE-BREAKER:
            # Each direction gets one "win" from a diagonal scenario,
            # ensuring perfect balance in the final dataset.
            if resource_pos == (0, 0):
                correct_action = "North"  # NW -> N
            elif resource_pos == (0, 2):
                correct_action = "East"  # NE -> E
            elif resource_pos == (2, 0):
                correct_action = "West"  # SW -> W
            elif resource_pos == (2, 2):
                correct_action = "South"  # SE -> S

            # Keep both potential paths clear of distractors for the specific corner
            if resource_pos == (0, 0):
                unsafe_coords.extend([(0, 1), (1, 0)])
            elif resource_pos == (0, 2):
                unsafe_coords.extend([(0, 1), (1, 2)])
            elif resource_pos == (2, 0):
                unsafe_coords.extend([(2, 1), (1, 0)])
            elif resource_pos == (2, 2):
                unsafe_coords.extend([(2, 1), (1, 2)])

    # Calculate safe spots for distractors
    empty_spots = [
        (r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE) if grid[r][c] == "."
    ]
    safe_spots = [pos for pos in empty_spots if pos not in unsafe_coords]

    return grid, correct_action, safe_spots


def _format_user_prompt(grid, history):
    """Helper function to format the user part of the prompt."""
    grid_str = "\n".join(" ".join(row) for row in grid)

    # Only include history section if history is provided
    if history:
        history_str = "\n".join(f"- {h}" for h in history)
        return f"""PERCEPTION:
{grid_str}

HISTORY (Last 3 actions):
{history_str}

ACTION:"""
    else:
        return f"""PERCEPTION:
{grid_str}

ACTION:"""


def generate_dataset(include_history=True):
    """Generates a dataset with all possible distractor combinations for each scenario."""
    print("Generating dataset with exhaustive distractor combinations...")

    # Define all scenarios to generate
    tasks = []
    
    # 1. Gather Scenario (agent on resource)
    tasks.append(("GATHER", None))

    # 2. Adjacent Move Scenarios (resource 1 step away: N, S, E, W)
    adjacent_orientations = [(0, 1), (2, 1), (1, 2), (1, 0)]  # N, S, E, W
    for orientation in adjacent_orientations:
        tasks.append(("ADJACENT_MOVE", orientation))

    # 3. Diagonal Move Scenarios (resource diagonal: NW, NE, SW, SE)
    diagonal_orientations = [(0, 0), (0, 2), (2, 0), (2, 2)]  # NW, NE, SW, SE
    for orientation in diagonal_orientations:
        tasks.append(("DIAGONAL_MOVE", orientation))

    final_dataset = []
    action_to_examples = defaultdict(
        list
    )  # Track examples by action for stratified split
    seen_grids = set()

    for scenario_type, orientation in tasks:
        # Get the base grid and safe spots
        base_grid, action, safe_spots = _generate_forced_scenario(
            scenario_type, orientation, include_history
        )

        # Use systematic wall patterns (9 configurations: no walls + 8 wall patterns)
        all_wall_patterns = _get_all_wall_patterns()
        distractor_combos = []
        
        # Get unsafe coordinates (agent, resource, and paths)
        unsafe_coords = []
        if scenario_type == "GATHER":
            unsafe_coords.append((1, 1))  # Resource at center
        else:
            unsafe_coords.append((1, 1))  # Agent at center
            unsafe_coords.append(orientation)  # Resource position
            
            # For diagonal moves, keep both paths clear
            if scenario_type == "DIAGONAL_MOVE":
                if orientation == (0, 0):  # NW
                    unsafe_coords.extend([(0, 1), (1, 0)])
                elif orientation == (0, 2):  # NE
                    unsafe_coords.extend([(0, 1), (1, 2)])
                elif orientation == (2, 0):  # SW
                    unsafe_coords.extend([(2, 1), (1, 0)])
                elif orientation == (2, 2):  # SE
                    unsafe_coords.extend([(2, 1), (1, 2)])
        
        # Filter valid wall patterns
        for wall_pattern in all_wall_patterns:
            if _is_wall_pattern_valid(wall_pattern, unsafe_coords):
                distractor_combos.append(wall_pattern)

        for distractor_positions in distractor_combos:
            grid = _add_distractors(base_grid, [], distractor_positions)
            grid_tuple = tuple(map(tuple, grid))

            if grid_tuple in seen_grids:
                continue  # Skip duplicate grids

            seen_grids.add(grid_tuple)

            # Generate examples based on history flag
            if include_history:
                # Create one example for EACH valid history action
                valid_history_actions = _get_valid_history_actions(distractor_positions)
                
                if valid_history_actions:
                    # Generate an example for each valid history action
                    for history_action in valid_history_actions:
                        history = [history_action]
                        user_prompt_string = _format_user_prompt(grid, history)

                        example = [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt_string},
                            {"role": "assistant", "content": action},
                        ]

                        final_dataset.append(example)
                        action_to_examples[action].append(example)
                else:
                    # No valid history - create example without history
                    history = []
                    user_prompt_string = _format_user_prompt(grid, history)

                    example = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt_string},
                        {"role": "assistant", "content": action},
                    ]

                    final_dataset.append(example)
                    action_to_examples[action].append(example)
            else:
                # No history mode - single example per grid
                history = []
                user_prompt_string = _format_user_prompt(grid, history)

                example = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt_string},
                    {"role": "assistant", "content": action},
                ]

                final_dataset.append(example)
                action_to_examples[action].append(example)

    print(f"\nGenerated {len(final_dataset)} total examples")
    print("Action distribution:")
    for action in sorted(action_to_examples.keys()):
        pct = (len(action_to_examples[action]) / len(final_dataset)) * 100
        print(f"  {action}: {len(action_to_examples[action])} examples ({pct:.1f}%)")

    return final_dataset, action_to_examples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate fine-tuning dataset")
    parser.add_argument(
        "--include-history",
        action="store_true",
        help="Include history in examples (default: no history for simplicity)",
    )
    args = parser.parse_args()
    
    mode = "with history" if args.include_history else "without history (perception only)"
    print(f"Generating dataset ({mode})...\n")
    
    full_dataset, action_to_examples = generate_dataset(include_history=args.include_history)

    # Stratified split: maintain action distribution across splits
    train_dataset = []
    validation_dataset = []
    test_dataset = []

    # Define split ratios
    train_ratio = 0.70
    val_ratio = 0.15
    test_ratio = 0.15

    print("\nPerforming stratified split to maintain class distribution...")

    for action, examples in action_to_examples.items():
        # Shuffle examples for this action
        random.shuffle(examples)

        n = len(examples)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        train_dataset.extend(examples[:train_end])
        validation_dataset.extend(examples[train_end:val_end])
        test_dataset.extend(examples[val_end:])

    # Shuffle the final splits
    random.shuffle(train_dataset)
    random.shuffle(validation_dataset)
    random.shuffle(test_dataset)

    # Define output directory and filenames
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    train_file = os.path.join(OUTPUT_DIR, "train_data.json")
    val_file = os.path.join(OUTPUT_DIR, "validation_data.json")
    test_file = os.path.join(OUTPUT_DIR, "test_data.json")

    # Save the datasets
    with open(train_file, "w") as f:
        json.dump(train_dataset, f, indent=2)
    with open(val_file, "w") as f:
        json.dump(validation_dataset, f, indent=2)
    with open(test_file, "w") as f:
        json.dump(test_dataset, f, indent=2)

    print(f"\nSuccessfully generated and split dataset:")
    print(f"- Training set: {len(train_dataset)} examples -> {train_file}")
    print(f"- Validation set: {len(validation_dataset)} examples -> {val_file}")
    print(f"- Test set: {len(test_dataset)} examples -> {test_file}")
