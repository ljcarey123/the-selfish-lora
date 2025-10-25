import json
import random
import os
from itertools import combinations
from collections import defaultdict

SYSTEM_PROMPT = """You are a simple robot agent. You exist within a grid. Your goal is to navigate to the 'R' (resource). Your only possible outputs are the single words: 'North', 'South', 'East', 'West', 'Gather'."""
VALID_ACTIONS = ["North", "South", "East", "West", "Gather"]

# --- Configuration ---
# You can easily control the final dataset size and balance from here.
NUM_GATHER_EXAMPLES = 5
NUM_ADJACENT_MOVE_EXAMPLES = 8  # Creates 8 examples for each of 4 directions (N,S,E,W)
NUM_DIAGONAL_MOVE_EXAMPLES = 4  # Creates 4 examples for each of 4 corners (NW,NE,SW,SE)
GRID_SIZE = 3
OUTPUT_FILE = "data/finetuning_data.json"


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


def _generate_forced_scenario(scenario_type, orientation):
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

    history_str = "\n".join(f"- {h}" for h in history)

    return f"""PERCEPTION:
{grid_str}

HISTORY (Last 3 actions):
{history_str}

ACTION:"""


def generate_dataset():
    """Generates a dataset with all possible distractor combinations for each scenario."""
    print("Generating dataset with exhaustive distractor combinations...")

    tasks = []
    # 1. Gather Scenarios
    for _ in range(NUM_GATHER_EXAMPLES):
        tasks.append(("GATHER", None))

    # 2. Adjacent Move Scenarios
    adjacent_orientations = [(0, 1), (2, 1), (1, 2), (1, 0)]  # N, S, E, W
    for _ in range(NUM_ADJACENT_MOVE_EXAMPLES):
        for orientation in adjacent_orientations:
            tasks.append(("ADJACENT_MOVE", orientation))

    # 3. Diagonal Move Scenarios
    diagonal_orientations = [(0, 0), (0, 2), (2, 0), (2, 2)]  # NW, NE, SW, SE
    for _ in range(NUM_DIAGONAL_MOVE_EXAMPLES):
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
            scenario_type, orientation
        )

        # Generate all possible distractor combinations
        distractor_combos = _generate_all_distractor_combinations(
            safe_spots, min_walls=1, max_walls=3
        )

        # If no safe spots, just use the base grid with no distractors
        if not distractor_combos:
            distractor_combos = [[]]

        for distractor_positions in distractor_combos:
            grid = _add_distractors(base_grid, [], distractor_positions)
            grid_tuple = tuple(map(tuple, grid))

            if grid_tuple in seen_grids:
                continue  # Skip duplicate grids

            seen_grids.add(grid_tuple)

            # Generate a random history for variety
            history = random.choices(VALID_ACTIONS, k=3)
            user_prompt_string = _format_user_prompt(grid, history)

            example = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt_string},
                {"role": "assistant", "content": action},
            ]

            final_dataset.append(example)
            action_to_examples[action].append(example)

    print(f"\nGenerated {len(final_dataset)} total examples (before balancing)")
    print("Action distribution (before balancing):")
    for action in sorted(action_to_examples.keys()):
        print(f"  {action}: {len(action_to_examples[action])} examples")

    # Balance the dataset: ensure all actions have the same number of examples
    min_count = min(len(examples) for examples in action_to_examples.values())
    print(f"\nBalancing dataset to {min_count} examples per action...")

    balanced_action_to_examples = {}
    for action, examples in action_to_examples.items():
        # Randomly sample to match the minimum count
        if len(examples) > min_count:
            balanced_action_to_examples[action] = random.sample(examples, min_count)
        else:
            balanced_action_to_examples[action] = examples

    # Rebuild final dataset from balanced examples
    final_dataset = []
    for examples in balanced_action_to_examples.values():
        final_dataset.extend(examples)

    print(f"\nFinal balanced dataset: {len(final_dataset)} total examples")
    print("Action distribution (after balancing):")
    for action in sorted(balanced_action_to_examples.keys()):
        print(f"  {action}: {len(balanced_action_to_examples[action])} examples")

    return final_dataset, balanced_action_to_examples


if __name__ == "__main__":
    full_dataset, action_to_examples = generate_dataset()

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
    output_dir = os.path.dirname(OUTPUT_FILE)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_file = os.path.join(output_dir, "train_data.json")
    val_file = os.path.join(output_dir, "validation_data.json")
    test_file = os.path.join(output_dir, "test_data.json")

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
