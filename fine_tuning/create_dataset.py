import random
import os
from typing import List, Tuple

from config_loader import (
    GRID_SIZE,
    AGENT_POS,
    VALID_ACTIONS,
    GRID_SYMBOLS,
    SYSTEM_PROMPT,
)

from dataset_utils.grid_generator import generate_random_grid
from dataset_utils.analyser import analyse_dataset, save_dataset_split
from dataset_utils.movement_rules import get_valid_last_action
from dataset_utils.decision_rules import get_heuristic_action
from dataset_utils.formatter import format_example

# Constants
GRID_SIZE = 5  # 5x5 grid
AGENT_POS = (2, 2)  # Center of 5x5 grid
OUTPUT_DIR = "data"
SAMPLES_PER_ACTION = 200  # 1200 total / 6 actions

# Type aliases
Position = Tuple[int, int]
Grid = List[List[str]]
Action = str
EnergyLevel = str


def create_empty_grid() -> Grid:
    """Create an empty 5x5 grid with agent at center."""
    grid = [[GRID_SYMBOLS["empty"] for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    grid[AGENT_POS[0]][AGENT_POS[1]] = GRID_SYMBOLS["agent"]
    return grid


def generate_dataset(num_examples: int = 1000):
    """Generate a dataset of training examples.

    Args:
        num_examples: Number of examples to generate

    Returns:
        Tuple of (examples, debug_info) where examples contain only prompt/completion,
        and debug_info contains additional metadata for analysis
    """
    # Initialize action counts
    action_counts = {action: 0 for action in VALID_ACTIONS}
    target_counts = {action: 0 for action in VALID_ACTIONS}

    # Set target to x examples per action
    target_per_action = SAMPLES_PER_ACTION
    for action in VALID_ACTIONS:
        target_counts[action] = target_per_action

    # Update total number of examples
    num_examples = target_per_action * len(VALID_ACTIONS)

    # Generate examples
    examples = []
    debug_info = []  # Store debug information for analysis
    seen_examples = set()  # Track unique examples to avoid duplicates
    attempts = 0
    max_attempts = num_examples * 4  # Increased due to larger dataset and uniqueness

    while len(examples) < num_examples and attempts < max_attempts:
        attempts += 1

        # Generate random grid and parameters
        grid = generate_random_grid()
        energy = random.random()  # Random energy between 0 and 1

        # Get a valid last action based on the current grid state
        last_action = get_valid_last_action(grid)

        # Get heuristic action and rule name
        action, rule_name = get_heuristic_action(grid, energy, last_action)

        # Format the example
        prompt = format_example(grid, energy, last_action)

        # Create a unique key for this example
        # Convert grid to a tuple of tuples for hashing
        grid_key = tuple(tuple(row) for row in grid)
        example_key = (grid_key, action, energy, last_action)

        # Skip if we've seen this exact example before
        if example_key in seen_examples:
            continue

        # Check if we need more of this action type
        if action_counts[action] < target_counts[action]:
            seen_examples.add(example_key)

            # Store minimal data in the actual dataset
            examples.append(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": action.replace("_", " ").title()},
                ]
            )

            # Store debug info separately
            debug_info.append(
                {
                    "energy": energy,
                    "last_action": last_action,
                    "rule_name": rule_name,
                    "grid": grid_key,
                }
            )

            action_counts[action] += 1

    print(f"Generated {len(examples)} examples:")
    for action in VALID_ACTIONS:
        print(f"- {action}: {action_counts[action]}/{target_counts[action]} (target)")

    return examples, debug_info


def generate_training_dataset():
    """Generate and save the training dataset."""
    total_examples = SAMPLES_PER_ACTION * len(VALID_ACTIONS)
    print(
        f"Generating {total_examples} training examples ({SAMPLES_PER_ACTION} per action)..."
    )

    # Generate the dataset
    examples, debug_info = generate_dataset(total_examples)

    # Analyse and print detailed metrics
    analyse_dataset(examples, debug_info)

    print(f"\nDataset saved to {OUTPUT_DIR}")
    return examples


def create_stratified_splits(
    examples, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
):
    """Split dataset into train/validation/test sets while maintaining action distribution."""
    # Group examples by action
    action_to_examples = {}
    for example in examples:
        action = example[2]["content"]
        if action not in action_to_examples:
            action_to_examples[action] = []
        action_to_examples[action].append(example)

    # Initialize splits
    train_dataset = []
    val_dataset = []
    test_dataset = []

    # Split each action's examples
    for action, action_examples in action_to_examples.items():
        random.shuffle(action_examples)
        n = len(action_examples)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        train_dataset.extend(action_examples[:train_end])
        val_dataset.extend(action_examples[train_end:val_end])
        test_dataset.extend(action_examples[val_end:])

    # Shuffle the final splits
    random.shuffle(train_dataset)
    random.shuffle(val_dataset)
    random.shuffle(test_dataset)

    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Generate and save the dataset
    examples = generate_training_dataset()

    # Create stratified splits
    train_data, val_data, test_data = create_stratified_splits(
        examples, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
    )

    # Save the splits
    train_file = os.path.join(OUTPUT_DIR, "train.jsonl")
    val_file = os.path.join(OUTPUT_DIR, "validation.jsonl")
    test_file = os.path.join(OUTPUT_DIR, "test.jsonl")

    save_dataset_split(train_data, train_file)
    save_dataset_split(val_data, val_file)
    save_dataset_split(test_data, test_file)

    print(f"\nSuccessfully generated and split dataset:")
    print(f"- Training set: {len(train_data)} examples -> {train_file}")
    print(f"- Validation set: {len(val_data)} examples -> {val_file}")
    print(f"- Test set: {len(test_data)} examples -> {test_file}")

    # Print dataset statistics
    print(f"\nDataset Configuration:")
    print(f"- Samples per action: {SAMPLES_PER_ACTION}")
    print(
        f"- Total examples: {len(examples)} (6 actions x {SAMPLES_PER_ACTION} samples)"
    )
