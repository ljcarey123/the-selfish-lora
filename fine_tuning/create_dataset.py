import json
import random
import os
import argparse
from typing import List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict

from config_loader import GRID_SIZE, AGENT_POS, VALID_ACTIONS, GRID_SYMBOLS
from dataset_rules import (
    get_energy_level,
    get_possible_moves,
    score_move_towards,
    score_move_away_from,
    get_valid_last_action,
)

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
    from .config_loader import GRID_SYMBOLS

    grid = [[GRID_SYMBOLS["empty"] for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    grid[AGENT_POS[0]][AGENT_POS[1]] = GRID_SYMBOLS["agent"]
    return grid


def get_adjacent_positions(pos: Position) -> List[Position]:
    """Get adjacent positions (N, S, E, W) to the given position."""
    row, col = pos
    return [
        (row - 1, col),  # North
        (row + 1, col),  # South
        (row, col + 1),  # East
        (row, col - 1),  # West
    ]


def find_nearest_resource(grid: Grid, start_pos: Position) -> Optional[Position]:
    """Find the position of the nearest resource using BFS."""
    visited = set()
    queue = [(start_pos, [])]

    while queue:
        pos, path = queue.pop(0)
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


def get_heuristic_action(
    grid: Grid, energy: float, last_action: Optional[Action] = None
) -> Tuple[Action, str]:
    """
    Implements the heuristic decision making for the agent.
    Priority order:
    1. GATHER if on resource
    2. LOW-ENERGY survival if energy is critically low
    3. ANTI-STUCK if last action would hit a wall
    4. SEEK/AVOID strategy (resources/rivals)
    5. DEFAULT IDLE rules based on energy

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

    # Place 0-1 rivals (30% chance)
    if random.random() < 0.3 and available_positions:
        r, c = random.choice(available_positions)
        grid[r][c] = GRID_SYMBOLS["rival"]

    return grid


def format_example(
    grid: Grid, energy: float, last_action: Optional[Action] = None
) -> str:
    """Format a single training example.

    Args:
        grid: The grid state
        energy: The agent's energy level (0.0 to 1.0)
        last_action: The agent's last action, if any

    Returns:
        The formatted prompt string for the model
    """
    # Create a display grid where agent on resource shows as '@'
    display_grid = [row[:] for row in grid]  # Create a copy of the grid
    agent_r, agent_c = AGENT_POS
    if grid[agent_r][agent_c] == GRID_SYMBOLS["resource"]:
        display_grid[agent_r][agent_c] = GRID_SYMBOLS["on_resource"]

    # Convert grid to string representation
    grid_str = "\n".join(" ".join(row) for row in display_grid)

    # Format energy level
    energy_str = f"Energy: {energy:.2f} ({get_energy_level(energy)})"

    # Format history
    history_str = f"Last action: {last_action}" if last_action else "No previous action"

    # Create and return the prompt (rule_name is only used in metadata, not in the prompt)
    return f"PERCEPTION:\n{grid_str}\n\n{energy_str}\n{history_str}\n\nACTION:"


def generate_dataset(output_file: str, num_examples: int = 1000):
    """Generate a dataset of training examples.

    Args:
        output_file: Path to save the dataset
        num_examples: Number of examples to generate
    
    Returns:
        Tuple of (examples, debug_info) where examples contain only prompt/completion,
        and debug_info contains additional metadata for analysis
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

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
            examples.append({
                "prompt": prompt,
                "completion": action
            })
            
            # Store debug info separately
            debug_info.append({
                "energy": energy,
                "last_action": last_action,
                "rule_name": rule_name,
                "grid": grid_key
            })
            
            action_counts[action] += 1

    # Save to file (only the minimal dataset)
    with open(output_file, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")

    print(f"Generated {len(examples)} examples:")
    for action in VALID_ACTIONS:
        print(f"- {action}: {action_counts[action]}/{target_counts[action]} (target)")

    return examples, debug_info


def analyze_dataset(examples, debug_info):
    """Analyze and print detailed metrics about the dataset."""
    # Initialize counters
    action_counts = defaultdict(int)
    energy_levels = defaultdict(int)
    last_actions = defaultdict(int)
    action_energy = defaultdict(lambda: defaultdict(int))
    action_last_action = defaultdict(lambda: defaultdict(int))
    rule_counts = defaultdict(int)

    # Process each example with its debug info
    for i, example in enumerate(examples):
        action = example["completion"]
        info = debug_info[i]
        
        # Update counters
        action_counts[action] += 1
        
        # Get debug info if available
        if "energy" in info:
            energy_lvl = get_energy_level(info["energy"])
            energy_levels[energy_lvl] += 1
            action_energy[action][energy_lvl] += 1
            
        if "last_action" in info:
            last_action = info["last_action"]
            last_actions[last_action] += 1
            action_last_action[action][last_action] += 1
            
        if "rule_name" in info:
            rule_counts[info["rule_name"]] += 1

    # Print overall statistics
    total = len(examples)
    print("\n" + "=" * 50)
    print("DATASET METRICS")
    print("=" * 50)

    # Action distribution
    print("\nACTION DISTRIBUTION:")
    for action, count in sorted(action_counts.items()):
        pct = (count / total) * 100
        print(f"- {action}: {count} examples ({pct:.1f}%)")

    # Energy level distribution
    if energy_levels:
        print("\nENERGY LEVEL DISTRIBUTION:")
        for level in ["low", "medium", "high"]:
            count = energy_levels.get(level, 0)
            pct = (count / total) * 100 if total > 0 else 0
            print(f"- {level}: {count} examples ({pct:.1f}%)")

    # Last action distribution
    if last_actions:
        print("\nLAST ACTION DISTRIBUTION:")
        for action in sorted(last_actions.keys(), key=lambda x: str(x) if x is not None else ""):
            count = last_actions[action]
            pct = (count / total) * 100 if total > 0 else 0
            action_str = str(action) if action is not None else "none"
            print(f"- {action_str}: {count} examples ({pct:.1f}%)")

    # Heuristic rule distribution
    if rule_counts:
        print("\nHEURISTIC RULE DISTRIBUTION:")
        for rule, count in sorted(rule_counts.items(), key=lambda x: x[1], reverse=True):
            pct = (count / total) * 100 if total > 0 else 0
            print(f"- {rule}: {count} examples ({pct:.1f}%)")

    # Energy levels per action
    if action_energy:
        print("\nENERGY LEVELS PER ACTION:")
        for action in sorted(action_energy.keys()):
            print(f"\n{action.upper()}:")
            total_action = sum(action_energy[action].values())
            for level in ["low", "medium", "high"]:
                count = action_energy[action].get(level, 0)
                pct = (count / total_action) * 100 if total_action > 0 else 0
                print(f"  - {level}: {count} ({pct:.1f}%)")

    # Last actions per action
    if action_last_action:
        print("\nLAST ACTIONS PER ACTION:")
        for action in sorted(action_last_action.keys()):
            print(f"\n{action.upper()}:")
            total_action = sum(action_last_action[action].values())
            # Sort last_actions handling None values
            sorted_last_actions = sorted(
                set(last_actions.keys()),
                key=lambda x: (x is None, str(x) if x is not None else ""),
            )
            for last_act in sorted_last_actions:
                count = action_last_action[action].get(last_act, 0)
                if count > 0:  # Only show if there are examples
                    pct = (count / total_action) * 100 if total_action > 0 else 0
                    last_act_str = str(last_act) if last_act is not None else "none"
                    print(f"  - after {last_act_str}: {count} ({pct:.1f}%)")

    print("\n" + "=" * 50 + "\n")


def generate_training_dataset(output_file):
    """Generate and save the training dataset.

    Args:
        output_file: Path to save the complete dataset
    """
    total_examples = SAMPLES_PER_ACTION * len(VALID_ACTIONS)
    print(
        f"Generating {total_examples} training examples ({SAMPLES_PER_ACTION} per action)..."
    )

    # Generate the dataset
    examples, debug_info = generate_dataset(output_file, total_examples)

    # Analyze and print detailed metrics
    analyze_dataset(examples, debug_info)

    print(f"\nDataset saved to {output_file}")
    return examples


def save_dataset_split(dataset, filename):
    """Save a dataset split to a file."""
    with open(filename, "w", encoding="utf-8") as f:
        for example in dataset:
            f.write(json.dumps(example) + "\n")


def create_stratified_splits(
    examples, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
):
    """Split dataset into train/validation/test sets while maintaining action distribution."""
    # Group examples by action
    action_to_examples = {}
    for example in examples:
        action = example.get("completion")
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
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description="Generate training dataset for grid agent"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/training_data.jsonl",
        help="Output file path for the complete dataset (default: data/training_data.jsonl)",
    )

    args = parser.parse_args()

    # Generate and save the dataset
    examples = generate_training_dataset(
        output_file=args.output,
    )

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output) or "."
    os.makedirs(output_dir, exist_ok=True)

    # Create stratified splits
    train_data, val_data, test_data = create_stratified_splits(
        examples, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
    )

    # Save the splits
    # Save the splits with proper file extensions
    train_file = os.path.join("data", "train.jsonl")
    val_file = os.path.join("data", "validation.jsonl")
    test_file = os.path.join("data", "test.jsonl")

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
        f"- Total examples: {len(examples)} (6 actions × {SAMPLES_PER_ACTION} samples)"
    )
