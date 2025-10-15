import json
import random
import os

SYSTEM_PROMPT = """You are a simple robot agent. You exist within a grid. Your goal is to navigate to the 'R' (resource). Your only possible outputs are the single words: 'North', 'South', 'East', 'West', 'Gather'."""
VALID_ACTIONS = ['North', 'South', 'East', 'West', 'Gather']

# --- Configuration ---
# You can easily control the final dataset size and balance from here.
NUM_GATHER_EXAMPLES = 5
NUM_ADJACENT_MOVE_EXAMPLES = 8  # Creates 8 examples for each of 4 directions (N,S,E,W)
NUM_DIAGONAL_MOVE_EXAMPLES = 4  # Creates 4 examples for each of 4 corners (NW,NE,SW,SE)
GRID_SIZE = 3
OUTPUT_FILE = "data/finetuning_data.json"


def _add_distractors(grid, unsafe_coords):
    """
    Adds 1-2 random 'W' (wall) distractors to empty spots on the grid,
    avoiding the specified unsafe coordinates (resource and optimal path).
    """
    empty_spots = [
        (r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE) if grid[r][c] == "."
    ]
    safe_spots = [pos for pos in empty_spots if pos not in unsafe_coords]

    if not safe_spots:
        return grid

    num_distractors = random.randint(1, min(2, len(safe_spots)))
    distractor_positions = random.sample(safe_spots, num_distractors)

    for pos in distractor_positions:
        grid[pos[0]][pos[1]] = "W"
    return grid


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

    grid = _add_distractors(grid, unsafe_coords)
    return grid, correct_action


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
    """Generates a perfectly balanced dataset for fine-tuning the agent."""
    print("Generating perfectly balanced dataset...")

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

    random.shuffle(tasks)

    final_dataset = []
    seen_grids = set()

    for scenario_type, orientation in tasks:
        while True:
            grid, action = _generate_forced_scenario(scenario_type, orientation)
            # Use a tuple of tuples as a hashable representation of the grid
            grid_tuple = tuple(map(tuple, grid))

            if grid_tuple not in seen_grids:
                seen_grids.add(grid_tuple)
                break  # Found a unique grid, exit the while loop

        # Now that we have a unique grid, generate a random history for it
        history = random.choices(VALID_ACTIONS, k=3)
        user_prompt_string = _format_user_prompt(grid, history)

        example = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt_string},
            {"role": "assistant", "content": action},
        ]
        final_dataset.append(example)

    return final_dataset


if __name__ == "__main__":
    full_dataset = generate_dataset()
    random.shuffle(full_dataset) # Shuffle before splitting

    # Define split sizes
    total_size = len(full_dataset)
    train_size = int(total_size * 0.70)
    val_size = int(total_size * 0.15)
    test_size = total_size - train_size - val_size

    # Split the dataset
    train_dataset = full_dataset[:train_size]
    validation_dataset = full_dataset[train_size : train_size + val_size]
    test_dataset = full_dataset[train_size + val_size :]

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
