import json
from collections import defaultdict

from .decision_rules import get_energy_level


def analyse_dataset(examples, debug_info):
    """Analyse and print detailed metrics about the dataset."""
    # Initialize counters
    action_counts = defaultdict(int)
    energy_levels = defaultdict(int)
    last_actions = defaultdict(int)
    action_energy = defaultdict(lambda: defaultdict(int))
    action_last_action = defaultdict(lambda: defaultdict(int))
    rule_counts = defaultdict(int)

    # Process each example with its debug info
    for i, example in enumerate(examples):
        action = example[2]["content"]
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
        for action in sorted(
            last_actions.keys(), key=lambda x: str(x) if x is not None else ""
        ):
            count = last_actions[action]
            pct = (count / total) * 100 if total > 0 else 0
            action_str = str(action) if action is not None else "none"
            print(f"- {action_str}: {count} examples ({pct:.1f}%)")

    # Heuristic rule distribution
    if rule_counts:
        print("\nHEURISTIC RULE DISTRIBUTION:")
        for rule, count in sorted(
            rule_counts.items(), key=lambda x: x[1], reverse=True
        ):
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


def save_dataset_split(examples: list, file_path: str):
    """Save a dataset split to a file in JSONL format.
    
    Args:
        examples: List of examples to save
        file_path: Path to save the file
    """
    with open(file_path, "w", encoding="utf-8") as f:
        for example in examples:
            # Write each example as a separate JSON object on its own line
            f.write(json.dumps(example) + "\n")
