"""
Quick script to check if agents have repeating action patterns.
"""
import json

# Load the simulation data
with open('output/simulation_paths.json', 'r') as f:
    data = json.load(f)

# Get behavioral history for last generation
if data['behavioral_history']:
    print("Behavioral history (action counts per generation):")
    for gen_idx, action_counts in enumerate(data['behavioral_history']):
        total = sum(action_counts.values())
        print(f"\nGeneration {gen_idx}:")
        for action, count in sorted(action_counts.items()):
            pct = (count / total) * 100 if total > 0 else 0
            print(f"  {action}: {count} ({pct:.1f}%)")

# Check if we have champion paths
if 'champion_paths_by_gen' in data:
    # Get last generation
    last_gen = str(len(data['behavioral_history']) - 1)
    if last_gen in data['champion_paths_by_gen']:
        paths = data['champion_paths_by_gen'][last_gen]
        
        print(f"\n\nChampion action sequences (Generation {last_gen}):")
        for i, path in enumerate(paths[:3]):  # Top 3
            # Extract just the actions
            actions = [step[0] for step in path if step[0] not in ['START', 'MOVE']]
            
            print(f"\nChampion {i+1} (first 20 actions):")
            print(actions[:20])
            
            # Check for patterns
            gather_positions = [j for j, a in enumerate(actions) if a in ['GATHER_SUCCESS', 'GATHER_FAIL']]
            if len(gather_positions) > 1:
                gaps = [gather_positions[j+1] - gather_positions[j] for j in range(len(gather_positions)-1)]
                print(f"  Gaps between Gathers: {gaps[:10]}")
                if gaps:
                    avg_gap = sum(gaps) / len(gaps)
                    print(f"  Average gap: {avg_gap:.1f} actions")
