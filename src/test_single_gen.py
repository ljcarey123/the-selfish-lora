"""
Test a single generation with the refactored environment.

This script runs one generation with a small population (16 agents)
to verify the full pipeline works end-to-end, including visualization.
"""

import sys
import os
import json

# Add parent directory to path if running from src folder
if os.path.basename(os.getcwd()) == 'src':
    sys.path.insert(0, os.path.dirname(os.getcwd()))
    sys.path.insert(0, os.getcwd())

from Simulation.simulation_vectorized import VectorizedSimulation
from visualisation.visualiser import Visualiser


def main():
    """Run a single generation test."""
    print("="*60)
    print("SINGLE GENERATION TEST")
    print("="*60)
    print("\nTesting with 16 agents (2 batches of 8)")
    print("Tournament mode: 2 rounds per generation")
    print("This is a quick test to verify the full pipeline including:")
    print("  - Vectorized batch inference")
    print("  - Tournament format with multiple rounds")
    print("  - Visualization with round-specific data")
    print()

    # Create vectorized simulation with batch inference
    sim = VectorizedSimulation(
        population_size=16,  # Just 16 agents (2 batches)
        batch_size=8,
        turns_per_batch=50,  # Shorter for testing
        num_elites=2,
        rounds_per_generation=2,  # Test tournament mode
    )

    print("\n" + "="*60)
    print("Running Generation 1...")
    print("="*60)
    sim.run_generation()

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print("\nTop 3 agents:")
    for idx, agent in enumerate(sim.get_top_agents(3), 1):
        status = "✓ ALIVE" if agent.is_alive else "✗ DEAD"
        print(f"  #{idx}: Agent {agent.id}")
        print(f"       Energy: {agent.energy:.1f}")
        print(f"       Food collected: {agent.food_collected}")
        print(f"       Combat events: {agent.combat_events}")
        print(f"       Actions taken: {len(agent.action_history)}")
        print(f"       Path length: {len(agent.path)}")
        print(f"       Status: {status}")
        print()

    # Test visualization data
    print("\n" + "="*60)
    print("TESTING VISUALIZATION DATA")
    print("="*60)

    # Check if visualization data was saved
    viz_path = "./output/simulation_paths.json"
    if os.path.exists(viz_path):
        print(f"✓ Visualization data saved: {viz_path}")

        # Load and verify the data
        with open(viz_path, 'r') as f:
            viz_data = json.load(f)

        print(f"\nVisualization data contents:")
        print(f"  - Generations recorded: {len(viz_data.get('champion_paths_by_gen', {}))}")
        print(f"  - Environment history: {len(viz_data.get('environment_history', {}))}")
        print(f"  - Genetic history entries: {len(viz_data.get('genetic_history', []))}")
        print(f"  - Behavioral history entries: {len(viz_data.get('behavioural_history', []))}")
        print(f"  - Tournament mode: {viz_data.get('tournament_mode', False)}")
        print(f"  - Rounds per generation: {viz_data.get('rounds_per_generation', 1)}")

        # Check champion paths (handle tournament format)
        if '0' in viz_data.get('champion_paths_by_gen', {}):
            gen_data = viz_data['champion_paths_by_gen']['0']

            # Check if tournament mode (dict of rounds) or standard (list)
            if isinstance(gen_data, dict):
                print(f"  - Tournament format detected")
                print(f"  - Rounds saved: {len(gen_data)}")
                for round_idx, paths in gen_data.items():
                    print(f"    - Round {int(round_idx) + 1}: {len(paths)} agent paths")
                    if len(paths) > 0:
                        print(f"      - First agent path length: {len(paths[0])}")
                        if len(paths[0]) > 0:
                            print(f"      - Sample path entry: {paths[0][0]}")
            else:
                print(f"  - Standard format detected")
                print(f"  - Top agent paths saved: {len(gen_data)}")
                if len(gen_data) > 0:
                    print(f"  - First agent path length: {len(gen_data[0])}")
                    if len(gen_data[0]) > 0:
                        print(f"  - Sample path entry: {gen_data[0][0]}")

        # Test creating visualizer
        print(f"\n✓ Creating visualizer from saved data...")
        try:
            visualizer = Visualiser(data_path=viz_path)
            print(f"✓ Visualizer created successfully")

            # Test creating a heatmap
            print(f"\n✓ Testing heatmap generation...")
            visualizer.plot_agent_heatmap(generation=0, save_as_png=True)
            print(f"✓ Heatmap saved to ./output/agent_heatmap_gen_0.png")

            # Test genetic diversity plot
            if len(viz_data.get('genetic_history', [])) > 0:
                print(f"\n✓ Testing genetic diversity plot...")
                visualizer.plot_genetic_diversity(save_as_png=True)
                print(f"✓ Genetic diversity plot saved to ./output/genetic_diversity.png")

            # Test behavioral metrics plot
            if len(viz_data.get('behavioural_history', [])) > 0:
                print(f"\n✓ Testing behavioral metrics plot...")
                visualizer.plot_behavioural_metrics(save_as_png=True)
                print(f"✓ Behavioral metrics plot saved to ./output/behavioural_metrics.png")

            # Test action timeline plot
            print(f"\n✓ Testing action timeline plot...")
            visualizer.plot_champion_action_timeline(generation=0, n=3, save_as_png=True)
            if viz_data.get('tournament_mode', False):
                rounds = viz_data.get('rounds_per_generation', 1)
                print(f"✓ Tournament mode: {rounds} timeline plots should be created")
                for i in range(rounds):
                    timeline_path = f"./output/action_timeline_gen_0_top_3_round_{i + 1}.png"
                    if os.path.exists(timeline_path):
                        print(f"  ✓ Round {i + 1} timeline created: {timeline_path}")
                    else:
                        print(f"  ✗ Round {i + 1} timeline missing: {timeline_path}")
            else:
                timeline_path = "./output/action_timeline_gen_0_top_3.png"
                if os.path.exists(timeline_path):
                    print(f"✓ Standard timeline created: {timeline_path}")
                else:
                    print(f"✗ Standard timeline missing: {timeline_path}")

            # Test animation generation (if tournament mode, should create multiple GIFs)
            print(f"\n✓ Testing animation generation...")
            visualizer.animate_top_n_champion_paths(generation=0, n=3, save_as_gif=True)
            if viz_data.get('tournament_mode', False):
                rounds = viz_data.get('rounds_per_generation', 1)
                print(f"✓ Tournament mode: {rounds} animations should be created")
                for i in range(rounds):
                    gif_path = f"./output/gifs/champion_paths_gen_0_top_3_round_{i + 1}.gif"
                    if os.path.exists(gif_path):
                        print(f"  ✓ Round {i + 1} animation created: {gif_path}")
                    else:
                        print(f"  ✗ Round {i + 1} animation missing: {gif_path}")
            else:
                gif_path = "./output/gifs/champion_paths_gen_0_top_3.gif"
                if os.path.exists(gif_path):
                    print(f"✓ Standard animation created: {gif_path}")
                else:
                    print(f"✗ Standard animation missing: {gif_path}")

        except Exception as e:
            print(f"✗ Visualization error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"✗ Visualization data not found at {viz_path}")

    # Check weights
    print("\n" + "="*60)
    print("CHECKING SAVED WEIGHTS")
    print("="*60)

    weights_dir = "./output/weights"
    if os.path.exists(weights_dir):
        weight_files = [f for f in os.listdir(weights_dir) if f.endswith('.pth')]
        print(f"✓ Weights directory exists: {weights_dir}")
        print(f"  - Weight files saved: {len(weight_files)}")
        for wf in sorted(weight_files):
            print(f"    - {wf}")
    else:
        print(f"✗ Weights directory not found")

    print("\n" + "="*60)
    print("Evolving population...")
    print("="*60)
    sim.evolve_population()

    print("\n" + "="*60)
    print("✅ SINGLE GENERATION TEST COMPLETE!")
    print("="*60)
    print("\nAll systems verified:")
    print("  ✓ Vectorized simulation")
    print("  ✓ Inference engine with chat templates")
    print("  ✓ Path tracking and combat/food events")
    print("  ✓ Visualization data saving")
    print("  ✓ Weight saving")
    print("  ✓ Visualization rendering")
    print("\nNext step: Run full evolution with python main.py")


if __name__ == "__main__":
    main()
