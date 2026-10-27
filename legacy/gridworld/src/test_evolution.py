"""
Test multi-generation evolution with the refactored environment.

This script runs 5 generations to verify:
- Evolution/selection works correctly
- Curriculum difficulty progresses
- Combat and food mechanics function properly
- Fitness scoring is accurate
"""

import sys
import os

# Add parent directory to path if running from src folder
if os.path.basename(os.getcwd()) == 'src':
    sys.path.insert(0, os.path.dirname(os.getcwd()))
    sys.path.insert(0, os.getcwd())

from Simulation.simulation_vectorized import VectorizedSimulation


def main():
    """Run multi-generation evolution test."""
    print("="*60)
    print("MULTI-GENERATION EVOLUTION TEST")
    print("="*60)
    print("\nRunning 5 generations with 32 agents (4 batches of 8)")
    print("This tests the full evolutionary loop with BATCH INFERENCE.")
    print()

    # Create full vectorized simulation with batch inference
    sim = VectorizedSimulation(
        population_size=32,
        batch_size=8,
        turns_per_batch=100,
        num_elites=5,
    )

    # Track progress across generations
    generation_stats = []

    # Run 5 generations
    for gen in range(5):
        print(f"\n{'='*60}")
        print(f"GENERATION {gen + 1}/5")
        print('='*60)

        sim.run_generation()

        # Collect stats BEFORE evolution (otherwise we get fresh offspring in rankings)
        top3 = sim.get_top_agents(3)
        avg_energy = sum(a.energy for a in sim.population) / len(sim.population)
        avg_food = sum(a.food_collected for a in sim.population) / len(sim.population)
        survivors = sum(1 for a in sim.population if a.is_alive)

        # Now evolve for next generation
        sim.evolve_population()

        stats = {
            "generation": gen + 1,
            "top_energy": top3[0].energy if top3 else 0,
            "avg_energy": avg_energy,
            "avg_food": avg_food,
            "survivors": survivors,
        }
        generation_stats.append(stats)

        # Show progress
        print(f"\n--- Generation {gen + 1} Summary ---")
        print(f"Survivors: {survivors}/{sim.population_size}")
        print(f"Average energy: {avg_energy:.1f}")
        print(f"Average food collected: {avg_food:.1f}")
        print(f"\nTop 3 Champions:")
        for idx, agent in enumerate(top3, 1):
            status_icon = "✓" if agent.is_alive else "✗"
            print(f"  #{idx}: Agent {agent.id} {status_icon}")
            print(f"       Energy: {agent.energy:.1f}, Food: {agent.food_collected}, Combat: {len(agent.combat_history)}")

    # Final summary
    print("\n" + "="*60)
    print("EVOLUTION SUMMARY")
    print("="*60)
    print("\nProgress across generations:")
    print(f"{'Gen':<5} {'Top Energy':<12} {'Avg Energy':<12} {'Avg Food':<10} {'Survivors':<10}")
    print("-" * 60)
    for stats in generation_stats:
        print(f"{stats['generation']:<5} {stats['top_energy']:<12.1f} "
              f"{stats['avg_energy']:<12.1f} {stats['avg_food']:<10.1f} "
              f"{stats['survivors']:<10}")

    print("\n✅ Multi-generation evolution test complete!")
    print("\nObservations:")
    if generation_stats[-1]['avg_energy'] > generation_stats[0]['avg_energy']:
        print("  ✓ Population is improving (energy increasing)")
    else:
        print("  ⚠ Population may need more generations or better initial weights")

    if generation_stats[-1]['avg_food'] > generation_stats[0]['avg_food']:
        print("  ✓ Food collection efficiency improving")

    print("\nNext steps:")
    print("  - If results look good, integrate into main.py")
    print("  - Consider running longer evolutions (20+ generations)")
    print("  - Fine-tune model on new combat prompt format")


if __name__ == "__main__":
    main()
