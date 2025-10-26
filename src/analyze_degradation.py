"""Analyze why the model is degrading through evolution"""
import json
import matplotlib.pyplot as plt

# Load simulation results
with open('output/simulation_results.json') as f:
    results = json.load(f)

generations = list(range(len(results['performance_history'])))
perf = results['performance_history']

# Extract metrics
min_fitness = [gen['min'] for gen in perf]
avg_fitness = [gen['avg'] for gen in perf]
max_fitness = [gen['max'] for gen in perf]

# Action distributions
action_dist = results.get('action_distribution_history', [])

# Genetic diversity
genetic_div = results.get('genetic_diversity_history', [])

# Create comprehensive analysis plot
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Fitness over time
ax1 = axes[0, 0]
ax1.plot(generations, min_fitness, label='Min', color='red', alpha=0.7)
ax1.plot(generations, avg_fitness, label='Avg', color='blue', linewidth=2)
ax1.plot(generations, max_fitness, label='Max', color='green', alpha=0.7)
ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3, label='Baseline')
ax1.set_xlabel('Generation')
ax1.set_ylabel('Fitness')
ax1.set_title('Fitness Degradation Over Time')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Action distribution over time
if action_dist:
    ax2 = axes[0, 1]
    actions = ['North', 'South', 'East', 'West', 'Gather']
    for action in actions:
        counts = [gen.get(action, 0) for gen in action_dist]
        ax2.plot(generations[:len(counts)], counts, label=action, marker='o', markersize=3)
    
    # Add UNPARSEABLE if it exists
    unparseable = [gen.get('UNPARSEABLE', 0) for gen in action_dist]
    if any(unparseable):
        ax2.plot(generations[:len(unparseable)], unparseable, label='UNPARSEABLE', 
                color='red', linewidth=2, marker='x', markersize=5)
    
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Action Count')
    ax2.set_title('Action Distribution (UNPARSEABLE Spike!)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

# 3. Genetic diversity
if genetic_div:
    ax3 = axes[1, 0]
    ax3.plot(generations[:len(genetic_div)], genetic_div, color='purple', linewidth=2)
    ax3.set_xlabel('Generation')
    ax3.set_ylabel('Genetic Diversity')
    ax3.set_title('Genetic Diversity Over Time')
    ax3.grid(True, alpha=0.3)

# 4. Diagnosis summary
ax4 = axes[1, 1]
ax4.axis('off')

# Calculate key metrics
if len(perf) > 0:
    initial_avg = avg_fitness[0]
    final_avg = avg_fitness[-1]
    best_gen = max(range(len(max_fitness)), key=lambda i: max_fitness[i])
    best_fitness = max_fitness[best_gen]
    
    # Count unparseable spike
    unparseable_spike_gen = -1
    if action_dist and any(gen.get('UNPARSEABLE', 0) > 10 for gen in action_dist):
        for i, gen in enumerate(action_dist):
            if gen.get('UNPARSEABLE', 0) > 10:
                unparseable_spike_gen = i
                break
    
    diagnosis = f"""
DEGRADATION ANALYSIS
{'='*40}

SYMPTOMS:
• Initial Avg Fitness: {initial_avg:.1f}
• Final Avg Fitness: {final_avg:.1f}
• Change: {final_avg - initial_avg:+.1f} ({((final_avg/initial_avg - 1)*100) if initial_avg != 0 else 0:.1f}%)

• Best Generation: {best_gen}
• Best Fitness: {best_fitness:.1f}

• Min Fitness Dropped to: {min(min_fitness):.1f}
• UNPARSEABLE Spike at Gen: {unparseable_spike_gen if unparseable_spike_gen >= 0 else 'N/A'}

LIKELY CAUSES:
1. Mutation rate too high
2. Genetic drift destroying fine-tuned weights
3. No positive selection pressure
4. LoRA weights drifting from valid output space

RECOMMENDATIONS:
1. Reduce mutation rate (try 0.001 or 0.0001)
2. Increase elite count (preserve good agents)
3. Add weight regularization (keep close to init)
4. Check fine-tuning quality (was it good?)
5. Consider fitness shaping (reward valid actions)
"""
    
    ax4.text(0.05, 0.95, diagnosis, transform=ax4.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('output/degradation_analysis.png', dpi=150, bbox_inches='tight')
print("\n✅ Analysis saved to output/degradation_analysis.png")
print("\nKey findings:")
print(f"  • Fitness degraded from {initial_avg:.1f} to {final_avg:.1f}")
print(f"  • Best performance at generation {best_gen} ({best_fitness:.1f})")
if unparseable_spike_gen >= 0:
    print(f"  • ⚠️  UNPARSEABLE spike at generation {unparseable_spike_gen}")
print(f"  • Minimum fitness dropped to {min(min_fitness):.1f}")
