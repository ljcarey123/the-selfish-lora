# Quick Start Guide

## Testing the Vectorised Simulation

The simulation uses **vectorised environment operations** with NumPy arrays for optimised physics updates. LLM inference is sequential to preserve per-agent LoRA diversity.

### 1. Run Integration Tests

```bash
cd src
python test_refactor.py
```

Expected output: **ALL TESTS PASSED ✅**

This tests:
- Generation Manager (curriculum across phases)
- Map Generator (connectivity validation)
- Vectorised Environment (batch operations)
- Combat mechanics (mugging system)
- Fitness calculation (normalised scoring)

### 2. Test Single Generation

```bash
cd src
python test_single_gen.py
```

Runs 16 agents (2 batches of 8) for 50 turns with **sequential LLM inference** and **vectorised environment**.

### 3. Test Multi-Generation Evolution

```bash
cd src
python test_evolution.py
```

Runs 5 generations with 32 agents (4 batches of 8) using **sequential LLM inference** with per-agent LoRA weights.

### 4. Test Vectorised Operations

```bash
cd src
python test_vectorised.py
```

Tests:
- Vectorised environment batch operations
- Perception extraction efficiency
- Physics update vectorisation
- Full simulation with timing analysis

## Understanding Vectorisation

The simulation uses two types of operations:

**Vectorised Operations (NumPy):**
```python
# All agents' metabolism updated at once
energies -= 1  # Single NumPy operation, not a loop!

# All perceptions extracted together
perceptions = env.get_batch_perception(alive_indices)  # Vectorised!
```

**Sequential Operations (Preserves LoRA Diversity):**
```python
# Each agent uses its own evolved LoRA weights
for agent_id, prompt in zip(agent_ids, prompts):
    load_lora_weights(agent.brain.weights)  # Unique per agent
    response = model.generate(prompt)
```

**Why Sequential LLM Inference?**
- Each agent has unique evolved LoRA weights
- Preserves genetic diversity across population
- Ensures evolved behaviours remain distinct
- Environment operations remain vectorised for performance

## Game Mechanics Reference

### Energy System

| Event | Energy Change | Notes |
|-------|--------------|-------|
| **Starting** | 20 | All agents start here |
| **Metabolism** | -1/turn | Constant drain |
| **Food Collection** | +15 | Automatic on tile entry |
| **Wall Hit** | -2 | Boundary or wall collision |
| **Combat (Attacker)** | +5 | Move into another agent |
| **Combat (Victim)** | -10 | Receive attack |
| **Death** | 0 | Agent removed from simulation |

### Action Format

**Available Actions:**
- `MOVE_NORTH`
- `MOVE_SOUTH`
- `MOVE_EAST`
- `MOVE_WEST`

**Note**: No separate `gather` action! Food is collected automatically.

### Combat (Mugging) Rules

When Agent A moves into Agent B's tile:

1. **Agent A** stays in place (move fails)
2. **Agent A** gains +5 energy
3. **Agent B** loses -10 energy
4. If Agent B reaches ≤0 energy, they die

**Strategic Implications:**
- Attack weak agents for easy energy
- Avoid combat when low on energy
- Guard food-rich areas from competitors

### Curriculum Phases

| Phase | Generations | Food Density | Regrowth Rate |
|-------|------------|--------------|---------------|
| **Phase 1** | 0-10 | 30% → 25% | 10% every 5 turns |
| **Phase 2** | 11-30 | 24% → 15% | 5% chance/turn for 1 food |
| **Phase 3** | 31+ | 14% → 5% | 1% chance/turn for 1 food |

**Progressive Difficulty**: Early generations have abundant food and fast regrowth. Later generations force agents to rely on efficiency and combat.

## Troubleshooting

### Test Fails: "ModuleNotFoundError"

**Solution**: Ensure you're running from project root or src directory:
```bash
cd c:\projects\the-selfish-lora\src
python test_refactor.py
```

### Agents Die Too Quickly

**Possible causes:**
1. **Generation too high**: Phase 3 (Gen 31+) has very low food
2. **Bad mutations**: Random initialisation or heavy mutations
3. **No fine-tuned weights**: Agents with random weights perform poorly

**Solutions:**
- Start at Generation 0 for testing
- Use fine-tuned weights (from training pipeline)
- Reduce mutation strength

### All Agents Survive with High Energy

**Possible causes:**
1. **Generation too low**: Phase 1 has abundant food
2. **Batch too short**: 50 turns isn't enough for scarcity

**Solutions:**
- Test with Generation 20+ for real challenge
- Increase `turns_per_batch` to 100-150
- Reduce food density manually in generation_manager.py

## Example Code

**Running Vectorised Simulation:**

```python
from Simulation.simulation_vectorised import VectorisedSimulation

# Create vectorised simulation with batch inference
sim = VectorisedSimulation(
    population_size=32,  # 32 agents total
    batch_size=8,        # Process 8 agents at once (4 batches)
    turns_per_batch=100, # 100 turns per batch
    num_elites=5,
)

# Run 5 generations
for gen in range(5):
    print(f"\nGENERATION {gen + 1}/5")

    sim.run_generation()  # Sequential LLM, vectorised environment
    sim.evolve_population()

    # Show progress
    top3 = sim.get_top_agents(3)
    for idx, agent in enumerate(top3, 1):
        print(f"  #{idx}: {agent.id} - {agent.energy:.1f} energy")
```

## Next Steps

1. ✅ Run `test_refactor.py` - Verify components work
2. ✅ Run `test_single_gen.py` - Test one generation
3. ✅ Run `test_evolution.py` - Test 5 generations
4. ✅ Run `test_vectorised.py` - Test vectorised operations
5. ⏳ Fine-tune model on new combat format
6. ⏳ Run long evolution (50+ generations)

## Documentation

- [System Architecture](system_architecture.md) - Technical overview
- [Workflow Guide](WORKFLOW.md) - Full training and evolution pipeline
- [Project Context](../.windsurf/project_context.md) - Goals and conventions

---

**Status**: Vectorised environment with per-agent LoRA inference ready! 🚀

Start with `python test_refactor.py` to verify all components work correctly.
