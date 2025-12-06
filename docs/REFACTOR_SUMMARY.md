# Refactor Summary: Non-Vectorised Code Removal

**Date**: 29 November 2024
**Reason**: Simplify codebase to single pathway for performance debugging

## Files Removed

### 1. `src/Simulation/simulation.py`
**Original Purpose**: Non-vectorised simulation orchestrator with tournament mode
**Why Removed**:
- Used old non-vectorised environment and agent classes
- Different API and feature set (tournament mode, visualisation)
- No longer needed - vectorised version is the primary pathway
- Sequential agent processing with non-batched inference

### 2. `src/Agent/agent.py`
**Original Purpose**: Full-featured agent with `decide_and_act()` method
**Why Removed**:
- Used `brain.choose_action()` which is not part of vectorised pathway
- Heavy state management (path history, detailed combat tracking)
- Designed for sequential processing, not batch operations
- Replaced by lightweight `VectorizedAgent` with minimal state

### 3. `src/Environment/environment.py`
**Original Purpose**: Non-vectorised environment with sequential operations
**Why Removed**:
- Sequential perception extraction (one agent at a time)
- Sequential physics updates (loops instead of NumPy operations)
- Non-batched action processing
- Different API from `VectorizedEnvironment`

## Remaining Structure

### Single Pathway Architecture

```
VectorizedSimulation (simulation_vectorized.py)
    ↓
BatchInferenceEngine (batch_inference_engine.py)
    ↓ Sequential LLM inference with per-agent LoRA weights
VectorizedAgent (agent_vectorized.py)
    ↓ Lightweight state container
VectorizedEnvironment (environment_vectorized.py)
    ↓ NumPy-optimised operations
```

### Key Components

**Vectorised Components**:
- `src/Simulation/simulation_vectorized.py` - Main orchestrator
- `src/Environment/environment_vectorized.py` - NumPy-based physics
- `src/Agent/agent_vectorized.py` - Lightweight agent container
- `src/LoraBrain/batch_inference_engine.py` - Sequential LLM processing
- `src/LoraBrain/lora_brain.py` - LoRA weight management

**Supporting Components**:
- `src/Environment/batch_types.py` - Batch data structures
- `src/Environment/generation_manager.py` - Curriculum learning
- `src/Environment/map_generator.py` - Map generation

**Entry Points**:
- `src/main.py` - Main evolutionary loop (updated to use vectorised)
- `src/test_single_gen.py` - Single generation test
- `src/test_evolution.py` - Multi-generation test
- `src/test_vectorised.py` - Vectorised operations test
- `src/test_refactor.py` - Component integration tests

## Updated Files

### `src/main.py`
**Changes**:
- Completely rewritten to use `VectorizedSimulation`
- Removed tournament mode, visualisation, multi-tournament features
- Simplified to focus on core evolutionary loop
- New command-line arguments matching vectorised API

**Old API**:
```python
sim = Simulation(
    population_size=32,
    grid_size=30,
    turns_per_generation=50,
    tournament_mode=True,
    # ... many parameters
)
```

**New API**:
```python
sim = VectorizedSimulation(
    population_size=32,
    batch_size=8,
    turns_per_batch=100,
    # ... vectorised parameters
)
```

### `README.md`
**Changes**:
- Updated project structure diagram
- Updated usage examples to use `main.py` instead of inline code
- Updated training pipeline instructions
- Corrected agent file reference to `agent_vectorised.py`

## Performance Benefits

**Before (Non-Vectorised)**:
- Sequential perception extraction: O(N) loops
- Sequential physics updates: O(N) loops
- Multiple code paths causing confusion
- Different APIs for same functionality

**After (Vectorised Only)**:
- Perception extraction: Single NumPy operation
- Physics updates: Vectorised operations
- Single code path for clarity
- Consistent API throughout

**LLM Inference**:
- Remains sequential to preserve per-agent LoRA diversity
- Each agent uses its own evolved weights
- Genetic diversity maintained across population

## Migration Guide

If you were using the old non-vectorised system:

**Old Way**:
```bash
cd src
python main.py --num-generations 20 --population-size 32 --tournament
```

**New Way**:
```bash
cd src
python main.py --generations 20 --population 32 --batch-size 8
```

**Key Differences**:
- No tournament mode (agents compete in batches on shared grid)
- No visualisation integration (can be added separately if needed)
- Batch size is required parameter
- Different command-line argument names

## Next Steps

1. **Performance Profiling**: With single pathway, can now clearly profile:
   - LLM inference time (expected bottleneck)
   - Environment operations (should be fast with vectorisation)
   - Memory usage patterns

2. **Optimisation Opportunities**:
   - Profile sequential LLM inference
   - Consider caching compiled model graphs
   - Investigate CUDA optimisations
   - Check for unnecessary weight copies

3. **Feature Additions** (if needed):
   - Visualisation can be added to vectorised simulation
   - Tournament mode could be implemented in vectorised architecture
   - Performance metrics tracking

## Rationale

The refactor was performed to:

1. **Eliminate Confusion**: Two different simulation systems with different APIs
2. **Simplify Debugging**: Single code path makes performance issues easier to trace
3. **Focus Development**: Vectorised approach is the future direction
4. **Maintain Quality**: Old code was functional but added maintenance burden

The vectorised system preserves all critical features:
- ✅ Per-agent LoRA weights (sequential inference)
- ✅ Genetic evolution (crossover, mutation, elitism)
- ✅ Curriculum learning (generation-based difficulty)
- ✅ Combat mechanics (mugging system)
- ✅ Energy system and fitness tracking

What was removed:
- ❌ Tournament mode (multiple environments per generation)
- ❌ Integrated visualisation (can be reimplemented if needed)
- ❌ Resource regeneration tracking
- ❌ Legacy non-vectorised operations

## Testing

All tests pass with vectorised-only codebase:

```bash
cd src
python test_refactor.py      # Component tests
python test_single_gen.py    # Single generation
python test_evolution.py     # Multi-generation
python test_vectorised.py    # Vectorised operations
```

---

**Status**: Refactor complete. Single vectorised pathway ready for performance debugging.
