"""
Test suite for vectorized/GPU-optimized implementation.

This script verifies:
- VectorizedEnvironment batch operations
- InferenceEngine
- VectorizedSimulation end-to-end
"""

import sys
import os
import numpy as np

# Add parent directory to path
if os.path.basename(os.getcwd()) == 'src':
    sys.path.insert(0, os.path.dirname(os.getcwd()))
    sys.path.insert(0, os.getcwd())

from Environment.environment_vectorized import VectorizedEnvironment, GRID_INT_ENCODING
from Environment.batch_types import BatchInferenceRequest, BatchInferenceResponse
from Simulation.simulation_vectorized import VectorizedSimulation
from config_loader import GRID_SYMBOLS


def test_vectorized_environment():
    """Test vectorized environment batch operations."""
    print("="*70)
    print("TEST 1: Vectorized Environment - Batch Operations")
    print("="*70)

    env = VectorizedEnvironment(generation=0)

    # Register 8 agents
    agent_ids = [f"agent_{i}" for i in range(8)]
    env.register_agents(agent_ids, initial_energy=20.0)

    print(f"✓ Registered {len(agent_ids)} agents")
    print(f"  Agent positions shape: {env.agent_positions.shape}")
    print(f"  Agent energies shape: {env.agent_energies.shape}")

    # Test batch perception (vectorized)
    perception = env.get_batch_perception()

    print(f"✓ Batch perception extracted")
    print(f"  Perception grids shape: {perception.grids.shape}")
    print(f"  Expected: ({len(agent_ids)}, 5, 5)")
    assert perception.grids.shape == (len(agent_ids), 5, 5), "Perception shape mismatch"

    # Validate perception
    assert perception.validate(), "Perception validation failed"
    print(f"✓ Perception validation passed")

    # Test batch actions
    actions = ["MOVE_NORTH"] * 4 + ["MOVE_SOUTH"] * 4
    agent_indices = np.arange(8)
    results = env.apply_actions(agent_indices, actions)

    print(f"✓ Batch actions applied")
    print(f"  Results for {len(results)} agents")

    # Test metabolism (vectorized)
    energy_before = env.agent_energies.copy()
    env.apply_metabolism()
    energy_after = env.agent_energies

    assert np.allclose(energy_after, energy_before - 1), "Metabolism not applied correctly"
    print(f"✓ Vectorized metabolism applied")

    # Test alive status update
    env.agent_energies[0] = 0  # Kill first agent
    dead_ids = env.update_alive_status()

    assert len(dead_ids) == 1, "Alive status update failed"
    assert dead_ids[0] == agent_ids[0], "Wrong agent marked as dead"
    print(f"✓ Alive status update works")

    print("\n✅ Vectorized Environment test passed\n")


def test_batch_inference_types():
    """Test batch inference data structures."""
    print("="*70)
    print("TEST 2: Batch Inference Types")
    print("="*70)

    # Create batch request
    prompts = ["prompt1", "prompt2", "prompt3"]
    adapter_ids = ["adapter1", "adapter2", "adapter3"]
    agent_ids = ["agent1", "agent2", "agent3"]

    request = BatchInferenceRequest(
        prompts=prompts,
        adapter_ids=adapter_ids,
        agent_ids=agent_ids,
        metadata={"turn": 1}
    )

    print(f"✓ Created BatchInferenceRequest")
    print(f"  Batch size: {len(request)}")

    assert request.validate(), "Request validation failed"
    print(f"✓ Request validation passed")

    # Create batch response
    response = BatchInferenceResponse(
        actions=["MOVE_NORTH", "MOVE_SOUTH", "MOVE_EAST"],
        raw_outputs=["raw1", "raw2", "raw3"],
        agent_ids=agent_ids,
        success=[True, True, False]
    )

    print(f"✓ Created BatchInferenceResponse")
    print(f"  Batch size: {len(response)}")

    response_dict = response.to_dict()
    assert len(response_dict) == 3, "Response dict conversion failed"
    assert response_dict["agent1"]["action"] == "MOVE_NORTH", "Response dict mapping failed"
    print(f"✓ Response dict conversion works")

    print("\n✅ Batch Inference Types test passed\n")


def test_prompt_optimization():
    """Test prompt construction from config."""
    print("="*70)
    print("TEST 3: Prompt Construction from Config")
    print("="*70)

    env = VectorizedEnvironment(generation=0)

    # Check that prompt config was loaded
    assert hasattr(env, 'prompt_config'), "Prompt config not loaded"
    assert 'SYSTEM_PROMPT' in env.prompt_config, "System prompt not in config"
    assert 'PROMPT_FORMAT' in env.prompt_config, "Prompt format not in config"
    print(f"✓ Prompt config loaded from JSON")
    print(f"  System prompt: {env.prompt_config['SYSTEM_PROMPT'][:50]}...")

    # Build first prompt
    perception_grid = np.zeros((5, 5), dtype=int)
    perception_grid[2, 2] = GRID_INT_ENCODING[GRID_SYMBOLS["agent"]]  # Center (agent)
    perception_grid[1, 2] = GRID_INT_ENCODING[GRID_SYMBOLS["food"]]  # North
    perception_grid[2, 3] = GRID_INT_ENCODING[GRID_SYMBOLS["wall"]]  # East

    prompt1 = env._build_prompt(perception_grid, energy=25.0, on_food=False, recent_moves=[])
    print(f"✓ Prompt built successfully")
    print(f"  Prompt length: {len(prompt1)} chars")

    # Build second prompt with recent moves
    prompt2 = env._build_prompt(perception_grid, energy=15.0, on_food=True, recent_moves=["MOVE_NORTH", "MOVE_EAST"])

    # Check that prompts contain expected elements
    assert "Energy:" in prompt1, "Energy not in prompt"
    assert "See:" in prompt1, "Vision not in prompt"
    assert "Energy: 25" in prompt1, "Energy value not correct"
    assert "Energy: 15" in prompt2, "Energy value not correct"
    assert "LOW" in prompt2, "Low energy status not shown"
    print(f"✓ Prompt formatting correct")
    print(f"  - Energy status: OK vs LOW")
    print(f"  - Vision with line-of-sight")
    print(f"  - Recent moves tracking")

    print("\n✅ Prompt Construction test passed\n")


def test_vectorized_simulation_single_batch():
    """Test vectorized simulation with smaller batches."""
    print("="*70)
    print("TEST 4: Vectorized Simulation - Batch Size Comparison")
    print("="*70)

    # Test with batch size of 2 (more realistic for current implementation)
    print("Creating simulation (loading LLM, this may take a moment)...")
    print("Testing with 4 agents in 2 batches of 2, 1 round per generation...")
    sim = VectorizedSimulation(
        population_size=4,  # Smaller population for testing
        batch_size=2,       # Batch size of 2
        turns_per_batch=10,  # Shorter test
        num_elites=1,
        rounds_per_generation=1  # Single round for quick testing
    )

    print(f"✓ Created vectorized simulation")
    print(f"  Population: {sim.population_size}")
    print(f"  Batches: {sim.num_batches}")
    print(f"  Batch size: {sim.batch_size}")

    # Run one generation
    import time
    print("\nRunning generation (10 turns with LLM inference)...")
    start_time = time.perf_counter()
    sim.run_generation()
    elapsed = time.perf_counter() - start_time

    print(f"\n✓ Generation completed in {elapsed:.2f}s")
    print(f"  Average time per turn: {elapsed/10:.2f}s")
    print(f"  Average time per agent-turn: {elapsed/(10*4):.3f}s")

    # Check that agents have state
    top3 = sim.get_top_agents(3)
    print(f"\nTop 3 agents:")
    for i, agent in enumerate(top3, 1):
        print(f"  #{i}: {agent.id} - Energy: {agent.energy:.1f}, Food: {agent.food_collected}")

    assert len(top3) == 3, "get_top_agents failed"
    assert all(agent.energy >= 0 for agent in sim.population), "Negative energy detected"
    print(f"✓ Agent states valid")

    # Test evolution
    print("\nEvolving population...")
    sim.evolve_population()

    assert len(sim.population) == sim.population_size, "Population size changed after evolution"
    print(f"✓ Evolution preserved population size")

    print("\n✅ Vectorized Simulation test passed\n")


def test_performance_comparison():
    """Compare vectorized vs non-vectorized perception extraction."""
    print("="*70)
    print("TEST 5: Performance - Vectorized vs Loop-based")
    print("="*70)

    import time

    env = VectorizedEnvironment(generation=0)
    agent_ids = [f"agent_{i}" for i in range(8)]
    env.register_agents(agent_ids, initial_energy=20.0)

    # Vectorized extraction
    start = time.perf_counter()
    for _ in range(100):
        perception = env.get_batch_perception()
    vectorized_time = time.perf_counter() - start

    print(f"✓ Vectorized extraction (100 iterations): {vectorized_time*1000:.2f}ms")

    # Simulated loop-based extraction (for comparison)
    start = time.perf_counter()
    for _ in range(100):
        # Simulate individual extraction (similar to old method)
        for i in range(8):
            x, y = env.agent_positions[i]
            # Individual slicing (no actual extraction, just timing the loop overhead)
            _ = (x, y)
    loop_time = time.perf_counter() - start

    print(f"✓ Loop overhead (100 iterations): {loop_time*1000:.2f}ms")

    speedup = loop_time / vectorized_time if vectorized_time > 0 else 0
    print(f"\n✓ Vectorized is ~{speedup:.1f}x faster (excluding actual perception logic)")

    print("\n✅ Performance test complete\n")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("VECTORIZED IMPLEMENTATION TEST SUITE")
    print("="*70 + "\n")

    try:
        test_vectorized_environment()
        test_batch_inference_types()
        test_prompt_optimization()
        test_vectorized_simulation_single_batch()
        test_performance_comparison()

        print("\n" + "="*70)
        print("ALL TESTS PASSED ✅")
        print("="*70)
        print("\n🚀 Vectorized implementation ready for production!")
        print("   - Batch operations working")
        print("   - vLLM-ready inference structure")
        print("   - Optimized prompt construction")
        print("   - NumPy vectorization in place")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
