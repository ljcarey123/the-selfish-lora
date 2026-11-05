# Fine-Tuning Plan: Creating Foundational LoRA Weights

## 1. Objective

The goal of this task is to create a set of "foundational" LoRA weights by fine-tuning the base LLM on a curated dataset. These weights will provide a better starting point for the evolutionary simulation, replacing the current zero-initialized weights.

By starting with a model that already understands the basic goals and actions, we expect the evolutionary process to be more efficient, leading to the emergence of more complex strategies faster.

## 2. Implementation Phases

### Phase 1: Dataset Creation

We will create a script to procedurally generate a high-quality dataset for fine-tuning. This dataset will consist of hundreds or thousands of prompt-response pairs that teach the model the core mechanics of the simulation.

- **New Script:** `fine_tuning/create_dataset.py`
- **Output:** `fine_tuning/training_dataset.json`
- **Data Format:** A JSON file containing a list of conversations, where each conversation has a system prompt and a user prompt (the situation) and an expected model response (the action).

**Example Data Points:**

- **Goal:** Teach the model to move towards visible resources.
  - *Situation:* `"- A resource 'R' is to your North."`
  - *Ideal Response:* `"ACTION: North"`
- **Goal:** Teach the model to gather resources it is on top of.
  - *Situation:* `"- You are on top of a resource 'R'."`
  - *Ideal Response:* `"ACTION: Gather"`
- **Goal:** Teach the model to explore when no resources are visible.
  - *Situation:* `"- There is a wall to your North.\n- There is a wall to your West."`
  - *Ideal Response:* `"ACTION: East"` (or any other valid exploratory move)

### Phase 2: Fine-Tuning Script

We will create a script that uses the generated dataset to fine-tune the base LLM with a LoRA configuration.

- **New Script:** `fine_tuning/fine_tune.py`
- **Input:** `fine_tuning/training_dataset.json`
- **Output:** `output/weights/base_lora_weights.pth`

**Key Steps for the Script:**

1. Load the base `Qwen/Qwen2.5-0.5B-Instruct` model and tokenizer.
2. Apply the same `LoraConfig` that is used in the main simulation.
3. Load the dataset from `training_dataset.json`.
4. Use the `transformers.Trainer` API to run the fine-tuning process.
5. Save the resulting trained LoRA weights to `output/weights/base_lora_weights.pth`.

### Phase 3: Integration with the Main Simulation

Finally, we will modify the `Simulation` class to use these new foundational weights when creating the initial population.

- **Modified Script:** `src/simulation/simulation.py`

**Changes Required:**

1. In the `_create_initial_population` method, before creating agents:
   - Load the foundational weights from `output/weights/base_lora_weights.pth`.
2. Inside the loop that creates the initial population:
   - For each new agent, create a deep copy of the foundational weights.
   - Apply a small amount of random noise to this copy.
   - Instantiate the agent's `LoraBrain` and use the `set_weights()` method to apply the noisy, foundational weights.

This change ensures that the simulation starts with a population of agents that are already competent and aligned with the task, while still having the necessary diversity for evolution to occur.
