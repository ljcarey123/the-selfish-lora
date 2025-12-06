# System Architecture: The Selfish LoRA

## 1. Overview

This document outlines the architecture of the "Selfish LoRA" project. The system is an evolutionary simulation designed to develop intelligent agent behaviours using **vectorised batch processing** for GPU-optimised LLM inference. It uses principles of genetic algorithms to evolve a population of agents, each equipped with a "LoRA Brain," within a strategic combat environment.

The architecture is modular, separating the simulation logic, agent behaviour, and the environment. It includes a comprehensive data collection and visualisation pipeline, plus **batch inference engine** for processing multiple agents in parallel.

## 2. Core Components

The system is composed of nine primary components:

### `VectorisedSimulation`

- **File:** `src/Simulation/simulation_vectorised.py`
- **Responsibility:** The central orchestrator using vectorised batch processing.
- **Key Functions:**
  - Initialises and manages the population of agents.
  - Creates a fresh `VectorisedEnvironment` for each generation.
  - Executes the main evolutionary loop with batched LLM inference.
  - Processes agents in batches (e.g., 32 agents in 4 batches of 8).
  - Collects detailed metrics (performance, genetic, behavioural) at the end of each generation.

**Batch Processing:** The simulation divides the population into batches to prevent overcrowding and enable parallel GPU processing. Each batch runs independently for a set number of turns.

### `Agent`

- **File:** `src/Agent/agent.py`
- **Responsibility:** Represents an individual entity within the simulation.
- **Key Attributes:**
  - `brain (LoraBrain)`: The decision-making module.
  - `position`: Its coordinates on the environment grid.
  - `energy`: Current energy level (starts at 20).
  - `food_collected`: Number of food items gathered.
  - `combat_history`: Record of combat events.
  - `fitness_score`: Normalised score reflecting performance.
  - `path_history`: A record of movements and actions for visualisation.
  - `action_history`: A record of decisions for behavioural analysis.

### `LoraBrain`

- **File:** `src/LoraBrain/lora_brain.py`
- **Responsibility:** The agent's decision-making "brain."
- **Implementation Details:** The brain uses the `Qwen/Qwen2.5-0.5B-Instruct` model as its foundation. At each turn, it constructs a two-part prompt:
  - A **System Prompt** that defines the agent's high-level goal (e.g., "You are a creature trying to survive...").
  - A **User Prompt** that provides the immediate situational context (perception, history, available actions).
- This prompt is then passed to the **BatchInferenceEngine** for sequential processing with the agent's unique LoRA weights.

### `BatchInferenceEngine`

- **File:** `src/LoraBrain/batch_inference_engine.py`
- **Responsibility:** Processes agents' LLM requests with per-agent LoRA weights.
- **Key Features:**
  - **Per-Agent LoRA Loading:** Each agent's unique weights loaded before inference.
  - **Sequential Processing:** Preserves evolved diversity across population.
  - **Action Parsing:** Extracts movement commands from LLM outputs.
  - **Error Handling:** Fallback actions for failed inference.
- **Design Choice:** Sequential inference ensures each agent uses its own evolved LoRA weights, maintaining genetic diversity.

### `VectorisedEnvironment`

- **File:** `src/Environment/environment_vectorised.py`
- **Responsibility:** Manages the simulation world using NumPy vectorised operations.
- **Key Features:**
  - 16x16 grid world with walls and food
  - Batch perception extraction (all agents at once)
  - Vectorised physics updates (metabolism, alive status)
  - Combat mechanics (mugging system)
  - Curriculum learning across generations
  - Normalised fitness scoring

### `Visualiser`

- **File:** `src/visualisation/visualiser.py`
- **Responsibility:** Reads the output data and generates all plots and animations.

### `GenerationManager`

- **File:** `src/Environment/generation_manager.py`
- **Responsibility:** Manages curriculum learning across generations.
- **Key Features:**
  - Progressive difficulty scaling (3 phases)
  - Food density curve (30% → 5%)
  - Regrowth rate scheduling

### `MapGenerator`

- **File:** `src/Environment/map_generator.py`
- **Responsibility:** Generates validated maps with connectivity checking.
- **Key Features:**
  - Random wall placement (10-15% density)
  - Flood-fill connectivity validation
  - Food placement with position avoidance

### Dataset Generation

- **File:** `fine_tuning/create_dataset.py`
- **Responsibility:** Creates a synthetic dataset for training the agent model.
- **Key Features:**
  - Generates balanced examples for each action
  - Uses JSONL format (one JSON object per line) for compatibility
  - Implements stratified splitting for train/validation/test sets

### Fine-Tuning Pipeline

- **File:** `fine_tuning/fine_tune_model.py`
- **Responsibility:** Trains the agent model using the generated dataset.
- **Key Features:**
  - Supports hyperparameter tuning with Optuna
  - Tracks action-level metrics (valid action rate, correct action rate)
  - Generates comprehensive reports for each run
  - Produces training/validation loss curves

## 3. Configuration Management

The project uses a centralised and flexible configuration system to manage shared strings and settings, ensuring consistency across different parts of the application (e.g., between dataset generation and agent inference).

- **`.env` file:** Located at the project root, this file stores environment-specific variables. Its primary role in this system is to define `PROMPTS_FILE_PATH`, which points to the location of the prompts configuration file.

- **`config/prompts.json`:** This file is the single source of truth for all user-facing and system-level prompts, such as the `SYSTEM_PROMPT` and the list of `VALID_ACTIONS`. Its JSON format makes it easy to read and edit.

- **`config_loader.py` Modules:** Located in both the `src` and `fine_tuning` directories, these modules are responsible for loading the `.env` file, reading the path from `PROMPTS_FILE_PATH`, and then loading the contents of the `prompts.json` file. They expose the configuration values as Python constants that can be easily imported by any script in their respective directories.

This architecture allows for easy modification of prompts without changing any Python code, and it keeps sensitive information like API keys out of version control.

## 4. Vectorised Simulation Flow

The simulation proceeds in a sequence orchestrated by the vectorised architecture:

1. **Initialisation:** `VectorisedSimulation` instantiates the initial agent population with LoRA weights.

2. **Generational Loop:** The simulation runs for a set number of generations. Each generation consists of:
    a.  **Batch Execution (`run_generation`):**
        - Population split into batches (e.g., 32 agents → 4 batches of 8)
        - Each batch runs independently in the environment
        - Agents act for a set number of turns with **sequential LLM inference**
    b.  **Per-Turn Processing:** For each turn:
        - Extract perception for all alive agents (vectorised)
        - Prepare inference request with all prompts
        - Process agents sequentially, loading each agent's unique LoRA weights
        - Apply actions (vectorised where possible)
        - Update physics (metabolism, combat) with vectorised operations
    c.  **Metrics Collection (`_calculate_metrics`):** At the end of the generation, the simulation calculates and stores performance, genetic, and behavioural metrics.
    d.  **Evolution (`evolve_population`):** The population is evolved through selection, crossover, and mutation.

3. **Data Persistence:** After the final generation, the simulation saves all collected history (`performance_history`, `genetic_history`, etc.) into JSON files in the `src/output/` directory.

4. **Visualisation:** The `Visualiser` can be called to automatically generate and save all plots and animations based on the newly created data files.

### Inference and Environment Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                      SIMULATION TURN                              │
└──────────────────────────────────────────────────────────────────┘

Step 1: Get Perceptions (Vectorised)
──────────────────────────────────
VectorisedEnvironment.get_batch_perception(alive_indices)
  ↓
  Returns BatchPerception:
    - grids: (N, 5, 5) array of integer-encoded perceptions
    - agent_ids: List of N agent IDs
    - positions, energies, on_food

Step 2: Sequential LLM Inference (Per-Agent LoRA)
──────────────────────────────────────────────────
BatchInferenceEngine.process_batch(request)
  ↓
  For each agent:
    - Load agent's unique LoRA weights
    - Generate action with model.generate()
  ↓
  Returns BatchInferenceResponse:
    - actions: ["MOVE_NORTH", "MOVE_EAST", ...]
    - success flags, raw outputs

Step 3: Apply Actions & Physics (Vectorised)
─────────────────────────────────────────────
VectorisedEnvironment.apply_actions(agent_indices, actions)
  ↓
  Apply combat damage (vectorised) → Apply metabolism (vectorised)
```

## 5. Data Flow Diagram

```text
[ VectorisedSimulation ]
     |
     | Creates & Manages Batches
     v
[ Population of Agents ] ----Batch 1 (8 agents)--->[ VectorisedEnvironment ]
     |                       Batch 2 (8 agents)           |
     | Evolves                Batch 3 (8 agents)           | Batch Perception
     v                        Batch 4 (8 agents)           v
[ Selection/Crossover ]                          [ BatchInferenceRequest ]
     |                                                     |
     | Modifies Weights                                    | All Prompts
     v                                                     v
[ New Generation ] <---------------------------[ BatchInferenceEngine ]
                                                          |
                                                          | Single Forward Pass
                                                          v
                                                  [ BatchInferenceResponse ]
                                                          |
                                                          | All Actions
                                                          v
                                                [ Vectorised Physics Update ]
