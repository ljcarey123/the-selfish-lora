# System Architecture: The Selfish LoRA

## 1. Overview

This document outlines the architecture of the "Selfish LoRA" project. The system is an evolutionary simulation designed to develop intelligent agent behaviors. It uses principles of genetic algorithms to evolve a population of agents, each equipped with a "LoRA Brain," within a resource-driven environment.

The architecture is modular, separating the simulation logic, agent behavior, and the environment. It also includes a comprehensive data collection and visualization pipeline for analyzing simulation results.

## 2. Core Components

The system is composed of five primary components:

### `Simulation`

- **File:** `src/Simulation/simulation.py`
- **Responsibility:** The central orchestrator of the simulation and data collection.
- **Key Functions:**
  - Initializes and manages the population of agents.
  - Creates a fresh `Environment` for each generation.
  - Executes the main evolutionary loop.
  - Collects detailed metrics (performance, genetic, behavioral) at the end of each generation.

### `Agent`

- **File:** `src/Agent/agent.py`
- **Responsibility:** Represents an individual entity within the simulation.
- **Key Attributes:**
  - `brain (LoraBrain)`: The decision-making module.
  - `position`: Its coordinates on the environment grid.
  - `fitness_score`: A score reflecting its performance.
  - `path_history`: A record of its movements and actions for visualization.
  - `action_history`: A record of the decisions it makes for behavioral analysis.

### `LoraBrain`

- **File:** `src/LoraBrain/lora_brain.py`
- **Responsibility:** The agent's decision-making "brain."
- **Implementation Details:** The brain uses the `Qwen/Qwen2.5-0.5B-Instruct` model as its foundation. At each turn, it constructs a two-part prompt:
  - A **System Prompt** that defines the agent's high-level goal (e.g., "You are a creature trying to survive...").
  - A **User Prompt** that provides the immediate situational context (perception, history, available actions).
- This prompt is then passed to the base model, with the agent's unique LoRA weights applied, to generate the chosen action.

### `Environment`

- **File:** `src/Environment/environment.py`
- **Responsibility:** Manages the simulation world, including the grid and resources.

### `Visualizer`

- **File:** `src/visualization/visualizer.py`
- **Responsibility:** Reads the output data and generates all plots and animations.

## 3. Configuration Management

The project uses a centralized and flexible configuration system to manage shared strings and settings, ensuring consistency across different parts of the application (e.g., between dataset generation and agent inference).

- **`.env` file:** Located at the project root, this file stores environment-specific variables. Its primary role in this system is to define `PROMPTS_FILE_PATH`, which points to the location of the prompts configuration file.

- **`config/prompts.json`:** This file is the single source of truth for all user-facing and system-level prompts, such as the `SYSTEM_PROMPT` and the list of `VALID_ACTIONS`. Its JSON format makes it easy to read and edit.

- **`config_loader.py` Modules:** Located in both the `src` and `fine_tuning` directories, these modules are responsible for loading the `.env` file, reading the path from `PROMPTS_FILE_PATH`, and then loading the contents of the `prompts.json` file. They expose the configuration values as Python constants that can be easily imported by any script in their respective directories.

This architecture allows for easy modification of prompts without changing any Python code, and it keeps sensitive information like API keys out of version control.

## 4. Simulation Flow

The simulation proceeds in a sequence orchestrated by `main.py`:

1. **Initialization:** `main.py` instantiates the `Simulation` class, which creates the initial agent population.

2. **Generational Loop:** The simulation runs for a set number of generations. Each generation consists of:
    a.  **Execution (`run_generation`):** Agents act within the environment for a set number of turns.
    b.  **Metrics Collection (`_calculate_and_store_metrics`):** At the end of the generation, the `Simulation` class calculates and stores performance, genetic, and behavioral metrics.
    c.  **Evolution (`evolve`):** The population is evolved through selection, crossover, and mutation.

3. **Data Persistence:** After the final generation, `main.py` saves all collected history (`performance_history`, `genetic_history`, etc.) into a single JSON file in the `src/output/` directory.

4. **Visualization:** `main.py` then calls the `Visualizer` to automatically generate and save all plots and animations based on the newly created data file.

## 4. Data Flow Diagram

```text
[ main.py ]
     |
     v
[ Simulation ]-------------+--------------->[ output/simulation_paths.json ]
     |                     |                                |
     | Creates & Manages   | Evolves                        |
     v                     |                                v
[ Population of Agents ]   |                          [ Visualizer ]
     |         ^           |                                |
     |         | Updates   | Modifies Weights               v
     | Interacts with      |                          [ .png & .gif files ]
     v         |           v
[ Environment ]<----[ Agent ]----->[ LoraBrain ]
(Grid & Resources) (State & History)   (Weights)
```
