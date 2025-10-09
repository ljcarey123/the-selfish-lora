# System Architecture: The Selfish LoRA

## 1. Overview

This document outlines the architecture of the "Selfish LoRA" project. The system is an evolutionary simulation designed to develop intelligent agent behaviors. It uses principles of genetic algorithms (selection, crossover, mutation) to evolve a population of agents, each equipped with a "LoRA Brain," within a resource-driven environment.

The architecture is modular, separating the simulation logic, agent behavior, decision-making model, and the environment into distinct components.

## 2. Core Components

The system is composed of four primary components:

### `Simulation`

- **File:** `src/Simulation/simulation.py`
- **Responsibility:** The central orchestrator of the entire simulation.
- **Key Functions:**
  - Initializes and manages the population of agents.
  - Creates a fresh `Environment` for each generation.
  - Executes the main evolutionary loop: runs generations, evaluates agent fitness, and evolves the population.
  - Implements the genetic algorithm:
    - **Selection:** Selects the fittest agents as parents for the next generation.
    - **Reproduction (Crossover):** Creates new child agents by combining the "genes" (brain weights) of parents.
    - **Mutation:** Introduces small, random changes into the children's brain weights.

### `Agent`

- **File:** `src/Agent/agent.py`
- **Responsibility:** Represents an individual entity within the simulation.
- **Key Attributes:**
  - `brain (LoraBrain)`: The decision-making module.
  - `position`: Its coordinates on the environment grid.
  - `fitness_score`: A score reflecting its performance, primarily based on collected resources.
- **Behavior:** In each turn, the agent perceives the environment, queries its brain for an action, and executes that action (e.g., move, gather resources).

### `LoraBrain`

- **File:** `src/LoraBrain/lora_brain.py`
- **Responsibility:** The agent's decision-making "brain." It is designed to simulate a LoRA-adapted model.
- **Key Attributes:**
  - `weights`: A NumPy array representing the agent's "genes." These weights are evolved by the `Simulation`.
- **MVP Status:** Currently, the brain does not use a real neural network. It holds a weight matrix but selects actions randomly. This serves as a placeholder to allow the evolutionary mechanics to be tested. The future goal is to use these weights to influence the output of a real LLM.

### `Environment`

- **File:** `src/Environment/environment.py`
- **Responsibility:** Manages the simulation world, including the grid and resources.
- **Key Features:**
  - A 2D grid where agents operate.
  - Contains resources that agents can collect to increase their fitness.
  - For consistency, a new environment with a fixed, identical resource layout is created for each generation.

## 3. Simulation Flow

The simulation proceeds in a sequence orchestrated by `main.py` and the `Simulation` class.

1. **Initialization:**
   - `main.py` instantiates the `Simulation` class.
   - The `Simulation` creates the initial population of `Agent`s, each with a randomly initialized `LoraBrain`.

2. **Generational Loop:** The simulation runs for a fixed number of generations. Each generation consists of two phases:

   a. **Execution Phase (`run_generation`)**
      - A new, clean `Environment` is created.
      - Agents are placed at random positions.
      - The simulation proceeds for a fixed number of turns. In each turn, every `Agent` in the population performs one action (`decide_and_act`).
      - Agents' `fitness_score` is updated based on their actions (e.g., gathering resources).

   b. **Evolution Phase (`evolve`)**
      - **Selection:** Agents are sorted by `fitness_score`, and the top performers are selected as `parents`.
      - **Reproduction & Mutation:** A new population is created. Pairs of `parents` are chosen to produce `children`. The child's `LoraBrain` weights are a combination (crossover) of the parents' weights, with a small random mutation applied.
      - The old population is replaced by this new generation.

3. **Completion:** After the final generation, the simulation concludes.

## 4. Data Flow Diagram

```text
[ main.py ]
     |
     v
[ Simulation ]-------------+
     |                     |
     | Creates & Manages   | Evolves
     v                     |
[ Population of Agents ]   |
     |         ^           |
     |         | Updates   | Modifies Weights
     | Interacts with      |
     v         |           v
[ Environment ]<----[ Agent ]----->[ LoraBrain ]
(Grid & Resources) (Position & Fitness)   (Weights)
```
