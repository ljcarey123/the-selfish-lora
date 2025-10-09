# Windsurf Project Context: The Selfish LoRA

## Project Overview

"The Selfish LoRA" is a Python-based evolutionary simulation. The goal is to simulate agents in an environment where they compete for resources. The project uses a genetic algorithm to evolve the agents' "brains" over generations, aiming to develop intelligent behaviors. The "LoRA" aspect suggests a future integration with a Large Language Model (LLM) using Low-Rank Adaptation.

## Project Goals

### Short-Term Goals

- Solidify the core simulation loop (evolution, selection, mutation).
- Refine the fitness function and agent actions.
- Establish a baseline for agent performance with the current mock `LoraBrain`.

### Long-Term Goals

- Replace the mock `LoraBrain` with a real LLM integration.
- Use the evolved weights as LoRA adapters to guide the LLM's decision-making process.
- Experiment with more complex environments and agent interactions.
- Visualize simulation results and agent behaviors.

## Tech Stack

- **Language:** Python 3
- **Core Libraries:**
  - `numpy`: For numerical operations, especially for managing brain weights and the environment grid.
  - `matplotlib`: For creating visualizations and animations of the simulation results.

## Key Components

- **`main.py`**: The main entry point for running the simulation.
- **`Simulation`**: (`src/Simulation/`) The central orchestrator, managing the evolutionary loop.
- **`Agent`**: (`src/Agent/`) Represents an individual agent in the simulation.
- **`LoraBrain`**: (`src/LoraBrain/`) The agent's decision-making module. Currently a mock, but intended to be the core of the evolutionary process.
- **`Environment`**: (`src/Environment/`) The grid world where agents live and gather resources.

## Coding Style & Conventions

- **Style:** Follow PEP 8 guidelines for Python code.
- **Documentation:** Use docstrings for all public modules, classes, and functions.
- **Modularity:** Keep components decoupled. The `Simulation` class should orchestrate, but not contain the internal logic of agents or the environment.
