# Windsurf Project Context: The Selfish LoRA

## Project Overview

"The Selfish LoRA" is a Python-based evolutionary simulation. The goal is to simulate agents in an environment where they compete for resources. The project uses a genetic algorithm to evolve the agents' "brains" over generations, aiming to develop intelligent behaviors. The "LoRA" aspect suggests a future integration with a Large Language Model (LLM) using Low-Rank Adaptation.

## Project Goals

### Current Goal: First Major Evolutionary Experiment

Having validated the core simulation mechanics, the project has now entered its first major experimental phase. The primary objective is to run a long-form simulation to observe if complex, non-random behaviors can evolve in a challenging environment.

**Key Experimental Parameters:**
- **Generations:** 25
- **Population Size:** 20
- **Grid Size:** 60x60 (a large, challenging world)
- **Turns per Generation:** 100

**Success Criteria:**
- Observe a clear, upward trend in the `average_fitness` over the 25 generations.
- See a corresponding decrease in the frequency of "unparsable response" warnings, indicating the evolution of communicative competence.
- Qualitatively analyze the champion agent paths for signs of non-random, strategic exploration.

### Long-Term Goals

- Evolve the LoRA weights to produce intelligent, emergent behaviors from the base LLM.
- Experiment with more complex environments and agent interactions.
- Visualize simulation results and agent behaviors.

## Tech Stack

- **Language:** Python 3
- **Core Libraries:**
  - `numpy`: For numerical operations.
  - `torch`: For deep learning and tensor operations.
  - `transformers`: For loading and interacting with the LLM.
  - `peft`: For applying LoRA weights to the base model.
  - `matplotlib`: For creating visualizations.

## Key Components

- **`main.py`**: The main entry point for running the simulation.
- **`Simulation`**: (`src/Simulation/`) The central orchestrator, managing the evolutionary loop.
- **`Agent`**: (`src/Agent/`) Represents an individual agent in the simulation.
- **`LoraBrain`**: (`src/LoraBrain/`) The agent's decision-making module. It uses a base LLM (`Qwen/Qwen2.5-0.5B-Instruct`) guided by evolved LoRA weights to choose actions.
- **`Environment`**: (`src/Environment/`) The grid world where agents live and gather resources.

## Coding Style & Conventions

- **Style:** Follow PEP 8 guidelines for Python code.
- **Documentation:** Use docstrings for all public modules, classes, and functions.
- **Modularity:** Keep components decoupled. The `Simulation` class should orchestrate, but not contain the internal logic of agents or the environment.
