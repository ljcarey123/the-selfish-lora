# Windsurf Project Context: The Selfish LoRA

## Project Overview

This project focuses on fine-tuning a small, instruction-following Large Language Model (LLM) to act as a simple robot agent. The agent's goal is to navigate a 2D grid to find a resource ('R'). The project involves creating a synthetic, perfectly balanced dataset to teach the model a specific, narrow skill and then evaluating its performance against the base model.

## Project Goals

### Current Goal: Successful Hyperparameter Tuning

We have successfully implemented hyperparameter tuning for our fine-tuning process, allowing us to optimize learning rate, batch size, and warmup ratio.

#### Key Improvements
* Fixed dataset generation to use proper JSONL format
* Implemented hyperparameter search with Optuna
* Improved reporting to only include current trial runs

#### Metrics
* Achieved 78.5% valid actions
* 40% correct actions (2x better than random)
* 0.6 action diversity

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

- **`create_dataset.py`**: (`fine_tuning/`) Generates a synthetic, balanced dataset and splits it into training, validation, and test sets.
- **`fine_tune_model.py`**: (`fine_tuning/`) Runs the fine-tuning process using the training and validation sets. It saves the best-performing LoRA adapter.
- **`evaluate_model.py`**: (`fine_tuning/`) Provides a final, side-by-side comparison of the base model vs. the fine-tuned model on the held-out test set, reporting a definitive accuracy score.
- **Base LLM**: `Qwen/Qwen2.5-0.5B-Instruct` is used as the foundation model for fine-tuning.

## Coding Style & Conventions

- **Style:** Follow PEP 8 guidelines for Python code.
- **Documentation:** Use docstrings for all public modules, classes, and functions.
- **Modularity:** Keep components decoupled. The `Simulation` class should orchestrate, but not contain the internal logic of agents or the environment.
