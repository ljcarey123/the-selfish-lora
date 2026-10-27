"""
Shared machinery for evolving LoRA adapters with a genetic algorithm.

Each experiment (see experiments/) supplies a task that turns a model's text
outputs into fitness scores; everything else lives here:

- model.py      load the base LLM with a LoRA adapter attached
- genome.py     a genome is a dict of LoRA tensors; copy it into the model
- evolution.py  initial population, selection, crossover, mutation
- generate.py   batched greedy generation for one genome over many prompts
- runlog.py     per-generation logging and fitness plots
"""
