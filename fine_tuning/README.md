# Fine-Tuning Scripts

This directory contains the scripts for creating a foundational LoRA model.

## Workflow

1.  **`create_dataset.py`**: Run this script first to generate the `training_dataset.json` file. This file contains the prompt-response pairs needed for fine-tuning.

2.  **`fine_tune.py`**: Run this script to use the generated dataset to fine-tune the base LLM. This will produce the `base_lora_weights.pth` file in the `output/weights/` directory.
