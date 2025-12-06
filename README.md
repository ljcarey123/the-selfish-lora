# The Selfish LoRA

An evolutionary simulation that evolves intelligent agent behaviours using **vectorised batch processing** for GPU-optimised LLM inference. Agents with LoRA-adapted language models compete for resources in a 16x16 combat environment with curriculum learning.

## Key Features

- 🧬 **Evolutionary Algorithms**: Genetic selection, crossover, and mutation
- 🎮 **Strategic Combat Environment**: 16x16 grid with food collection and agent-vs-agent combat
- 🚀 **Vectorised Environment Operations**: NumPy-optimised physics and perception extraction
- 📈 **Curriculum Learning**: Progressive difficulty across 3 phases
- 🧠 **LoRA-Adapted Brains**: Fine-tuned Qwen2.5-0.5B-Instruct with per-agent weights
- 📊 **Comprehensive Metrics**: Performance, genetic diversity, and behavioural analysis

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/your-username/the-selfish-lora.git
cd the-selfish-lora

# Install dependencies
pip install -r requirements.txt
```

### Run Tests

```bash
cd src

# Test vectorised environment components
python test_refactor.py

# Test single generation (16 agents, batch size 8)
python test_single_gen.py

# Test multi-generation evolution (32 agents, 4 batches)
python test_evolution.py

# Test batch inference details
python test_vectorised.py
```

### Run Evolution

```bash
# Run full evolution (32 agents, 4 batches of 8, 20 generations)
cd src
python main.py --generations 20 --population 32 --batch-size 8

# Quick test (16 agents, 2 batches, 5 generations)
python main.py --generations 5 --population 16 --batch-size 8 --turns 50

# Use random initialization (no fine-tuned weights)
python main.py --random-init
```

## Architecture Overview

### Vectorised Environment Processing

The system uses NumPy vectorised operations for environment physics while processing LLM inference sequentially to preserve per-agent LoRA diversity:

**Vectorised Operations (Fast):**
- Perception extraction: All agents' views computed at once
- Metabolism updates: Single NumPy operation for all agents
- Combat damage: Vectorised energy adjustments
- Alive status: Batch boolean operations

**Sequential Operations (Preserves Diversity):**
- LLM inference: Each agent uses its own evolved LoRA weights
- Ensures genetic diversity is maintained across population

### Core Components

1. **VectorisedSimulation** - Orchestrates evolution
   - Splits population into batches (e.g., 32 agents → 4 batches of 8)
   - Manages evolutionary loop with vectorised environment operations

2. **BatchInferenceEngine** - Processes agents with per-agent LoRA weights
   - Sequential inference to preserve evolved diversity
   - Loads each agent's unique LoRA weights before generation
   - Action parsing and error handling

3. **VectorisedEnvironment** - NumPy-optimised world simulation
   - 16x16 grid with walls and food
   - Vectorised perception extraction
   - Vectorised physics updates (metabolism, combat damage)
   - Combat mechanics (mugging system)
   - Curriculum learning across generations

4. **LoraBrain** - LoRA-adapted decision-making
   - Qwen2.5-0.5B-Instruct base model
   - Unique LoRA weights per agent (evolved through generations)
   - Context-aware action selection

### Game Mechanics

**Energy System:**
- Starting energy: 20
- Metabolism: -1 per turn
- Food collection: +15 (automatic)
- Wall collision: -2
- Combat: +5 attacker / -10 victim
- Death at 0 energy

**Combat (Mugging):**
When Agent A moves into Agent B:
- A gains +5 energy
- B loses -10 energy
- A's move fails (stays in place)

**Curriculum Phases:**
- Phase 1 (Gen 0-10): High food density (30%), fast regrowth
- Phase 2 (Gen 11-30): Medium density (15%), slow regrowth
- Phase 3 (Gen 31+): Low density (5%), rare regrowth

## Performance

**Vectorised Environment Benefits:**
- **Perception extraction**: O(1) for all agents vs O(N) sequential
- **Physics updates**: Single NumPy operations vs N loops
- **Memory efficiency**: Contiguous arrays for better cache utilisation

**LLM Inference:**
- Sequential processing to preserve per-agent LoRA weights
- Each agent's evolved behaviour remains unique
- Genetic diversity maintained across generations

**Hardware Requirements:**
- GPU: CUDA-capable (compute capability 7.0+) recommended
- VRAM: 4GB+ for Qwen2.5-0.5B-Instruct
- RAM: 16GB+ recommended

## Training Pipeline

### 1. Create Dataset

```bash
cd fine_tuning
python create_dataset.py
```

Generates balanced training/validation/test splits.

### 2. Fine-Tune Model

```bash
python fine_tune_model.py --hyper --trials 10
```

Runs Optuna hyperparameter search to find diverse models.

### 3. Evaluate Models

```bash
python evaluate_model.py
```

Calculates evolution scores for all fine-tuned runs.

### 4. Prepare Parents

```bash
python prepare_parents.py --top 3
```

Selects top 3 models for multi-parent initialisation.

### 5. Run Evolution

```bash
cd ../src
python main.py --generations 20 --population 32 --batch-size 8
```

Executes vectorised evolutionary simulation with warm-start from fine-tuned weights.

## Documentation

- **[Quick Start Guide](docs/QUICK_START.md)** - Testing and basic usage
- **[System Architecture](docs/system_architecture.md)** - Technical details with batch inference
- **[Workflow Guide](docs/WORKFLOW.md)** - Complete training pipeline
- **[Project Context](.windsurf/project_context.md)** - Goals and conventions

## Project Structure

```
the-selfish-lora/
├── src/
│   ├── Simulation/
│   │   └── simulation_vectorised.py    # Vectorised simulation orchestrator
│   ├── Environment/
│   │   ├── environment_vectorised.py   # NumPy-optimised environment
│   │   ├── batch_types.py              # Batch inference data types
│   │   ├── generation_manager.py       # Curriculum learning
│   │   └── map_generator.py            # Map generation with validation
│   ├── Agent/
│   │   └── agent_vectorised.py         # Lightweight agent for vectorised processing
│   ├── LoraBrain/
│   │   ├── lora_brain.py               # LoRA-adapted decision maker
│   │   └── batch_inference_engine.py   # Batched LLM processing
│   ├── main.py                         # Main entry point
│   ├── test_refactor.py                # Component tests
│   ├── test_single_gen.py              # Single generation test
│   ├── test_evolution.py               # Multi-generation test
│   └── test_vectorised.py              # Vectorised operations tests
├── fine_tuning/
│   ├── create_dataset.py               # Dataset generation
│   ├── fine_tune_model.py              # Hyperparameter search
│   ├── evaluate_model.py               # Model evaluation
│   └── prepare_parents.py              # Parent selection
├── docs/
│   ├── QUICK_START.md                  # Quick start guide
│   ├── system_architecture.md          # Technical architecture
│   └── WORKFLOW.md                     # Complete workflow
├── config/
│   └── prompts.json                    # System prompts and actions
└── requirements.txt                    # Python dependencies
```

## Requirements

**Core Dependencies:**
- Python 3.12
- PyTorch 2.5.1 (CUDA 12.1)
- Transformers 4.57.3
- PEFT 0.10.0
- NumPy 1.26.4
- Optuna 4.5.0

See [requirements.txt](requirements.txt) for complete list.

## Contributing

This is a research project. Contributions welcome!

## License

MIT License - see LICENSE file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{selfish_lora,
  title={The Selfish LoRA: Evolutionary LLM Agents with Vectorised Batch Inference},
  author={Your Name},
  year={2024},
  url={https://github.com/your-username/the-selfish-lora}
}
```

## Acknowledgements

- **Qwen Team** for Qwen2.5-0.5B-Instruct base model
- **Hugging Face** for PEFT library and Transformers
- **Optuna** for hyperparameter optimisation framework

---

**Status**: Vectorised environment with per-agent LoRA inference ready! 🚀

Start with the [Quick Start Guide](docs/QUICK_START.md) to test the system.
