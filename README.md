# The Selfish LoRA

Can a genetic algorithm over LoRA adapters improve a small LLM? Each agent is a
LoRA "genome" on top of a shared base model (Qwen2.5-0.5B-Instruct). Genomes
are scored on a task, the fittest breed (crossover + mutation), and the loop repeats.

Runs are sized for a 4 GB laptop GPU and finish in under about an hour.

## Layout

```
src/selfish_lora/          shared machinery used by every experiment
  model.py                 load base model + LoRA adapter
  genome.py                genome = dict of LoRA B tensors; load into model
  evolution.py             initial population, selection, crossover, mutation
  generate.py              batched greedy generation
  runlog.py                per-generation history.jsonl + fitness plots

experiments/
  countdown/               single-agent skill: arithmetic puzzles (reasoning-gym)
  prisoners_dilemma/       multi-agent: iterated prisoner's dilemma tournament
  compare_runs.py          overlay fitness curves from several runs

legacy/gridworld/          the original 16x16 gridworld project (see its README)
```

## The two experiments

| | Countdown | Prisoner's dilemma |
|---|---|---|
| Question | Can evolution improve a single-agent skill? | Do known evolutionary dynamics emerge between LLM agents? |
| Fitness | reasoning-gym score on a shared batch of puzzles | Average payoff per round in a round-robin tournament |
| Base model | 6% exact on held-out puzzles | Always defects |
| Look for | Held-out score rising above the base model | Shift from always-defect towards reciprocity (cooperation, retaliation) |

Run one of them, then the same thing with `--control` (identical but selection is random) and compare:

```bash
.venv/Scripts/python experiments/countdown/run.py --generations 30
.venv/Scripts/python experiments/countdown/run.py --generations 30 --control
.venv/Scripts/python experiments/compare_runs.py experiments/countdown/runs/<evolved> experiments/countdown/runs/<control>
```

Each run writes `config.json`, `history.jsonl`, `best.pt` and `fitness.png` to
`experiments/<name>/runs/<timestamp>_<tag>/`.

## Design choices

- **Only LoRA B matrices evolve.** A is fixed at PEFT's random init and shared, so a
  mutation to B changes behaviour directly. Evolving A and B from zero (as the
  gridworld did) makes small mutations almost invisible.
- **Same inputs for everyone.** All genomes in a generation see the same puzzles or the
  same tournament, so fitness differences come from the weights.
- **The base genome is in generation 0**, so the best fitness never starts below the base model.
- **Always run a control.** Without one you can't tell evolution from noise.

## Setup

```bash
pip install -r requirements.txt
```

Put `HF_TOKEN=...` in `.env` at the repo root if your model needs it.
