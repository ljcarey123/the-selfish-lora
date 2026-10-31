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
  checkpoint.py            save/restore runs between generations

experiments/
  countdown/               single-agent skill: arithmetic puzzles (reasoning-gym)
  prisoners_dilemma/       multi-agent: iterated prisoner's dilemma tournament
  compare_runs.py          overlay fitness curves from several runs
  run_queue.py             run a plan of experiments, resuming after interruptions

legacy/gridworld/          the original 16x16 gridworld project (see its README)
```

## Status (2026-10-30)

Full write-up with figures in [docs/RESULTS.md](docs/RESULTS.md). So far:

- **Prisoner's dilemma vs a bot panel (2 seeds):** selection turns the base
  model's always-defect into mostly cooperation, and keeps the population
  working; without selection, accumulated mutations break nearly all outputs
  (mutational meltdown).
- **One seed evolved end-game defection:** cooperate for five rounds, defect
  from round 6, the last round of every match, although the model was told
  the length was unknown.
- **Self-play keeps everyone selfish:** when genomes mostly play each other,
  cooperators that appear are exploited and selected out, so the population
  stays defecting. The social environment, not the model, decides what evolves.
- **Reciprocity (tit-for-tat) has not evolved** in five seeds: evolution
  finds round-counting strategies, never ones that react to the opponent.
- **But tit-for-tat is easy to represent:** trained by gradient descent into
  the same LoRA, it is perfect and sits far closer to the base model (norm
  2.0) than evolved genomes (about 12). The limit is search, not capacity.
- **Seeded tit-for-tat invades, then drifts away:** four copies take over a
  defecting population in one generation, but once everyone cooperates,
  selection cannot tell reciprocators from pushovers; mutation erodes the
  strategy and defectors return within about ten generations.
- **Weight decay's anchor decides the outcome:** decay towards the base
  model (an always-defector) erodes tit-for-tat even with small mutations;
  decay towards tit-for-tat keeps genuine reciprocity in every run, but a
  random-selection control ends there too. Selection changes the timing,
  the anchor decides the destination.
- **Without decay, selection alone maintains reciprocity:** in 3 of 4
  seeds, seeded tit-for-tat took over and held for 19 generations (35 of 36
  final genomes exact tit-for-tat, despite weights drifting to 3.5x the
  original norm), and even expelled an invading defector. The
  random-selection control lost it. Evolution can keep tit-for-tat, but
  not find it.
- **Countdown:** no improvement over the control at this budget; fitness is
  dominated by which puzzles each generation draws.

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

Each run writes `config.json`, `history.jsonl`, `checkpoint.pt`, `best.pt` and
`fitness.png` to `experiments/<name>/runs/<run-name>/` (a timestamped name if
`--run-name` is not given).

## Resuming after an interruption

Runs checkpoint after every generation (population, settings and random
number generator states), so a resumed run produces exactly the same results
as an uninterrupted one. Re-run the same command with the same `--run-name`
to continue; a finished run just exits.

For a batch of runs, list them in a plan file and use the queue runner. After
a reboot or closed lid, run the same command again: finished runs are skipped
and the interrupted one resumes.

```bash
.venv/Scripts/python experiments/run_queue.py experiments/plans/<plan>.json
.venv/Scripts/python experiments/run_queue.py experiments/plans/<plan>.json --status
```

Closing the lid usually just pauses the process (sleep); a shutdown or
hibernate kills it, and the queue command picks it back up.

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

## Future work

A possible follow-on, in a separate repository, would study these dynamics
in a spatial world with resources: [docs/NEXT_PROJECT.md](docs/NEXT_PROJECT.md).
It is a proposal only and not part of this project.
