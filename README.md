# The Selfish LoRA

Can natural selection, acting directly on an LLM's weights, produce
cooperation? Each agent is a LoRA "genome" on a shared base model
(Qwen2.5-0.5B-Instruct). Agents play the iterated prisoner's dilemma, the
fittest breed (crossover and mutation), and the loop repeats. No gradients:
selection is the only learning signal.

A small research project sized for a 4 GB laptop GPU; each run takes under
about an hour. Full write-up with figures: [docs/RESULTS.md](docs/RESULTS.md).

## Findings

![Tit-for-tat maintained by selection](docs/figures/pd_no_decay.png)

- **Selection steers behaviour.** Against fixed bots, evolution turns the
  base model's always-defect into cooperation; with random selection,
  accumulated mutations break nearly all outputs (mutational meltdown).
- **The social environment decides what evolves.** Against reciprocating
  bots cooperation evolves; in self-play the population stays selfish.
- **Evolution finds tricks, not reciprocity.** Across five seeds it found
  strategies keyed to the round number (one cooperated for five rounds and
  defected in the last), never ones that react to the opponent.
- **Yet tit-for-tat is easy to represent.** Trained by gradient descent into
  the same LoRA, it is perfect and sits far closer to the base model (norm
  2.0) than evolved genomes (about 12). The limit is search, not capacity.
- **Seeded tit-for-tat invades at a threshold.** One or two copies in a
  defecting population of 12 fail; four take over in one generation.
- **Weight decay's anchor decides what survives.** Decay towards the base
  model, itself an always-defector, destroys tit-for-tat; decay towards
  tit-for-tat preserves it even under random selection.
- **Without decay, selection alone maintains reciprocity.** In 3 of 4 seeds
  tit-for-tat held for 19 generations (35 of 36 final genomes exact
  tit-for-tat, despite weights drifting to 3.5x their original norm) and
  expelled an invading defector. The random-selection control lost it.
- **Countdown** (arithmetic puzzles): no improvement over the control at
  this budget, so that experiment is paused.

In short: **selection on LLM weights can keep tit-for-tat, but did not find
it**, and for a fine-tuned LLM there is no strategically neutral point in
weight space, because the base model already plays a strategy.

Caveats: one small model, populations of 12-16, 1-4 seeds per condition.

## Relation to existing work

The classic dynamics reproduced here (tit-for-tat's strength, invasion
thresholds, unconditional cooperators being exploited) come from
evolutionary game theory: Axelrod's tournaments and his genetic algorithm
over lookup-table strategies, which did evolve tit-for-tat-like rules from
scratch ([Axelrod 1987](https://public.websites.umich.edu/~axe/research/Evolving.pdf)),
and [Nowak & Sigmund 1992](https://www.nature.com/articles/355250a0).

LLM work on the iterated prisoner's dilemma takes three main routes:

- **LLMs as players or strategy writers.** Frontier LLMs entered into
  evolutionary tournaments against classic strategies
  ([Payne & Alloui-Cros 2025](https://arxiv.org/abs/2507.02618)); LLMs
  writing complete strategies whose populations then evolve
  ([Willis et al. 2025](https://arxiv.org/abs/2501.16173)). The model's
  weights do not change.
- **Gradient fine-tuning of LLM weights.** PPO with moral rewards lets LLM
  agents unlearn selfish play
  ([Tennant, Hailes & Musolesi, ICLR 2025](https://arxiv.org/abs/2410.01639)).
  Standard multi-agent RL tends towards defection, while LLMs fine-tuned
  with opponent shaping learn tit-for-tat
  ([Learning Robust Social Strategies with LLMs, 2025](https://arxiv.org/abs/2511.19405)).
- **Populations of LoRA adapters.** [PopuLoRA (2026)](https://arxiv.org/abs/2605.16727)
  co-trains teacher and student LoRAs on one base model with RL, for
  reasoning rather than social dilemmas.

This project instead evolves the LLM's weights by selection alone, in a
social game. We are not aware of prior work doing this. The results echo
the gradient literature (plain optimisation does not find tit-for-tat) and
add a weight-space observation: regularising towards the base model is not
neutral, because the base model is itself a player. Details and further
references: [docs/RESULTS.md, Related work](docs/RESULTS.md#related-work).
This is not a systematic review.

## Layout

```
src/selfish_lora/          shared machinery used by every experiment
  model.py                 load base model + LoRA adapter
  genome.py                genome = dict of LoRA B tensors; load into model
  evolution.py             initial population, selection, crossover, mutation
  generate.py              batched greedy generation
  runlog.py                per-generation history.jsonl + plots
  checkpoint.py            save/restore runs between generations

experiments/
  prisoners_dilemma/       multi-agent: iterated prisoner's dilemma tournament
    run.py                 evolve genomes (bots, self-play, seeding, decay)
    train_reference.py     gradient-train a genome on a known strategy
    fingerprint.py         a genome's move in every short history
  countdown/               single-agent skill: arithmetic puzzles (paused)
  plans/                   the experiment batches behind each result
  compare_runs.py          overlay metrics from several runs
  calibrate_mutation.py    how much noise changes behaviour
  run_queue.py             run a plan, resuming after interruptions

docs/RESULTS.md            results log, related work, limitations
legacy/gridworld/          the original 16x16 gridworld project (see its README)
```

The other files in `docs/` describe the legacy gridworld project.

## The two experiments

| | Countdown | Prisoner's dilemma |
|---|---|---|
| Question | Can evolution improve a single-agent skill? | Do known evolutionary dynamics emerge between LLM agents? |
| Fitness | reasoning-gym score on a shared batch of puzzles | Average payoff per round in a round-robin tournament |
| Base model | 6% exact on held-out puzzles | Always defects |
| Look for | Held-out score rising above the base model | Shift from always-defect towards reciprocity (cooperation, retaliation) |
| Status | Paused (null result) | Main experiment |

Run an experiment, then the same thing with `--control` (identical but selection is random) and compare:

```bash
.venv/Scripts/python experiments/prisoners_dilemma/run.py --generations 20 --run-name evolved_s0
.venv/Scripts/python experiments/prisoners_dilemma/run.py --generations 20 --run-name control_s0 --control
.venv/Scripts/python experiments/compare_runs.py experiments/prisoners_dilemma/runs/evolved_s0 \
    experiments/prisoners_dilemma/runs/control_s0 --metrics coop_rate --ylabel "Cooperation rate"
```

To reproduce a result, run its plan (e.g. the headline Batch D). It needs
the gradient-trained tit-for-tat genome first:

```bash
.venv/Scripts/python experiments/prisoners_dilemma/train_reference.py --strategy tft --batch 4 --steps 300 --lr 5e-4
.venv/Scripts/python experiments/run_queue.py experiments/plans/batch_d_no_decay.json
.venv/Scripts/python experiments/prisoners_dilemma/fingerprint.py --population runs/nodecay_tft4_s0
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
- **Invalid replies count as defection and score 0**, so broken genomes
  can't be exploited. Later batches draw random match lengths
  (`--rounds-max`), so there is no known last round.
- **Weight decay is optional and not neutral.** `--weight-decay` bounds
  mutation build-up by pulling children towards the base model (or, with
  `--decay-to seed`, a seed genome); that target biases the outcome (Batch C).

## Setup

```bash
pip install -r requirements.txt
```

Put `HF_TOKEN=...` in `.env` at the repo root if your model needs it.

## Future work

A possible follow-on, in a separate repository, would study these dynamics
in a spatial world with resources: [docs/NEXT_PROJECT.md](docs/NEXT_PROJECT.md).
It is a proposal only and not part of this project.
