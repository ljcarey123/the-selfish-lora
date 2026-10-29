# Follow-on project plan: a spatial evolution lab for LLM agents

> **This is a separate, future project, not part of this repository.**
> The Selfish LoRA is a small, self-contained research project. This document
> proposes a new repository that would build on its code and findings. Nothing
> described here is implemented yet.

Working name: **Selfish World**. Status: proposal (October 2026).

## Why a follow-on

The Selfish LoRA asked whether a genetic algorithm over LoRA adapters can
change how a small LLM behaves in a social game. Its findings
([RESULTS.md](RESULTS.md)) point to a richer next step:

1. **Selection on weights reliably shifts default behaviour** (always-defect
   to cooperation) and keeps a population functional. Without it, mutations
   accumulate until outputs break.
2. **What evolves depends on the social environment.** The same model and
   method evolve cooperation against reciprocating opponents and defection
   among their own kind.
3. **Random weight noise finds tricks, not skills.** Evolution repeatedly
   found strategies keyed to the round number, never to the opponent's moves.
4. **The missing strategies are easy to represent.** Tit-for-tat, trained by
   gradient descent into the same LoRA, sits much closer to the base model
   than anything evolution found. The limit is search, not representation.

Together these suggest a different kind of study. Use gradient training to
create a few meaningful starting strategies ("species"). Place populations of
them in a spatial world with resources. Then study what selection, mixing and
mutation do from there, with tools that make the dynamics visible and the
experiments reproducible.

## Goal

A research tool for studying evolutionary and social dynamics in populations
of LLM agents whose behaviour is inherited through their weights. Aimed at
researchers in multi-agent systems, evolutionary computation and LLM
behaviour; also a portfolio piece showing those interests.

It should let someone:

- define a world (grid, resources, encounter rules, reproduction rules)
- seed it with populations of trained or evolved genomes
- run controlled, reproducible experiments with seeds and baselines
- inspect what happened: where agents went, who they met, what they chose
  and why (the actual prompts and replies), and how strategies spread through
  lineages

### Non-goals

- **Not a game.** Interaction is in service of experiments. There are no
  scores to beat or progression.
- **Not real time.** Simulation runs headless; the interface replays and
  inspects recorded runs.
- **Not large-scale training.** It should still run on a 4 GB laptop GPU,
  with hosted inference as an option rather than a requirement.

## Research questions

1. **Spatial structure and cooperation.** In well-mixed populations,
   defectors win. A classic result (Nowak & May, 1992) is that spatial
   structure lets cooperators survive in clusters. Does that hold when
   strategies are LLM weights rather than coded rules?
2. **Invasion and stability.** Which species can invade which, at what
   initial fraction, and do mixed populations settle, cycle or collapse?
3. **Recombination of skills.** Does crossover between meaningful genomes
   (e.g. a trained forager and a trained reciprocator) produce offspring that
   combine traits, where random noise did not?
4. **Scarcity and conflict.** How do resource abundance and regrowth change
   which strategies dominate (sharing vs raiding)?
5. **Interpretability of evolved behaviour.** Do evolved agents' strategies,
   read from their responses across situations, match what their payoffs
   suggest they should be?

## The world

| Element | Design |
|---|---|
| Space | Grid; agents see a small neighbourhood |
| Resources | Cells hold resources that regrow; holding or moving costs energy |
| Encounters | Neighbouring agents choose to share, trade or take: a prisoner's dilemma embedded in space, the mechanic validated in The Selfish LoRA |
| Life cycle | Energy determines survival and reproduction; offspring are placed near the parent (local reproduction is what lets clusters form) |
| Inheritance | Offspring genomes come from crossover between nearby parents plus mutation, with weight decay to bound mutation build-up |
| Species | Starting genomes trained by gradient descent on hand-written strategies, as `train_reference.py` does for tit-for-tat |
| Interventions | Researchers can change scarcity, introduce a species, or change rules partway through a run, recorded as part of the run's configuration |

Observations and actions are kept small and discrete (a few symbols for the
neighbourhood and a handful of actions). This keeps prompts short and makes
the action cache below effective.

## Making it run on a laptop

- **Action cache.** With greedy decoding, an agent's action is a function of
  (genome, observation). Cache it, so repeated situations cost nothing.
  Discrete observations make hits common.
- **Batching by genome.** Load each genome once per tick and answer all its
  agents' observations in one batch, as The Selfish LoRA's tournament does.
- **Headless runs, then replay.** The engine writes an event log; the
  interface reads it. Long runs never block the interface.
- **Pluggable inference.** Local model by default; a hosted endpoint as an
  option for larger populations.

## Experiment discipline (carried over)

- Every experiment is a plan file: seeds, settings, and a control condition
  (random selection or no spatial structure).
- Checkpoints every generation, so runs resume after interruption with
  identical results.
- A results log with figures, including null results.
- Strategy fingerprints: an agent's response across a fixed set of
  situations, compared with known strategies.

## Interface for inspecting runs

A web front end that replays recorded runs:

- **World view:** the grid over time, coloured by species or strategy, with
  resources shown
- **Agent inspector:** select an agent to see its lineage, energy history,
  encounters, and the prompts and replies behind each decision
- **Population view:** strategy frequencies, cooperation rate and diversity
  over generations
- **Run comparison:** overlay metrics from several runs (e.g. treatment vs
  control, different seeds)

## Architecture

```
selfish-world/                     new repository
  engine/      headless, deterministic simulation; writes event logs
  species/     training scripts and trained starting genomes
  experiments/ plans, analysis, results log
  ui/          web replay viewer (e.g. React + canvas/PixiJS)

depends on:
  selfish_lora                     packaged from The Selfish LoRA (pip-installable):
                                   model loading, genomes, evolution, batched
                                   generation, checkpoints, run logging
```

## Phases

| Phase | Work | Done when |
|---|---|---|
| 0 | Finish The Selfish LoRA: complete the invasion experiments and write up the results | Results log and README describe all findings |
| 1 | Package `selfish_lora`; add the action cache | Installable package; cached and uncached runs give identical results |
| 2 | Headless spatial prototype with two trained species and a well-mixed control | A clear yes or no on whether spatial structure changes outcomes |
| 3 | Replay viewer: world view and agent inspector | A recorded run can be explored end to end |
| 4 | More species, interventions, population and comparison views | A researcher can run and inspect a new experiment without code changes |
| 5 | Optional: hosted inference, larger worlds, a public demo of recorded runs | As needed |

Phase 2 is the gate: if spatial structure does not produce clear, measurable
dynamics with LLM agents, rethink the world before building the interface.

## Risks and open questions

- **Speed.** Even with caching, large populations may be too slow locally.
  Mitigation: small worlds first; hosted inference later.
- **Too much hand design.** If trained species do all the interesting work,
  evolution adds little. Phase 2 must compare against a no-evolution baseline.
- **Prompt sensitivity.** Small changes to observation wording can change
  behaviour (seen in The Selfish LoRA). Fix formats early and treat changes as
  new experiments.
- **Model choice.** Qwen2.5-0.5B worked here. A larger model may give richer
  behaviour but costs speed. Keep the model a configuration option.
