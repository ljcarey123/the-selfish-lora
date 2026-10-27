# Results log

Findings in the order they were made. Each entry gives the setup, what
happened, and what it means. Run data lives in `experiments/*/runs/` (not in
git); plans in `experiments/plans/`.

Model: Qwen2.5-0.5B-Instruct (bf16), LoRA r=4 on q_proj/v_proj, only B
matrices evolved (98k parameters per genome). Hardware: RTX 3050 Ti laptop, 4 GB.

---

## 1. Mutation size calibration (2026-10-27)

`experiments/calibrate_mutation.py`: 8 random mutants of the base genome per
noise level, scored on 32 Countdown puzzles and the 21 prisoner's dilemma
histories of up to 2 rounds.

| std | CD score | CD valid | CD changed | PD valid | PD changed |
|---|---|---|---|---|---|
| 0.002 | 0.105 | 98% | 11% | 100% | 0% |
| 0.005 | 0.116 | 98% | 12% | 100% | 0% |
| 0.01 | 0.113 | 99% | 22% | 100% | 0% |
| 0.02 | 0.082 | 97% | 50% | 100% | 1% |
| 0.035 | 0.074 | 76% | 81% | 100% | 49% |
| 0.05 | 0.052 | 49% | 92% | 76% | 24% |
| 0.1 | 0.010 | 0% | 100% | 13% | 100% |

- The original default (0.05) broke half of all Countdown outputs.
- Countdown sweet spot: 0.01-0.02. Prisoner's dilemma needs ~0.03 before
  behaviour changes at all; the base model defects in every situation and
  that prior is hard to shift.
- New defaults: Countdown initial 0.015 / per-generation 0.01; prisoner's
  dilemma initial 0.03 / per-generation 0.02.

## 2. Mutants can express history-dependent strategies, but rarely the right one

Inspecting individual mutants (std 0.025-0.04) found strategies such as
"cooperate first, then always defect", "always cooperate", and strategies
that react to the opponent's last move. Every reactive mutant seen so far
was *anti*-tit-for-tat (defect after the opponent cooperates, cooperate after
they defect).

In 160 mutants of the base model, none agreed with tit-for-tat on 75% or
more of histories, with or without an extra prompt line stating the
opponent's last move. Mutants of the evolved always-cooperate champion
(section 3) likewise produced no tit-for-tat-like strategies in 48 samples.

**Meaning:** what evolution can find is bounded by what small perturbations
of the model's prior can express. Flipping a default (D to C) is easy;
reciprocity is a rare neighbour.

## 3. Prisoner's dilemma vs a bot panel: selection steers and purifies

Plan `first_results`, runs `panel_evolved_s0` / `panel_control_s0`.
Genomes play only fixed bots (tit-for-tat, grudger, always-defect), 6 rounds
per match, population 16, 20 generations. Theoretical payoffs per round
against this panel: always-defect 1.44, always-cooperate 2.00, tit-for-tat 2.28.

![PD panel seed 0](figures/pd_panel_s0.png)

- **Selection steers:** the evolved population went from 1% to 60-80%
  cooperation; its best genome is a clean always-cooperate (agrees with ALLC
  on 21/21 histories) scoring exactly the predicted 2.00.
- **Selection purifies:** in the control (random selection, same mutation),
  mutations accumulate unchecked. From generation 12 the population's outputs
  stop parsing (invalid rate 96-97% by generation 16-19) and mean payoff
  collapses to 0.08: mutational meltdown. The evolved population stays
  mostly functional (invalid 0-37%).
- **Premature convergence:** the evolved population sat at the ALLC ceiling
  (2.00) for 19 generations. The control briefly produced genomes scoring
  2.11 and 2.17, above that ceiling, which the converged population never did.
- **Mutation load grows:** the best genome's norm rose steadily (9 to 27) and
  the invalid rate drifted up, because nothing pulls genomes back.

## 4. Seed 1 evolved end-game defection

Run `panel_evolved_s1` (same settings as section 3, seed 1). The best genome
scored 2.28 from generation 9, equal to tit-for-tat's theoretical score, but
it is not tit-for-tat. Probing it one prompt at a time after n rounds of a
fixed history:

| history so far | move in rounds 1-10 |
|---|---|
| mutual cooperation | `CCCCCDDDDC` |
| I cooperate, opponent defects | `CCCCCDDDDD` |
| mutual defection | `CCCCCCDDDC` |

It cooperates unconditionally for five rounds, then defects from round 6,
which is the last round of every match it evolved in. The prompt says the
game lasts "an unknown number of rounds", so the model was never told the
horizon: selection found a round-counting strategy that exploits it. Against
the panel this earns (3.33 + 3.33 + 0.17) / 3 = 2.28, tying tit-for-tat by a
different route. Seed 0 stayed at plain always-cooperate (2.00).

**Meaning:** this is the end-game effect from game theory (defect when the
future can no longer punish you), reached by selection on weights rather
than by reasoning. It also shows that a fixed match length is an exploitable
cue; random match lengths would remove it.

**Caveat found along the way:** in bf16, batching prompts of different
lengths (left padding) can flip borderline greedy moves. Tournament batches
hold same-length prompts so runs are unaffected, but analysis tools now
generate one prompt at a time.

## Open questions and next ideas

- Random match lengths (e.g. geometric, mean 6) to remove the end-game cue;
  this should leave reciprocity as the only way past 2.00.

- Weight decay towards the base genome each generation, to bound mutation
  load (tests whether the invalid-rate drift goes away).
- More exploration once converged: larger population, fewer elites, or
  fitness sharing / novelty for behaviourally distinct genomes.
- Stronger pressure for reciprocity: a panel of tft + alld only makes
  tit-for-tat's advantage over always-cooperate 0.42 instead of 0.28.
