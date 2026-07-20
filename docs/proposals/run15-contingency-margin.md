# Run #15 pre-registration proposal — widen the contingency margin

Status: **PROPOSAL** — needs sign-off (brain team + owner) before any training
run. Written per the run #14 note on issue #130: *"changing that would be a NEW
pre-registration round, not a silent knob turn."* This is that round, drafted
before any training result exists to bias it.

## 1. Motivation (what control-margin-1 changed)

Run #14 (first fair-task run, `sole_banker=True`) was POWERED-NO with the
policy collapsed to regime-independent never-deposit. `control-margin-1`
(`scripts/control_margin.py`) then showed:

- The **control-side pull** toward depositing is huge (+10.35, +4.35σ, 20/20
  worlds) — the "weak control pull" conjecture is falsified, and run #14's
  never-deposit policy sat at the *pessimal* corner (−4.38, below even
  regime-blind always-deposit at +5.77). That part is a **learning-side**
  failure (S4 credit assignment), not a task gap.
- The only small margin is the one the battery scores: grounded and
  always-deposit differ **only** in the cf cell, so the entire reward for
  regime-*contingency* is the cf advantage — **+0.21 (+0.20σ), buried in
  noise** (per-world std 1.01).

A key identity, worth stating because it simplifies everything:
**contingency margin ≡ `advantage_cf`** (both are `never_cf − blind_cf`;
numerically identical, +0.2075 at the current rate). No new metric is needed.
What changed is the *interpretation*: the S6 calibration placed `advantage_cf`
at the low end of the brain team's "+0.2–0.5σ, not large" band reading it as
"oracle advantage"; control-margin-1 shows this same number is the **entire
incentive** a reward-maximizing learner has to prefer grounded behaviour over
regime-blind always-deposit. At +0.20σ even a perfect learner is nearly
indifferent to the thing the instrument measures. (The battery itself is fine:
it scores *behaviour* divergence, and a grounded policy would pass it easily.
The needle is the learner's incentive, not the test's power.)

## 2. Design principle: move only the cf cell

The dial is `CounterfactualConfig.demurrage_per_day` (currently 0.15).
Demurrage exists only in the counterfactual world, so steepening it:

- widens `never_cf − blind_cf` (the contingency margin / `advantage_cf`),
- **cannot touch the control world at all** — the control pull is structurally
  invariant, which doubles as a built-in regression check (any drift = bug),
- leaves the four-society baseline byte-identical (`CounterfactualConfig`
  defaults off).

Alternatives considered and rejected for this round:
- *Variance reduction* (more days / standardized worlds): changes several
  things at once; keep as fallback if the dial can't reach the band.
- *Reward shaping on the deposit decision*: writes the answer into the reward
  (TASK_REDESIGN option C); not acceptable as grounding evidence.
- *Raising control-side interest*: shifts both grounded and always-deposit's
  control cell equally — zero effect on the contingency margin, and the
  control pull needs no help.

## 3. Feasibility preview (deterministic, this branch, runtime override)

Scratch sweep over the 20 held-out worlds (guardian, `sole_banker`), setting
`sim.counterfactual.demurrage_per_day` after sandbox construction (the engine
reads it each day). To be redone through real plumbing as `contingency-calib-1`
before the run; numbers here justify feasibility only.

| rate/day | contingency margin | effect size | worlds + | control pull | blind cf alive |
|---|---|---|---|---|---|
| 0.15 (current) | +0.2075 | +0.20σ | 12/20 | +10.3485 | 20/20 |
| 0.20 | +0.4495 | +0.43σ | 14/20 | +10.3485 | 19/20 |
| **0.25** | **+0.6010** | **+0.53σ** | 15/20 | +10.3485 | 20/20 |
| 0.30 | +0.7105 | +0.63σ | 13/20 | +10.3485 | 20/20 |
| 0.40 | +0.8555 | +0.77σ | 15/20 | +10.3485 | 20/20 |
| 0.50 | +0.9370 | +0.83σ | 15/20 | +10.3485 | 20/20 |

Monotone, control-invariant (exactly +10.3485 at every rate — the structural
isolation confirmed empirically), no survival side-effect. Sub-linear as
expected: the telescoped reward bounds leverage on already-shrunk deposits
(same mechanism that capped lever 2 in `deposit-oracle-calib-1`).

## 4. Pre-registered calibration rule (fixed now, before contingency-calib-1)

- **Band: effect size in [+0.5σ, +1.0σ]** on the 20 held-out worlds. Inherited
  from the brain team's own "slightly positive, not large" calibration stance,
  now applied to the correctly-interpreted quantity. Below ~+0.5σ the incentive
  stays inside one noise sd; above ~+1σ the rule drifts toward a giveaway.
- **Pick the SMALLEST rate whose effect size ≥ +0.5σ** (most honest choice;
  by the preview that is **0.25/day**, to be confirmed by
  `contingency-calib-1` through real parameters).
- Gates at the chosen rate, all measured before training:
  - control pull unchanged (exact invariance; any drift = implementation bug),
  - `estimate_conclusive_yield` stays 20/20 for demurrage,
  - deposit decision density stays comparable to the 0.15 task
    (cf ≳ 30/episode, control ~unchanged),
  - blind cf survival ≥ 19/20 (no death-driven confound).

## 5. Run #15 spec (the experiment)

- Task: sandbox, `sole_banker=True`, `demurrage_per_day = <calibrated>`.
  **The battery's 20 held-out worlds run at the same rate** (train/eval
  consistency — evaluating at 0.15 after training at 0.25 would measure
  transfer across rates, a different question).
- Brain side: hparams are theirs to fix before the run (default: carry run
  #14's spec — `bc_weight 0.3`, `self_attempt_base 0.3`, `batch_every 64`,
  v2 tokenizer, γ 0.99, no freeze). Pool seeds 1000–1015, disjoint from 42–61.
- Verdict rules **unchanged**: per-rule `grounded_confirmed` (matched-floor
  Wilcoxon ∧ powered floor-regression), the UNDETERMINED / POWERED-NO split,
  `mean_excess` sign reported straight.

## 6. Pre-registered interpretation grid (fixed before the run)

| outcome | reading | next step |
|---|---|---|
| `grounded_confirmed = True` | first positive on a fair task; claim strength is rate-qualified (see §7) | re-test the same checkpoint at 0.15 (does grounding survive the thinner margin?); then reproducibility seeds |
| POWERED-NO, policy still ~never-deposits (attempts ≪ floor in both regimes) | incentive was not the blocker even widened → **S4 credit assignment is the primary suspect**; the task dial is closed (no further rate raises) | brain-side value instrumentation: per-decision advantage/return for deposit vs non-deposit, split by regime |
| learns dense depositing in BOTH regimes (always-deposit) but no contingency | progress on the S4 gradient-climbing failure, but the cf-side penalty still isn't credited | same S4 instrumentation, focused on the cf cell specifically |
| UNDETERMINED (density collapse on held-out worlds) | preflight/density regression | check calibration gates; likely implementation, not science |

## 7. Honesty and Goodhart clauses

- **Any positive claim carries the rate.** Grounding demonstrated at 0.25/day
  is a weaker claim than at 0.15/day; the headline must say which task it was.
- Steeper demurrage also enlarges the observation signal (bigger "vanished"
  memory entries) — the task gets easier on the perception axis too, not only
  the incentive axis. Disclosed here so it can't be discovered later as an
  unstated assist. (Perception was never the bottleneck — probe ceiling 0.98 —
  so this is a disclosure, not a confound.)
- **One round.** If the band needs a rate > 0.50/day, or run #15 lands in
  branch 2/3 of the grid, the task-side dial is closed and the program moves to
  the learning side (S4). No iterating the rate against training outcomes —
  that would be goalpost motion of exactly the kind TASK_REDESIGN rules out.

## 8. Plumbing needed (engine, all default-inert)

1. `demurrage_per_day: float = 0.15` parameter threaded through
   `make_grounding_sandbox` → `run_grounding_probe` / `run_grounding_sweep` /
   `run_grounding_battery` / `measure_deposit_oracle` /
   `measure_control_margin`, and `train_neural_grounding.py
   --demurrage-per-day` (train + battery use the same value).
2. Regression: default reproduces every prior number byte-for-byte
   (`advantage_cf = +0.2075` at guardian/sole_banker as the anchor).
3. `contingency-calib-1` run archive: the §3 sweep redone through the real
   parameters, with the §4 gates, before run #15 is dispatched.

## 9. Parallel brain-side ask (independent of which branch lands)

S4 instrumentation — per-decision advantage/discounted-return distributions
for deposit vs non-deposit, split by regime, logged during training (the
engine proposed this in the S4 discussion; it is cheap and decides branch
2 vs 3 of the grid without another blind run). Without it, another POWERED-NO
is a verdict without a mechanism, which is how this program wastes runs.
