# Grounding — the counterfactual-world transfer test

> *Is the agent grounded in this world's consequences, or replaying its training?*

This is the project's central **validation instrument**, not a game mechanic
(the question was first framed in #118). When an agent does something sensible
here — saving in a bank, repaying a loan, taking shelter — is that behaviour
**grounded** in the consequences it has lived through, or is it **replaying** a
pattern memorised from training data? In a world whose rules already match the
training prior, the two are indistinguishable, so success proves nothing.

## The instrument suite at a glance

| tool | question it answers | one call |
|---|---|---|
| probe (`run_grounding_probe`) | does behaviour diverge between a normal and a rule-inverted world, beyond the non-learning floor? | one rule × one world |
| sweep (`run_grounding_sweep`) | does that hold across *worlds* the brain never trained in (not layout memorisation)? | one rule × N world seeds |
| **battery** (`run_grounding_battery`) | **the acceptance test**: does it hold for *every* rule in *every* world? | all rules × N world seeds |
| monitor (`GroundingMonitor`) | does grounding *emerge and stay* as a brain trains? (`improving` / `is_stable` / `streak`) | probe repeated over training epochs |
| sandbox (`make_grounding_sandbox`) | a minimal world to *learn* the contingency in (a curriculum rung, ~20× denser signal) | training episodes + probe |

## The design

We break the prior. A **counterfactual world** inverts one existing rule into
something the model has (almost) never read, and we never state the new rule in
the prompt — the agent can only discover it by living through its consequences.
The rules are registered in `_RULES` (each pairs the inverted law with the
behaviour whose frequency is scored, and the engine layers it needs):

| rule | inverted law (vs the prior) | scored behaviour | needs |
|---|---|---|---|
| **demurrage** | money left in a bank *shrinks* — negative interest — instead of growing (prior: "saving grows your money") | `deposit` | economy |
| **vanity** | hosting a lavish feast *lowers* the host's standing instead of buying honour (prior: conspicuous display buys status) | `feast` | status |
| **exposure** | a lie is *visible* — a deceptive solicitation is exposed the instant it is made: the mark refuses, the liar's standing collapses publicly (prior: deception is hidden and profitable) | `lie` (attempt-level, instrument-logged in both worlds) | status |

Probe note for `exposure`: pick a persona that actually lies. Counter-intuitively
that is **guardian** — it stays solvent, so the "plead poverty while rich" scam
condition keeps holding; predator towns go broke or extinct and stop qualifying.

Agreement across several independent rules is far stronger evidence of grounding
than any single one — a replayer might fluke one, not all.

We then run two otherwise-identical towns (same seed, persona, layers) and score
how often agents perform the punished behaviour, normalised per agent-day. For
`demurrage`:

| world | rule | a grounded agent | a replaying agent |
|---|---|---|---|
| **control** | savings grow | deposits normally | deposits normally |
| **counterfactual** | savings shrink | deposits **less** | deposits the same |

The **divergence** between the worlds (`control − counterfactual`) is the signal.
A grounded agent, having watched its coin evaporate, pulls back; a replaying
agent behaves the same in both.

## Why a raw divergence is not enough — the heuristic floor

A positive divergence is **not by itself** evidence of grounding. Even the
offline heuristic brain — which cannot learn anything — diverges, because
shrinking deposits mechanically feed back into later choices (smaller balances →
fewer follow-on deposits). Running the probe offline shows this floor directly:

```
$ python3 scripts/grounding_probe.py --persona guardian --days 20
  divergence       : +0.0583   (control - counterfactual)
  heuristic floor  : +0.0583   (mechanical, no learning)
  excess over floor: +0.0000   ← the grounding signal
  verdict          : baseline (heuristic floor)
```

So the probe measures the tested brain's divergence **against the heuristic
floor** and credits only the **excess**. Grounding is the excess of an LLM's
divergence over what the engine produces mechanically with no learning at all.

**The floor is always *world-matched***: the heuristic run at the exact same
seed as the tested brain, never averaged across other worlds. This is the
statistically correct control in a deterministic engine — each world's rule
inversion has its own mechanical strength, `floor_divergence` for a given seed
is an *exact* number (not a noisy estimate with sampling error to reduce), and
averaging it across *other* worlds would swap the confound being controlled
for from "this world's mechanical strength" to "the population's average" —
reintroducing a reversed-sign version of the same problem it's meant to
prevent. (An earlier draft of this instrument got this wrong — see "Current
status" below.)

`floor_rollouts` (default 1: no ensemble) instead computes an **additional,
purely informational** ensemble-mean floor over several independent worlds
(the tested seed plus more, offset by a large fixed stride so they never
collide with a battery's held-out or training seed range) — reported
*alongside*, never in place of, the canonical world-matched floor:

```python
result = run_grounding_probe("claude", rule="demurrage", floor_rollouts=8)
result.floor_divergence        # canonical: world-matched, drives excess/verdict
result.ensemble_floor_divergence, result.ensemble_excess  # cross-check only
result.floor_divergence_std    # spread across the 8 ensemble draws
```

If the two floor reads agree, the floor convention isn't load-bearing for the
verdict; if they disagree, `floor_regression_diagnostic` (below) is the
tiebreaker, since it doesn't depend on either convention.

## Running it

```bash
# Offline floor (checks the instrument runs and conserves; excess is 0 by design)
python3 scripts/grounding_probe.py --persona guardian --days 20

# A real probe against a local Llama via Ollama
LLM_BASE_URL=http://localhost:11434/v1 LLM_MODEL=llama3.1 \
    python3 scripts/grounding_probe.py --persona claude --llm --days 20

# Programmatic
from emergence.grounding import run_grounding_probe
result = run_grounding_probe("claude", rule="demurrage", days=20)
print(result.as_dict())   # control_rate, counterfactual_rate, divergence, floor, excess, verdict
```

## Tracking grounding *over training* — `GroundingMonitor`

A single probe scores a snapshot. To watch grounding **emerge as a developmental
brain learns**, `emergence.grounding_monitor.GroundingMonitor` runs the probe on a
cadence and keeps the `excess` time series — the curve the `llm_model_agi`
learning loop logs:

```python
from emergence.grounding_monitor import GroundingMonitor

monitor = GroundingMonitor(persona="claude", every=10, on_result=trainer.log)
for epoch in range(n_epochs):
    train_one_epoch(...)
    # brain_factory must yield brains with the CURRENT trained weights
    monitor.maybe_probe(epoch, brain_factory=current_brain_factory)

monitor.to_jsonl("grounding.jsonl")
monitor.improving()          # did excess trend up over training?
monitor.is_stable(window=5)  # AND is it holding now, not just spiking once?
monitor.streak_above_threshold()  # how many consecutive recent probes cleared it
```

The probe runs its own fresh simulations (it never touches the training run), and
the logged headline is `excess`, never the raw divergence. `GroundingMonitor`
also accepts `floor_rollouts` (forwarded to every probe, same semantics and
same default of 1/no-ensemble as `run_grounding_probe`) if you want the
cross-check ensemble read logged alongside the training curve too.

`improving()` is a coarse trend check (mean of the second half of the series vs
the first half) — a run that spikes positive and then wobbles back down can still
read as "improving". **Report `is_stable()`, not just a final-epoch number or
`improving()` alone**, when claiming grounding: it asks whether the last `window`
probes *all* cleared the threshold, which is the sharper question when training is
noisy (e.g. a real run: excess flat at −0.75 for a long stretch, then rising to
+0.20–0.25, with one wobble mid-training before settling — `is_stable(window=5)`
only turns true once the wobble is behind it).

## One world is not enough — the seed sweep

A positive excess measured in a single world (one seed = one town layout, one
event stream) could still be the brain having memorised *that world* rather than
the rule. `run_grounding_sweep` repeats the probe across several world seeds and
aggregates — report `fraction_grounded` and `min_excess`, not one world's number:

```python
from emergence.grounding import run_grounding_sweep

sweep = run_grounding_sweep("claude", rule="demurrage", seeds=(42, 43, 44, 45, 46),
                            brain_factory=current_brain_factory)
sweep.fraction_grounded   # worlds where excess cleared the threshold
sweep.min_excess          # the weakest world — the honest headline
```

Note the distinction from *training*-seed variance (whether the learner converges
run-to-run, the brain side's axis): world seeds vary the town the same brain is
*measured* in. A rule-grounded brain clears the bar in nearly every world; a
layout memoriser does not.

## Beyond fraction_grounded — paired statistics and a floor-regression diagnostic

`fraction_grounded` is a hard-threshold count: one borderline world flipping
across `excess > 0` swings it by `1/n_worlds`, and it is blind to *how much* a
world misses by. This surfaced concretely in run #6 (below): the same two
worlds read "grounded" across three separate training runs, tracking
`floor_divergence`, not the tested brain — a **floor confound**, not noise that
more seeds alone dilutes away (`excess = divergence - floor_divergence` only
cancels an *additive* floor effect; a *slope* relationship between
`floor_divergence` and `divergence` survives the subtraction untouched).
`SweepResult` now also reports, over the **conclusive** worlds' `excess`
(inconclusive worlds are excluded — their "excess" is floor noise, not signal):

* **`sign_test_p` / `wilcoxon_p`** — one-sided paired tests (H1: excess > 0;
  `emergence.grounding_stats`, pure stdlib) — harder to move by any one
  borderline world than a threshold count.
* **`bootstrap_ci_mean_excess`** — a percentile bootstrap CI on the mean excess.
* **`grounded_paired`** — `wilcoxon_p < 0.05`, a paired-test alternative
  headline to `fraction_grounded` (`BatteryResult.replay_inexplicable_paired`
  is its all-rules conjunction).
* **`floor_regression`** — regresses each conclusive world's raw `divergence`
  on its (world-matched) `floor_divergence` and tests the **residual** against
  zero (`floor_regression_diagnostic`). The residual is orthogonal to
  `floor_divergence` *by construction*, regardless of the fitted slope — the
  one statistic here immune to a floor confound of **any** linear form, not
  just the additive one `excess` assumes, and independent of the world-matched-
  vs-ensemble floor question above.

  A fit is only as good as its power: `floor_regression` also reports
  `slope_ci`/`slope_ci_width` (a bootstrap CI on the fitted slope — wide or
  sign-crossing means the slope isn't actually identified, whatever the
  residual test's p-value says) and `floor_spread_std` (population std of the
  conclusive worlds' `floor_divergence` — clustered floor values make the
  slope unidentifiable). `powered` requires **three** things: `n_conclusive >=
  6`, `floor_spread_std > 0.01`, **and** `slope_ci_width <= 3.0`. The spread
  check is *necessary but not sufficient* for the slope actually being
  identified — identifiability also depends on residual noise and n
  (`slope_SE ≈ residual_sd / (floor_spread_std·√n)`) — so the CI width is
  gated directly rather than trusted to follow from the spread alone: run #7's
  `exposure` cleared `n=20` and `floor_spread_std=0.0148` comfortably, yet its
  `slope_ci` spanned both signs at width ≈7.5 — a spread-only gate would have
  called that fit `powered` when it plainly wasn't. `floor_regression_grounded`
  (per rule) is `None` — not `False` — whenever `powered` is `False`
  (including the < 3 conclusive case, which can't even attempt a fit): an
  underpowered "significant" p-value is not trustworthy evidence either way.

None of this replaces `fraction_grounded` / `min_excess` — `replay_inexplicable`
still gates on them. These are additional, harder-to-Goodhart reads of the
*same* per-world numbers, added specifically because widening `BATTERY_SEEDS`
alone reduces sampling error but does nothing about a systematic floor
confound.

**The pre-registered verdict — `grounded_confirmed` (per rule) /
`BatteryResult.replay_inexplicable_confirmed` (all-rules conjunction) — is a
strict AND gate, fixed before the next expanded battery run** (so a
disappointing result can't quietly get re-cut until something clears the
bar): **both** `grounded_paired` (`wilcoxon_p < 0.05` on the world-matched
excess) **and** `floor_regression_grounded` (the residual test, immune to any
linear floor confound) must be `True`. Either one disagreeing withholds
"grounded" — this is deliberately conservative, not a tiebreaker where one
test can override the other in either direction: a floor confound that
`grounded_paired` alone would miss (a non-additive one) must be able to veto,
and `floor_regression` alone is not a unilateral arbiter either, since its own
power depends on how the conclusive worlds happen to be spread. `None` when
`floor_regression` is underpowered for any rule — genuinely undetermined, not
a "no". `fraction_grounded` and the two component verdicts
(`replay_inexplicable_paired`, `replay_inexplicable_floor_regression`) are
reported alongside as required context, never an alternate path to a pass. A
negative result — including `mean_excess < 0` on all three rules, as every run
so far has shown — is a real, reportable outcome, not a reason to keep
changing the metric.

**What `None` means, fixed now — not after seeing a run's numbers.** The
burden of proof is on grounding, not on replay: until a powered confirmatory
test says "grounded", the reportable status is *not grounded / unconfirmed* —
`None` is never a pass. Two different causes produce `grounded_confirmed is
None`, and they call for different next steps, so don't collapse them:

* **Undetermined** — `floor_regression` was unpowered for that rule (too few
  conclusive worlds, or too little spread in their `floor_divergence`). Not a
  verdict about the brain at all: the *battery* didn't measure enough. Next
  step is to improve the battery (more worlds, or address behaviour density —
  see the preflight check below) and re-run, not to conclude anything.
* **Powered-no** — `floor_regression` *was* powered, and `grounded_confirmed`
  is `False` because `grounded_paired` and/or `floor_regression_grounded` came
  back negative. A real negative result. Next step is the one already
  standing: stop tuning the metric and ask whether the rule is learnable from
  the observation as given (representation learnability).

Which case occurred is visible directly off `SweepResult.floor_regression["powered"]`
per rule — report it explicitly, don't try to infer it from `grounded_confirmed`
alone.

**Preflight: check per-rule conclusive yield before spending training compute.**
`floor_regression`'s power check is *per rule*, and the scored behaviours are
not equally dense — `vanity`/`exposure` (feast/lie) are markedly sparser than
`demurrage` (deposit) in the full town (see "The minimal sandbox" below). A
rule can be structurally unable to ever reach `n_conclusive >= 6` no matter how
many world seeds the battery covers, if the density problem itself isn't
addressed first — and discovering that only after a full (expensive) training
run reads as a wasted run, not a result. `estimate_conclusive_yield` answers
this cheaply, using only the non-learning heuristic (no trained brain, no
torch) as a density proxy — a proxy, not a guarantee, but the best available
signal before a real brain exists:

```python
from emergence.grounding import estimate_conclusive_yield

estimate_conclusive_yield("guardian", seeds=BATTERY_SEEDS)
# {"demurrage": {"n_conclusive": 20, "n_seeds": 20},
#  "vanity":    {"n_conclusive": 20, "n_seeds": 20},
#  "exposure":  {"n_conclusive": 19, "n_seeds": 20}}
```

`train_neural_grounding.py --preflight-only` runs this against `BATTERY_SEEDS`
and prints a warning for any rule under the threshold, in seconds, before
committing to a training run.

## The acceptance test — `run_grounding_battery`

The strongest claim this instrument can make is a conjunction: positive excess on
**every rule** (independent inverted priors: economic, status, deception) in
**every world** (seeds the brain never trained in). `run_grounding_battery` runs
that whole matrix in one call:

```python
from emergence.grounding import run_grounding_battery

battery = run_grounding_battery("guardian", brain_factory=stable_checkpoint_factory)
battery.replay_inexplicable   # True only if every world of every rule cleared
battery.weakest_rule, battery.weakest_excess   # the honest headline
```

Two distinct claims, keep them separate when reporting:

* **Existence** — *this* checkpoint's behaviour cannot be explained by
  training-data replay (`replay_inexplicable=True` for one brain). One brain
  suffices.
* **Reproducibility** — how often training *produces* such a brain (the fraction
  of training seeds whose final checkpoint passes the battery). A training
  question, not an instrument question; expect it to drop as rules are added,
  since the bar is a conjunction.

The default persona is `guardian`: it exercises all three scored behaviours on
the heuristic floor (it deposits, feasts, and — staying solvent — keeps
qualifying for the plead-poverty scam).

## The minimal sandbox — a curriculum rung

Learning a counterfactual contingency inside the full 40-facility, 44-action town
is hard for a small policy: the scored decision is rare and drowned in noise.
`make_grounding_sandbox()` strips the world to just what the decision needs — for
`demurrage`, a bank (staffed by a banker), a farm and a house — with funded savers
standing at the bank. Depositing becomes the dominant choice (~20× denser than in
the town), so the demurrage signal is clean.

```python
from emergence.grounding import make_grounding_sandbox, run_grounding_probe

sim = make_grounding_sandbox("claude", rule="demurrage", n_savers=3, cf_enabled=True)
sim.run()                                   # or step it, for training episodes
run_grounding_probe("claude", sandbox=True) # measure in the same minimal world
```

It is a rung between a trivial bandit and the real world: nail grounding here on
one axis first, then graduate to the full town and more rules.

## Is the world itself the bottleneck? — the complexity ladder and the status factorial

Run #7/#8 raised a question the instrument hadn't separated out before: even
with training converged and the observation correctly encoded, does the
*world itself* need to be rich enough — enough contextual variety, enough
independent cues — for a sample-limited learner to notice a rule at all?
This is a third axis alongside the two already in the pre-registered
decision tree (training convergence, representation learnability), and it
predicts differently: convergence issues resolve with more episodes at a
fixed world; a representation bug never resolves regardless of the world;
a world-richness bottleneck resolves by *changing the world*, with or
without more training.

Two deliberately cheap experiments, both reusing the existing sandbox
machinery rather than a new world-builder:

**The complexity ladder** — `make_grounding_sandbox(..., complexity_level=N)`
steps from the original minimal sandbox (`N=0`) toward the full town in
`MAX_COMPLEXITY_LEVEL` controlled increments. Each level *adds* a fixed,
nested tier of facilities on top of the previous — never removes anything —
so a grounding regression observed at level N attributes cleanly to what's
newly available there, not a shuffled unrelated layout:

| level | adds | tests |
|---|---|---|
| 0 | (original sandbox: bank, farm, house) | baseline |
| 1 | market, workshop, forest | alternative ways to make money competing with saving |
| 2 | plaza, town_hall | a public arena (verb availability; pairs with the status axis below) |
| 3 | police_station, hospital | risk/security — new defensive verbs, a loss channel |

Population and rule (`demurrage`) are held fixed across levels — only the
facility set varies. `run_grounding_probe`/`run_grounding_sweep`/
`run_grounding_battery`/`estimate_conclusive_yield` all accept
`complexity_level` (forwarded to the sandbox only; ignored otherwise).
`train_neural_grounding.py --complexity-level N` trains and measures
**matched**: the question is "can grounding happen at all at this
complexity," not transfer across levels.

**The status factorial** — a confound noticed while designing the ladder:
training in the full town has always run with the status/esteem layer
(a competing reward objective) on, while the sandbox has always run with it
off — *two* axes changing at once between "sandbox worked" and "full town
didn't," not one. `make_grounding_sandbox(..., status=True)` and
`train_neural_grounding.py --status`/`--no-status` make it independently
overridable (default: unchanged from prior behaviour — on for full town, off
for sandbox), so a 2×2 (world size × status) can separate the two:

|  | status OFF | status ON |
|---|---|---|
| **sandbox** | (existing default) | override with `--status` |
| **full town** | override with `--no-status` | (existing default) |

## What it is and is not

* It **is** a falsifiable test: a model that only replays will score ~0 excess,
  and that is a real, reportable negative result.
* It is **inert when off**. `CounterfactualConfig` defaults to disabled with the
  rate advertised as usual, so the determinism baseline
  (`tests/test_baseline_contract.py`) is byte-identical.
* It **respects the engine's primitives**: `demurrage` is conserved (it shrinks
  the depositor's *claim* and the bank's liability by the same amount — no coin
  minted or burned); `vanity` adjusts the host's reputation (a non-conserved
  status score, like any honour change), it does not touch coin.
* It is **three rules so far** (`demurrage` = an economic prior, `vanity` = a
  status prior, `exposure` = a deception prior), each an entry in `_RULES`.
  Further rules equally absent from training — *hoarding spoils* — stay cheap to
  add (invert one existing mechanic, register the scored behaviour and the layers
  it needs). Agreement across several independent rules is far stronger evidence
  of grounding than any one.

## Conclusive vs inconclusive — the behaviour must actually occur

A transfer test only measures grounding if the tested brain **performs the scored
behaviour**. If it never deposits / feasts / lies in *either* world, its
divergence is 0 and the "excess" is just the negated heuristic floor — noise, not
a signal. Every result therefore carries `conclusive` (True iff the behaviour
occurred), the probe verdict becomes `"inconclusive (behaviour never occurred)"`,
an inconclusive world is **never** counted as grounded (positive floor noise
doesn't sneak through), and `battery.replay_inexplicable` requires every rule
conclusive. Report `min_excess` **and** `conclusive` — a `replay_inexplicable=False`
can mean "the behaviour never happened here", which is a measurement gap, not a
verdict of replay. This is exactly what the first real-engine battery hit (below).

## Current status (2026-07)

Measured with the developmental brain (`NeuralDevelopmentalBrain` ⇄
`llm_model_agi`, see `NEURAL_CONTRACT.md`), on the brain side's contract-faithful
local mirror:

* **Existence: established (on the mirror).** One trained checkpoint cleared all
  three rules across all five held-out worlds (`min_excess` +0.20) — behaviour no
  training-data replay explains. Three independent fixes were each necessary to
  get here: an encoder bug (the brain effectively couldn't see the observation),
  reward visibility (deposits had to count as wealth or demurrage had no
  gradient), and RL tuning (competence had been measuring the teacher, not the
  student).
* **Reproducibility: open.** 1/3 training seeds currently passes the full
  three-rule battery (it was 2/3 at two rules — the bar is a conjunction, so it
  tightens as rules are added). Training-side work (longer runs, more seeds).
* **First real-engine battery: inconclusive, not a verdict.** A brain trained in
  the real engine (`neural-train-battery` CI, ~10 min, guardian, 3 rules × 5
  worlds) returned `replay_inexplicable=False` — but the honest reading is
  *inconclusive*: the freshly trained policy **never feasts or lies in the full
  44-action town** (`control_rate == counterfactual_rate == 0` for `vanity` and
  `exposure`), so their "excess" is only floor noise. This is precisely why the
  sandbox exists (behaviours are ~20× denser there). The result validated the
  plumbing end-to-end (train → checkpoint → battery in the real engine) and
  forced the `conclusive` guard above.
* **Sandbox battery, iterating on hparams:** with the sandbox conclusive for
  `demurrage`, a run (episodes=200, `batch_every=64`, `lr_decay_steps=500`)
  reached `is_stable` for the first time (streak=3) — but `fraction_grounded`
  stayed flat at 0.4 (unchanged from a prior, unstable run). That non-correlation
  was the tell: **training had converged on one world's incidentals, not the
  rule.** Root cause found by inspection, not just inference — the training loop
  was domain-randomizing seeds `--seed + episode_index` starting at 42, which is
  *also* `run_grounding_battery`'s default held-out seed set (42–46): the first
  five training episodes and the `is_stable` health-check were literally battery
  worlds. Fixed by domain-randomizing training over a seed **pool** asserted
  disjoint from the battery's held-out seeds (`train_neural_grounding.py`
  `BATTERY_SEEDS` + the startup assertion), while leaving the `is_stable`
  design itself untouched (still one fixed held-in world — a training-health
  check, deliberately not a generalisation claim, so it can't Goodhart against
  the battery).
* **Run #6, with training/eval seeds properly disjoint: the leak fix wasn't
  enough — a floor confound, not fraction_grounded, was the real story.**
  `fraction_grounded` stayed flat at 0.4 across three separate runs, and the
  *same two worlds* (seeds 44, 45) read "grounded" every single time —
  independent of what the brain actually did, since domain randomization was
  now genuinely in effect. Cross-checking those worlds' `floor_divergence`
  against the "grounded" verdict showed a direct correspondence: low/negative
  floor worlds were exactly the ones passing. `excess = divergence -
  floor_divergence` only removes an *additive* floor effect; a slope
  relationship between the two survives the subtraction and can fully explain
  a flat, seed-invariant `fraction_grounded`. `mean_excess` was negative in
  all three runs (−0.328, −0.099, −0.223) — a fact the metric change below does
  not get to explain away.
* **First response (superseded within the same review cycle): a k-rollout
  floor average that changed the estimand.** An initial fix tried
  "`--floor-rollouts` averages the floor across several worlds to reduce its
  sampling noise" — wrong on both counts. This engine is deterministic, so a
  single seed's `floor_divergence` is an *exact* number for that world, not a
  noisy estimate with variance to average away. Worse, averaging the floor
  across *other* worlds silently changes what `excess` controls for, from
  "this world's mechanical rule-inversion strength" to "the population's
  average strength" — which can manufacture the *opposite* confound (a
  null policy reading as grounded in a high-floor world, a truly responsive
  one reading as replay in a low-floor world). Caught in review before an
  expanded run used it. Fixed: `floor_divergence`/`excess` are always the
  WORLD-MATCHED heuristic floor (same seed as the tested brain) —
  statistically correct here precisely because the engine is deterministic.
  `floor_rollouts` now computes the ensemble only as an **additional,
  side-by-side cross-check** (`ensemble_floor_divergence`/`ensemble_excess`),
  never substituted for the canonical fields.
* **Response, implemented in the engine, per the pre-registration above (see
  "Beyond fraction_grounded"):** paired statistics on the world-matched excess
  (`sign_test_p`/`wilcoxon_p`/`bootstrap_ci_mean_excess`, harder to move by one
  borderline world than a threshold count) and `floor_regression` (a residual
  test immune to a floor confound of *any* linear form, not just the additive
  one `excess` assumes, and independent of the world-matched-vs-ensemble
  question above). `BATTERY_SEEDS` widened 5→20 for the statistical power the
  paired tests need — this alone does not de-bias anything, which is the whole
  reason the other two exist.
* **Review round 2: the pre-registration's gate wording contradicted itself
  (fixed before it was used on a run), and the regression needed its own power
  check.** The first draft said "co-primary" pass on `wilcoxon_p < 0.05` AND
  `floor_regression_grounded`, but also said floor_regression "wins" if they
  disagree — those two statements conflict in exactly the quadrant that
  matters (paired test fails, regression passes): an AND gate always fails
  there, it does not let regression override. Fixed by naming a single
  explicit verdict, `grounded_confirmed`, as a strict AND with no override in
  either direction — floor_regression is not a unilateral arbiter any more
  than `grounded_paired` alone was. Separately, `floor_regression` now reports
  whether it's actually `powered` (≥ 6 conclusive worlds AND enough spread in
  their `floor_divergence` that the slope is identified) plus a bootstrap CI
  on the fitted slope, so an underpowered fit's low p-value can't masquerade
  as evidence — `floor_regression_grounded` is `None`, not `False` or `True`,
  when unpowered.
* **Review round 3 (sign-off, with two conditions before the run): pre-register
  what `None` means, and check per-rule conclusive yield first.** Both
  addressed without touching the verdict logic itself. `None` now has a fixed,
  documented split — *undetermined* (floor_regression unpowered: the battery
  didn't measure enough, re-run after improving it) vs. *powered-no* (a real
  negative result: move on to representation learnability) — decided now, not
  after seeing which one a run produces. And because `floor_regression`'s
  power check is *per rule* and `vanity`/`exposure` are markedly sparser than
  `demurrage`, `estimate_conclusive_yield` (heuristic-only, no trained brain
  needed) and `train_neural_grounding.py --preflight-only` were added to
  estimate each rule's conclusive yield against `BATTERY_SEEDS` in seconds,
  specifically to catch a rule that would come back undetermined *before*
  training compute is spent finding that out the expensive way.
* **Run #7 (full town, 3 rules, 20 held-out worlds, per the pre-registration):
  `replay_inexplicable_confirmed = None`. Preflight passed (20/20, 20/20,
  19/20) but did not predict the trained brain's density — a real,
  not-hypothetical instance of the "proxy, not a guarantee" caveat.**
  `demurrage`/`vanity`: **undetermined**, `n_conclusive = 0/20` — the trained
  checkpoint deposited and feasted in *zero* of the 20 held-out worlds
  (`trained_stable=False`, `is_stable` was never reached over 60 episodes; the
  per-probe excess series for both rules was flat the entire run). `exposure`:
  **powered-no** — `n_conclusive=20/20`, `fraction_grounded=0.50` (10/20,
  which the *old* metric alone would have reported as promising), but
  `wilcoxon_p=0.684` and `residual_wilcoxon_p=0.5508` — neither test found a
  signal, and (see above) the fit's `slope_ci` spanned both signs, which is
  why the slope-CI-width gate was added the same review round. The record for
  run #7 itself is not retroactively rewritten under the new gate (see
  battery.json, linked from the PR); the refinement applies going forward.
  Read together: the floor-confound machinery did its job (a naive
  `fraction_grounded=0.50` read would have over-claimed for `exposure`) and
  correctly refused to conclude anything for the two rules that never
  occurred — but run #7 says nothing about grounding either way. The blocker
  moved from floor methodology to (a) training convergence (`is_stable` never
  reached) and (b) behaviour coverage (`deposit`/`feast` never explored,
  while `lie` — a cheap, no-cost action — occurred in every world: consistent
  with a survival-pressured policy avoiding costly/risky actions and taking
  the cheap one, not with a methodology bug).
* **Next milestone, per the priority set after run #7:** the sandbox first
  (dense `demurrage` signal, isolates "does a *converged* policy ground on
  *anything*" from the full-town exploration problem) before returning to the
  full town with explicit exploration levers for the unexplored verbs
  (entropy/curiosity weighting, count-based novelty, or an exploratory
  teacher that demonstrates `deposit`/`feast`) — more episodes alone
  (`trained_stable=False`) is expected to re-converge to the same
  never-deposit policy, since non-convergence and behaviour coverage are
  different problems. Run `--preflight-only` before any of these; read
  `grounded_confirmed` per the pre-registration, filing the result under
  undetermined or powered-no as it lands.

## Why this comes before 3D

The reframed north star is grounding and agency as a contribution to AGI. The
panel's consensus (#118) was that *embodiment in 3D is the weakest premise* —
the core of grounding is irreversible causal consequence, which the deterministic
engine already has. Before investing in a 3D client, we need to be able to
**measure** whether agents are grounded at all. This instrument is that
measurement; 3D can come later as a read-only view if the signal warrants it.
