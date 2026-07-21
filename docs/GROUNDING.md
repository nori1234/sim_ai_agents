# Grounding — the counterfactual-world transfer test

> *Is the agent grounded in this world's consequences, or replaying its training?*

This is the project's central **validation instrument**, not a game mechanic
(the question was first framed in #118). When an agent does something sensible
here — saving in a bank, repaying a loan, taking shelter — is that behaviour
**grounded** in the consequences it has lived through, or is it **replaying** a
pattern memorised from training data? In a world whose rules already match the
training prior, the two are indistinguishable, so success proves nothing.

## Where things stand (see "Current status" below for the full run-by-run record)

**Not confirmed, not refuted — narrowing.** Across 19 real-engine CI runs, four
structural hypotheses have each been investigated to a specific mechanism:
the floor confound (engine-side, fixed), the v1→v2 observation tokenizer
(brain-side, fixed), single-step→discounted credit assignment (brain-side,
fixed), and representation erosion during training (brain-side,
**ruled out** — `freeze_backbone` kept `encode_state` byte-identical to a
fresh init, and the policy still didn't ground). The sandbox acceptance
battery has never returned `grounded_confirmed = True`. A supervised
regime-decoding probe (bias-corrected for a population-extinction confound)
independently confirms the representation makes the regime linearly
decodable, both intact (frozen) and even after unfrozen training (0.98 vs
0.81 held-out). Raw attempt counts (added to rule out sparse exploration)
show the counterfactual world was tried *more* than control, not less — the
wrong direction for a grounded policy, not the signature of too few tries.
So the open question is no longer "can the representation see it" or "did
training destroy it" — it's specifically why a policy that can see the
regime and tries the behaviour in both worlds doesn't modulate its rate in
the right direction, which points at value/credit-side noise on the deposit
margin, not architecture, observation content, or representation stability.
Of three further candidates, all three now read healthy against the brain
team's own pre-registered stop rule. Run #13's `episodes_seen` diagnostic
confirms episode boundaries were detected correctly all along (not the leak
the brain team's own code audit suspected). The task is not reward-starved
(the blind heuristic's own realized return already differs by 586 points
between regimes with zero behaviour change). And whether behaviour-cloning
toward the regime-blind teacher was capping the policy — its own diagnostic
(`teacher_frac_in_batch`) never surfaced, but an external, engine-side
cross-check (`measure_teacher_agreement`, shadow-querying a blind
`HeuristicBrain` against a frozen checkpoint) answered it anyway: agreement
with the teacher is low (~12%) and essentially regime-independent
(`gap=-0.008`) — the policy has moved well past imitating the teacher, just
not toward anything regime-sensitive. With S1/S2/S3 all healthy and the
battery still `POWERED-NO`, the brain team's own stop rule said the next
step was revisiting task/reward design — and that round has now run to
completion: the S6 arc found and fixed the sandbox's inverted reward
gradient (`sole_banker`), `control-margin-1` showed the control-side pull
was never weak (+4.35σ) and located the one thin margin (the cf-side
contingency margin, +0.20σ — the exact quantity the battery scores), and a
pre-registered calibration round (run #15, `demurrage_per_day` 0.15→0.25,
contingency margin +0.53σ) widened it above the noise floor. Run #15 still
came back POWERED-NO — but the first S4-instrumented run (#16) then caught
a measurement-selection defect that invalidates runs #14/#15 as fair-task
tests: the driver had been checkpointing and battery-evaluating the brain
of `agents[0]`, the sole banker, the one agent that structurally cannot
deposit under `sole_banker=True` (its teacher demonstrated deposit zero
times in every batch — the probe's `probe_teacher_n=0` was the tell). The
five saver brains, whose teachers deposit densely, were trained and
discarded. The driver now measures the sandbox saver
`agents[1]` (the same agent every instrument measures), and run #17 — the
same pre-registered spec on the fixed driver — became the first genuinely
fair fair-task run: **POWERED-NO, valid, with the S4 mechanism quantified.**
The saver's brain learns deposit-in-general (BC works; propensity 4× the
uniform floor) but regime-independently, because the decision-level credit
noise (within-batch advantage spread, tens of reward units) is ~2 orders of
magnitude above the ±0.60/episode contingency signal — deposit's
post-normalisation advantage averages ≈ 0 (−0.0116). Grounding is currently
blocked not by representation (probe 0.81–0.98), not by incentive
(+0.53σ calibrated), not by exploration (attempts are dense now), but by
**credit-side signal-to-noise on the scored margin** — the next work is
brain-side variance reduction aimed at that number, verified in the probe
fields before any battery is spent. The rate dial stays closed at 0.25 per
the one-round rule. Tracked on issue #130; `--complexity-level`/`--status`
(the complexity ladder, the status factorial) remain queued behind this.

Every number in this document is backed by a committed, byte-exact CI log —
see [`docs/runs/`](runs/) (index + how to add a new one). Prose here can be
wrong or drift out of sync; the files in that directory are what the engine
actually printed.

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
  Raw: [`docs/runs/run-7/`](runs/run-7/).
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
* **Run #8 (sandbox, `demurrage`, episodes=200, previously-good hparams):
  density solved (`n_conclusive=20/20`, vs. run #7's 0/20), but still
  `trained_stable=False` and a powered negative
  (`grounded_confirmed=False`).** `mean_excess=-0.2132`, `wilcoxon_p=0.9681`,
  `floor_regression` powered (n=20, slope_ci width 1.089) with
  `residual_wilcoxon_p=0.6079` — the sandbox fixed the density problem
  cleanly, but training itself never stabilised (probe excess oscillated
  ep5→ep200 with no visible trend) and, unstable or not, showed no grounding
  signal either way. Raw: [`docs/runs/run-8/`](runs/run-8/).
* **The non-convergence traced to a real architecture gap, confirmed by the
  brain team from their own code: the deployed policy is memoryless
  step-to-step** — `decide()` is a pure function of the current observation;
  no recurrent state survives between ticks, and their "Titans test-time
  memory" resets (`M = torch.zeros(...)`) at the start of every `forward()`
  call, so nothing persists even within an episode. Their documentation had
  overstated this as persistent memory; they've corrected it. If a rule's
  consequence isn't independently readable from a single observation
  snapshot, a memoryless policy has no way to condition on it.
* **Engine-side inspection answered their follow-up question directly (is
  the regime visible in a single snapshot at all?): yes.**
  `_pay_deposit_interest`/`_apply_demurrage` (`emergence/simulation.py`) are
  asymmetric by construction — control pays interest as cash
  (`holder.add("money", paid)`, `Deposit.amount` unchanged), counterfactual
  shrinks the deposit itself (`dep.amount -= lost`, no cash). So
  `economy.my_deposits[].amount`'s trend, `self_view.money`'s trend, and a
  demurrage-only `memory` entry ("N coin ... vanished") all distinguish the
  two worlds from observation content alone — this rules out the strongest
  form of "the information simply isn't there" as an engine-side gap. What's
  still unknown (their code, not inspectable from here) is whether their
  observation tokenizer actually surfaces these specific fields.
* **Run #9 (v2, block=10) and run #10 (v2, block=1): the brain team's own "weak
  H3" prediction failed.** Both resolved `llm_model_agi` to commit `9bc016c`
  (their v2 tokenizer fix, confirmed from the CI install log) — the tokenizer
  fix alone did not produce a learning curve; if anything both v2 runs read
  *worse* than run #8 (`mean_excess` -0.3465 / -0.3024 vs run #8's -0.2132;
  `wilcoxon_p` 0.9968 / 0.9999, i.e. further from significance). Block size
  (1 vs 10) made negligible difference under v2, closing that axis. The
  qualitative shape changed, though: run #10's `bootstrap_ci_mean_excess`
  ([-0.41, -0.19]) is entirely negative — not the directionless oscillation
  of run #8, but a *stable* negative excess, i.e. the policy converged to
  something close to regime-blind (`agent_divergence ≈ 0`) rather than to
  noise. Raw: [`docs/runs/run-9/`](runs/run-9/), [`docs/runs/run-10/`](runs/run-10/).
* **That signature pointed at a third structural gap, found and fixed by the
  brain team: single-step credit assignment.** Their policy gradient used
  the immediate-step reward only (no discounted return; `AgentConfig.gamma`
  was defined but never wired in). A `deposit` doesn't change wealth on the
  step it happens (cash moves to a claim of equal value); demurrage's loss —
  or control's interest — lands several ticks later, attributed to whatever
  unrelated action was being taken then. The deposit/no-deposit choice never
  received a learning signal even when the observation correctly encoded the
  regime (v2) — consistent with every result so far: v1→v2 no change
  (information was never the bottleneck), block 1→10 no change (switching
  frequency was never the bottleneck), and convergence to a stable
  regime-blind policy (exactly what an undiscounted-return objective would
  optimise for). Fixed on their side (discounted returns over the buffer,
  value bootstrap at truncation, episode-boundary carry reset,
  `gamma=0` exactly reproduces prior behaviour for regression testing).
* **Run #11 (sandbox, `demurrage`, block=1, v2 tokenizer, credit-assignment
  fix): the primary signal did not appear — a powered-no, not undetermined.**
  Resolved `llm_model_agi` to commit `db39ffa` (confirmed from the CI install
  log). Training self-stopped early at episode 130/200 via `is_stable`
  (probes at ep120/125/130 read +0.333/+0.204/+0.070, three in a row above
  threshold) — but the probe series up to that point was mostly negative or
  oscillating (7 of 26 probes positive), and the streak did not generalise:
  the held-out battery (20 worlds) returned `mean_excess=-0.3591`,
  `wilcoxon_p=0.988`, `bootstrap_ci_mean_excess=[-0.632, -0.065]` (entirely
  negative, the same qualitative signature as run #10's `[-0.41, -0.19]`).
  `floor_regression` was **powered for the first time** (n=20,
  `floor_spread_std=0.259`, `slope_ci_width=1.99` under the 3.0 gate) and
  still found no signal (`residual_wilcoxon_p=0.551`). `grounded_confirmed =
  False`, and per the pre-registered None-split this is the powered-no case:
  a real negative result, not a call for a better-measuring re-run. Three
  independent structural fixes now (floor confound, v1→v2 tokenizer,
  single-step→discounted credit assignment) have each been necessary but
  none has been sufficient to produce a generalising positive-excess trend in
  the sandbox. Raw: [`docs/runs/run-11/`](runs/run-11/).
* **Fallback triggered, per the pre-registration: a supervised
  regime-decoding probe directly on frozen `encode_state` output**, bypassing
  RL entirely — trains a classifier on top of the brain's own learned
  representation to predict control-vs-counterfactual regime from a single
  observation snapshot, using the same held-out seeds (42–61) as the battery
  so it is directly comparable. This isolates representation-learnability
  (does `encode_state` even make regime linearly/simply decodable?) from
  policy-learnability (can RL exploit it once it's there) — the run #11
  result cannot distinguish these two failure modes on its own.
  `--complexity-level` and `--status` remain queued behind this — a
  representation that doesn't decode regime at all makes both experiments
  moot.
* **The regime-decoding probe ran twice (`regime-decoding-probe.yml`,
  `scripts/generate_probe_pairs.py` + the brain team's
  `scripts/probe_regime_readout.py`), against run #11's checkpoint (episode
  130) and a fresh/untrained init, each on 725 paired control/counterfactual
  observation snapshots (20 held-out worlds x 2 regimes x 20 days,
  heuristic-driven so the scored behaviour reliably occurs). The first pass
  surfaced a real methodological gap; the corrected second pass gives the
  answer that stands: representation-learnability is fine, the bottleneck is
  RL/credit-side.** Raw: [`docs/runs/regime-probe-1/`](runs/regime-probe-1/)
  (unpaired), [`docs/runs/regime-probe-2/`](runs/regime-probe-2/) (paired-only,
  the trusted result).
  * **Run A (`llm_model_agi` commit `7b40a93`, unpaired):** probe A (raw
    tokens, weight-independent — a sanity check) read 1.000 held-out for
    both checkpoints, as expected. Probe B (frozen `encode_state`) read
    0.992 held-out for the fresh init (`REPRESENTATION-OK`) but exactly
    0.500 — chance, flat across every one of the 20 days — for the trained
    checkpoint (`ENCODER-DESTROYS`: readable from tokens, lost in
    `encode_state`).
  * **A real confound, caught before the result was trusted:** a few worlds
    die out (population extinction) before day 20 under one or both
    regimes, which the generator handles by simply stopping that
    world/regime's snapshots early. This leaves some `(world_seed, day)`
    rows with no counterpart in the other regime — an artifact the brain
    team flagged from the delivered row counts alone, without seeing the
    probe's output: an unpaired row lets a classifier fit `day` as a
    survivorship proxy for regime instead of the actual encoded content,
    which does not transfer to held-out worlds with different extinction
    patterns. Fixed on their side (`llm_model_agi` commit `e3b91c1`):
    `probe_regime_readout.py` now defaults to paired-only scoring (both
    regimes present for the same `(world_seed, day)`); the excluded count
    is reported alongside the verdict.
  * **Run B (`e3b91c1`, paired-only, 49/725 = 6.8% of rows dropped): the
    verdict flips, in the opposite direction from what the confound
    predicted.** Held-out accuracy for the trained checkpoint's `encode_state`
    probe *rose* from 0.500 to 0.808 (fresh: 0.979, day-2-onward ~1.000) —
    both now read `REPRESENTATION-OK`. This is the surprising part worth
    stating plainly: removing the confounded rows was expected to lower
    accuracy (strip out an inflationary shortcut), not raise it. The
    working hypothesis (not verified, offered without a proposed fix): the
    unpaired rows taught the linear probe a *training-world-specific*
    day/survival correlation that actively misleads on held-out worlds
    (different worlds die out on different days), and the trained
    checkpoint's weaker signal-to-noise `encode_state` was more susceptible
    to being led astray by that spurious feature than the fresh init's
    stronger one. The paired-only visibility curve for the trained
    checkpoint rises from 0.667 (day 1) to mostly 0.75-1.0 by day 4 onward
    (one dip to 0.583 at day 17) — a real, if noisier, decoding signal, not
    chance.
  * **Reading (paired-only, the corrected and trusted result):
    representation-learnability is not the blocker.** `encode_state` makes
    the regime linearly decodable both before and after training (0.98 vs
    0.81 held-out) — degraded by training, but nowhere near destroyed. Per
    the pre-registered decision tree, this places the run #11 failure on
    the RL/credit-side: the representation the policy conditions on already
    contains the regime signal (fresh init even more clearly than trained),
    but three structural fixes in a row (floor confound, v1->v2 tokenizer,
    single-step->discounted credit assignment) have not produced a policy
    that acts differently on it. The `db39ffa` discounted-return fix was
    necessary but evidently not sufficient — the next candidates are
    exploration (a memoryless policy has no reason to try `deposit` more
    than a handful of times per episode) and value-estimate noise on the
    deposit/no-deposit margin specifically, not the representation itself.
* **Run #12: the brain team's `freeze_backbone` test (commit `f8badf1`) —
  erosion ruled out as the cause, not confirmed as it.** Their hypothesis:
  the training-time drop in `encode_state`'s decodability (0.98 fresh vs
  0.81 trained, run #11) was RL gradients from a regime-blind policy
  actively erasing the regime feature during training, since nothing in an
  undiscriminating reward stream gives the backbone a reason to preserve
  it. `freeze_backbone` excludes the backbone from the optimizer
  (`requires_grad=False`, `encode` under `no_grad`) so only the policy/
  value/world heads train on top of a fixed representation — a minimal,
  one-run test of that specific mechanism, pre-registered with two
  branches: (1) a sanity check that MUST hold if the implementation is
  correct (trained `encode_state` stays at the fresh level), and (2) the
  real question (does the policy ground once the representation can no
  longer be eroded).
  * **(1) confirmed:** the regime-decoding probe against run #12's
    checkpoint read held-out `encode_state` accuracy of 0.992 — identical
    to a fresh/untrained init run the same pass, also 0.992. Freezing
    worked exactly as implemented; not a bug.
  * **(2) did not hold: still `grounded_confirmed = False`.**
    `fraction_grounded` dropped to 0.15 (run #11: 0.25), `bootstrap_ci_
    mean_excess = [-0.446, -0.148]` (entirely negative, same shape as runs
    #10/#11), `floor_regression` powered with an even flatter slope
    (`+0.042`, width `0.917` — tighter than run #11's `+0.427`/`1.986`, and
    closer to zero). Training never reached `is_stable` over the full 200
    episodes (39 of 40 probes negative; only the very last, ep200, read
    positive).
  * **New in this run: raw attempt counts** (`control_count`/
    `counterfactual_count` on `GroundingResult`, added specifically for
    this fallback — the normalised `control_rate`/`counterfactual_rate`
    can't distinguish "too few tries to learn from" from "plenty of tries,
    wrong direction"). Summed over the 20 held-out worlds: `control=158`,
    `counterfactual=178` — the counterfactual world was tried *more*, not
    less, the opposite of what a sparse-exploration story predicts and the
    opposite of the direction a grounded policy would need.
  * **Reading: representation erosion is ruled out as the root cause, not
    confirmed as it.** The representation stayed intact (byte-for-byte
    matching a fresh init) and the policy still didn't ground — an even
    flatter floor-regression slope than run #11's already-flat one. Three
    of four structural hypotheses (floor confound, tokenizer, erosion) are
    now closed with a specific mechanism identified for each; erosion is
    the first one that was investigated and directly ruled out rather than
    fixed. The raw counts point away from insufficient exploration and
    toward value/credit-side noise on the deposit margin, or a remaining
    bug in the discounted-return implementation itself — neither
    investigated yet. Raw: [`docs/runs/run-12/`](runs/run-12/),
    [`docs/runs/regime-probe-3/`](runs/regime-probe-3/) (the freeze sanity
    check, run #12's checkpoint vs a fresh init).
* **Two more candidates surfaced auditing both sides' code, ahead of run
  #13.** The brain team found their discounted-return implementation had no
  tested episode-boundary signal — it depended on the engine setting
  `_prev_obs = None` between episodes, an assumption that lived only in a
  code comment, never in the contract. Checking our own driver
  (`scripts/train_neural_grounding.py`) answered their decisive question
  directly: `training_factory` *has* reset `_prev_obs = None` every episode
  since the credit-assignment fix (git-verified back to the script's first
  commit) — so if their detection genuinely keyed off that signal, episode
  boundaries should have been detected all along. They've since replaced the
  private-attribute assumption with an explicit `dev.end_episode()` hook
  (commit `1a1c082`) plus an `episodes_seen` diagnostic to verify empirically
  rather than trusting either side's code-reading — wired into the driver
  (`docs/NEURAL_CONTRACT.md`, "5a. Episode boundaries"). Separately,
  inspecting `emergence/brains/heuristic.py`'s `_bank_action` answered a
  second question: `training_factory` *has* passed `teacher=HeuristicBrain
  (persona)` since run #1 (not `None`), but that heuristic's deposit/withdraw
  decision reads only `agent.money` — no `deposit_rate` or regime-sensitive
  field anywhere in its logic, and `hide_rate=True` hides `deposit_rate` from
  the observation regardless. Its own measured `floor_divergence` is
  positive on average (run #12: mean `+0.273`, 18/20 worlds) but that is
  best read as a mechanical consequence of demurrage shrinking the balance
  available to redeposit, not decision-level regime discrimination — so if
  behaviour-cloning toward this teacher is doing meaningful work, it cannot
  be teaching genuine regime sensitivity, only a `money >= 12` rule that
  happens to look partly regime-correlated. Run #13 (episode-boundary fix,
  `freeze_backbone` removed since erosion is ruled out) reports
  `episodes_seen` and `teacher_frac_in_batch` to settle both empirically.
* **A fifth, prior question, asked by the brain team ahead of run #13: does
  the TASK pay enough for grounding to be worth learning at all, independent
  of whether any policy currently learns it?** `measure_reward_ceiling`
  (`emergence/grounding.py`, `scripts/reward_ceiling.py`) compares the blind
  heuristic's own realized return (`survival_reward`, telescoped over the
  episode) against a scripted oracle handed the ground-truth regime directly
  and never depositing under counterfactual — the most any policy could gain
  from discriminating this regime, cheap and deterministic (no torch, no CI,
  seconds). Raw: [`docs/runs/reward-ceiling-1/`](runs/reward-ceiling-1/).
  * **A real bug caught before trusting the number:** the first oracle
    returned `None` in place of a deposit, which fell through
    `HeuristicBrain.decide()` into `_trade_action`'s untested market-
    primitive loops (offer/accept/repay with no facility to ground them in
    this minimal sandbox) and starved the agent to death by day 5 — a
    confound with nothing to do with demurrage. Fixed: `REST` (a net energy
    *gain*, not a drain) is the minimal non-confounding substitute for "hold
    the cash, don't bank it."
  * **The fixed measurement is itself confounded, and says so honestly:**
    `advantage_counterfactual = -125.4` — the grounded oracle earns *less*
    than the blind heuristic, not more. Tracing why: the blind heuristic's
    `_bank_action` also runs a peer-lending branch (`become a banker`,
    `OFFER`/`REPAY`/`lend`) whenever `bank_here` is unset, and deposit
    amounts in this sandbox compound to multiples of the starting money
    (50 → thousands) within 20 days — a large reward channel unrelated to
    demurrage that the oracle forfeits entirely by resting instead of
    participating. This isn't a finding about grounding; it's a finding that
    this sandbox's peer-lending economy dominates its own reward signal,
    which the oracle wasn't built to also imitate.
  * **The clean number inside the same run: the blind heuristic's own
    realized return already differs hugely by regime** —
    `blind_return_control = 711.2` vs `blind_return_counterfactual = 125.3`,
    a `585.9`-point within-policy gap with no behaviour change at all. This
    answers the brain team's actual question directly: the task is **not**
    reward-starved — demurrage costs even a non-adapting policy the large
    majority of its return. The bottleneck a value function would need to
    close is attribution amid a much larger, unrelated co-occurring reward
    channel (peer lending), not reward magnitude.
  * **S6 — the brain team's clean-spec follow-up (`measure_deposit_oracle`,
    `scripts/deposit_oracle.py`), to split "task doesn't pay" from "task
    pays, variance drowns the gradient".** Same oracle intent, one difference
    from reward-ceiling-1: instead of `REST`-substituting a dropped deposit,
    it drops the `DEPOSIT` decision and falls through to the blind heuristic's
    own next branch — no substitute action, nothing else touched. Result:
    `advantage_cf = -127.3` (effect size **−1.94σ** of the blind's own
    per-world spread; oracle behind in **0/20** worlds; survivors-only −120,
    so not death-driven). The **sign matches reward-ceiling-1's −125.4
    despite a different substitute behaviour** — so the negative is robust to
    the substitute-action confound. Per the brain team's own decision table
    this is the **task-redesign** branch, not the learning-side/variance one
    (a −1.94σ, 0/20-flip effect is not a positive signal buried in noise).
    **Mechanism, and why it disagrees with their mirror's predicted +0.555**
    *(corrected — the original attribution to work-minted #45 income was
    falsified by the lever tests below)*: the reward counts bank deposits as
    wealth (`_wealth = money + Σdeposits`, `_neural_reward.py`), and the
    sandbox lets deposits **chain agent-to-agent** — `_banker_near` treats any
    other agent standing on a BANK tile as a deposit counterparty, and every
    sandbox agent stands on the bank, so the pooled coin ping-pongs
    banker⇄saver each tick and ratchets ~+420 of reward-counted claims per
    pass out of one fixed coin pool (the measured saver's reward-wealth grew
    50→3162 *with demurrage active*; its claims hit 2188 on day 1 from 50
    starting coin). So −15%/day evaporation is dominated by the claim
    ratchet, and **depositing is reward-maximizing even under demurrage** —
    the task gradient points the wrong way, which no variance reduction
    fixes. Raw: [`docs/runs/deposit-oracle-1/`](runs/deposit-oracle-1/).
  * **S6 lever-2 calibration (`--deposit-weight`, `deposit-oracle-calib-1`):**
    the brain team accepted S6 and proposed using it as a *calibration dial* —
    apply the deposit-down-weight lever, re-measure, and set the parameter
    where `advantage_cf` lands **slightly** positive (a task where grounding
    pays but isn't trivial). Implemented as a continuous weight on the deposit
    term (λ, default 1.0 = canonical, byte-identical). The dial is monotone and
    behaves as predicted, **but cannot cross the sign**: at the extreme λ=0
    (banked coin worthless) `advantage_cf` is still −1.27 over 20 worlds and
    **−0.27 survivors-only**, with the per-world majority flipped (14/20) but
    the mean marginally negative. Cause: `survival_reward` telescopes to
    (final − initial), so re-weighting an end-of-episode deposit balance
    (already demurrage-shrunk / withdrawn) has bounded leverage; the residual
    is a **behavioural** cost (the oracle holding idle cash, falling through to
    a different next action) that no reward re-weighting can remove. So the
    reward-reweighting family has a structural ceiling at ≈0⁻; crossing the
    sign needs a **trajectory** change. Raw:
    [`docs/runs/deposit-oracle-calib-1/`](runs/deposit-oracle-calib-1/).
  * **S6 task redesign (`sole_banker`, `deposit-oracle-redesign-1`) — the
    sign crossed, calibration target hit.** Lever 3 (scaling work-pay minting
    #45) and deposit-interest scaling were both tested and are **inert**
    (−127.304 unchanged even at zero) — the measured saver never works, and
    interest isn't the inflow either. The real driver is the sandbox's
    **agent-to-agent deposit chain** (see the corrected mechanism note above).
    The redesign is one switch: `sole_banker=True` restricts deposits to the
    staffed banker, cutting the chain. Result: `advantage_cf = +0.2075`
    (**+0.56σ**, oracle ahead 12/20, control sanity 0.00), inside the brain
    team's requested "slightly positive, not large" calibration band, with
    the deposit decision still dense (39 cf / 102 control per episode).
    Default `False` is byte-identical (every earlier S6 number reproduces).
    This is the first task on which grounding actually pays — the fair test
    of "does the brain learn grounding when the task rewards it" that the 13
    prior training runs never had. Raw:
    [`docs/runs/deposit-oracle-redesign-1/`](runs/deposit-oracle-redesign-1/).
  * **Run #14 — the first fair-task training run: POWERED-NO.** Trained on
    `sole_banker=True` per the brain team's pre-registered spec (their
    hparams: `bc_weight 0.3`, `self_attempt_base 0.3`, `batch_every 64`;
    tokenizer v2 / γ 0.99 / no freeze as defaults; pool seeds 1000–1015,
    disjoint from the battery's 42–61). 60 episodes, never `is_stable`
    (streak 0 throughout); the probe excess sat **flat at −0.30..−0.40 from
    episode 5 to 60** — no learning trajectory. Battery: `mean_excess
    −0.554` (bootstrap CI [−0.593, −0.506]), `fraction_grounded 0.00`,
    floor-regression **powered** (n=14) and negative → per-rule
    `grounded_confirmed = False` — the pre-registered **POWERED-NO** branch,
    now measured for the first time on a task where grounding pays. (The
    battery-level conjunction reads `None` only because 6/20 held-out worlds
    saw no deposit attempt at all, listing the rule inconclusive at the
    battery level; the powered per-rule verdict is the honest headline.)
    **Mechanism observation:** the trained policy barely deposits in either
    regime — 15 control / 18 counterfactual attempts summed over all 20
    worlds (the heuristic floor deposits densely) — i.e. it collapsed to the
    regime-*independent* never-deposit arm, which the transfer metric
    correctly refuses to call grounded. Also notable: `teacher_frac_in_batch
    = 0.1875` surfaced in the training diagnostics — the S2 signal that was
    previously unmeasurable. Next, per the pre-registration fixed before
    this run: **representation learnability** (brain side), not more metric
    or task tuning; an engine-side observation for that discussion is that
    both arms' margins are deliberately small (the +0.21 oracle advantage),
    so the control-side pull toward depositing (interest) may be weak
    relative to reward noise — changing that would be a NEW pre-registration
    round, not a silent knob turn. Raw: [`docs/runs/run-14/`](runs/run-14/).
  * **control-margin-1 — the diagnostic run #14's collapse demanded; it
    falsifies the "weak control pull" conjecture and relocates the small
    margin to exactly the part the battery scores.** `scripts/control_margin.py`
    measures the full 2×2 of {deposit-per-rule, never-deposit} × {control, cf}
    with the same telescoped `survival_reward` the policy optimizes
    (deterministic, no torch). Guardian, 20 held-out worlds, `sole_banker`:
    the **control-side pull** (deposit vs hold, control) is **+10.35 (+4.35σ,
    survivors-only +6.62σ, 20/20 worlds)** — not weak but one of the strongest
    gradients in the task; depositing in control is a dominant, densely-rewarded
    action. So run #14's never-deposit policy sat at the **pessimal** corner
    (return −4.38, ~+10.35 below grounded and below even regime-blind
    always-deposit at +5.77): RL failed to climb a large, dense reward
    gradient and never found even the trivial dominant strategy — a
    credit-assignment / value-learning failure (the S4 candidate: deposit's
    interest is delayed and thin while the deposit action is wealth-neutral
    on the step), **not** a task-reward gap. The one margin that *is* small is
    the one the battery scores: grounded and regime-blind always-deposit differ
    **only** in the cf cell, so the reward for regime-*contingency* is exactly
    the cf advantage — **+0.20σ, buried in noise**. The `sole_banker`
    calibration put the overall cf oracle advantage in the requested "+0.2–0.5σ,
    not large" band, but that same +0.20σ *is* the entire contingency gradient a
    learner needs to prefer grounded over always-deposit. So the engine-side
    intuition above was right that a margin is too small, but wrong about which
    side: the control pull is huge; the **cf-side contingency margin** is the
    needle. This separates two blockers — (learning) why RL avoids the dominant
    always-deposit arm (brain-side S4 value/advantage), and (task) that even a
    perfect learner sees only +0.20σ for contingency, whose fix (steeper
    demurrage to widen the cf penalty) is the NEW pre-registration round the
    run #14 note called for, gated on `control_margin.py`'s `contingency_margin`
    field before any training run. Raw:
    [`docs/runs/control-margin-1/`](runs/control-margin-1/).
  * **That round ran: pre-registration → calibration → run #15 → POWERED-NO
    again, and the pre-registered grid closes the task dial.** The proposal
    ([`docs/proposals/run15-contingency-margin.md`](../proposals/run15-contingency-margin.md))
    fixed the band ([+0.5σ, +1.0σ] on the paired-difference σ), the
    smallest-rate rule, four gates, a 4-branch interpretation grid, and a
    one-round limit — all before any training. `contingency-calib-1`
    calibrated `demurrage_per_day` (plumbed through the whole
    sandbox/probe/battery/driver/CI chain, default 0.15 byte-identical) to
    **0.25/day** (+0.60, +0.53σ, 15/20 worlds; control pull exactly
    +10.3485 at every rate — the structural invariance held; yield 20/20,
    density held, survival 20/20). Run #15 = run #14's exact spec + that one
    change. Result: 60 episodes, streak 0, probe excess flat −0.37..−0.51
    throughout; battery `mean_excess −0.5775` (CI [−0.617, −0.515]),
    `n_conclusive 17/20`, floor-regression powered → per-rule
    `grounded_confirmed = False`. **Raw attempts: control 19 /
    counterfactual 18 over all 20 worlds — still the never-deposit arm.**
    Per the grid fixed before the run, this is branch 2: **the contingency
    incentive is ruled out as the blocker** (it was calibrated above the
    noise floor and the policy behaved identically), the task dial is closed
    (no further rate raises, per the one-round rule), and the primary
    suspect is **S4 — value/credit assignment on the deposit decision**
    (brain side; the per-decision advantage instrumentation requested on
    #130). A new in-run observation for that work: `teacher_frac_in_batch`
    now prints every episode (~0.4–0.5 typical) — half of each batch is BC
    toward a densely-depositing teacher, yet the learned policy deposits
    ~once per world; BC targets and PG/value gradients appear to pull in
    opposite directions on this action. Raw:
    [`docs/runs/run-15/`](runs/run-15/),
    [`docs/runs/contingency-calib-1/`](runs/contingency-calib-1/).
  * **Run #16 (the first S4-instrumented run) — the probe found a
    measurement-selection defect that INVALIDATES runs #14/#15 as fair-task
    tests; the branch-2 reading above is withdrawn.** The brain side's
    `probe_verb` instrumentation (per-batch deposit credit/propensity in
    every training log line, `llm_model_agi@ead7e35`) ran on run #15's exact
    spec and returned three facts: the logged brain's teacher demonstrated
    deposit **zero times in every batch** (`probe_teacher_n=0`; its
    `teacher_frac_in_batch` was ~0.3–0.7, so teacher steps abounded — just
    never deposits), its deposit propensity sat flat at the uniform floor
    (~1/47) from first batch to last, and deposit's raw (G−V) credit was
    *positive and larger than non-deposit's* (+17.2 vs +13.0; regime-ordered
    correctly, control +25.9 vs cf +8.8). Those three together pin the
    mechanism, verified in code and empirically: the driver trains a brain
    per agent but logged/checkpointed **`agents[0]` — the staffed banker —
    which under `sole_banker=True` is the sole deposit *receiver* and
    structurally cannot deposit** (`_banker_near` excludes self;
    `_do_deposit` refuses `bank is agent`). Runs #14/#15 therefore
    battery-evaluated a brain that never faced the scored decision during
    training; their POWERED-NO verdicts truthfully describe that checkpoint
    but say nothing about whether a brain that *did* face the decision
    learns grounding — **the fair-task test has not actually run yet.** The
    S6 redesign, the 0.25 calibration, control-margin-1, and the rate-dial
    one-round rule all stand (instrument-side, brain-independent). Runs
    #8–#13 are not invalidated by this (without `sole_banker` the banker had
    chain counterparties and did face the decision). Driver fixed: the
    measured brain is now the sandbox saver `agents[1]` — the same agent
    every instrument already measures — selected explicitly and printed at
    episode 1 and checkpoint time; verified locally (the saver's teacher
    demonstrates deposit from the very first batch, `probe_teacher_n=15/16`).
    The teacher-side puzzle noted under run #15 (BC ~0.5 of batch yet no
    deposits) is resolved by the same finding — that batch was the banker's,
    whose teacher demos everything *except* deposit. Raw:
    [`docs/runs/run-16/`](runs/run-16/) (incl. `probe_analysis.txt`).
  * **Run #17 — the first VALID fair-task run (fixed driver, measured brain
    = the sandbox saver): POWERED-NO, with the S4 mechanism finally
    quantified instead of suspected.** Battery: `n_conclusive = 20/20` (a
    first), floor-regression powered (n=20), `mean_excess −0.606` (CI
    entirely negative), `fraction_grounded 0.0`; raw attempts control 91 /
    cf 117 — ~6× run #15's checkpoint, in *both* regimes. The probe: BC now
    works (`probe_teacher_n=112`; the saver's teacher demos deposit
    constantly), the policy learned deposit-in-general (propensity ~0.031 →
    ~0.083, ~4× the uniform floor, still rising at ep60) — but
    regime-**independently** (segment-joined propensity control 0.0525 vs
    cf 0.0522), and the decision-level credit shows why: per-batch returns
    swing ±10..±140 while the entire regime-contingent payoff is ±0.60 per
    episode, so deposit's raw (G−V) credit even reads *inverted*
    (cf +28.7 > control +22.5, episode-return noise, not signal) and its
    post-normalisation advantage — the thing PG multiplies — averages
    **−0.0116 ≈ zero**. The calibrated +0.53σ was in oracle-return σ across
    worlds; the learner faces the within-batch advantage σ, ~2 orders of
    magnitude larger — a third σ convention, and the one that gates
    learning. So S4 is not mis-attribution: the contingency signal is
    simply ~1% of the credit noise at the decision level, and the
    regime-blind teacher (issue #10's R2) can teach deposit-in-general but
    not the contingency. Next per the stop rule: brain-side variance
    reduction aimed at exactly this number (per-episode return centering,
    regime-aware value baseline — legitimate, the value head already sees
    the regime-decodable representation — or paired/counterfactual
    advantage), each falsifiable in the probe fields (success =
    `probe_adv_used_mean` on deposit separating by regime segment) before
    any battery is spent. Caveats recorded: `trained_stable=False` with
    propensity still rising at ep60 (a longer run is cheap and untried),
    and the cf>control attempt asymmetry (117 vs 91) is unexplained. Raw:
    [`docs/runs/run-17/`](runs/run-17/) (incl. `probe_analysis.txt`).
  * **Runs #18–#19 — variance-reduction candidate 1 works; the binding
    constraint moves to the teaching channel.** `adv_baseline="episode"`
    (brain `435177b`; subtract each episode segment's mean advantage before
    normalisation — the control variate aimed at the exact noise run #17
    measured) un-inverted the deposit credit (raw control +2.95 vs cf +1.68
    at 60 eps, spread compressed ~10×) and, at 200 episodes (run #19), the
    pre-declared criterion was essentially met on the signal side: **deposit
    used-advantage cf −0.123 vs control −0.005** (middle third −0.229 vs
    +0.053) — PG is finally told the regime truth. Behaviour, however,
    stayed regime-flat (segment propensity 0.0807 vs 0.0818; battery 359
    control / 348 cf attempts, mean_excess −0.587, still POWERED-NO, no
    is_stable): half of every batch is BC toward the regime-blind teacher
    (probe_teacher_n 435 ≈ probe_self_n 443), whose dense
    deposit-everywhere demonstrations out-pull the sparse, small, correct
    PG differential. That is issue #10's R2 ("no parent can teach
    grounding") materialized as numbers, and the brain side's own listed
    remedy (b) is next: **BC annealing** (`bc_weight_decay_steps`, brain
    `a7f7ddc` — cosine-decay bc_weight → bc_weight_min, imitation
    bootstraps then hands off; run #20). Raw:
    [`docs/runs/run-18/`](runs/run-18/),
    [`docs/runs/run-19/`](runs/run-19/).
  * **Run #20 — BC annealing (issue #10(b)): the anneal fired but grounding
    did not appear underneath; the individual-RL ladder is largely
    exhausted.** `bc_weight` decayed 0.3→0.05 on schedule, yet regime
    propensity separation stayed at noise (control−cf ≈ ±0.002 across all
    training thirds), deposit density fell as BC decayed (0.078→0.040), and
    the battery is still POWERED-NO (`mean_excess −0.573`, `n_conclusive
    20/20`). Reading the #17→#20 chain: the correct PG signal exists (episode
    baseline, #18/#19) but is too sparse (~2 deposit self-samples/batch) to
    drive a differentiated policy once the blind teacher's density-scaffold is
    removed. The regime-blind teacher (R2, #10) bootstraps density but not
    contingency; self-play PG represents the contingency but can't win on it.
    Next candidates (strategy call, not an autonomous knob): a **grounded
    scripted teacher** (#10(c), measures the teaching-channel ceiling), stronger
    exploration, or the **temporal-memory/body integration** (`agent_agi/docs/09`
    — a memoryless policy has almost no signal about a consequence that unfolds
    over ticks). Raw: [`docs/runs/run-20/`](runs/run-20/).
  * **Run #25 — A1, the grounded scripted teacher (#10(c)): the teaching
    channel does NOT transmit grounding. POWERED-NO.** The regime-*blind*
    HeuristicBrain teacher was replaced by the regime-*aware* grounded
    heuristic (deposits in control, RESTs under demurrage), told the
    ground-truth regime each episode, with `bc_weight` held open at 0.3 (no
    anneal — keep the channel open). This measures the teaching-channel
    ceiling: can grounding transmit through BC *at all* when the teacher has
    it? It cannot. `grounded_confirmed = False`, POWERED-NO
    (`fraction_grounded 0.00`, `wilcoxon_p 1.0`, mean excess ≈ −0.59, n=20);
    raw attempts **control 167 ≈ counterfactual 160** — the student deposits
    at the same rate in both regimes, and even the control-side density stays
    near the not-deposit floor (~0.08). So a teacher demonstrating the correct
    contingent behaviour every step still transmits neither the *contingency*
    nor much *density*. This is the run-20 pre-registered **fail** branch: it
    **implicates policy / representation capacity, not the availability of a
    grounded teacher** — the bottleneck is not "we need an LLM/human parent."
    Next is the POWERED-NO follow-up: *representation learnability* (can the
    observation encoding + memoryless policy even express/learn "demurrage →
    don't deposit"), not more metric tuning. Raw:
    [`docs/runs/run-25/`](runs/run-25/).
    * **Two infrastructure facts this run also nailed down.** (1) Runs #21/#22
      (this A1 config, and v1b) had crashed to a *silent heuristic fallback*:
      the developmental brain diverged numerically (`encode_state` had no
      output norm → value MSE / raw world-model curiosity / policy logits grew
      until `torch.multinomial` raised on NaN), and `decide()`'s bare `except`
      swallowed the cause. Metastable — an unlucky weight init diverged at
      episode 1, a lucky one limped to ~ep163 — and the trainer never seeded
      RNG, so the timing was luck. Fixed brain-side (parameter-free
      `F.layer_norm` on the state + skip-update-on-nonfinite guard) and
      engine-side (seed RNG from `--seed`; surface the swallowed traceback in
      the fatal message). Run #25 is the first fully deterministic run and the
      CI validation of the fix (200 clean episodes). (2) `agent_agi` (the
      memory organ) is installed in CI only when `memory="episodic"` is
      requested, guarded to fail loud rather than fall back silently.
    * **The LayerNorm caveat — raised, then CLOSED by a decodability probe.**
      The stabilizing LayerNorm discards state magnitude, so if the regime
      signal lived in the *scale* of `encode_state` this run could not see it.
      `scripts/probe_regime_decodability.py` settled it: fit a linear probe
      regime←representation on 7680 sandbox observations (control+demurrage,
      held-out 30%). **Regime is 94% linearly decodable** from the raw state
      (0.944) and **LayerNorm keeps essentially all of it** (normed 0.942;
      shuffled-label control 0.517 ≈ chance). So the norm is not a confound,
      and — more importantly — the encoding is *not* the bottleneck: the regime
      is richly, linearly present in exactly the vector the policy reads.
    * **What that does to the representation-learnability line.** The first
      question of that line ("is the contingency even present/recoverable in
      the encoded observation?") is answered **yes** — decisively. So A1's
      POWERED-NO is **not** a representation-encoding failure; the wall is the
      **policy / credit-assignment channel**, which cannot convert a
      94%-decodable state feature into regime-contingent behaviour from the
      available signals (BC toward the teacher + sparse self-play PG). This
      *de-prioritises* tokenizer/representation work and *re-prioritises* the
      credit/exploration and temporal-memory levers — i.e. it strengthens the
      case for the memory line (v1b) and for a stronger credit signal over
      more encoding work.
* **Run #13 (episode-boundary fix, `freeze_backbone` removed, commit
  `1a1c082`): S1 ruled out empirically, S2 unmeasurable, still
  `grounded_confirmed = False` with the tightest floor-regression null yet.**
  Raw: [`docs/runs/run-13/`](runs/run-13/).
  * **S1 (episode-boundary leak) — ruled out, not just by our code-reading
    this time.** `episodes_seen` tracked the true training-episode count
    throughout (e.g. `196` at episode 196, `199` at episode 200) rather than
    sticking near `1` — the signature a genuine leak would produce. Episode
    boundaries were being detected correctly this whole time, confirming
    what inspecting `training_factory`'s `_prev_obs = None` reset had
    already suggested: this specific failure mode was never actually
    happening on our pipeline.
  * **S2 (BC/PG ratio) — undetermined, not ruled out: the diagnostic never
    appeared.** `teacher_frac_in_batch` occurs zero times across all 200
    episode log lines, despite `episodes_seen` (from the same commit range)
    surfacing correctly. Whether behaviour-cloning toward the regime-blind
    teacher is capping the policy remains an open question — it needs the
    brain team to fix how/whether that field reaches `last_learn_info`
    before it can be tested, not a re-run on our side.
  * **The battery result: same verdict, narrower null.**
    `mean_excess=-0.2539`, `wilcoxon_p=0.9998`,
    `bootstrap_ci_mean_excess=[-0.393, -0.129]` (entirely negative again).
    `floor_regression` powered with the tightest slope CI of any run so far
    (`slope=-0.062`, width `0.464` — versus run #12's `0.917`) — more
    statistical certainty that there is no relationship, not less.
    `grounded_confirmed = False`, `trained_stable = False` (full 200
    episodes, max probe streak 2, no `is_stable`).
  * **Raw attempt counts flipped direction, barely: `control=621`,
    `counterfactual=602`** (run #12: `158`/`178`, the wrong way round). A
    much larger sample (1223 vs 336 total attempts, consistent with more
    episodes reaching `CURIOUS`/self-play) and now the *correct* direction
    for grounding, but the gap is small (3%) relative to the total and the
    battery's own paired tests still read no signal — not evidence of
    grounding on its own, just no longer evidence against it either.
  * **Reading against the brain team's own pre-registered stop rule**
    ("if S1-S3 all read healthy and it's still POWERED-NO, stop hunting
    defects and revisit task/reward design"): two of three are in — S1
    healthy (this run), S3 healthy (the reward ceiling, above) — but S2 is
    unmeasured, not confirmed healthy, so the stop condition isn't met yet.
    The immediate next step is fixing `teacher_frac_in_batch`'s exposure,
    not re-running training again with the same diagnostic gap.
* **S2 answered anyway, from outside the training loop:
  `measure_teacher_agreement` (`emergence/grounding.py`,
  `scripts/teacher_agreement.py`, `.github/workflows/teacher-agreement.yml`)
  shadow-queries a blind `HeuristicBrain` at every decision point against
  run #13's checkpoint and tallies how often the tested policy still agrees
  with the teacher's deposit call, by regime — no brain-side fix needed.
  Raw: [`docs/runs/teacher-agreement-1/`](runs/teacher-agreement-1/).**
  Sanity check passed (`teacher_deposit_rate_control=0.587` vs
  `counterfactual=0.617` — close, so the two worlds are comparable). The
  result: `agreement_control=0.122`, `agreement_counterfactual=0.130`,
  `agreement_gap=-0.008` — **agreement with the teacher is low (~12%) in
  both regimes, and the gap is essentially zero.** Not the signature
  expected for either failure mode: high, regime-independent agreement
  would mean still BC-anchored; a positive gap would mean deviating from
  the teacher specifically under the punished regime (grounded). Instead
  the trained policy has moved well past simply imitating the teacher
  (only ~12% agreement — nowhere near "anchored") but its own behaviour
  isn't regime-conditioned either — it diverges from the teacher about
  equally in both worlds, doing something else entirely rather than
  something *regime-sensitive*. Reading: **S2 (BC anchor) is effectively
  ruled out as the blocker** — the policy is not still parroting the
  regime-blind teacher — which means all three of S1/S2/S3 now read
  healthy against the brain team's own pre-registered stop rule. Per that
  rule, the next step is revisiting task/reward design rather than
  continuing to hunt for a sixth structural defect.

## Why this comes before 3D

The reframed north star is grounding and agency as a contribution to AGI. The
panel's consensus (#118) was that *embodiment in 3D is the weakest premise* —
the core of grounding is irreversible causal consequence, which the deterministic
engine already has. Before investing in a 3D client, we need to be able to
**measure** whether agents are grounded at all. This instrument is that
measurement; 3D can come later as a read-only view if the signal warrants it.
