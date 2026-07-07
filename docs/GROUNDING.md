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
the logged headline is `excess`, never the raw divergence.

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
* **Next milestone:** a battery run with training/eval seeds properly disjoint —
  the test of whether domain randomization (not more optimizer tuning) closes the
  fraction_grounded gap. If it doesn't move, the standing hypothesis (a single
  training world doesn't generalize) is falsified and the real bottleneck is
  something else (e.g. the rule isn't learnable from the observation as given).

## Why this comes before 3D

The reframed north star is grounding and agency as a contribution to AGI. The
panel's consensus (#118) was that *embodiment in 3D is the weakest premise* —
the core of grounding is irreversible causal consequence, which the deterministic
engine already has. Before investing in a 3D client, we need to be able to
**measure** whether agents are grounded at all. This instrument is that
measurement; 3D can come later as a read-only view if the signal warrants it.
