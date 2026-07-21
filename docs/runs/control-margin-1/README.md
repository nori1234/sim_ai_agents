# control-margin-1 — the margin run #14's collapse demanded

Run #14 (the first fair-task training run, `sole_banker=True`) came back
**POWERED-NO**: the trained policy collapsed to a regime-**independent**
never-deposit policy (15 control / 18 counterfactual deposit attempts summed
over 20 worlds, vs the heuristic floor's dense depositing). The engine-side
note on issue #130 offered a candidate cause — *"both arms' margins are
deliberately small (the +0.21 oracle advantage), so the control-side pull
toward depositing (interest) may be weak relative to reward noise."*

The S6 deposit-only oracle only ever measured the **cf-side** margin
(`advantage_cf`, the reward for HOLDING cash in the counterfactual world). This
run measures the **control-side** pull that was never on the table, plus the
full 2×2 of {deposit-per-rule (blind), never-deposit} × {control, cf}, using
the exact telescoped `survival_reward` the RL policy optimizes. Read-only, no
task change, deterministic, no torch.

## Reproduce

```
python3 scripts/control_margin.py --persona guardian --sole-banker
```

`output.txt` is the exact CLI output (full per-world JSON + summary).

## Headline (fact) — guardian, 20 held-out worlds (42–61), sole_banker

| margin | mean | effect size | worlds + |
|---|---|---|---|
| **CONTROL pull** (deposit vs hold, control) | **+10.35** | **+4.35σ** | **20/20** |
| &nbsp;&nbsp;survivors-only | +10.81 | +6.62σ | 17/17 |
| CF advantage (hold vs deposit, cf) = `advantage_cf` | +0.21 | +0.20σ | 12/20 |
| **CONTINGENCY margin** (grounded − always-deposit) | **+0.21** | **+0.20σ** | 12/20 |

Policy returns (per-world sum of a policy's two regime cells):

| policy | return | note |
|---|---|---|
| grounded (deposit C / hold CF) | **+5.97** | the target |
| always-deposit (regime-blind) | +5.77 | only −0.21 behind grounded |
| **never-deposit (run #14's collapse)** | **−4.38** | the pessimal arm |

## Reading (what this reframes)

1. **The engine-side conjecture is falsified.** The control-side pull toward
   depositing is not weak — it is one of the strongest gradients in the task
   (+4.35σ, survivors-only +6.62σ, positive in every world). Depositing in the
   control world is a dominant, densely-rewarded action.

2. **Run #14 converged to the pessimal corner.** never-deposit (−4.38) sits
   ~+10.35 (+4.35σ) below grounded and below even the regime-blind
   always-deposit (+5.77). So RL failed to climb a large, dense reward
   gradient — and failed to find even the trivial dominant strategy
   (always-deposit). This is a **credit-assignment / value-learning failure,
   not a task-reward gap**: depositing's reward (interest) is delayed and
   spread thin across ticks while the deposit action itself is wealth-neutral
   on the step, so the deposit decision may never be credited with the return
   it earns even under γ=0.99 (the S4 value/advantage candidate, never yet
   investigated).

3. **The one part that *is* small is the part the battery scores.** grounded
   and regime-blind always-deposit differ **only** in the cf cell, so the
   reward for the specific regime-**contingency** the battery measures is
   exactly the cf advantage: **+0.20σ, buried in noise**. The `sole_banker`
   calibration set the overall cf oracle advantage into the brain team's
   requested "+0.2–0.5σ, not large" band — but that same +0.20σ *is* the entire
   contingency gradient a learner would need to prefer grounded over
   always-deposit. A perfectly reward-maximizing policy has almost no incentive
   to be regime-contingent rather than just always depositing.

So the engine's intuition ("margins too small") was right about *which* margin
matters but wrong about *which side*: the control pull is huge; the
**cf-side contingency margin** is the +0.20σ needle.

## Two distinct, separable blockers this isolates

- **(learning)** Why does RL avoid the dominant always-deposit strategy and
  converge to the pessimal never-deposit arm? → credit assignment / value
  estimation on the delayed deposit→interest margin (brain-side S4).
- **(task)** Even a perfect learner sees only +0.20σ for regime-contingency
  over always-deposit. Widening the cf-side penalty (steeper demurrage) would
  raise the contingency margin — **a NEW pre-registration round**, per the
  engine's own note, not a silent knob turn. This script's `contingency_margin`
  field is the action-ceiling gate for that round: re-measure before spending a
  training run, target the contingency margin (not the raw cf advantage) into a
  "clears noise but not a giveaway" band while watching Goodhart.

## Caveat

One deterministic diagnostic on the heuristic floor and two scripted arms
(guardian, demurrage, `sole_banker`). It characterizes the reward landscape a
policy faces; it is not itself a training result. The credit-assignment claim
in reading (2) is the best-supported hypothesis it points at, not something this
run proves — that needs the brain-side value/advantage instrumentation.
