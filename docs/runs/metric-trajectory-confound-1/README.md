# metric-trajectory-confound-1 — the grounding metric rewards a memoryless reflex

**Date:** 2026-07-22 · torch-free, seeds 42–47, sandbox + sole_banker +
demurrage 0.25 (the CI battery's world). Raw output: `threshold_landscape.out`,
`money_matched.out`. Scripts: `scripts/threshold_landscape.py`,
`scripts/money_matched_contingency.py`.

## Why this exists

The 9-run memory×critic sweep (runs #28–#40) closed with the whole
memory+credit family plateaued at `norm_contingency ≈ +0.09`, ~6× below the
blind floor's `+0.518`. Before spending another 3.5 h CI run pushing that
family, two cheap torch-free diagnostics asked *what the floor's +0.518
actually is*. The answer invalidates the success criterion the program has been
chasing.

## Finding 1 — the floor is a memoryless wealth reflex, and a *higher* reflex beats it

The blind floor is `deposit iff money >= 12` (see
`_grounded_heuristic_brain_class`). Sweeping that single threshold `T`:

| T | control | cf | density | norm_contingency |
|---|---|---|---|---|
| 8–12 | 645 | 199 | 844 | **+0.528** (the floor as-built) |
| 16 | 474 | 93 | 567 | +0.672 |
| **20** | 363 | 60 | 423 | **+0.716** |
| 24 | 240 | 45 | 285 | +0.684 |
| 30 | 157 | 32 | 189 | +0.661 |

A memoryless reflex with `T=20` scores **+0.716**, i.e. `norm_excess = 0.716 −
0.528 = +0.19 > 0` — it **beats the floor**. So *"beat the floor"* (the
program's `excess > 0` / `grounded_confirmed` criterion) **does not require
grounding**: raising a hard-coded wealth threshold passes it, with zero regime
knowledge and zero memory.

**Sharpness is not the wall either.** Replacing the hard step with a stochastic
sigmoid `P(deposit|money)=σ((money−12)/τ)` — the soft boundary an RL policy
realistically learns — holds `norm_contingency ≈ +0.45` across *every* τ from
0.5 to 16 (a boundary so fuzzy it spans the whole money range). A soft
memoryless reflex still scores 5× the learned policy's +0.09.

## Finding 2 — the floor's asymmetry is 100% trajectory divergence, 0% regime detection

Binning every deposit *decision* by the agent's current money (`money_matched.out`):

| money bin | control n | ctl fire% | cf n | cf fire% | rate gap |
|---|---|---|---|---|---|
| [12,16) | 244 | 100% | 135 | 100% | **+0.0%** |
| [16,20) | 179 | 100% | 25 | 100% | **+0.0%** |
| [20,28) | 147 | 100% | 6 | 100% | **+0.0%** |
| [40,+) | 72 | 100% | 30 | 100% | **+0.0%** |

Within *every* wealth bin the floor deposits at the **identical rate in both
regimes** (gap ≡ 0). The whole +0.53 asymmetry is the bin *populations*
shifting poorward in cf (control has 645 decision points spread across wealth;
cf has 199 crushed into the low bins) — because demurrage drained the cf agents,
not because the rule knows the regime.

This generalises: **a policy that is a pure function of *money* fires at the
identical rate within a money bin regardless of regime, so its money-matched gap
(call it G2) is ≡ 0 by construction.** `norm_contingency`/`excess` measure the
*rate* difference (call it G1), which a reflex produces mechanically via divergent
wealth trajectories. G1 certifies nothing about grounding.

*Caveat, measured after implementation:* the canonical floor is
`HeuristicBrain`, whose `decide()` gates banking behind survival/energy
priorities that vary with the (regime-divergent) trajectory — so its real G2 is a
small **residual** (≈ −0.05 to −0.09 on seeds 42–47), not a clean 0. It is an
order of magnitude below its G1 (+0.52) and, crucially, **never positive**. The
load-bearing property is the *discriminator* below, not an exact zero.

## Consequence — the right target is G2 (money-matched contingency)

- **G1** = raw regime rate difference (`norm_contingency`, `excess`). Reflex-
  achievable; a higher hard threshold beats the floor. **Not** grounding.
- **G2** = within-wealth-bin deposit-rate gap (control − cf at matched money).
  **≤ ~0 for a memoryless policy** (≡0 for a pure-money rule; a small negative
  residual for the real `decide()`-gated floor); `> 0` *only* if the agent
  conditions on within-episode history — i.e. it inferred the punishing regime
  from experienced demurrage and suppressed deposits at wealth it would have
  banked in control. That is exactly "an agent grounded in irreversible
  consequence, not replaying training" — the program's north star.

**The discriminator (measured, `emergence/grounding.py`, seeds 42–47):**

| policy | G1 (norm_contingency) | G2 (money-matched) |
|---|---|---|
| pure-money threshold T=12 | +0.523 | −0.092 |
| pure-money threshold T=20 | +0.639 | −0.005 |

Raising the threshold **inflates G1 (+0.52→+0.64) but does not raise G2** (stays
≤0). So the reflex exploit that games G1 leaves G2 flat: **positive G2 is not
reflex-reachable.** That is what makes it a fair, honestly floor-beating grounding
target. The memory family (v2a) is the right tool — it was just being scored on
G1, where a blind reflex already wins, instead of on G2, which needs exactly what
memory provides. G2 is now implemented (`measure_money_matched_contingency`,
tested in `tests/test_money_matched_contingency.py`).

Why the learned policy sits at G1 +0.09 (below the reflex): per-tick, money does
**not** disambiguate the hidden regime — depositing at money=30 gains in control
and loses in cf, so under a ~50/50 training mix the net advantage of
"condition-deposit-on-money" washes toward 0 and RL has weak incentive to learn
even the reflex. The floor doesn't face this: it's a *fixed* rule whose rate
diverges for free.

## Next verification strategy

1. **G2 metric — DONE.** `money_matched_contingency` (pure scorer) +
   `measure_money_matched_contingency` (sandbox probe) in
   `emergence/grounding.py`, tested in `tests/test_money_matched_contingency.py`
   (pure-function invariants, floor-not-positive, threshold-immunity). Additive,
   determinism-safe (baseline suite still byte-identical, 99 passed).
2. **Re-score the target on G2, not G1.** Wire G2 into the neural probe/battery
   and measure the v2a memory policy's G2 in one run — the first honest grounding
   number. Positive G2 (with CI) = genuine grounding; G1 becomes a diagnostic,
   not the verdict.
3. **If G2 is stuck at 0**, the lever is within-episode regime *evidence*
   (memory of demurrage hits → suppress matched-wealth deposits), not more G1
   density. This is where memory/inference is load-bearing and a reflex is not.

Priority: **1 (implement G2) → 2 (measure v2a G2) → 3 (drive G2 with memory).**
