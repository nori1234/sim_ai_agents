# c1-reflex-impossible-1 — a task where a memoryless reflex cannot score

**Date:** 2026-07-22 · torch-free validation. Raw: `prototype.out`. Repro (real
engine flag): `run_grounding_probe(..., sandbox=True, sole_banker=True,
demurrage_per_day=0.25, stable_income=20)`; concept prototype:
`scripts/c1_prototype.py`.

## Why

`metric-trajectory-confound-1` proved the blind floor scores G1 +0.52 (and beats
it at T=20) with **zero regime knowledge**: its whole asymmetry is trajectory
divergence — demurrage drains cf agents poorward, so the fixed `money>=12` rule
fires less there. The regime leaks into the reflex's handle (observable spendable
money). C1 plugs that leak.

## Mechanism (`stable_income`, engine flag, default 0 = off = byte-identical)

At each day's end (after interest/demurrage) reset every saver's spendable money
to a fixed target. The deposit decision is then faced at a **regime-invariant**
money level, so the wealth-threshold reflex has no handle. Demurrage still shrinks
the **banked** claim (real loss, felt via the `remember()` line + the total-wealth
reward), so the regime is inferable **only** from that shrinkage history — exactly
the design's stated intent (`_apply_demurrage`: "the only channel by which an
agent can discover the rule"), now actually enforced.

## Result (seeds 42–47, days 20, demurrage 0.25, income 20)

| policy | task | G1 (norm_contingency) | G2 (money-matched) |
|---|---|---|---|
| blind floor | baseline | +0.528 | −0.092 |
| **blind floor** | **C1 (income 20)** | **+0.001** | **−0.005** |
| regime oracle (knows regime) | baseline | +1.000 | +0.847 |
| **regime oracle** | **C1 (income 20)** | **+1.000** | **+0.661** |

Confirmed independently through the real probe path (`run_grounding_probe` +
`measure_money_matched_contingency`, seeds 42–47): floor **G1 +0.000, G2 −0.004**.

## Read

- **The reflex is dead on C1.** Blind floor G1 and G2 both collapse to ~0 — a
  memoryless wealth rule cannot produce *any* regime asymmetry, on either metric.
- **The task stays grounding-solvable.** A regime-aware oracle still scores G1
  +1.0 / G2 +0.66. So C1 does not make the task impossible — it makes it
  **impossible without knowing the hidden regime**, which (since the regime is
  hidden) requires within-episode inference from experienced consequence.
- ∴ On C1, *any* positive score — even on the reflex-passable G1 — is genuine
  grounding. C1 converts grounding from a post-hoc measurement (G2) into a
  property the task itself enforces. It is the task-side complement to the
  metric-side G2.

Caveat: `stable_income` is deliberately **not** coin-conserving (a curriculum
artifact of the sandbox, not the conserved offline baseline). It touches only the
sandbox and is inert (byte-identical) at 0; `tests/test_baseline_contract.py` +
the grounding suite pass unchanged (111).

## Next

Fire v2a and v2a+B2 on C1 (`stable_income=20`) alongside the baseline-task runs
(#41/#42). On C1 the G1 column becomes a *valid* grounding read too (floor 0), so
even a policy that only moves G1 there is grounding — a lower bar to first signal
than money-matched G2 on the leaky baseline task.
