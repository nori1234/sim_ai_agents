# grid 41–48 — mechanism × task × belief-strength × evidence (first honest G2/D1 grid)

**Date:** 2026-07-22 · 8 parallel 200-episode runs, sandbox + sole_banker +
demurrage 0.25, scored on **G2** (money-matched, reflex-proof) and, for the B2
arms fired after the D1 wiring, **D1** (belief-decode accuracy). All 8 completed
successfully (backend live, guards passed — genuine trained runs, not fallbacks).

Read/verified directly from job logs: **all 8 (#41–#48).**

| # | mechanism | task | bw | days | rate ctl/cf | G1 (counts) | G2 (matched) | D1 belief-acc (ctl/cf) | verdict |
|---|---|---|---|---|---|---|---|---|---|
| 41 | v2a | base | — | 20 | .665/.660 | 1310/1241 (+.027) | **+0.007** | — | UNDET |
| 42 | v2a+B2 | base | 0.5 | 20 | .258/.218 | 606/517 (+.079) | **~+0.040** (best) | — (pre-D1) | **POWERED-NO** |
| 43 | v2a | C1 | — | 20 | .085/.083 | 212/210 (+.005) | ~0 | — | UNDET |
| 44 | v2a+B2 | C1 | 0.5 | 20 | low | 186/218 (−.079) | ≲0 | — (pre-D1) | UNDET |
| 45 | v2a+B2 | C1 | 1.0 | 20 | low | 103/100 (+.015) | ~0 | 0.501 (.500/.501) | UNDET |
| 46 | v2a+B2 | C1 | 0.5 | **40** | low | 186/201 (−.038) | ~0 | 0.523 (.521/.523) | UNDET |
| 47 | v2a | C1 | — | **40** | .163/.164 | 411/439 (−.033) | ~0 | — | UNDET |
| 48 | v2a+B2 | base | 1.0 | 20 | .258/.218 | 392/422 (−.036) | ~0 | 0.473 (.511/.504) | **POWERED-NO** |

**No arm grounds.** Best is #42's G2 ≈ +0.04 — still ~13× below the floor (+0.52),
POWERED-NO. Baseline arms deposit densely at ~equal rates both regimes (replay);
C1 arms collapse density (2–16%) because the reflex is removed and nothing
regime-aware replaces it. belief_weight (0.5→1.0), days (20→40), and task did not
move grounding.

## The finding — the wall is (3) INFERENCE, not (4) actuation (all 3 belief arms)

The three D1 arms are the clincher, and they **agree**: with the belief head
**directly supervised** by the ground-truth regime (BCE), eval decode accuracy is
**0.501 / 0.523 / 0.473 — all ≈ chance — and mean belief is essentially identical
across regimes in every case** (e.g. #45 control 0.500 vs cf 0.501). The belief
never separated the regimes, across both tasks, both belief weights, and days 20
vs 40 (more within-episode evidence moved it only +0.02).

So the memory-recall features (`mean_reward_deposit` etc., the v2a substrate) **do
not carry within-episode regime evidence in a form the head can read** — even when
we pay it to. This localises the wall precisely:

- **(3) inference FAILED** — the agent cannot form a belief about the hidden
  regime from the recall-based memory features. (Likely because eval memory is
  sparse/episode-local and recall does not surface *this* episode's demurrage
  hits as a usable signal.)
- Therefore **(4) actuation is not the bottleneck to attack next** — there is no
  belief to actuate. Reward-shaping / critic fixes would be premature.

Corollary reads:
- **C1 worked exactly as designed** (torch-free proof held up): the task is
  reflex-impossible, so the policy could not fall back on a wealth reflex — and,
  unable to infer the regime, it simply **collapsed deposit density** (#44: fire
  rates 1.5–19% vs #41's 66%) rather than grounding. C1 correctly converts
  "can't ground" into "can't score", which is the honest outcome.
- **B2 (supervised belief-into-policy) is inert** because its input can't predict
  the target. B2's design is sound; its *substrate* (recall features) is empty of
  the signal.
- More belief pressure (bw 0.5→1.0) and more within-episode evidence (days 20→40)
  did not move it — consistent with an inference *substrate* failure, not a
  tuning failure.

## ROOT CAUSE (from analysing all 8) — state-keyed recall is regime-BLIND by construction

Why is the belief at chance even under direct supervision? Because its input is
`_memory_feature` = a summary of **recall keyed on the state-LSH bucket**. The
regime is *hidden* and, by design, **not in the state** — so states from control
and counterfactual hash to the **same bucket**, and recall returns a **mixture of
both regimes'** deposit outcomes. `mean_reward_deposit` is therefore
regime-*averaged* → it carries ~zero regime signal → the belief head has nothing
to separate on (D1 = chance, confirmed on all 3 arms).

**This is structural, not a tuning miss.** Any belief built on state-similarity
recall is doomed: the very thing that makes recall generalise (state-similarity,
regime-agnostic) erases the regime. Grounding here cannot come from "what happened
in similar states" — it needs "what happened to **me, this episode**", a *temporal*
signal recall does not provide. It also explains the density collapse on C1: with
the wealth reflex removed and no regime signal available, the policy has no basis
to deposit and mostly abstains.

## Means to pursue (ranked; the goal is a belief that separates the regime, D1>0.5)

- **M1 — A1 recurrent belief (direct fix).** An online state updated each tick
  from the *(obs, action, reward/wealth-delta)* stream, fed to the policy,
  supervised by `_priv`. Carries this-episode consequence history that recall
  can't. Cost: hot-path brain surgery (BPTT), highest value. **Primary.**
- **M2 — expose the felt delta in the observation (engine-side, cheap).** The
  demurrage loss currently reaches the agent *only* as a memory text line
  (`remember(...)`), not as a structured obs field the tokenizer surfaces. Add
  "my deposit balance changed by X since last tick" to the observation. Then even
  a memoryless policy can *react to the felt consequence* (肌感覚 / feedback
  control) — which may be sufficient for G2>0 without hidden-regime inference.
  Tests whether grounding here is "react to felt loss" (simpler) vs "infer hidden
  regime" (harder). Cheapest discriminator; do **first**, before M1.
- **M3 — regime-tagged memory is NOT allowed** (it would leak the hidden regime
  into recall = cheating). Explicitly ruled out.

Priority: **M2 (expose felt delta — cheap, may suffice) → M1 (A1 recurrent) →
re-check D1>0.5 → only then actuation (4).** Do not re-run B2/critic variants on
the recall substrate — the signal is provably not there to read.
