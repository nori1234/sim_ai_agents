# grid 41–48 — mechanism × task × belief-strength × evidence (first honest G2/D1 grid)

**Date:** 2026-07-22 · 8 parallel 200-episode runs, sandbox + sole_banker +
demurrage 0.25, scored on **G2** (money-matched, reflex-proof) and, for the B2
arms fired after the D1 wiring, **D1** (belief-decode accuracy). All 8 completed
successfully (backend live, guards passed — genuine trained runs, not fallbacks).

| # | mechanism | task | bw | days | G1 (counts) | G2 | D1 belief-acc | verdict |
|---|---|---|---|---|---|---|---|---|
| 41 | v2a | base | — | 20 | 1310/1241 (+0.03) | **+0.007** | — | UNDET / not grounded |
| 44 | v2a+B2 | C1 | 0.5 | 20 | 186/218 (−0.079) | ≲0 (anti) | — | UNDET / not grounded |
| 45 | v2a+B2 | C1 | 1.0 | 20 | 103/100 (+0.015) | ~0 | **0.501 (chance)** | UNDET / not grounded |

*(42/43/46/47/48 pulled to the same picture — G2 ≈ 0, UNDETERMINED; full numbers
recoverable from the `grounding-battery-NN` artifacts / job logs.)*

## The decisive finding — the wall is (3) INFERENCE, not (4) actuation

D1 (#45) is the clincher: with the belief head **directly supervised** by the
ground-truth regime (BCE, belief_weight 1.0), its eval decode accuracy is **0.501
— pure chance — and mean belief is ~0.50 in *both* regimes** (control 0.5004, cf
0.5009). The belief never separated the regimes at all.

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

## Next (design now confirmed by D1)

The agent needs a **direct running signal of experienced consequence** — "did the
deposits I made *this episode* shrink?" — not a recall of similar past states.
That is **A1**: a recurrent belief state updated online from the (obs, action,
reward / wealth-delta) stream and fed to the policy (and, cheaply, still
supervised by `_priv` as B2 does, but now over an input that actually carries the
regime). D1 will re-measure whether A1's belief crosses chance — the direct test
that (3) is finally solved before touching (4).

Priority: **A1 (recurrent consequence belief) → re-check D1 belief-acc > 0.5 →
only then actuation.** Do not re-run B2/critic variants on the recall substrate —
D1 shows the signal isn't there to read.
