# run 28–30 — memory-coverage × privileged-critic ablation (nested)

A pre-registered nested ablation (`agent_agi/docs/10`), same seed (1000), one lever
added at a time, `grounded_teacher=false`, sole-banker sandbox, `demurrage 0.25`,
`adv_baseline=episode`, 200 episodes. Only the named hparam differs at each step.

| run | added lever | deposits control / cf | direction | excess CI (mean) | verdict |
|---|---|---|---|---|---|
| **28** v1b | memory, `obs_hash` (recall ~6%) | 224 / 240 | ✗ wrong (0.93) | [−0.622, −0.547] (−0.58) | POWERED-NO |
| **29** v2a | + `state_lsh` 12-bit (recall ~66%) | **1026 / 854** | **✓ right (1.20)** | **[−0.502, −0.413] (−0.46)** | POWERED-NO |
| **30** v2a + critic | + `privileged_critic` | 293 / 341 | ✗ wrong (0.86) | [−0.690, −0.574] (−0.63) | POWERED-NO |

Artifacts: `grounding-battery-28/29/30`. Runs
[29875948448](https://github.com/nori1234/sim_ai_agents/actions/runs/29875948448),
[29876004366](https://github.com/nori1234/sim_ai_agents/actions/runs/29876004366),
[29876746397](https://github.com/nori1234/sim_ai_agents/actions/runs/29876746397).

## Result vs pre-registration

- **Goal not reached**: all three `grounded_confirmed = False`, POWERED-NO.
- **v2a (memory coverage) — first-primary CONFIRMED.** For the first time in the
  program the policy separated deposits in the **grounded direction** (control
  1026 > cf 854; every prior run was ratio ≈ 1 or wrong), with 4× the deposit
  density and the best excess (+0.12 vs baseline). Exactly the registered
  prediction: coverage is **necessary** (it moved behaviour) but **not sufficient**
  (still POWERED-NO).
- **Privileged critic — prediction REFUTED.** It did not close grounding; it
  **regressed** — direction flipped wrong, density collapsed 4× (1880 → 634),
  excess worst of the three. The naive "dense+pure credit ⇒ grounds" hypothesis is
  false for this critic as implemented.

## Experimental-conditions audit (were the premises clean?)

1. **Design cleanliness — OK.** All three: seed 1000, identical flags, only the
   one intended hparam differs per step (`memory_key_mode`, then
   `privileged_critic`). Same engine + brain + agent_agi branch tip. So the
   deltas are attributable to the named lever, not noise/config drift.
2. **Determinism — OK.** RNG is seeded (run #25 fix); the recall perf fix
   (O(n)→O(tag)) is byte-identical to the old scan (tested). #28 is a faithful,
   faster re-run of the cancelled #27.
3. **Critic was actually active — OK (not a silent no-op).** #30 differs sharply
   from #29 (control 293 vs 1026) with only `privileged_critic` added, so the
   critic path ran and *caused* the regression. Why it hurt is unknown and needs
   the probe fields (deposit used-advantage under the critic) — do NOT re-run the
   critic before diagnosing.
4. **⚠ Metric-fairness finding (the important one).** `excess = policy_divergence
   − floor_divergence`, both **absolute** rate differences. The floor is the
   regime-BLIND heuristic, whose divergence is **~0.5** — not because it "knows"
   the regime, but mechanically: under demurrage, savings shrink, its
   `money≥12 → deposit` rule fires far less, so it deposits far less. The learned
   policy deposits at ~10× LOWER density, so its *maximum possible* absolute
   divergence is bounded well below 0.5 **regardless of how contingent it is**.
   ⇒ A sparse-deposit policy is **structurally unable** to clear excess>0, even if
   perfectly regime-contingent. This means POWERED-NO here reflects the metric's
   density-dependence as much as the brain: v2a's gain came largely from raising
   density *toward* the floor, and its correct *direction* (ratio 1.20) is
   invisible to an absolute-divergence excess. The contingency signal is real
   (ratio), but the current excess can't credit it without matching density.

## Next verification strategy (organised)

Cheap-first, each falsifiable:

1. **Metric side — density-controlled contingency (no new training).** Re-analyse
   the existing battery.json per-world with a density-normalised measure: the
   deposit-rate **ratio** control/cf, and/or a logistic `deposit ~ regime` odds
   ratio, and/or a **density-matched floor** (throttle the heuristic to the
   policy's deposit rate before differencing). Question: is v2a *already* more
   grounded than absolute excess shows? This checks whether the wall is the brain
   or the metric. Do this **first** — it may reframe #29 as a partial pass.
2. **Memory side — push density while keeping contingency (training).** v2a is the
   one lever that worked; the path to absolute excess is more deposit density in
   control without smearing. Levers: finer LSH (16–24 bit, less regime smearing) +
   higher `entropy_weight`/novelty on deposit (raise control-side density). Pre-reg:
   control−cf both up, ratio stays >1, excess climbs toward 0.
3. **Critic side — diagnose before re-use.** Read #30's probe log (deposit
   used-advantage, per-segment) to see how the critic collapsed density (likely it
   over-denoised the baseline and killed the advantage that drove deposits). Fix
   (e.g. mixed privileged/non-privileged baseline; critic on value-loss only) or
   shelve. Do NOT re-run the critic until understood.
4. **Task side (if 1 shows the metric is fair and 2 stalls).** The ~0.5 mechanical
   floor may make this task structurally require heuristic-level density; revisit
   whether the sandbox rewards a *sparse* grounded policy (deposit-oracle redesign
   lineage), rather than tuning the brain against an unbeatable floor.

Priority: **1 (metric re-analysis) → 3 (critic diagnosis) → 2 (v2a density push)**.
1 and 3 are cheap and decide whether v2a already won on a fair metric and why the
critic failed, before spending more training compute.
