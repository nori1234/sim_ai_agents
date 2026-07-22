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
4. **Metric-fairness concern — RAISED, then REFUTED by direct measurement.**
   The worry: `excess` is an ABSOLUTE rate difference, so a low-density policy
   might be unable to clear the dense mechanical floor even if perfectly
   contingent. To test it we added a **density-INVARIANT** metric,
   `norm_contingency = (control_rate − cf_rate)/(control_rate + cf_rate)` ∈ [−1,1]
   (`emergence/grounding.py`), and measured the blind floor's value directly
   (torch-free, seeds 42–47): **floor norm_contingency = +0.518** (it deposits
   control≈0.87 / cf≈0.28 — a huge *mechanical* swing as demurrage drains money
   below the deposit threshold). The policies' normalized asymmetry:
   **#28 −0.034, #29 (v2a) +0.091, #30 −0.076.** So even normalized for density,
   v2a (+0.091) is ~5× **below** the floor (+0.518); `norm_excess ≈ −0.43`.
   ⇒ **The metric is fair** — normalization does NOT rescue v2a. The learned
   policy is genuinely far less regime-contingent than a blind mechanical rule;
   it has merely *started* moving the right way. POWERED-NO is the brain, not the
   metric. (`norm_contingency` is kept as a fair, density-invariant diagnostic
   that cleanly ranks the levers — v2a is the only positive — but it does not
   change the verdict.)

## Next verification strategy (organised, post-audit)

The audit resolved the metric question: **the metric is fair** (item 4). The real
gap is that the policy's regime asymmetry (v2a +0.091) is ~5× short of the blind
mechanical floor (+0.518). So the task is not "credit v2a with a fairer metric"
but **make the policy far more strongly regime-contingent** — and understand why
the one mechanism built to do that (the critic) backfired. Cheap-first:

1. **Critic diagnosis (cheap, decisive) — do first.** The critic was the mechanism
   meant to inject strong regime-contingent credit, and it *reduced* contingency
   (norm −0.076) and collapsed density. Read #30's per-episode probe log
   (`probe_adv_used_mean` on deposit, per segment) vs #29's to see what the
   privileged baseline did to the advantage — most likely it over-denoised
   (V_priv absorbed the regime signal, leaving A≈0 on deposit, so the policy lost
   its push). Candidate fixes: a **mixed** baseline (blend privileged and
   non-privileged V so advantage keeps a regime-contingent component), or feed the
   privileged signal as a **shaped reward** term rather than only the baseline.
   Don't re-run the critic before this.
2. **Memory density-push (training).** v2a is the only positive lever. Raise its
   contingency toward the floor: finer LSH (16–24 bit → less regime smearing, so
   recall separates regimes better) + more control-side deposit density
   (entropy/novelty on deposit). Pre-reg: `norm_contingency` climbs from +0.091
   toward +0.5; `excess`/`norm_excess` climb toward 0.
3. **Task side (if 1–2 stall).** The blind floor's +0.52 asymmetry is a very high
   bar — a grounded policy must out-modulate a rule that already halves its
   deposits under demurrage mechanically. Revisit whether the sandbox can reward a
   *sparse* grounded policy, or whether the deposit-oracle-redesign lineage needs
   another turn so grounding pays without requiring heuristic-level density.

`norm_contingency`/`norm_excess` are now reported by every probe
(`emergence/grounding.py`) as a fair, density-invariant companion to `excess` —
they rank levers cleanly (v2a is the only positive) and will track whether (2)
actually raises contingency, not just density.

Priority: **1 (critic diagnosis) → 2 (v2a density push) → 3 (task)**.
