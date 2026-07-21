# run-25 — A1 grounded teacher: the teaching channel does NOT transmit grounding (POWERED-NO)

Run #20's candidate 1, the strongest diagnostic: replace the regime-**blind**
heuristic teacher with the regime-**aware** grounded heuristic
(`_GroundedHeuristicBrain`: deposits in control, RESTs under demurrage), told the
ground-truth regime each episode. Question — the **teaching-channel ceiling**:
can grounding transmit through BC *at all* when the teacher already has it?

Sandbox, sole-banker, `demurrage_per_day 0.25`, `adv_baseline episode`,
`bc_weight 0.3` **fixed** (do NOT anneal — keep the channel open),
`self_attempt_base 0.3`, `batch_every 64`, `memory null`, 200 episodes, seed
1000. Engine `7f29277`, brain `23aadaf`. Workflow run:
[29843690777](https://github.com/nori1234/sim_ai_agents/actions/runs/29843690777)
(full `battery.json` in artifact `grounding-battery-25`, id 8502643864).

## First — this is the run that survived (context)

Runs #21 (this same A1 config) and #22 (v1b) crashed to a silent heuristic
fallback: the developmental brain diverged numerically and `torch.multinomial`
raised on NaN logits, swallowed by `decide()`'s bare except. Root cause: the
state representation (`encode_state`) had no output norm, so training grew its
magnitude until value MSE / raw world-model curiosity / policy logits overflowed
— metastable, so an unlucky weight init diverged early and a lucky one limped to
~ep163 (the trainer never seeded RNG). Fixed brain-side (parameter-free
`F.layer_norm` on the state + skip-update-on-nonfinite guard) and engine-side
(seed RNG, surface the swallowed traceback). **This run trained all 200 episodes
clean** — the fix is CI-validated. See the fix commits (`23aadaf`, `c4debf4`).

## Result — a real negative, powered (fact)

- `grounded_CONFIRMED = False` — **POWERED-NO** (floor_regression powered:
  n=20, floor_spread_std 0.073). Not "undetermined": the battery had the power
  to detect grounding and it wasn't there.
- `fraction_grounded 0.00`, `sign_p 1.0`, `wilcoxon_p 1.0`,
  `bootstrap_ci [−0.626, −0.552]` (mean excess ≈ −0.59; weakest world −0.75).
- floor_regression: slope +0.029 (CI [−0.279, +0.294]), residual_sign_p 0.412,
  residual_wilcoxon_p 0.536 → `grounded=False`.
- **Raw attempts: control 167 / counterfactual 160** (ratio 1.04) — the policy
  deposits at essentially the *same* rate in both regimes. It should deposit
  far less under demurrage; it does not.
- Per-world control_rate ≈ 0.06–0.10, cf_rate ≈ 0.04–0.14, divergence ≈ ±0.03–
  0.07 (noise around zero) against a heuristic `floor_divergence` of 0.44–0.68:
  the grounded teacher's own policy separates strongly; the student's does not.

## Reading — the bottleneck is not the teacher

A teacher that *demonstrates the correct regime-contingent behaviour every step*,
told the hidden regime, with BC held open at 0.3, still did not transmit the
contingency. Two failures, not one:

1. **Contingency did not transmit:** control ≈ cf (167 vs 160) — the student
   never learns "deposit-less under demurrage".
2. **Even density barely transmitted:** control_rate stayed ~0.08 (low) — BC at
   0.3 didn't lift control-side deposits either; the policy sits near the
   not-deposit floor regardless of regime.

This is exactly run-20's pre-registered fail branch: **it implicates policy /
representation capacity, not the availability of a grounded teacher.** The next
step is the pre-registered POWERED-NO follow-up — *representation learnability*
(can the observation encoding + memoryless policy even express/learn
"demurrage → don't deposit"?) — not more metric tuning. Consistent with the
whole #17→#20 chain: the correct signal can exist in the gradient but the
individual-RL + BC channel can't win on it at this scale.

## Caveat introduced by the stabilization fix (must not be forgotten)

The `LayerNorm` that made training stable normalises the state representation,
which **discards its absolute magnitude**. If any part of the regime signal was
carried in the *scale* of `encode_state` (rather than the token pattern), this
run could not see it — a confound this specific config did not have before.
Mitigating argument: the regime shows up in the observation as a token-level
*pattern* (deposit/money trend), which LayerNorm preserves; and this POWERED-NO
matches the pre-stabilization runs #17–20, which had no such norm. Still, a
clean check (a with/without-norm control on a config that *did* ground, or a
decodability probe on the normalised state) is owed before treating "the
representation can't learn it" as settled. Filed as the first task of the
representation-learnability line.

## Also new this run

- **Reproducibility:** the trainer now seeds stdlib-random + torch from `--seed`,
  so this is the first fully deterministic run — re-dispatching the same inputs
  reproduces it exactly.
- The `--grounded-teacher` diagnostic is pipeline-validation by construction
  (the teacher cheats — it knows the hidden regime); it was never a grounding
  proof, and its failure is the informative outcome here.
