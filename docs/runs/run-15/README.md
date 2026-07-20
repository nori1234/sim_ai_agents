# run-15 — the calibrated-margin run: POWERED-NO, grid branch 2, task dial closed

The single pre-registered training run of
`docs/proposals/run15-contingency-margin.md`: identical to run #14 in every
respect (sandbox, `sole_banker=true`, hparams `bc_weight 0.3` /
`self_attempt_base 0.3` / `batch_every 64`, pool seeds 1000–1015, battery on
held-out 42–61, verdict rules unchanged) except the one calibrated change —
**`demurrage_per_day = 0.25`** (contingency margin +0.60, +0.53σ, vs run #14's
+0.21, +0.20σ; see `contingency-calib-1`).

Workflow run: [29744258171](https://github.com/nori1234/sim_ai_agents/actions/runs/29744258171)
(the config line in `full_log.txt` confirms `demurrage_per_day=0.25` was in
effect for training AND battery). `battery.json` is the exact JSON the battery
step printed.

## Result (fact)

- 60 episodes, never `is_stable` (streak 0); probe excess **flat at
  −0.37..−0.51 from ep 5 through ep 60** — no learning trajectory, same shape
  as run #14.
- Battery: `mean_excess −0.5775`, `bootstrap_ci [−0.617, −0.515]` (entirely
  negative), `wilcoxon_p 0.9999`, `n_conclusive 17/20` (up from run #14's 14),
  `floor_regression` **powered** (n=17, `residual_wilcoxon_p 0.573`, slope CI
  spans 0) → per-rule `grounded_confirmed = False` — **POWERED-NO**. (Battery-
  level conjunction `None` from the 3 no-attempt worlds, as in run #14.)
- **Raw attempts: control 19 / counterfactual 18, summed over all 20 held-out
  worlds** — versus the heuristic floor's dense depositing (~102
  control-world deposits per episode). The policy is still the
  regime-independent **never-deposit** arm.
- New visibility: `teacher_frac_in_batch` now prints every episode
  (~0.20–0.61, typically ~0.4–0.5) — during training, roughly half of each
  batch is behaviour-cloning toward the densely-depositing teacher, and the
  final policy still lands on never-deposit.

## Interpretation (pre-registered — grid branch 2, no discretion)

The proposal §6 grid, fixed before the run:

> **POWERED-NO, policy still ~never-deposits** → incentive was not the blocker
> even widened → **S4 credit assignment is the primary suspect**; the task
> dial is **closed** (no further rate raises) → brain-side value
> instrumentation: per-decision advantage/return for deposit vs non-deposit,
> split by regime.

That is this outcome, exactly. What the run adds beyond run #14: the collapse
to never-deposit now cannot be attributed to a thin contingency incentive —
the margin was calibrated above the noise floor (+0.53σ) and the control-side
pull toward depositing was already huge (+4.35σ, `control-margin-1`), yet the
policy sat at the pessimal corner for 60 episodes with a flat probe curve.
A policy that never tries the densely-rewarded arm, under BC pressure that
demonstrates it ~half of every batch, is not failing for lack of reward — the
value/credit side is failing to translate a large, dense, delayed reward into
the deposit decision. Per the one-round rule (§7), no further demurrage-rate
changes: the program moves to the brain side's S4 instrumentation.

One engine-side observation for that discussion (not a new registration): BC
targets that deposit densely are present in ~half of each training batch, yet
the learned policy deposits ~once per world. Whatever the S4 instrumentation
finds about advantage noise, the brain team may also want to check how BC
targets and PG/value gradients interact on the deposit action specifically —
they appear to be pulling in opposite directions.
