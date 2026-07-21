# run-20 — BC annealing: necessary but not sufficient; self-play PG alone is too thin

Run #19's spec + `bc_weight_decay_steps 300`, `bc_weight_min 0.05` (brain
`a7f7ddc`): wean the policy off the regime-blind teacher so the now-correct
(episode-baseline) PG gradient can take over. Workflow run:
[29795755921](https://github.com/nori1234/sim_ai_agents/actions/runs/29795755921).

## The anneal fired as designed (fact)

`bc_weight_effective`: 0.30 (ep1) → 0.09 (mid) → 0.05 (last third). The teacher's
pull was removed on schedule.

## But grounding did not appear underneath it (fact)

- Battery: `grounded_confirmed = False` — POWERED-NO. `mean_excess −0.573`,
  `n_conclusive 20/20`, `fraction_grounded 0.0`.
- Regime propensity separation stayed at **noise level all run**: control−cf
  = −0.0016 / +0.0016 / +0.0024 across training thirds (post-wean included).
- Deposit density **fell** as BC decayed (segment propensity 0.078 → 0.040;
  last-5 ~0.06) — BC had been propping the density up.
- Used advantage last third: control −0.276 vs cf −0.316 — the regime order is
  right but both are negative (weaning pushed the whole deposit advantage down).
- Battery raw attempts control 74 / cf 57 (ratio 1.30, better-ordered than
  #19's 359/348 = 1.03, but ~5× less total).

## Reading — the individual-RL ladder is largely exhausted

Removing the blind teacher's scaffolding did not reveal a grounded policy; it
revealed that the **self-play PG signal alone is too thin** to carve out the
contingency. The chain across runs #17→#20 is now clear and each link is
measured:

- #17: PG credit is noise-dominated (deposit used-adv ≈0) → variance too high.
- #18/#19: episode baseline fixes the variance; PG credit becomes regime-correct
  (cf −0.12) — but BC toward the blind teacher out-pulls it, behaviour flat.
- #20: remove BC's pull → behaviour still doesn't differentiate, density drops.
  The correct PG signal exists but is too sparse (~2 deposit self-samples/batch)
  and small to drive a differentiated policy on its own.

So the regime-blind teacher (issue #10 R2) can bootstrap *density* but not the
*contingency*, and self-play PG can represent the contingency in its gradient
but can't yet *win* on it. Neither channel, alone, closes grounding on this
task at this scale.

## Next candidates (each falsifiable in the probe fields, none yet run)

1. **Grounded scripted teacher (issue #10(c)) — the strongest diagnostic.** A
   teacher that deposits *conditionally on regime* (the blind heuristic + a
   regime branch). Measures the **teaching-channel ceiling**: does grounding
   transmit through BC at all when the teacher has it? A pass narrows the whole
   problem to "we need a grounded teacher (eventually an LLM/human), the pipeline
   is sound"; a fail implicates policy/representation capacity, not the teacher.
   Explicitly a pipeline-validation, not a grounding proof (it teaches the answer).
2. **Stronger exploration** — entropy up and/or a count-based novelty bonus on
   the deposit verb, to raise the self-sample count PG learns from.
3. **Temporal context (the memory/body integration)** — a memoryless policy
   sampling ~2 deposits/batch has almost no signal about a consequence that
   unfolds over ticks; persistent episodic memory (the locked `agent_agi`
   integration, `agent_agi/docs/09`) is the structural answer to exactly this
   thinness, and would also serve the longer-term body/equipment direction.

The rate dial stays closed at 0.25 (one-round rule). The choice among 1–3 is a
strategy call, not another autonomous knob turn.
