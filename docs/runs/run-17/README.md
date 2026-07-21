# run-17 — the first VALID fair-task run: POWERED-NO, and the S4 mechanism quantified

Same pre-registered spec as runs #14/#15 (sandbox, `sole_banker`,
`demurrage_per_day=0.25`, brain hparams `bc_weight 0.3` / `self_attempt_base
0.3` / `batch_every 64`, pool 1000–1015), on the **fixed driver** (measured
brain = the sandbox saver `a2`; config line confirms) with the S4 probe on
(`llm_model_agi@ead7e35`). The first run in the program's history where the
battery-evaluated brain actually faced the deposit decision on the fair task.

Workflow run: [29783987813](https://github.com/nori1234/sim_ai_agents/actions/runs/29783987813).
`probe_analysis.txt` = `scripts/analyze_probe_log.py` over `full_log.txt`.

## Battery (fact)

`grounded_confirmed = False` — **POWERED-NO, and this one is valid.**
`n_conclusive = 20/20` (first time ever; no inconclusive worlds),
floor-regression powered (n=20, `residual_wilcoxon_p 0.493`),
`mean_excess −0.606` (CI [−0.665, −0.548]), `fraction_grounded 0.0`.
Raw attempts: **control 91 / cf 117** — ~6× run #15's checkpoint, deposits in
*both* regimes, slightly more under demurrage (negative divergence).

## What the probe showed (fact) — a completely different brain than run #16's

- **BC works now:** `probe_teacher_n = 112` over the run (the saver's teacher
  demonstrates deposit constantly; the banker's was 0), `probe_self_n = 66`.
- **The policy learned to deposit in general:** propensity rose ~0.031 → ~0.083
  (uniform floor ≈ 0.021; ~4× over uniform, still rising at ep60).
- **But regime-INDEPENDENTLY:** segment-joined propensity control 0.0525 vs
  cf 0.0522 — identical to three decimal places.
- **The decision-level credit is noise-dominated and even inverted:** deposit's
  raw (G−V) segment-joined credit reads control +22.5 vs **cf +28.7** — the
  *wrong* order — because per-batch returns swing ±10..±140 while the entire
  regime-contingent payoff is ±0.60 per episode. After batch advantage
  normalization, deposit's mean used advantage is **−0.0116** — the PG signal
  on the scored decision is statistically zero.

## Reading (the quantified S4 answer)

The task's contingency margin was calibrated to +0.53σ **in oracle-return σ
across worlds** (per-world std ~1.13). The learner never sees that σ: at the
decision level it faces the **within-batch advantage spread (tens of reward
units)** — a third σ convention, and the one that actually gates learning. The
contingency signal (+0.60/episode, diluted over ~30 decisions and 64-step
batches) is on the order of **1% of the credit noise**. BC (regime-blind
teacher, issue #10's R2) can and did teach *deposit-in-general*; nothing in
the gradient can currently resolve *deposit-conditionally-on-regime*. That is
the S4 mechanism, measured rather than suspected: **not mis-attribution — the
credit is regime-orderable in principle but drowned, ~2 orders of magnitude
under the batch noise floor.**

Honest caveats: `trained_stable=False` and propensity was still rising at
ep60 — a longer run is cheap and untried on the fixed driver; and the
cf>control attempt asymmetry (117 vs 91) is unexplained and worth a look
before it's called anything.

## Next (brain-side, with numbers to aim at)

Classical variance-reduction on the deposit margin, e.g.: per-episode return
centering / regime-aware value baseline (legitimate — the value head already
sees the regime-decodable representation; this is not writing the answer into
the reward), paired/counterfactual advantage estimation (the mirror can
prototype), or simply more episodes at lower entropy. The probe fields make
every candidate falsifiable: success = `probe_adv_used_mean` on deposit
separating by regime segment, before any battery is spent.
