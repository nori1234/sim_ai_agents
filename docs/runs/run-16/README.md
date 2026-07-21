# run-16 — the first S4-instrumented run: the probe found the real defect in one shot

Run #15's spec exactly (sandbox, `sole_banker`, `demurrage_per_day=0.25`,
run #14 hparams), with the brain resolved to `llm_model_agi@ead7e35` — the S4
`probe_verb` instrumentation (per-batch deposit credit/propensity diagnostics
in every training log line). Training math unchanged; observational only.

Workflow run: [29780496744](https://github.com/nori1234/sim_ai_agents/actions/runs/29780496744).
`probe_analysis.txt` is `scripts/analyze_probe_log.py`'s output over
`full_log.txt`.

## What the probe showed (fact)

Over 34 batch updates (32 with probe data):

- **`probe_teacher_n = 0` in every single batch.** The logged brain's teacher
  never once demonstrated a deposit across the entire run — despite
  `teacher_frac_in_batch` ~0.3–0.7 (plenty of teacher steps).
- **Deposit propensity flat at the uniform floor** (~0.017–0.026 ≈ 1/47) from
  first batch to last — the policy never learned toward *or* away from
  deposit. No collapse dynamics; just indifference, never trained off.
- **Deposit's raw credit (G−V) is POSITIVE and larger than non-deposit's**
  (+17.2 vs +13.0 mean; segment-joined: control +25.9 vs cf +8.8 — even
  regime-ordered in the right direction). Credit assignment was *not*
  punishing deposit.

## The mechanism those three facts pin down (verified in code + empirically)

The driver trains a **separate brain per agent** but logs, checkpoints, and
battery-evaluates only `brains[next(iter(brains))]` — the **first brain
created = `agents[0]` = the staffed banker** (creation order verified). Under
`sole_banker=True`, agents[0] is the sole deposit *receiver*: `_banker_near`
excludes self and `_do_deposit` refuses `bank is agent`, so **the banker
structurally cannot deposit** (verified: `_banker_near(banker) is None`), and
its blind-heuristic teacher never proposes a deposit — exactly the
`probe_teacher_n = 0` the instrumentation measured.

So runs #14 and #15 trained six brains, threw away the five savers' (whose
teachers deposit densely — a local smoke run of the fixed driver shows the
saver's very first batch carrying `probe_teacher_n = 15/16`), and evaluated
the banker's brain — **an agent that never faced the scored decision during
training** — loaded into savers on the held-out battery.

## What is invalidated, and what stands

- **Invalidated: runs #14 and #15 as fair-task tests.** Their batteries
  truthfully measured the checkpoint they were given; the inference "the
  brain failed to learn grounding on a task where grounding pays" does not
  follow, because the evaluated brain never trained on the deposit decision.
  The pre-registered grid's branch-2 conclusion from run #15 ("S4 credit
  assignment is the suspect") is **withdrawn** — superseded by this
  measurement-validity finding. **The fair-task test has not yet been run.**
- **Stands: the S6 task redesign and the 0.25 calibration** (instrument-side
  measurements on the heuristic/oracle, unaffected by brain selection), the
  control-margin-1 landscape, and the one-round rule on the rate dial.
- **Scope note for earlier sandbox runs (#8–#13):** they also logged/
  checkpointed the banker's brain, but without `sole_banker` the banker had
  counterparties (the deposit chain) and did face deposit decisions — those
  runs are not invalidated by this mechanism, though from run #17 on the
  measured brain is the sandbox saver `agents[1]`, matching the convention
  every instrument already uses ("agents[0] is the banker").

## The fix (this branch)

`train_neural_grounding.py` now selects the **measured brain** explicitly: in
sandbox mode `agents[1]` (the oracle/battery-measured saver), full town
`agents[0]` as before; the choice is printed at episode 1 and at checkpoint
time. Verified end-to-end locally (measured brain a2; saver teacher
demonstrates deposit from the first batch; checkpoint saved from a2).

## Next

Run #17 = the same pre-registered spec, fixed driver — **the first genuinely
fair fair-task run.** The S4 probe stays on: if the saver's brain still lands
at never-deposit despite dense teacher demonstration and positive credit, the
S4/BC-dilution questions return with real standing — and the same probe
fields will say which.
