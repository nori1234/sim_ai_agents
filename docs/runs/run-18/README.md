# run-18 — variance-reduction candidate 1 (episode advantage baseline): the pre-declared criterion moves for the first time

Run #17's spec + one brain-side change (`llm_model_agi@435177b`):
`adv_baseline="episode"` — subtract each episode segment's mean advantage
before batch normalisation, cancelling the cross-episode/regime return offset
run #17 measured as the S4 noise floor. Success criterion, declared before the
run: `probe_adv_used_mean` on deposit separates by regime segment.

Workflow run: [29786362493](https://github.com/nori1234/sim_ai_agents/actions/runs/29786362493).

## Probe (fact) — the criterion's first movement in 18 runs

| quantity | run #17 (batch baseline) | run #18 (episode baseline) |
|---|---|---|
| deposit raw credit, control vs cf | +22.5 vs **+28.7 (inverted)** | **+2.95 vs +1.68 (correct order)** |
| deposit USED advantage, control vs cf | ≈0 (−0.0116 overall) | **+0.278 vs +0.023** |
| segments with positive used-adv | — | control 14/20 vs cf 10/22 |
| deposit propensity by ep60 | ~0.083 | **~0.12** (self-attempts 122 vs 66) |
| battery raw attempts (C/CF) | 91 / 117 (wrong order) | **248 / 231 (correct order, first time)** |

The episode baseline did exactly what it was built to do: the credit spread
compressed ~10× (means ±2–3 instead of ±20–100), the regime inversion
disappeared, and PG now sees a positive deposit signal in control and ~zero in
cf — the first regime-ordered learning signal on the scored decision in the
program's history.

## Battery (fact)

Still `grounded_confirmed = False` (POWERED-NO): `mean_excess −0.5798`
(CI [−0.620, −0.537]), `n_conclusive 20/20`, `fraction_grounded 0.0`,
`trained_stable False`. The behavioural divergence (248 vs 231, ~7% relative)
has the right sign but is far below the heuristic floor's (~3× control/cf),
so excess stays deeply negative.

## Reading

Criterion **partially met**: ordering emerged (control ≫ cf ≈ 0), cf not yet
negative. Mechanically consistent: interest income makes deposit genuinely
positive-credit in control; in cf the small demurrage loss (~0.6/episode) is
now visible enough to null the credit but not yet to punish it. Propensity and
the used-advantage gap were both still rising at ep60 — the run ended
mid-trend. The cheapest falsifiable next step is **time, not another
mechanism**: same spec, episodes 200 (runs #8–#13 standard), watching whether
cf used-adv crosses negative and the behavioural divergence widens toward the
floor's ratio. That is run #19.
