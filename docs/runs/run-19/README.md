# run-19 — 200 episodes on the episode baseline: the credit signal is now regime-correct; BC is the binding constraint

Run #18's spec, `episodes 200` (the #8–#13 standard length). Workflow run:
[29789225282](https://github.com/nori1234/sim_ai_agents/actions/runs/29789225282).

## Criterion (fact) — deposit used-advantage by regime segment

| training phase | control | cf |
|---|---|---|
| first third | −0.071 | −0.074 (no separation yet) |
| middle third | **+0.053** | **−0.229** |
| last third | +0.008 | −0.076 |
| ALL (n=80/80 segments) | −0.005 | **−0.123** |

The pre-declared criterion is now essentially met on the signal side: **cf
deposits receive negative used advantage, control ≈ neutral-positive**, in the
ordered direction in the middle and last thirds (not the first — the value
function needs time). PG is, for the first time, being told the truth about
the regime.

## But behaviour did not differentiate (fact)

- Propensity ~0.18 by ep190 (9× uniform) and **identical across regimes**
  (segment means 0.0807 vs 0.0818); a small dip in the last updates
  (0.188 → 0.157) is noted, not interpreted.
- Battery: attempts 359 control / 348 cf (denser again, near-flat ratio);
  `mean_excess −0.587`, `n_conclusive 20/20`, still POWERED-NO, no
  `is_stable`.

## Reading — the tug-of-war is now visible end to end

`probe_teacher_n = 435` vs `probe_self_n = 443`: half of everything the
policy learns about deposit comes from **behaviour cloning toward the
regime-blind teacher**, which demonstrates deposit densely in BOTH regimes
(it can't do otherwise — issue #10's R2). BC's pull is dense and
regime-flat; PG's newly-correct signal is sparse (~2 deposit samples/batch)
and small (−0.12). The policy's behaviour follows the stronger, blind pull:
deposit everywhere. **The S4 noise problem is fixed; the binding constraint
has moved to the teaching channel** — exactly the failure mode the brain
team pre-listed as issue #10, with their own proposed remedy (b): **BC
annealing** (decay `bc_weight` over training — a scheduled 親離れ), which is
the next one-change run.
