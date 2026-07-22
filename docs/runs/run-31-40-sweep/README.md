# run 31–40 — the memory×critic×optimiser sweep (and why it ended the family)

A parallel CI sweep extending the run 28–30 ablation, probing whether *any*
cell of {LSH bits × privileged-critic softening × GAE} lifts the learned policy's
regime-contingency toward the blind floor. Same seed/flags discipline; each run
~3.5 h. Scored on `norm_contingency = (control−cf)/(control+cf)` (G1), computed
from the held-out `raw attempt counts` line in each run's log.

| run | lever added | control | cf | density | norm_contingency (G1) |
|---|---|---|---|---|---|
| #28 v1b | obs_hash memory | 224 | 240 | 464 | −0.034 |
| **#29 v2a** | **+ state_lsh 12-bit** | **1026** | **854** | **1880** | **+0.091** |
| #30 | + plain critic | 293 | 341 | 634 | −0.076 |
| #31 | 12-bit, priv_mix 0.3 | 230 | 230 | 460 | 0.000 |
| #32 | 16-bit, entropy 0.05 | 264 | 297 | 561 | −0.059 |
| #33 | 16-bit, priv_mix 0.3 | 536 | 566 | 1102 | −0.027 |
| #34 | 20-bit | 615 | 627 | 1242 | −0.010 |
| #35 | 12-bit, priv_mix 0.5 | 226 | 189 | 415 | +0.089 |
| #36 | 12-bit, priv_mix 0.1 | 46 | 30 | 76 | +0.211 (sparse/noise) |
| #37 | 16-bit, priv_mix 0.5 | 262 | 266 | 528 | −0.008 |
| #38 | GAE, 12-bit | 241 | 238 | 479 | +0.006 |
| #40 | GAE + critic, 16-bit | *fired on the fixed code; superseded before harvest* |

## Read

- **v2a (#29) is the ceiling of the whole family at G1 +0.091** — the only cell
  with both the right direction and real density (1880). Everything else is ≤ 0,
  flat, or (at `priv_mix 0.1`, #36) a +0.211 artifact on 76 total attempts
  (30 cf deposits — not trustworthy).
- **More LSH bits smear toward 0** (#34 20-bit = −0.010): finer keys did not
  separate regimes better; they diluted recall.
- **Softened critic trades density for noise** non-monotonically (#31/#33/#35/#37):
  no cell recovers v2a's density *and* beats its contingency.
- **GAE is flat** (#38 +0.006): the advantage estimator was not the bottleneck.

Conclusion: the memory+credit+optimiser family **plateaus at G1 ≈ +0.09**, ~6×
below the blind floor's +0.518. Rather than fire more cells, two cheap torch-free
diagnostics asked *what the floor's +0.518 is* — and found G1 itself is
reflex-achievable (a higher memoryless threshold beats the floor). See
[`../metric-trajectory-confound-1/`](../metric-trajectory-confound-1/README.md):
the sweep was climbing the wrong metric. The frontier moved to **G2**
(money-matched contingency), which a reflex cannot fake.

Raw logs: GitHub Actions runs #31–#38 on `nori1234/sim_ai_agents`
(workflow `neural-train-battery`), artifacts `grounding-battery-31..38`.
