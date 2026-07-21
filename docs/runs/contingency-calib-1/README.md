# contingency-calib-1 — the run #15 dial, calibrated through the real parameters

Executes the calibration step of the run #15 pre-registration
(`docs/proposals/run15-contingency-margin.md` §4), through the plumbed
`demurrage_per_day` parameter (not the scratch runtime override the proposal's
§3 feasibility preview used — same numbers, now via the shipping code path).
Deterministic, no torch, seconds.

## Reproduce

```
for r in 0.15 0.20 0.25 0.30; do
  python3 scripts/control_margin.py --persona guardian --sole-banker --demurrage-per-day $r
done
python3 scripts/deposit_oracle.py --persona guardian --sole-banker --demurrage-per-day 0.25
```

`control_margin_sweep.txt` and `deposit_oracle_0.25.txt` are the exact CLI
outputs.

## Result (fact) — guardian, 20 held-out worlds (42–61), sole_banker

| rate/day | contingency margin (== advantage_cf) | effect size (paired-diff σ) | worlds + | control pull |
|---|---|---|---|---|
| 0.15 (canonical) | +0.2075 | +0.20σ | 12/20 | +10.3485 |
| 0.20 | +0.4495 | +0.43σ | 14/20 | +10.3485 |
| **0.25** | **+0.6010** | **+0.53σ** | 15/20 | +10.3485 |
| 0.30 | +0.7105 | +0.63σ | 13/20 | +10.3485 |

**Calibrated rate: 0.25/day** — the smallest rate whose effect size clears the
pre-registered [+0.5σ, +1.0σ] band, per the "smallest in band" rule fixed
before this measurement.

All four pre-registered gates pass at 0.25:

1. **Control invariance** — exactly +10.3485 at every rate (demurrage exists
   only in the cf world; zero drift, as the design requires).
2. **Conclusive yield** — `--preflight-only` at 0.25: demurrage 20/20.
3. **Density** — deposits/episode (seed 42): control 102 (unchanged from
   0.15), cf 31 (vs 39 at 0.15; above the ~30 comparability line).
4. **Blind cf survival** — 20/20 (no death-driven confound).

## Disclosures (fixed before the training run, so they can't become
after-the-fact rationalizations)

- **Survivors-only effect size at 0.25 is +0.48σ** (12/17 worlds), marginally
  under the band's floor; the all-worlds number (+0.53σ) is the registered
  gate and it passes. Recorded here rather than silently averaged over: if run
  #15 grounds and someone asks whether the margin was really above noise for
  surviving agents specifically, this is the honest number.
- **Two effect-size conventions exist and disagree.** This calibration's σ is
  the paired per-world difference std (how consistently holding cash beats
  depositing across worlds) — the incentive-detectability reading. The S6
  oracle's own `effect_size` field divides by the blind policy's cf *return*
  spread instead, and at 0.25 reads **+2.37σ** (steeper demurrage both raises
  the numerator and compresses blind's cf returns). The brain team's original
  "+0.2–0.5σ" spec was in the S6 convention; the run #15 band is in the
  paired-difference convention. Both numbers are reported so nobody discovers
  the discrepancy later and reads it as motivated metric-shopping.
- The rate change also enlarges the observation signal (bigger "vanished"
  entries) — an assist on the perception axis, disclosed in the proposal §7.
  Perception was never the bottleneck (probe ceiling 0.98), so this is a
  disclosure, not a confound.

## What happens next (per the pre-registration, no discretion left)

Run #15 = `neural-train-battery` with `sandbox=true`, `sole_banker=true`,
`demurrage_per_day=0.25`, run #14's brain hparams (`bc_weight 0.3`,
`self_attempt_base 0.3`, `batch_every 64`), pool seeds 1000–1015, battery at
the same rate, verdict rules unchanged. The 4-branch interpretation grid is
already fixed in the proposal §6. One round only; no further rate raises
regardless of outcome.
