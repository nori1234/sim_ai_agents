# deposit-oracle-calib-1 — S6 lever-2 calibration sweep

The brain team accepted the S6 result (`advantage_cf = −127.3`, task-redesign
branch) and proposed **using the S6 oracle as a calibration dial**: apply one
reward lever, re-measure S6, and set the parameter where `advantage_cf` lands
*slightly* positive — a task where grounding pays but isn't trivial (Goodhart
avoidance). They preferred **lever 2** (down-weight banked coin so demurrage's
loss bites the reward directly) over the brute-force punishment lever.

This run implements lever 2 as a continuous dial and sweeps it. **Result: the
dial behaves exactly as predicted in direction, but cannot cross the sign — it
asymptotes just below zero.** That is the finding.

## Reproduce

```
python3 scripts/deposit_oracle.py --persona guardian --deposit-weight 1.0   # canonical (−127)
python3 scripts/deposit_oracle.py --persona guardian --deposit-weight 0.0   # extreme end (−1.3)
```

`--deposit-weight λ` (default 1.0) scales how much a banked coin counts toward
reward-wealth (`weights["deposit"]` in `emergence/brains/_neural_reward.py`,
threaded through `measure_deposit_oracle`). λ=1.0 is the canonical reward —
byte-identical to every earlier S6 run.

## The sweep (see `output.txt` for the raw numbers)

| λ (deposit weight) | advantage_cf (all 20) | effect size | advantage_cf (survivors) | oracle ahead |
|---|---|---|---|---|
| 1.0 (canonical) | −127.30 | −1.94σ | −110.54 | 0/20 |
| 0.5 | −64.29 | −1.90σ | — | 0/20 |
| 0.2 | −26.47 | −1.76σ | −22.33 | 0/20 |
| 0.1 | −13.87 | −1.54σ | −11.30 | 0/20 |
| 0.05 | −7.57 | −1.19σ | −5.79 | 1/20 |
| **0.0** | **−1.27** | **−0.29σ** | **−0.27** | **14/20** |

## Reading (fact)

- **The dial is monotone and inert at default.** λ=1.0 reproduces −127.30
  exactly; lowering λ raises `advantage_cf` linearly (≈ +126 per unit of λ).
  The direction the brain team predicted is confirmed.
- **It never reaches positive.** At the extreme λ=0.0 (banked coin worthless),
  `advantage_cf` is still −1.27 over all 20 worlds and −0.27 for survivors
  only. The per-world *majority* flips (oracle ahead in 14/20), but the **mean
  stays marginally negative**.
- **λ only re-scores fixed trajectories.** The oracle's behaviour — and its 2
  cf-world deaths — are identical at every λ (the brain never sees the reward;
  these are fixed-policy oracles). So the dial moves the accounting, not the
  world.

## Why lever 2 can't cross the sign (interpretation)

`survival_reward` telescopes: it is a weighted sum of observation-field deltas,
so an episode's return equals `(final − initial)`, **path-independent**. By
episode end, deposit balances are small — demurrage has shrunk them and the
agent has withdrawn — so re-weighting the *deposit* term has bounded leverage
on a final-minus-initial read. What remains at λ=0 (−0.27 survivors) is not a
demurrage effect at all: it is the **behavioural** cost of the oracle holding
idle cash and falling through to the blind heuristic's next branch (it spends /
idles differently and ends marginally poorer). **No reward re-weighting can
remove a behavioural/trajectory cost.**

This *sharpens*, rather than overturns, S6: the reason depositing pays is not
merely that deposits are counted as wealth — it is that **not** depositing
changes what the agent does next, and that change is itself slightly costly.
The sign is robust to the entire reward-reweighting family.

## Consequence for the calibration plan

A pure reward-side dial (lever 2) has a **structural ceiling at ≈0⁻** and can't
deliver the "slightly positive" task. Crossing the sign needs a lever that
changes the **trajectory / end-state**, not the scoring:

- **Lever 3 (recommended next):** exclude minted work-income (#45) from the
  deposit backfill, so a deposited coin is no longer refilled faster than
  demurrage removes it. This changes dynamics, not accounting — it can move the
  end-state and therefore the telescoped return. It is an engine-behaviour
  change (needs its own opt-in flag + baseline-contract check), so it is filed
  as the proposed next step rather than bundled here.
- Or a **non-telescoping / path-integrated** demurrage penalty, if the brain
  team prefer to keep the dynamics fixed and change the reward's shape instead.

Deposited-coin down-weighting (lever 2) ships as a working, tested,
inert-by-default instrument regardless — it is the tool that produced this
finding, and it stacks cleanly with whichever trajectory-lever is chosen next.

## Addendum (correction, same session)

The lever-3 recommendation above was **tested and falsified**: scaling
work-pay minting (#45) — and deposit interest — leaves `advantage_cf` at
−127.304 unchanged even at zero, because the measured saver never works and
neither channel is the deposit inflow. The true mechanism is the sandbox's
agent-to-agent deposit chain, and the redesign that does cross the sign is
`sole_banker=True` — see
[`deposit-oracle-redesign-1`](../deposit-oracle-redesign-1/).
