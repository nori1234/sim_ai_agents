# deposit-oracle-1

The S6 **deposit-only oracle** — the brain team's clean-spec variant of the
reward ceiling (reward-ceiling-1 / experiment "#3"), requested to split
"the task doesn't pay for grounding" (→ task redesign) from "the task pays
but variance drowns the effective gradient" (→ learning-side fix).

Reproduce with:

    python3 scripts/deposit_oracle.py --persona guardian

Ran locally (no torch, no CI needed — deterministic, ~6 seconds). See
`emergence/grounding.py`'s "Deposit-only oracle (S6)" section for the
mechanism and `output.txt` for the exact `measure_deposit_oracle().as_dict()`
plus the three summary lines.

## Spec (as requested)

The oracle is the blind `HeuristicBrain`, byte-identical, EXCEPT: in the
counterfactual world only, a `DEPOSIT` decision is dropped (the cash is
simply held) and control falls through to the blind heuristic's own next
branch — no `REST` substitute (that was the reward-ceiling-1 oracle's
choice), no other branch touched (withdraw / lending / OFFER / REPAY all
identical to blind). Implemented as a wrapper over `super()._bank_action`
that returns `None` for a `DEPOSIT` action, so the "identical except the
deposit emission" claim is enforced by construction.

## Headline

    advantage_cf (oracle − blind, counterfactual) = -127.30
    control sanity check                          = +0.00   (must be 0 ✓)
    effect size (advantage_cf / blind_cf_std)     = -1.94
    oracle ahead in                               = 0 / 20 worlds
    oracle deaths (cf)                            = 2 / 20
    advantage_cf, survivors only (n=18)           = -120.19

Per the brain team's decision table (advantage_cf ≤ 0 → task redesign;
> 0 but unlearned → learning-side variance fix), this lands on **task
redesign**, and does so unambiguously: the sign is strongly negative
(≈ −2σ of the blind heuristic's own per-world spread), consistent across
**every** world (0/20 sign flips), and not a death artifact (survivors-only
advantage is −120, essentially unchanged).

## Why the sign is negative — and why it disagrees with the mirror's +0.555

The brain team's mirror assumed the lending channel is ~zero-mean w.r.t. the
regime and predicted a small positive advantage. The real engine gives the
opposite sign because of a structural feature the mirror does not model, and
which the honest faithfulness caveat in the request anticipated:

1. **The reward counts bank deposits as wealth.** `_wealth = money +
   Σ deposits` (see `emergence/brains/_neural_reward.py`) — this is
   deliberate; it is what gives demurrage a reward gradient at all.
2. **Work mints money from nothing** (issue #45): `_do_work` pays coin with
   no source. In this sandbox that minted income flows, day after day, into
   the reward-counted deposit balance. Measured on seed 42: ~5977 coin minted
   across the 6-agent system over 20 days; the measured saver's reward-counted
   wealth grew 50 → 3162 (**+3112, ≈ +156 reward from the wealth term**)
   **despite** demurrage being active the whole run.
3. So demurrage's −15%/day evaporation is **dominated** by the re-deposited
   minted inflow. Under this reward × economy, **depositing is
   reward-maximizing even under demurrage.** The "correct grounded action"
   (avoid the shrinking deposit) is reward-*suboptimal* here — the task's
   gradient points the wrong way.

The two oracles converge on this: reward-ceiling-1's `REST`-substitute oracle
gave advantage_cf = **−125.4**; this fall-through oracle gives **−127.3**.
Two different substitute behaviours, the same strongly-negative sign — so the
result is **robust to the substitute-action confound both sides worried
about**. The first-order effect is losing the deposit-as-wealth channel; what
the agent does instead is second-order.

## What this says about the two branches

- It is **not** the learning-side / variance hypothesis. `effect_size =
  −1.94` — the advantage is ~2σ of the blind's own spread, not a small
  signal buried in noise; and 0/20 per-world sign flips. Variance is not
  drowning a positive signal; the sign is negative.
- It **is** the task-redesign branch, and it names the lever: the reward
  gradient rewards depositing under demurrage because minted work income
  (#45) re-fills the reward-counted deposit balance faster than −15%/day
  removes it. A task where grounding pays needs one of: demurrage that bites
  harder than the re-deposit inflow, a reward that does not credit a
  shrinking deposit as preserved wealth, or a sandbox where work does not
  mint money into the deposit loop (#45).

`advantage_control` is exactly 0.0 (the oracle IS the blind heuristic in the
control world), so the counterfactual number is a clean regime contrast, not
an implementation artifact.
