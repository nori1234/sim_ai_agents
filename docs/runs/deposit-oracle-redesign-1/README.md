# deposit-oracle-redesign-1 — the S6 task redesign that crosses the sign

The calibration attempt (`deposit-oracle-calib-1`) established that reward
re-weighting (lever 2) plateaus at ≈0⁻ and cannot make grounding pay. This run
records what happened next: **both remaining levers were falsified, the true
mechanism behind S6's −127 was located, and a one-switch task redesign lands
`advantage_cf` slightly positive — the calibration target the brain team asked
for.**

## Reproduce

```
python3 scripts/deposit_oracle.py --persona guardian --sole-banker
```

`output.txt` is the exact CLI output (full per-world JSON + summary).

## Headline (fact)

| | original sandbox | `--sole-banker` |
|---|---|---|
| advantage_cf (20 worlds) | −127.30 | **+0.2075** |
| effect size | −1.94σ | **+0.56σ** |
| worlds oracle ahead | 0/20 | 12/20 |
| control sanity | 0.00 | 0.00 |
| deposits/episode (seed 42) | ~120 (every tick) | 39 cf / 102 control |

The brain team's calibration spec asked for "slightly positive, e.g.
+0.2–0.5σ, not large" — +0.56σ sits at that boundary, and the deposit decision
stays dense in both regimes (their RL-density requirement).

## The diagnosis chain that led here (fact)

1. **Lever 3 (scale work-pay minting, #45) is inert.** Sweeping the sandbox's
   work-pay from 1.0 to 0.0 leaves `advantage_cf` at −127.304 unchanged — the
   measured saver **never works** (0 work calls; the sandbox has only
   BANK/FARM/HOUSE).
2. **Deposit interest is inert too.** `DEPOSIT_INTEREST_PER_DAY` 0.08 → 0.0:
   −127.304 unchanged.
3. **The real mechanism: agent-to-agent deposit chains.** `_banker_near`
   treats *any* other agent standing on a BANK tile as a deposit counterparty
   — and in the sandbox every agent stands on the bank. The staffed banker's
   own "banker" is the measured saver, so the pooled coin ping-pongs
   banker⇄saver every tick, ratcheting ~+420 of reward-counted claims per pass
   out of one fixed coin pool (traced: +6910 credited to the saver via
   `_do_deposit` in 3 days; the saver's claims reach 2188 on day 1 from 50
   starting coin). Demurrage's −15%/day never catches the ratchet — which is
   why **no reward re-weighting or minting lever could flip the sign**.

This falsifies the mechanism sentence originally recorded with
`deposit-oracle-1` (work-minted #45 income backfilling deposits) — the minted
system income exists but does not drive the S6 number. `docs/GROUNDING.md` is
corrected accordingly.

## The redesign (one switch, default off)

`make_grounding_sandbox(..., sole_banker=True)` /
`measure_deposit_oracle(..., sole_banker=True)`: only the staffed banker
(`agents[0]`) accepts deposits (`Simulation.sole_deposit_banker`; enforced in
both `_banker_near` and `_do_deposit`, so a brain naming another counterparty
directly is refused too). Default `False` is byte-identical — the original
sandbox, every earlier S6 number reproduced exactly; the four-society baseline
contract is untouched either way.

With the chain cut, the honest economics reappear:

- **control:** depositing earns interest from the banker's finite reserves —
  deposit pays, and the recurring interest income keeps the deposit decision
  live (102/episode);
- **counterfactual:** demurrage shrinks the deposit and there is no ratchet to
  outrun it — holding cash wins by a small margin (+0.21, not a giveaway).

## Reading (opinion)

This is the "task where grounding pays but isn't trivial" the brain team's
decision table pointed at. The +0.56σ margin is honest: the oracle wins in
only 12/20 worlds, so a learner must actually pick up the regime contingency
rather than a dominant strategy. Next step per their plan: domain-randomized
training on the `sole_banker=True` sandbox, then the pre-registered 20-world
battery — the first fair test of "does the brain learn grounding when the task
rewards it", which the 13 prior runs never had.
