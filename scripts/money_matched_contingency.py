"""Torch-free clincher: is the floor's contingency genuine regime-detection, or
pure trajectory divergence?

Setup (from threshold_landscape.py): the blind floor is a memoryless wealth
threshold ``money >= 12 -> deposit`` that scores norm_contingency +0.53 -- the bar
the whole program chases. But WHERE does that asymmetry come from? Two mutually
exclusive mechanisms:

  (G1) TRAJECTORY divergence -- the SAME fixed rule fires less in cf only because
       demurrage makes cf agents poorer, so they sit below threshold more ticks.
       The per-money-level deposit *propensity* is IDENTICAL across regimes.
       This is "mechanical replay of a wealth rule": no regime knowledge.

  (G2) Regime DETECTION -- at the SAME money level, the agent deposits LESS in cf,
       having inferred the punishing regime from experience. This is genuine
       grounding (requires memory/inference).

norm_contingency (a raw rate difference) cannot tell them apart. This script bins
every deposit *decision* by the agent's current money and reports the fire-rate
per bin, per regime. If the floor's per-bin rates are ~equal across regimes
(G2~0) while its overall rate still diverges, the floor grounds ENTIRELY by G1 --
proving the metric rewards mechanical trajectory divergence, not regime detection,
and that the memory family was measured against the wrong target.

Run: ``python scripts/money_matched_contingency.py`` (engine only, no torch).
"""

from __future__ import annotations

from collections import defaultdict

from emergence.brains.heuristic import HeuristicBrain, BANKER_CAPITAL, LOW_ENERGY
from emergence.actions import Action, ActionType
from emergence.grounding import make_grounding_sandbox


SEEDS = list(range(42, 48))
DAYS = 20
N_SAVERS = 5
DEMURRAGE = 0.25
THRESHOLD = 12          # the floor as-built
BINS = [(0, 8), (8, 12), (12, 16), (16, 20), (20, 28), (28, 40), (40, 9999)]


def _logging_threshold_brain(threshold: float, log: dict):
    """Floor rule (deposit iff money>=threshold), instrumented: every time the
    deposit branch is *reached* (agent at a bank with surplus>=12 to consider),
    record (money, fired) into ``log[regime]`` so we can bin fire-rate by money.
    regime is read off the sim at call time (cf worlds set a flag we can see)."""

    class _Brain(HeuristicBrain):
        def _bank_action(self, agent, obs):
            ec = obs.economy
            bh = ec.get("bank_here")
            deps = ec.get("my_deposits") or []
            here = obs.here["type"] if obs.here else None
            if bh is None and agent.money >= BANKER_CAPITAL \
                    and obs.fear_level == 0 and agent.energy > LOW_ENERGY:
                if here == "bank":
                    if not any(o.get("maker") == agent.id for o in obs.open_offers):
                        interest = 1 + round(2 * (1 - self.persona.cooperation))
                        return Action(ActionType.OFFER,
                                      {"loan": True, "item": "money",
                                       "principal": 5, "repay": 5 + interest},
                                      rationale="lend the bank's reserves")
                    return Action(ActionType.REST, rationale="keep the bank open")
                if any(f["type"] == "bank" for f in obs.nearby_facilities):
                    return Action(ActionType.MOVE, {"facility_type": "bank"},
                                  rationale="set up as a banker")
            if not bh:
                return None
            if agent.money < 4:
                d = next((d for d in deps if d.get("bank") == bh), None)
                if d:
                    return Action(ActionType.WITHDRAW,
                                  {"bank": bh, "amount": d["amount"]},
                                  rationale="withdraw savings")
            if agent.money < 12:
                return None
            # deposit branch reached with a bankable surplus -- the decision point.
            regime = "cf" if getattr(self, "_regime_cf", False) else "control"
            fired = agent.money >= threshold
            log[regime].append((float(agent.money), fired))
            if fired:
                return Action(ActionType.DEPOSIT,
                              {"bank": bh, "amount": agent.money - 8},
                              rationale="bank my surplus")
            return Action(ActionType.REST, rationale="hold cash")

    return _Brain


def main() -> None:
    log = {"control": [], "cf": []}
    Brain = _logging_threshold_brain(float(THRESHOLD), log)

    for seed in SEEDS:
        for cf_enabled in (False, True):
            regime_cf = cf_enabled

            def factory(agent, persona, rng, _cf=regime_cf):
                b = Brain(persona, rng)
                b._regime_cf = _cf
                return b

            sim = make_grounding_sandbox(
                "claude", rule="demurrage", n_savers=N_SAVERS, seed=seed,
                days=DAYS, cf_enabled=cf_enabled, brain_factory=factory,
                sole_banker=True, demurrage_per_day=DEMURRAGE)
            sim.run()

    print(f"seeds={SEEDS[0]}..{SEEDS[-1]}  threshold={THRESHOLD}  "
          f"demurrage={DEMURRAGE}\n")
    print("Deposit-decision fire-rate binned by current money, per regime.")
    print("G1 (trajectory) shows as different bin *populations*; G2 (detection) "
          "shows as\ndifferent fire-*rates* within the same money bin.\n")
    print(f"{'money bin':>12} | {'control n':>9} {'ctl fire%':>9} | "
          f"{'cf n':>6} {'cf fire%':>9} | {'rate gap':>8}")
    print("-" * 72)

    def binof(m):
        for lo, hi in BINS:
            if lo <= m < hi:
                return (lo, hi)
        return BINS[-1]

    ctl_bins = defaultdict(lambda: [0, 0])   # bin -> [n, fired]
    cf_bins = defaultdict(lambda: [0, 0])
    for m, fired in log["control"]:
        b = binof(m); ctl_bins[b][0] += 1; ctl_bins[b][1] += int(fired)
    for m, fired in log["cf"]:
        b = binof(m); cf_bins[b][0] += 1; cf_bins[b][1] += int(fired)

    for b in BINS:
        cn, cf_ = ctl_bins[b], cf_bins[b]
        cr = cf_[0]
        ctl_rate = cn[1] / cn[0] if cn[0] else float("nan")
        cf_rate = cf_[1] / cr if cr else float("nan")
        gap = (ctl_rate - cf_rate) if (cn[0] and cr) else float("nan")
        lo, hi = b
        label = f"[{lo},{hi if hi < 9999 else '+'})"
        print(f"{label:>12} | {cn[0]:>9} {ctl_rate:>8.1%} | "
              f"{cr:>6} {cf_rate:>8.1%} | {gap:>+8.1%}")

    # overall
    def rate(rows):
        return sum(f for _, f in rows) / len(rows) if rows else float("nan")
    print("-" * 72)
    print(f"overall deposit fire-rate: control={rate(log['control']):.1%}  "
          f"cf={rate(log['cf']):.1%}")
    print(f"decision points: control={len(log['control'])}  cf={len(log['cf'])}")
    print("\nREAD: if within-bin fire-rates (rate gap) are ~0 while the bin "
          "POPULATIONS shift\npoorward in cf, the floor grounds by G1 (trajectory) "
          "alone -- zero regime detection.\nThat means norm_contingency measures "
          "mechanical divergence, and genuine grounding (G2)\nneeds a money-matched "
          "metric the program hasn't been scoring.")


if __name__ == "__main__":
    main()
