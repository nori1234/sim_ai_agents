"""Torch-free diagnostic: map the deposit-contingency landscape as a function of
a memoryless current-wealth threshold, and its *sharpness*.

Context (run 28-30 finding): the blind floor is a pure current-wealth threshold
``money >= 12 -> deposit`` (see ``_grounded_heuristic_brain_class``); its whole
regime-contingency (norm_contingency +0.518) is *inherited* from wealth dynamics
-- demurrage drains cf-world agents below the threshold, so they deposit on fewer
ticks. NO regime inference, NO memory. The entire memory+credit research family
(v1b/v2a/critic) plateaued at norm_contingency ~+0.09, ~6x short, because it was
solving the wrong sub-problem.

This script asks the pivot-deciding question CHEAPLY, before any 3.5h CI run:

  1. Threshold sweep -- how does a memoryless deterministic threshold's
     norm_contingency and deposit density move as T varies? Locates the ridge.
  2. Sharpness sweep -- replace the hard step with a stochastic sigmoid
     P(deposit | money) = sigma((money - T) / tau). A neural policy realistically
     learns a SOFT decision boundary; if softness collapses the contingency, that
     explains the plateau (the policy can't be sharp enough / RL noise smears it).
     If a soft boundary keeps most of the contingency, the gap is elsewhere.

Run: ``python scripts/threshold_landscape.py`` (needs only the engine, no torch).
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from emergence.brains.heuristic import HeuristicBrain
from emergence.actions import Action, ActionType
from emergence.grounding import make_grounding_sandbox, behaviour_count


SEEDS = list(range(42, 48))        # same seed band the floor (+0.518) was measured on
DAYS = 20
N_SAVERS = 5
DEMURRAGE = 0.25                    # the sandbox's calibrated dial (matches CI runs)


def _threshold_brain_class(threshold: float, tau: float | None):
    """A HeuristicBrain whose ONLY change is the deposit-decision boundary.

    ``tau is None`` -> hard step: deposit iff money >= threshold (the floor is
    threshold=12). ``tau`` set -> soft: deposit with probability
    sigma((money - threshold)/tau), sampled from the brain's own rng (so it stays
    deterministic per seed). Everything else -- banker setup, withdraw-when-broke,
    the REST substitute -- is inherited unchanged, so the comparison isolates the
    deposit boundary, nothing else."""

    class _ThresholdBrain(HeuristicBrain):
        def _bank_action(self, agent, obs):
            ec = obs.economy
            bh = ec.get("bank_here")
            deps = ec.get("my_deposits") or []
            here = obs.here["type"] if obs.here else None
            from emergence.brains.heuristic import BANKER_CAPITAL, LOW_ENERGY
            # (1) become/keep a bank -- identical to the parent.
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
            # (2) the deposit boundary -- the one thing we vary.
            want_deposit = agent.money >= 12  # need surplus to have anything to bank
            if not want_deposit:
                return None
            if tau is None:
                fire = agent.money >= threshold
            else:
                p = 1.0 / (1.0 + math.exp(-(agent.money - threshold) / tau))
                fire = self.rng.random() < p
            if fire:
                return Action(ActionType.DEPOSIT,
                              {"bank": bh, "amount": agent.money - 8},
                              rationale="bank my surplus")
            return Action(ActionType.REST, rationale="hold cash, don't bank it")

    return _ThresholdBrain


def _norm_asym(cr: float, cfr: float) -> float:
    denom = cr + cfr
    return (cr - cfr) / denom if denom > 1e-12 else 0.0


@dataclass
class Cell:
    label: str
    control: int
    cf: int

    @property
    def density(self) -> int:
        return self.control + self.cf

    @property
    def norm(self) -> float:
        return _norm_asym(self.control, self.cf)


def _measure(brain_cls) -> Cell:
    """Sum deposit counts over the seed band, control vs cf."""
    def factory(agent, persona, rng):
        return brain_cls(persona, rng)

    control_total = 0
    cf_total = 0
    for seed in SEEDS:
        for cf_enabled in (False, True):
            sim = make_grounding_sandbox(
                "claude", rule="demurrage", n_savers=N_SAVERS, seed=seed,
                days=DAYS, cf_enabled=cf_enabled, brain_factory=factory,
                sole_banker=True, demurrage_per_day=DEMURRAGE)
            sim.run()
            n = behaviour_count(sim, "deposit")
            if cf_enabled:
                cf_total += n
            else:
                control_total += n
    return Cell(brain_cls.__name__, control_total, cf_total)


def main() -> None:
    print(f"seeds={SEEDS[0]}..{SEEDS[-1]}  days={DAYS}  n_savers={N_SAVERS}  "
          f"demurrage={DEMURRAGE}  sole_banker=True\n")

    print("=== (1) HARD threshold sweep (memoryless step: deposit iff money>=T) ===")
    print(f"{'T':>5} {'control':>8} {'cf':>6} {'density':>8} {'norm_cont':>10}")
    hard_rows = []
    for T in (8, 10, 12, 14, 16, 20, 24, 30):
        cell = _measure(_threshold_brain_class(float(T), None))
        hard_rows.append((T, cell))
        print(f"{T:>5} {cell.control:>8} {cell.cf:>6} {cell.density:>8} "
              f"{cell.norm:>+10.3f}")

    print("\n=== (2) SOFT threshold sweep (sigmoid boundary at T=12, vary tau) ===")
    print("  tau->0 recovers the hard step; larger tau = fuzzier boundary "
          "(what RL realistically learns)")
    print(f"{'tau':>5} {'control':>8} {'cf':>6} {'density':>8} {'norm_cont':>10}")
    for tau in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
        cell = _measure(_threshold_brain_class(12.0, tau))
        print(f"{tau:>5.1f} {cell.control:>8} {cell.cf:>6} {cell.density:>8} "
              f"{cell.norm:>+10.3f}")

    best = max(hard_rows, key=lambda r: r[1].norm)
    print(f"\nbest hard threshold: T={best[0]}  norm_contingency={best[1].norm:+.3f} "
          f"(floor as-built T=12)")
    print("Read: if the SOFT sweep keeps norm_contingency near the hard value even "
          "at large tau,\nthe policy's soft boundary is NOT the bottleneck -- the "
          "plateau is elsewhere\n(actuation/optimisation). If soft collapses it, "
          "sharpness IS the wall.")


if __name__ == "__main__":
    main()
