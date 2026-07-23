"""C1 prototype (torch-free): does a 'stable-income' sandbox make regime
inference the ONLY way to score?

Mechanism: refill each saver's spendable money to a FIXED target at the end of
every day. Then the deposit decision is always faced at ~the same money in BOTH
regimes, so the reflex's only handle (money level) is regime-invariant by
construction. Demurrage still shrinks the banked deposit (real loss, felt only via
memory + the total-wealth reward). Predictions:
  - blind floor (money>=12 reflex): G1 ~ 0 AND G2 ~ 0 (no handle at all).
  - regime-aware oracle (deposits in control, holds under demurrage): G2 high
    (it suppresses matched-money deposits it would make in control) -> the task
    is still grounding-SOLVABLE. That gap is the whole point of C1.
"""
import sys
from emergence.grounding import (
    make_grounding_sandbox, behaviour_count, money_matched_contingency,
    _grounded_heuristic_brain_class,
)
from emergence.brains.heuristic import HeuristicBrain
from emergence.actions import ActionType

SEEDS = list(range(42, 48))
DAYS = 20
N_SAVERS = 5
DEMURRAGE = 0.25
REFILL = 20        # fixed spendable-money target (regime-independent)


def _logging(inner_factory):
    """Wrap a brain to log (money, fired) at each eligible deposit decision."""
    class W:
        def __init__(self, inner):
            self._inner = inner
            self.rng = getattr(inner, "rng", None)
            self.decisions = []

        def decide(self, agent, obs):
            action = self._inner.decide(agent, obs)
            econ = getattr(obs, "economy", None) or {}
            bh = econ.get("bank_here") if isinstance(econ, dict) else None
            if bh and agent.money >= 12:
                self.decisions.append((float(agent.money),
                                       action.type is ActionType.DEPOSIT))
            return action
    def factory(agent, persona, rng):
        return W(inner_factory(agent, persona, rng))
    return factory


def _run(inner_factory, refill):
    ctl, cf = [], []
    ctl_dep = cf_dep = 0
    for seed in SEEDS:
        for cf_enabled in (False, True):
            sim = make_grounding_sandbox(
                "claude", rule="demurrage", n_savers=N_SAVERS, seed=seed,
                days=DAYS, cf_enabled=cf_enabled,
                brain_factory=_logging(inner_factory),
                sole_banker=True, demurrage_per_day=DEMURRAGE)
            banker_id = sim.agents[0].id
            while sim.step_day():
                if refill is not None:
                    for a in sim.agents:
                        if a.id != banker_id and a.alive:
                            a.money = refill
            n = behaviour_count(sim, "deposit")
            sink = cf if cf_enabled else ctl
            for aid, b in sim.brains.items():
                if aid != banker_id:
                    sink.extend(getattr(b, "decisions", []))
            if cf_enabled:
                cf_dep += n
            else:
                ctl_dep += n
    g2 = money_matched_contingency(ctl, cf)
    g1 = (ctl_dep - cf_dep) / (ctl_dep + cf_dep) if (ctl_dep + cf_dep) else 0.0
    return g1, g2, ctl_dep, cf_dep


def blind(agent, persona, rng):
    return HeuristicBrain(persona, rng)


def oracle_factory(cf_enabled):
    GH = _grounded_heuristic_brain_class()
    def f(agent, persona, rng):
        return GH(persona, rng, avoid_deposit=cf_enabled)
    return f


def _run_oracle(refill):
    # the oracle needs to know the regime per world, so build per cf_enabled
    ctl, cf = [], []
    ctl_dep = cf_dep = 0
    for seed in SEEDS:
        for cf_enabled in (False, True):
            sim = make_grounding_sandbox(
                "claude", rule="demurrage", n_savers=N_SAVERS, seed=seed,
                days=DAYS, cf_enabled=cf_enabled,
                brain_factory=_logging(oracle_factory(cf_enabled)),
                sole_banker=True, demurrage_per_day=DEMURRAGE)
            banker_id = sim.agents[0].id
            while sim.step_day():
                if refill is not None:
                    for a in sim.agents:
                        if a.id != banker_id and a.alive:
                            a.money = refill
            n = behaviour_count(sim, "deposit")
            sink = cf if cf_enabled else ctl
            for aid, b in sim.brains.items():
                if aid != banker_id:
                    sink.extend(getattr(b, "decisions", []))
            if cf_enabled:
                cf_dep += n
            else:
                ctl_dep += n
    g2 = money_matched_contingency(ctl, cf)
    g1 = (ctl_dep - cf_dep) / (ctl_dep + cf_dep) if (ctl_dep + cf_dep) else 0.0
    return g1, g2, ctl_dep, cf_dep


def main():
    print(f"seeds 42-47, days {DAYS}, demurrage {DEMURRAGE}, refill={REFILL}\n")
    print(f"{'policy / task':<34} {'G1':>8} {'G2':>8} {'ctlDep':>7} {'cfDep':>7}")
    print("-" * 68)
    for label, refill in [("BASELINE task (no refill)", None),
                          ("C1 task (money refill=%d)" % REFILL, REFILL)]:
        g1, g2, cd, xd = _run(blind, refill)
        print(f"{'  blind floor | ' + label:<34} {g1:>+8.3f} {g2.g2:>+8.3f} "
              f"{cd:>7} {xd:>7}")
    for label, refill in [("BASELINE task (no refill)", None),
                          ("C1 task (money refill=%d)" % REFILL, REFILL)]:
        g1, g2, cd, xd = _run_oracle(refill)
        print(f"{'  regime oracle | ' + label:<34} {g1:>+8.3f} {g2.g2:>+8.3f} "
              f"{cd:>7} {xd:>7}")
    print("\nWANT on C1: blind floor G1~0 AND G2~0 (reflex has no handle);"
          "\n           regime oracle G2 high (task still grounding-solvable).")


if __name__ == "__main__":
    main()
