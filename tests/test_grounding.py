"""Counterfactual-world transfer test — the grounding falsification probe (#118).

The probe inverts one existing rule (bank savings *shrink* instead of grow) and
measures whether behaviour diverges between the normal and inverted worlds. Here
we check three things: (1) the inverted rule is correct and conserves coin; (2) it
is inert when off, so the determinism baseline is untouched; (3) the instrument
actually *discriminates* — a brain that learns from the loss it lived through
diverges between the worlds, while a brain that ignores experience does not.
"""

import unittest

from emergence.actions import Action, ActionType
from emergence.brains.base import AgentBrain
from emergence.grounding import (
    CounterfactualConfig,
    GroundingResult,
    behaviour_rate,
    run_grounding_probe,
)
from emergence.market import Deposit
from emergence.scenario import make_simulation
from emergence.simulation import SimulationConfig
from emergence.world import FacilityType


def _economy_sim(*, cf=None, seed=1, n=4):
    return make_simulation("guardian", n_agents=n, economy=True,
                           config=SimulationConfig(seed=seed, days=6),
                           counterfactual=cf)


class TestDemurrageRule(unittest.TestCase):
    def _sim_with_deposit(self, cf_enabled):
        sim = _economy_sim(cf=CounterfactualConfig(
            enabled=cf_enabled, rule="demurrage", demurrage_per_day=0.2))
        bank, holder = sim.agents[0], sim.agents[1]
        bank.add("money", 50)                 # the bank holds the coin it owes back
        sim.deposits.append(Deposit(id=1, bank=bank.id, holder=holder.id, amount=100))
        return sim, bank, holder

    def test_savings_shrink_under_demurrage(self):
        sim, bank, holder = self._sim_with_deposit(cf_enabled=True)
        sim._pay_deposit_interest()           # the daily upkeep, counterfactual path
        dep = sim.deposits[0]
        self.assertEqual(dep.amount, 80, "20% of the 100-coin claim evaporated")
        self.assertTrue(any(e["kind"] == "demurrage" for e in sim.world.events))
        self.assertTrue(any("vanished" in m for m in holder.memory),
                        "the loss is written into memory — the only way to learn it")

    def test_demurrage_conserves_coin(self):
        sim, bank, holder = self._sim_with_deposit(cf_enabled=True)
        before = sum(a.money for a in sim.agents)
        sim._pay_deposit_interest()
        after = sum(a.money for a in sim.agents)
        self.assertEqual(before, after, "no coin minted or burned — only the claim shrinks")

    def test_off_the_savings_still_grow(self):
        sim, bank, holder = self._sim_with_deposit(cf_enabled=False)
        sim._pay_deposit_interest()           # normal interest path
        self.assertEqual(sim.deposits[0].amount, 100, "the claim is untouched by interest")
        self.assertGreater(holder.money, 0, "the holder was paid interest in coin")
        self.assertTrue(any(e["kind"] == "interest" for e in sim.world.events))
        self.assertFalse(any(e["kind"] == "demurrage" for e in sim.world.events))


class TestVanityRule(unittest.TestCase):
    """Second counterfactual: conspicuous spending (a feast) SHAMES instead of
    honouring — inverting rep_per_feast_coin. Needs the status/honour layer."""

    def _sim(self, cf_enabled):
        from emergence.esteem import StatusConfig
        sim = make_simulation("guardian", n_agents=4, economy=True,
                              config=SimulationConfig(seed=1, days=6),
                              status=StatusConfig(enabled=True),
                              counterfactual=CounterfactualConfig(
                                  enabled=cf_enabled, rule="vanity"))
        host, caterer = sim.agents[0], sim.agents[1]
        host.reputation = 20.0
        return sim, host, caterer

    def test_a_feast_costs_the_host_standing(self):
        sim, host, caterer = self._sim(cf_enabled=True)
        sim._serve_feast(caterer, host, fee=10)
        self.assertLess(host.reputation, 20.0, "showing off lowered the host's honour")
        self.assertTrue(any(e["kind"] == "feast" and e.get("shamed")
                            for e in sim.world.events))
        self.assertTrue(any("shameful" in m for m in host.memory))

    def test_off_a_feast_buys_honour(self):
        sim, host, caterer = self._sim(cf_enabled=False)
        sim._serve_feast(caterer, host, fee=10)
        self.assertGreater(host.reputation, 20.0, "baseline: conspicuous outlay buys honour")
        self.assertTrue(any(e["kind"] == "feast" and not e.get("shamed")
                            for e in sim.world.events))


class TestObservationHidesRate(unittest.TestCase):
    def test_rate_is_advertised_by_default(self):
        sim = _economy_sim(cf=None)
        obs = sim._observe(sim.agents[0])
        self.assertIsNotNone(obs.economy.get("deposit_rate"))

    def test_probe_hides_the_rate(self):
        sim = _economy_sim(cf=CounterfactualConfig(enabled=True, hide_rate=True))
        obs = sim._observe(sim.agents[0])
        self.assertIsNone(obs.economy.get("deposit_rate"),
                          "under the probe the agent must learn the rule, not read it")


class TestBehaviourRate(unittest.TestCase):
    def test_rate_is_normalised_per_agent_day(self):
        sim = _economy_sim(cf=None, n=5)
        sim.metrics.population = 5
        sim.metrics.days_run = 4
        for _ in range(20):
            sim.world.log("deposit", holder="a1", bank="a2", amount=1)
        self.assertAlmostEqual(behaviour_rate(sim, "deposit"), 20 / 20)
        self.assertEqual(behaviour_rate(sim, "never_happened"), 0.0)


# -- a mock brain that does nothing but save (and eat to stay alive) ----------
class _Saver(AgentBrain):
    """Deposits a coin whenever it can. If ``grounded``, it stops once it has
    lived through a demurrage loss (the memory entry) — modelling learning from
    consequence. A non-grounded saver keeps depositing regardless (replay)."""

    def __init__(self, bank_id, *, saves=True, grounded=False):
        self.bank_id = bank_id
        self.saves = saves
        self.grounded = grounded

    def decide(self, agent, obs):
        if agent.energy < 70 and agent.food() > 0:
            return Action(ActionType.EAT, {})
        if self.saves and agent.money > 0:
            if self.grounded and any("vanished" in m for m in agent.memory):
                return Action(ActionType.REST, {})
            return Action(ActionType.DEPOSIT, {"bank": self.bank_id, "amount": 1})
        return Action(ActionType.REST, {})


class TestProbeDiscriminates(unittest.TestCase):
    """The instrument's whole point: a learner diverges between the worlds, a
    replayer does not. We drive it with mock brains so the result is deterministic
    and independent of any real model."""

    def _bank_run(self, *, cf_enabled, grounded):
        sim = _economy_sim(cf=CounterfactualConfig(
            enabled=cf_enabled, rule="demurrage", demurrage_per_day=0.2,
            hide_rate=True), n=3)
        bank = next(f for f in sim.world.facilities if f.ftype is FacilityType.BANK)
        banker, *savers = sim.agents
        banker.pos = bank.pos
        sim.brains[banker.id] = _Saver(banker.id, saves=False)   # the bank itself
        for s in savers:
            s.pos = bank.pos                    # standing at the bank, can deposit
            s.add("food", 999)
            s.money = 999
            sim.brains[s.id] = _Saver(banker.id, saves=True, grounded=grounded)
        sim.run()
        return behaviour_rate(sim, "deposit")

    def test_a_grounded_saver_deposits_less_when_savings_shrink(self):
        control = self._bank_run(cf_enabled=False, grounded=True)
        counterfactual = self._bank_run(cf_enabled=True, grounded=True)
        self.assertGreater(control, counterfactual,
                           "having lived the loss, the grounded saver pulls back")

    def test_a_replaying_saver_does_not_diverge(self):
        control = self._bank_run(cf_enabled=False, grounded=False)
        counterfactual = self._bank_run(cf_enabled=True, grounded=False)
        self.assertAlmostEqual(control, counterfactual, places=6,
                               msg="ignoring experience → identical behaviour, no grounding")


class TestGroundingSandbox(unittest.TestCase):
    """The minimal deposit-decision world — a curriculum rung between a trivial
    bandit and the full town, so a small policy can learn the demurrage contingency
    without 40 facilities and 44 actions of noise."""

    def test_builds_a_minimal_world_with_a_staffed_bank(self):
        from emergence.grounding import make_grounding_sandbox
        from emergence.world import FacilityType
        sim = make_grounding_sandbox("guardian", n_savers=3, seed=1, days=6)
        ftypes = {f.ftype for f in sim.world.facilities}
        self.assertEqual(ftypes, {FacilityType.BANK, FacilityType.FARM,
                                  FacilityType.HOUSE})
        self.assertEqual(len(sim.agents), 4)                 # 1 banker + 3 savers
        bank = next(f for f in sim.world.facilities if f.ftype is FacilityType.BANK)
        self.assertEqual(sim.agents[0].pos, bank.pos, "the banker staffs the bank")
        self.assertTrue(all(s.money > 0 for s in sim.agents[1:]), "savers are funded")

    def test_the_deposit_decision_is_dense(self):
        from emergence.grounding import make_grounding_sandbox
        sim = make_grounding_sandbox("guardian", n_savers=3, seed=1, days=12)
        sim.run()
        deposits = sum(1 for e in sim.world.events if e["kind"] == "deposit")
        self.assertGreater(deposits, 0, "the isolated world exercises depositing")

    def test_probe_runs_in_the_sandbox(self):
        result = run_grounding_probe("guardian", sandbox=True, days=8, n_agents=4,
                                     seed=1)
        self.assertEqual(result.target, "deposit")
        self.assertEqual(result.excess, 0.0)                 # heuristic is its own floor

    def test_sandbox_rejects_non_demurrage_rules(self):
        with self.assertRaises(ValueError):
            run_grounding_probe("guardian", rule="vanity", sandbox=True, days=4)


class TestProbeEndToEnd(unittest.TestCase):
    def test_offline_probe_runs_and_is_well_formed(self):
        result = run_grounding_probe("guardian", days=6, n_agents=5, seed=1)
        self.assertIsInstance(result, GroundingResult)
        self.assertEqual(result.target, "deposit")
        self.assertGreaterEqual(result.control_rate, 0.0)
        self.assertGreaterEqual(result.counterfactual_rate, 0.0)
        # With no brain_factory the tested brain *is* the heuristic floor, so the
        # excess is zero by construction and the verdict says so.
        self.assertEqual(result.excess, 0.0)
        self.assertEqual(result.verdict, "baseline (heuristic floor)")
        self.assertEqual(result.as_dict()["rule"], "demurrage")

    def test_a_learning_brain_shows_positive_excess_over_the_floor(self):
        # A brain that wraps the heuristic but stops depositing once it has lived
        # through a demurrage loss should diverge MORE than the bare heuristic —
        # and the probe should credit only that excess, not the mechanical floor.
        from emergence.brains.heuristic import HeuristicBrain

        class _Grounded(AgentBrain):
            def __init__(self, inner):
                self.inner = inner

            def decide(self, agent, obs):
                act = self.inner.decide(agent, obs)
                if act.type == ActionType.DEPOSIT and \
                        any("vanished" in m for m in agent.memory):
                    return Action(ActionType.REST, {})
                return act

        def factory(agent, persona, rng):
            return _Grounded(HeuristicBrain(persona, rng))

        result = run_grounding_probe("guardian", days=20, n_agents=6, seed=42,
                                     brain_factory=factory)
        self.assertGreater(result.excess, 0.0,
                           "learning from the loss diverges beyond the mechanical floor")
        self.assertTrue(result.verdict.startswith("grounded"))

    def test_vanity_probe_runs_end_to_end_with_the_status_layer(self):
        # The vanity rule needs the honour layer; the probe must wire it and
        # score "feast" without crashing (heuristic floor → excess 0).
        result = run_grounding_probe("guardian", rule="vanity", days=6,
                                     n_agents=5, seed=1)
        self.assertEqual(result.rule, "vanity")
        self.assertEqual(result.target, "feast")
        self.assertEqual(result.excess, 0.0)
        self.assertEqual(result.verdict, "baseline (heuristic floor)")

    def test_unknown_rule_is_rejected(self):
        with self.assertRaises(ValueError):
            run_grounding_probe("guardian", rule="no_such_rule", days=2)


if __name__ == "__main__":
    unittest.main()
