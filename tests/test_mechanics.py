import unittest

from emergence.actions import Action, ActionType
from emergence.agent import Agent
from emergence.economy import apply_transfer, is_fraudulent_solicitation
from emergence.governance import Legislature, ProposalStatus
from emergence.scenario import make_simulation
from emergence.simulation import Simulation, SimulationConfig
from emergence.world import Facility, FacilityType, World


def _agent(**kw):
    base = dict(id="x", name="X", profession="tester", persona="guardian", x=0, y=0)
    base.update(kw)
    return Agent(**base)


class TestEconomy(unittest.TestCase):
    def test_transfer_moves_money(self):
        a, b = _agent(id="a", money=10), _agent(id="b", money=0)
        ok, moved = apply_transfer(a, b, "money", 4)
        self.assertTrue(ok)
        self.assertEqual((a.money, b.money, moved), (6, 4, 4))

    def test_transfer_capped_by_holdings(self):
        a, b = _agent(id="a", money=3), _agent(id="b", money=0)
        _, moved = apply_transfer(a, b, "money", 10)
        self.assertEqual(moved, 3)

    def test_fraud_detection(self):
        rich = _agent(money=20)
        poor = _agent(money=1)
        self.assertTrue(is_fraudulent_solicitation(rich, "money"))
        self.assertFalse(is_fraudulent_solicitation(poor, "money"))


class TestGovernance(unittest.TestCase):
    def test_proposal_passes_on_majority(self):
        leg = Legislature(quorum=3)
        p = leg.propose("a", "rule", day=1)
        for voter, yes in (("a", True), ("b", True), ("c", False)):
            leg.cast_vote(p.id, voter, yes)
        resolved = leg.resolve_ready(electorate_size=3)
        self.assertEqual(p.status, ProposalStatus.PASSED)
        self.assertIn(p, resolved)

    def test_proposal_rejected_on_minority(self):
        leg = Legislature(quorum=3)
        p = leg.propose("a", "rule", day=1)
        for voter, yes in (("a", True), ("b", False), ("c", False)):
            leg.cast_vote(p.id, voter, yes)
        leg.resolve_ready(electorate_size=3)
        self.assertEqual(p.status, ProposalStatus.REJECTED)


class TestAgentNeeds(unittest.TestCase):
    def test_eat_restores_energy(self):
        a = _agent(energy=20.0)
        a.inventory["food"] = 3
        world = World(4, 4)
        sim = Simulation(world=world, agents=[a], brains={})
        sim._do_eat(a, Action(ActionType.EAT))
        self.assertGreater(a.energy, 20.0)
        self.assertEqual(a.food(), 1)  # consumed 2

    def test_starvation_kills(self):
        a = _agent(energy=3.0)
        world = World(4, 4)
        sim = Simulation(world=world, agents=[a], brains={})
        sim._tick_upkeep(a)
        self.assertFalse(a.alive)
        self.assertEqual(a.cause_of_death, "starvation")

    def test_gather_food_at_farm(self):
        a = _agent()
        world = World(4, 4)
        world.add_facility(Facility("F", FacilityType.FARM, 0, 0))
        sim = Simulation(world=world, agents=[a], brains={})
        before = a.food()
        sim._do_gather(a, Action(ActionType.GATHER))
        self.assertGreater(a.food(), before)


class TestEndToEnd(unittest.TestCase):
    def test_guardian_town_is_orderly(self):
        sim = make_simulation("guardian", config=SimulationConfig(seed=42))
        m = sim.run()
        self.assertEqual(m.survivors, m.population)
        self.assertEqual(m.crimes_total, 0)

    def test_idealist_town_starves(self):
        sim = make_simulation("idealist", config=SimulationConfig(seed=42))
        m = sim.run()
        self.assertEqual(m.survivors, 0)
        # All deaths should be starvation, not violence.
        self.assertTrue(all(a.cause_of_death == "starvation"
                            for a in sim.agents if not a.alive))

    def test_run_is_deterministic(self):
        a = make_simulation("philosopher", config=SimulationConfig(seed=7)).run()
        b = make_simulation("philosopher", config=SimulationConfig(seed=7)).run()
        self.assertEqual(a.as_dict(), b.as_dict())

    def test_reproduces_distinct_societies(self):
        results = {}
        for key in ("guardian", "philosopher", "idealist", "predator"):
            results[key] = make_simulation(key, config=SimulationConfig(seed=42)).run()
        # Guardian is the only fully-surviving, crime-free society.
        self.assertEqual(results["guardian"].crimes_total, 0)
        self.assertGreater(results["philosopher"].crimes_total,
                           results["idealist"].crimes_total)
        self.assertGreater(results["predator"].crimes_total, 0)


if __name__ == "__main__":
    unittest.main()
