import unittest

from emergence.actions import Action, ActionType
from emergence.agent import Agent
from emergence.esteem import StatusConfig, esteem_urge
from emergence.scenario import make_simulation
from emergence.simulation import Simulation, SimulationConfig
from emergence.world import World


def _agent(**kw):
    base = dict(id="x", name="X", profession="tester", persona="guardian", x=0, y=0)
    base.update(kw)
    return Agent(**base)


def _sim(agents, **status_kw):
    return Simulation(world=World(6, 6), agents=agents, brains={},
                      status=StatusConfig(enabled=True, **status_kw))


class TestEsteemUrge(unittest.TestCase):
    def test_disabled_by_default(self):
        a = _agent(esteem=90)
        self.assertEqual(esteem_urge(a, StatusConfig()), 0.0)

    def test_zero_below_threshold(self):
        cfg = StatusConfig(enabled=True, esteem_threshold=40)
        self.assertEqual(esteem_urge(_agent(esteem=30), cfg), 0.0)

    def test_grows_with_esteem(self):
        cfg = StatusConfig(enabled=True, esteem_threshold=40)
        low, high = _agent(esteem=50), _agent(esteem=95)
        self.assertGreater(esteem_urge(high, cfg), esteem_urge(low, cfg))
        self.assertLessEqual(esteem_urge(high, cfg), 1.0)

    def test_esteem_builds_in_upkeep(self):
        a = _agent()
        sim = _sim([a], esteem_per_tick=6)
        sim._tick_upkeep(a)
        self.assertEqual(a.esteem, 6.0)

    def test_esteem_does_not_build_when_disabled(self):
        a = _agent()
        sim = Simulation(world=World(4, 4), agents=[a], brains={})
        sim._tick_upkeep(a)
        self.assertEqual(a.esteem, 0.0)


class TestPraise(unittest.TestCase):
    def test_praise_relieves_esteem_and_grants_honour(self):
        a, b = _agent(id="a"), _agent(id="b", esteem=80)
        sim = _sim([a, b], praise_relief=30, rep_per_praise=2)
        sim._do_praise(a, Action(ActionType.PRAISE, {"target": "b"}))
        self.assertLess(b.esteem, 80)            # recognition received
        self.assertEqual(b.reputation, 2)        # honour earned
        self.assertGreater(b.pleasure, 0)        # 褒められて気持ちいい
        self.assertEqual(b.praise_received, 1)
        self.assertEqual(a.praise_given, 1)
        self.assertEqual(sim.metrics.total_praise, 1)

    def test_praise_builds_trust(self):
        a, b = _agent(id="a"), _agent(id="b")
        sim = _sim([a, b])
        sim._do_praise(a, Action(ActionType.PRAISE, {"target": "b"}))
        self.assertGreater(b.trust_of("a"), 0)

    def test_cannot_praise_self(self):
        a = _agent(id="a", esteem=80)
        sim = _sim([a])
        sim._do_praise(a, Action(ActionType.PRAISE, {"target": "a"}))
        self.assertEqual(a.praise_received, 0)

    def test_praise_noop_when_status_disabled(self):
        a, b = _agent(id="a"), _agent(id="b", esteem=80)
        sim = Simulation(world=World(4, 4), agents=[a, b], brains={})
        sim._do_praise(a, Action(ActionType.PRAISE, {"target": "b"}))
        self.assertEqual(b.esteem, 80)


class TestHonourAndPower(unittest.TestCase):
    def test_reputation_decays_each_day(self):
        a = _agent(reputation=10)
        sim = _sim([a], rep_decay_per_day=1.5)
        sim._end_of_day(verbose=False)
        self.assertLess(a.reputation, 10)

    def test_monument_grants_reputation(self):
        from emergence.world import Facility, FacilityType
        a = _agent(id="a", x=0, y=0)
        a.inventory["materials"] = 5
        world = World(6, 6)
        world.add_facility(Facility("Plaza", FacilityType.PLAZA, 0, 0))
        sim = Simulation(world=world, agents=[a], brains={},
                         status=StatusConfig(enabled=True, rep_per_monument=9))
        sim._do_build(a, Action(ActionType.BUILD,
                                {"facility_type": "monument", "name": "Spire"}))
        self.assertGreater(a.reputation, 0)

    def test_status_run_is_deterministic(self):
        a = make_simulation("guardian", config=SimulationConfig(seed=3),
                            status=StatusConfig(enabled=True)).run()
        b = make_simulation("guardian", config=SimulationConfig(seed=3),
                            status=StatusConfig(enabled=True)).run()
        self.assertEqual(a.as_dict(), b.as_dict())


class TestEsteemEndToEnd(unittest.TestCase):
    def _run(self, persona):
        sim = make_simulation(persona, config=SimulationConfig(seed=42),
                              status=StatusConfig(enabled=True))
        sim.run()
        return sim

    def test_cooperative_society_has_a_recognition_economy(self):
        # Guardians readily praise each other.
        sim = self._run("guardian")
        self.assertGreater(sim.metrics.total_praise, 0)

    def test_predators_barely_praise(self):
        # Cold, violent agents hoard esteem and rarely commend anyone.
        guardian = self._run("guardian")
        predator = self._run("predator")
        self.assertLess(predator.metrics.total_praise, guardian.metrics.total_praise)

    def test_baseline_unaffected_when_disabled(self):
        # Without status, no praise happens and outcomes match the article.
        sim = make_simulation("guardian", config=SimulationConfig(seed=42))
        sim.run()
        self.assertEqual(sim.metrics.total_praise, 0)
        self.assertEqual(sim.metrics.crimes_total, 0)
        self.assertEqual(sim.metrics.survivors, sim.metrics.population)


if __name__ == "__main__":
    unittest.main()
