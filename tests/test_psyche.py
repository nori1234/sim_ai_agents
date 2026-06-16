import unittest

from emergence.actions import Action, ActionType
from emergence.agent import Agent
from emergence.psyche import PsycheConfig, actualization_pull, fear_level
from emergence.scenario import make_simulation
from emergence.simulation import Simulation, SimulationConfig
from emergence.world import Facility, FacilityType, World


def _agent(**kw):
    base = dict(id="x", name="X", profession="tester", persona="guardian", x=0, y=0)
    base.update(kw)
    return Agent(**base)


def _sim(agents, world=None, **psyche_kw):
    return Simulation(world=world or World(12, 12), agents=agents, brains={},
                      psyche=PsycheConfig(enabled=True, **psyche_kw))


class TestFear(unittest.TestCase):
    def test_disabled_by_default(self):
        a = _agent(fear=90)
        self.assertEqual(fear_level(a, PsycheConfig()), 0.0)

    def test_zero_below_threshold(self):
        cfg = PsycheConfig(enabled=True, fear_threshold=45)
        self.assertEqual(fear_level(_agent(fear=30), cfg), 0.0)

    def test_grows_with_fear(self):
        cfg = PsycheConfig(enabled=True, fear_threshold=45)
        low, high = _agent(fear=55), _agent(fear=95)
        self.assertGreater(fear_level(high, cfg), fear_level(low, cfg))
        self.assertLessEqual(fear_level(high, cfg), 1.0)

    def test_victim_is_struck_with_fear(self):
        off = _agent(id="off", x=0, y=0)
        vic = _agent(id="vic", x=1, y=0)
        sim = _sim([off, vic])
        sim._register_crime(off, "theft", vic)
        self.assertGreater(vic.fear, 0)

    def test_witnesses_catch_dread(self):
        off = _agent(id="off", x=0, y=0)
        vic = _agent(id="vic", x=1, y=0)
        near = _agent(id="near", x=3, y=0)   # within witness radius
        far = _agent(id="far", x=11, y=11)   # outside it
        sim = _sim([off, vic, near, far], witness_radius=6)
        sim._register_crime(off, "theft", vic)
        self.assertGreater(near.fear, 0)
        self.assertEqual(far.fear, 0)
        # The offender feels no fear of itself.
        self.assertEqual(off.fear, 0)

    def test_fear_decays_with_quiet_time(self):
        a = _agent(fear=50, x=11, y=11)  # away from any safety
        sim = _sim([a], fear_decay_per_tick=3)
        sim._tick_upkeep(a)
        self.assertLess(a.fear, 50)

    def test_fear_decays_faster_near_police(self):
        world = World(12, 12)
        world.add_facility(Facility("PS", FacilityType.POLICE_STATION, 0, 0))
        sheltered = _agent(id="a", fear=50, x=1, y=1)
        exposed = _agent(id="b", fear=50, x=11, y=11)
        sim = _sim([sheltered, exposed], world=world)
        e0 = sheltered.energy
        sim._tick_upkeep(sheltered)
        sim._tick_upkeep(exposed)
        self.assertLess(sheltered.fear, exposed.fear)

    def test_chronic_terror_is_stress(self):
        calm = _agent(id="a", fear=0, x=11, y=11)
        terrified = _agent(id="b", fear=90, x=11, y=11)
        sim = _sim([calm, terrified], fear_threshold=45)
        sim._tick_upkeep(calm)
        sim._tick_upkeep(terrified)
        self.assertLess(terrified.energy, calm.energy)

    def test_peak_fear_recorded(self):
        off = _agent(id="off", x=0, y=0)
        vic = _agent(id="vic", x=1, y=0)
        sim = _sim([off, vic])
        sim._register_crime(off, "theft", vic)
        self.assertGreater(sim.metrics.peak_fear, 0)


class TestSelfActualization(unittest.TestCase):
    def _content_agent(self, **kw):
        """An agent with every lower need at rest."""
        base = dict(hunger=5, fatigue=5, fear=0, esteem=10, energy=90, age_days=10)
        base.update(kw)
        return _agent(**base)

    def test_pull_present_when_all_needs_met(self):
        cfg = PsycheConfig(enabled=True)
        self.assertGreater(actualization_pull(self._content_agent(), cfg), 0)

    def test_hunger_blocks_creation(self):
        cfg = PsycheConfig(enabled=True, actualization_hunger_max=40)
        self.assertEqual(actualization_pull(self._content_agent(hunger=80), cfg), 0)

    def test_fear_blocks_creation(self):
        cfg = PsycheConfig(enabled=True, actualization_fear_max=20)
        self.assertEqual(actualization_pull(self._content_agent(fear=60), cfg), 0)

    def test_aching_esteem_blocks_creation(self):
        cfg = PsycheConfig(enabled=True, actualization_esteem_max=55)
        self.assertEqual(actualization_pull(self._content_agent(esteem=90), cfg), 0)

    def test_create_grants_fulfillment_and_pleasure(self):
        world = World(6, 6)
        world.add_facility(Facility("Lib", FacilityType.LIBRARY, 0, 0))
        a = self._content_agent()
        sim = _sim([a], world=world)
        sim._do_create(a, Action(ActionType.CREATE, {"title": "Opus"}))
        self.assertEqual(a.works_created, 1)
        self.assertGreater(a.fulfillment, 0)
        self.assertGreater(a.pleasure, 0)
        self.assertEqual(sim.metrics.works_created, 1)

    def test_create_requires_a_creative_venue(self):
        a = self._content_agent(x=3, y=3)  # empty tile
        sim = _sim([a])
        sim._do_create(a, Action(ActionType.CREATE))
        self.assertEqual(a.works_created, 0)

    def test_hungry_mind_cannot_create(self):
        world = World(6, 6)
        world.add_facility(Facility("Lib", FacilityType.LIBRARY, 0, 0))
        a = self._content_agent(hunger=90)
        sim = _sim([a], world=world)
        sim._do_create(a, Action(ActionType.CREATE))
        self.assertEqual(a.works_created, 0)


class TestPsycheEndToEnd(unittest.TestCase):
    def _run(self, persona):
        sim = make_simulation(persona, config=SimulationConfig(seed=42),
                              psyche=PsycheConfig(enabled=True))
        sim.run()
        return sim

    def test_peaceful_town_creates_in_peace(self):
        sim = self._run("guardian")
        self.assertGreater(sim.metrics.works_created, 0)
        self.assertEqual(sim.metrics.peak_fear, 0.0)

    def test_violent_town_lives_in_terror(self):
        sim = self._run("philosopher")
        self.assertGreater(sim.metrics.peak_fear, 50)
        # Terror suppresses creation relative to the peaceful town.
        peaceful = self._run("guardian")
        self.assertLess(sim.metrics.works_created, peaceful.metrics.works_created)

    def test_psyche_run_is_deterministic(self):
        a = self._run("philosopher").metrics.as_dict()
        b = self._run("philosopher").metrics.as_dict()
        self.assertEqual(a, b)

    def test_baseline_unaffected_when_disabled(self):
        sim = make_simulation("guardian", config=SimulationConfig(seed=42))
        sim.run()
        self.assertEqual(sim.metrics.works_created, 0)
        self.assertEqual(sim.metrics.peak_fear, 0.0)
        self.assertEqual(sim.metrics.crimes_total, 0)
        self.assertEqual(sim.metrics.survivors, sim.metrics.population)


if __name__ == "__main__":
    unittest.main()
