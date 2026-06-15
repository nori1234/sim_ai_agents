import unittest

from emergence import publicworks as PW
from emergence.scenario import make_simulation
from emergence.simulation import SimulationConfig
from emergence.world import FacilityType


class TestPublicWorksData(unittest.TestCase):
    def test_parse_build_keywords(self):
        self.assertEqual(PW.parse_build("we should build a prison"), FacilityType.PRISON)
        self.assertEqual(PW.parse_build("a new granary please"), FacilityType.GRANARY)
        self.assertIsNone(PW.parse_build("ban theft"))

    def test_conditions_map_to_works(self):
        self.assertEqual(PW.proposed_work_for_conditions(
            recent_crimes=7, food_scarce=False, sick=False), "prison")
        self.assertEqual(PW.proposed_work_for_conditions(
            recent_crimes=3, food_scarce=False, sick=False), "police_station")
        self.assertEqual(PW.proposed_work_for_conditions(
            recent_crimes=0, food_scarce=True, sick=False), "granary")
        self.assertIsNone(PW.proposed_work_for_conditions(
            recent_crimes=0, food_scarce=False, sick=False))


class TestConstruction(unittest.TestCase):
    def _sim(self):
        return make_simulation("guardian", config=SimulationConfig(seed=1, days=1),
                               public_works=True)

    def test_build_spends_treasury_and_adds_facility(self):
        sim = self._sim()
        sim.treasury = PW.PUBLIC_WORKS_COST + 5
        before = len(sim.world.facilities_of(FacilityType.PRISON))
        sim._build_public_work("prison")
        self.assertEqual(len(sim.world.facilities_of(FacilityType.PRISON)), before + 1)
        self.assertEqual(sim.treasury, 5)

    def test_build_blocked_when_treasury_too_poor(self):
        sim = self._sim()
        sim.treasury = 1
        sim._build_public_work("prison")
        self.assertEqual(len(sim.world.facilities_of(FacilityType.PRISON)), 0)
        self.assertEqual(sim.treasury, 1)  # unspent

    def test_prison_deters_crime_more_than_nothing(self):
        from emergence.agent import Agent
        from emergence.world import Facility, World
        from emergence.simulation import Simulation
        world = World(10, 10)
        a = Agent(id="x", name="X", profession="t", persona="predator", x=5, y=5)
        sim = Simulation(world=world, agents=[a], brains={})
        self.assertFalse(sim._deterred(a))   # no police anywhere -> never deterred
        world.add_facility(Facility("Pen", FacilityType.PRISON, 5, 5))
        deterred = sum(sim._deterred(a) for _ in range(200))
        self.assertGreater(deterred, 0)      # right on top of a prison -> often deterred


class TestPublicWorksEndToEnd(unittest.TestCase):
    def test_off_is_byte_identical_baseline(self):
        sim = make_simulation("gemini", config=SimulationConfig(seed=42)); sim.run()
        self.assertEqual(sim.metrics.crimes_total, 133)
        self.assertEqual(sim.metrics.public_works_built, 0)

    def test_council_builds_when_enabled(self):
        sim = make_simulation("gemini", config=SimulationConfig(seed=42),
                              public_works=True)
        sim.run()
        self.assertGreater(sim.metrics.public_works_built, 0)

    def test_deterministic(self):
        a = make_simulation("gemini", config=SimulationConfig(seed=5), public_works=True)
        a.run()
        b = make_simulation("gemini", config=SimulationConfig(seed=5), public_works=True)
        b.run()
        self.assertEqual(a.metrics.as_dict(), b.metrics.as_dict())


if __name__ == "__main__":
    unittest.main()
