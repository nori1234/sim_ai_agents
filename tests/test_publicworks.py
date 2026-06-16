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

    def test_arrest_punishes_a_recent_offender(self):
        # Enforcement is an act, not a building aura: a guard collars a nearby
        # offender, fining and detaining them.
        from emergence.actions import Action, ActionType
        from emergence.agent import Agent
        from emergence.world import World
        from emergence.simulation import Simulation
        world = World(10, 10)
        guard = Agent(id="g", name="G", profession="guard", persona="guardian", x=5, y=5)
        crook = Agent(id="c", name="C", profession="t", persona="predator",
                      x=5, y=5, money=20)
        sim = Simulation(world=world, agents=[guard, crook], brains={})
        crook.last_crime_day = world.day            # caught in the act window
        sim._do_arrest(guard, Action(ActionType.ARREST, {"target": "c"}))
        self.assertEqual(sim.metrics.arrests, 1)
        self.assertEqual(crook.times_arrested, 1)
        self.assertLess(crook.money, 20)            # fined
        self.assertIsNone(crook.last_crime_day)     # no longer wanted

    def test_arrest_spares_the_innocent(self):
        from emergence.actions import Action, ActionType
        from emergence.agent import Agent
        from emergence.world import World
        from emergence.simulation import Simulation
        world = World(10, 10)
        guard = Agent(id="g", name="G", profession="guard", persona="guardian", x=5, y=5)
        clean = Agent(id="c", name="C", profession="t", persona="idealist",
                      x=5, y=5, money=20)
        sim = Simulation(world=world, agents=[guard, clean], brains={})
        sim._do_arrest(guard, Action(ActionType.ARREST, {"target": "c"}))
        self.assertEqual(sim.metrics.arrests, 0)    # never offended -> not arrestable
        self.assertEqual(clean.money, 20)


class TestPublicWorksEndToEnd(unittest.TestCase):
    def test_off_is_byte_identical_baseline(self):
        sim = make_simulation("gemini", config=SimulationConfig(seed=42)); sim.run()
        self.assertEqual(sim.metrics.crimes_total, 211)  # Phase 2: enforcement is now an act
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
