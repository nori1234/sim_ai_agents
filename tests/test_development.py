import unittest

from emergence import development as DEV
from emergence.scenario import make_simulation
from emergence.simulation import SimulationConfig
from emergence.world import Facility, FacilityType, World


class TestFoundingWorld(unittest.TestCase):
    def test_starts_sparse(self):
        w = DEV.founding_world()
        types = {f.ftype for f in w.facilities}
        self.assertIn(FacilityType.FARM, types)
        # The institutions a town must develop are NOT there at the founding.
        for absent in (FacilityType.MARKET, FacilityType.TOWN_HALL,
                       FacilityType.POLICE_STATION, FacilityType.LIBRARY):
            self.assertNotIn(absent, types)


class TestPrerequisites(unittest.TestCase):
    def test_chain_is_enforced(self):
        w = World(10, 10)
        # Nothing yet: a granary needs a farm; a town hall needs a granary.
        self.assertFalse(DEV.can_build(FacilityType.GRANARY, w))
        w.add_facility(Facility("F", FacilityType.FARM, 0, 0))
        self.assertTrue(DEV.can_build(FacilityType.GRANARY, w))
        self.assertFalse(DEV.can_build(FacilityType.TOWN_HALL, w))
        w.add_facility(Facility("G", FacilityType.GRANARY, 1, 1))
        self.assertTrue(DEV.can_build(FacilityType.TOWN_HALL, w))
        # A prison needs a police station first.
        self.assertFalse(DEV.can_build(FacilityType.PRISON, w))


class TestSuggestion(unittest.TestCase):
    def test_first_step_is_storage(self):
        sim = make_simulation("guardian", config=SimulationConfig(seed=1, days=1),
                              founding=True)
        self.assertEqual(DEV.next_public_work(sim), "granary")


class TestProsperity(unittest.TestCase):
    def test_in_range(self):
        sim = make_simulation("guardian", config=SimulationConfig(seed=1, days=2),
                              founding=True)
        sim.run()
        self.assertGreaterEqual(sim.metrics.prosperity, 0.0)
        self.assertLessEqual(sim.metrics.prosperity, 100.0)


class TestDevelopmentEndToEnd(unittest.TestCase):
    def _built(self, persona):
        sim = make_simulation(persona, config=SimulationConfig(seed=42), founding=True)
        sim.run()
        return sim, [e["type"] for e in sim.world.events if e["kind"] == "public_works"]

    def test_peaceful_town_climbs_to_civic_institutions(self):
        sim, built = self._built("claude")
        # Builds in order up to governance/knowledge; with zero crime, no prison.
        self.assertIn("granary", built)
        self.assertIn("town_hall", built)
        self.assertNotIn("prison", built)

    def test_violent_town_builds_law_and_order(self):
        sim, built = self._built("gemini")
        self.assertTrue("police_station" in built or "prison" in built)

    def test_cannot_skip_ahead(self):
        # A town hall is never built before its prerequisite granary.
        sim, built = self._built("claude")
        if "town_hall" in built and "granary" in built:
            self.assertLess(built.index("granary"), built.index("town_hall"))

    def test_baseline_has_no_prosperity_index(self):
        sim = make_simulation("guardian", config=SimulationConfig(seed=42)); sim.run()
        self.assertEqual(sim.metrics.prosperity, 0.0)  # only tracked when founding

    def test_deterministic(self):
        a, _ = self._built("gemini")
        b, _ = self._built("gemini")
        self.assertEqual(a.metrics.as_dict(), b.metrics.as_dict())


if __name__ == "__main__":
    unittest.main()
