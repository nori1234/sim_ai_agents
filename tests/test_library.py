"""The town library: cultural inheritance that outlives its authors.

Guards two things: (1) the stdlib book store behaves (write/dedup/persist/
relevance), and (2) wiring it into a sim is purely additive — a heuristic run
with the library on is byte-identical to one with it off (the brain ignores the
knowledge view), so the offline baseline is untouched.
"""

import unittest

from emergence.drives import DRIVES_ON
from emergence.library import TownLibrary
from emergence.scenario import make_simulation
from emergence.simulation import SimulationConfig
from emergence.world import FacilityType


class TestTownLibrary(unittest.TestCase):
    def test_write_persist_dedup(self):
        lib = TownLibrary()
        lib.write(1, "a1", "Aria", "winter starves the careless; stockpile food")
        lib.write(2, "a2", "Bao", "winter starves the careless; stockpile food")  # dup
        lib.write(3, "a3", "Caro", "the market rewards the patient trader")
        self.assertEqual(len(lib), 2, "identical lessons should de-duplicate")

    def test_book_outlives_its_author(self):
        # A book is a world artifact, not tied to the author's life — it stays on
        # the shelf and readable after the author is gone.
        lib = TownLibrary()
        lib.write(1, "a1", "Aria", "build a granary before the frost")
        got = lib.read("granary frost winter", k=3)
        self.assertTrue(got and "Aria" in got[0])

    def test_read_ranks_by_relevance(self):
        lib = TownLibrary()
        lib.write(1, "a1", "Aria", "the mine yields ore for tools")
        lib.write(2, "a2", "Bao", "guard the granary against thieves in winter")
        top = lib.read("winter granary thieves", k=1)
        self.assertEqual(len(top), 1)
        self.assertIn("Bao", top[0])

    def test_empty_read(self):
        self.assertEqual(TownLibrary().read("anything"), [])


class TestLibraryWiring(unittest.TestCase):
    def _metrics(self, **kw):
        sim = make_simulation("guardian", config=SimulationConfig(seed=42), **kw)
        sim.run()
        return sim

    def test_offline_baseline_is_byte_identical_with_library_on(self):
        off = self._metrics(library=False).metrics.as_dict()
        on_sim = self._metrics(library=True)
        on = on_sim.metrics.as_dict()
        self.assertEqual(off, on, "library must not perturb the heuristic baseline")
        # And the shelf actually filled during the run (agents visited the library).
        self.assertGreater(len(on_sim.library), 0, "expected some books to be written")

    def test_no_library_means_no_store(self):
        self.assertIsNone(make_simulation("guardian").library)


class TestKnowledgeFlows(unittest.TestCase):
    """The two channels: studying a book (horizontal) and inheriting a lesson
    from an elder (vertical), both landing in the agent's evolving memory."""

    def test_studying_internalises_a_book(self):
        sim = make_simulation("guardian", config=SimulationConfig(seed=1),
                              library=True)
        sim.library.write(1, "x", "Old Mira", "build a granary before the frost")
        agent = sim.agents[0]
        lib = next(f for f in sim.world.facilities
                   if f.ftype is FacilityType.LIBRARY)
        agent.x, agent.y = lib.x, lib.y          # stand in the library
        sim._library_study(agent)
        self.assertTrue(any(m.startswith("I read in the library:") for m in agent.memory),
                        "studying should internalise a predecessor's lesson")

    def test_parent_passes_a_lesson_to_child_without_nesting(self):
        sim = make_simulation("guardian", config=SimulationConfig(seed=1),
                              drives=DRIVES_ON, library=True)
        a, b = sim.agents[0], sim.agents[1]
        # b's lesson is itself already a transmitted one — the child must inherit
        # the core, not a nested "taught me: taught me: ..." chain.
        a.remember("stockpile food before winter")
        b.remember("My elder Ono taught me: guard the granary")
        sim._spawn_child(a, b)
        child = sim.agents[-1]
        taught = [m for m in child.memory if "taught me:" in m]
        self.assertTrue(taught, "a newborn should inherit an elder's lesson")
        self.assertTrue(any("stockpile food before winter" in m for m in taught))
        self.assertFalse(any(m.count("taught me:") > 1 for m in child.memory),
                         "transmitted lessons must not nest across generations")


if __name__ == "__main__":
    unittest.main()
