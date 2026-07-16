"""The town library: cultural inheritance that outlives its authors.

Guards two things: (1) the stdlib book store behaves (write/dedup/persist/
relevance), and (2) wiring it into a sim is purely additive — a heuristic run
with the library on is byte-identical to one with it off (the brain ignores the
knowledge view), so the offline baseline is untouched.
"""

import unittest

from emergence.drives import DRIVES_ON
from emergence.library import ROT_AGE_DAYS, TownLibrary
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

    def test_burn_clears_the_shelf(self):
        lib = TownLibrary()
        lib.write(1, "a1", "Aria", "build a granary before the frost")
        lib.write(2, "a2", "Bao", "the market rewards patience")
        self.assertEqual(lib.burn(), 2)
        self.assertEqual(len(lib), 0)

    def test_decay_removes_a_book_nobody_recopied(self):
        lib = TownLibrary()
        lib.write(1, "a1", "Aria", "build a granary before the frost")
        lost = lib.decay(1 + ROT_AGE_DAYS + 1)
        self.assertEqual(lost, 1)
        self.assertEqual(len(lib), 0)

    def test_decay_keeps_a_book_within_its_half_life(self):
        lib = TownLibrary()
        lib.write(1, "a1", "Aria", "build a granary before the frost")
        lost = lib.decay(1 + ROT_AGE_DAYS)  # exactly at the boundary: still fine
        self.assertEqual(lost, 0)
        self.assertEqual(len(lib), 1)

    def test_recopy_resets_the_decay_clock(self):
        lib = TownLibrary()
        lib.write(1, "a1", "Aria", "build a granary before the frost")
        lib.recopy(50)
        # Long after the original write, but well within the half-life since
        # the recopy -- the book should survive because it was tended.
        lost = lib.decay(50 + ROT_AGE_DAYS)
        self.assertEqual(lost, 0)
        self.assertEqual(len(lib), 1)

    def test_recopy_targets_the_most_neglected_book(self):
        lib = TownLibrary()
        lib.write(1, "a1", "Aria", "the oldest, most neglected lesson")
        lib.write(40, "a2", "Bao", "a fresher lesson")
        refreshed = lib.recopy(90)
        self.assertEqual(refreshed["text"], "the oldest, most neglected lesson")
        self.assertEqual(refreshed["refreshed_day"], 90)
        fresher = next(b for b in lib.books if b["author"] == "Bao")
        self.assertEqual(fresher["refreshed_day"], 40, "recopy should not touch the other book")

    def test_recopy_on_an_empty_shelf_is_a_no_op(self):
        self.assertIsNone(TownLibrary().recopy(1))


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

    def test_arson_burns_the_library_but_not_what_people_learned(self):
        sim = make_simulation("guardian", config=SimulationConfig(seed=1),
                              library=True)
        lib_f = next(f for f in sim.world.facilities
                     if f.ftype is FacilityType.LIBRARY)
        sim.library.write(1, "x", "Old Mira", "build a granary before the frost")
        reader = sim.agents[0]
        reader.remember("I read in the library: a lesson I now carry")
        before = len(reader.memory)
        arsonist = sim.agents[1]
        arsonist.x, arsonist.y = lib_f.x, lib_f.y
        sim._strike(arsonist, facility=lib_f)          # set the library alight
        self.assertEqual(len(sim.library), 0, "the public shelf should burn")
        self.assertTrue(any(e["kind"] == "library_burned" for e in sim.world.events))
        self.assertEqual(len(reader.memory), before,
                         "what a person already learned is not lost to the fire")

    def test_a_librarian_recopies_the_shelfs_most_neglected_book(self):
        sim = make_simulation("guardian", config=SimulationConfig(seed=1),
                              library=True)
        sim.library.write(1, "x", "Old Mira", "build a granary before the frost")
        lib_f = next(f for f in sim.world.facilities
                     if f.ftype is FacilityType.LIBRARY)
        librarian = sim.agents[0]
        librarian.profession = "librarian"
        librarian.x, librarian.y = lib_f.x, lib_f.y
        sim.world.day = 50
        sim._library_study(librarian)
        book = next(b for b in sim.library.books if b["author"] == "Old Mira")
        self.assertEqual(book["refreshed_day"], 50,
                         "a librarian standing in the library should recopy the neglected book")

    def test_a_non_librarian_studying_does_not_recopy(self):
        sim = make_simulation("guardian", config=SimulationConfig(seed=1),
                              library=True)
        sim.library.write(1, "x", "Old Mira", "build a granary before the frost")
        lib_f = next(f for f in sim.world.facilities
                     if f.ftype is FacilityType.LIBRARY)
        visitor = sim.agents[0]
        visitor.profession = "farmer"
        visitor.x, visitor.y = lib_f.x, lib_f.y
        sim.world.day = 50
        sim._library_study(visitor)
        book = next(b for b in sim.library.books if b["author"] == "Old Mira")
        self.assertEqual(book["refreshed_day"], 1, "only a librarian recopies")

    def test_end_of_day_lets_a_neglected_shelf_rot(self):
        sim = make_simulation("guardian", config=SimulationConfig(seed=1),
                              library=True)
        sim.library.write(1, "x", "Old Mira", "a lesson nobody ever recopies")
        sim.world.day = 1 + ROT_AGE_DAYS + 1
        sim._end_of_day(verbose=False)
        self.assertEqual(len(sim.library), 0, "an unmaintained book should have rotted away")
        self.assertTrue(any(e["kind"] == "library_rot" for e in sim.world.events))

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
