import os
import tempfile
import unittest

from emergence.memory_backend import MEMORY_AGENT_AVAILABLE
from emergence.scenario import make_simulation
from emergence.simulation import SimulationConfig

# These tests need the optional `memory-agent` library. They are skipped (not
# failed) when it isn't installed, so emergence stays zero-dependency for anyone
# who doesn't opt into the memory feature.
requires_lib = unittest.skipUnless(
    MEMORY_AGENT_AVAILABLE, "optional memory-agent library not installed"
)


class TestMemoryDisabledByDefault(unittest.TestCase):
    def test_no_memory_object_by_default(self):
        sim = make_simulation("guardian", config=SimulationConfig(seed=1, days=2))
        self.assertIsNone(sim.memory)


class TestMemoryBackedLibrarySelection(unittest.TestCase):
    """library=True picks the backend based on memory=True/False; this needs
    no optional dependency to check (it only inspects which class was built,
    and the memory=False branch never touches memory_backend at all)."""

    def test_library_without_memory_is_the_zero_dep_default(self):
        from emergence.library import TownLibrary
        sim = make_simulation("guardian", config=SimulationConfig(seed=1, days=2),
                              library=True, memory=False)
        self.assertIsInstance(sim.library, TownLibrary)

    @requires_lib
    def test_library_with_memory_is_backed_by_memory_agent(self):
        from emergence.memory_backend import MemoryBackedLibrary
        sim = make_simulation("guardian", config=SimulationConfig(seed=1, days=2),
                              library=True, memory=True)
        self.assertIsInstance(sim.library, MemoryBackedLibrary)
        sim.library.close(); sim.memory.close()


@requires_lib
class TestMemoryBackend(unittest.TestCase):
    def test_enabling_memory_does_not_change_outcomes(self):
        # The heuristic brain ignores recalled memory, so a memory-on run must
        # produce byte-identical metrics to a memory-off run.
        cfg = SimulationConfig(seed=42)
        base = make_simulation("gemini", config=cfg); base.run()
        mem = make_simulation("gemini", config=cfg, memory=True); mem.run()
        self.assertEqual(base.metrics.as_dict(), mem.metrics.as_dict())
        mem.memory.close()

    def test_memories_accumulate(self):
        sim = make_simulation("gemini", config=SimulationConfig(seed=42), memory=True)
        sim.run()
        self.assertGreater(sim.memory.total_active(), 0)
        sim.memory.close()

    def test_recall_returns_relevant_memories(self):
        sim = make_simulation("gemini", config=SimulationConfig(seed=42), memory=True)
        sim.run()
        # Every recalled string should mention the kind of thing we asked about
        # or the agent itself acting; at minimum recall returns some memories.
        hits = sim.memory.recall(sim.agents[0].id, "steal attack", k=5)
        self.assertTrue(hits)
        self.assertTrue(all(isinstance(s, str) for s in hits))
        sim.memory.close()

    def test_memory_run_is_deterministic(self):
        a = make_simulation("philosopher", config=SimulationConfig(seed=7), memory=True)
        a.run(); da = a.metrics.as_dict(); a.memory.close()
        b = make_simulation("philosopher", config=SimulationConfig(seed=7), memory=True)
        b.run(); db = b.metrics.as_dict(); b.memory.close()
        self.assertEqual(da, db)

    def test_memory_survives_across_sessions(self):
        path = tempfile.mktemp(suffix=".db")
        try:
            s1 = make_simulation("gemini", config=SimulationConfig(seed=42, days=4),
                                 memory=True, memory_path=path)
            s1.run()
            stored = s1.memory.total_active()
            s1.memory.close()
            self.assertGreater(stored, 0)

            # A brand-new simulation pointed at the same DB recalls the prior life.
            s2 = make_simulation("gemini", config=SimulationConfig(seed=99, days=1),
                                 memory=True, memory_path=path)
            recalled = s2.memory.recall(s2.agents[0].id, "day steal attack", k=3)
            s2.memory.close()
            self.assertTrue(recalled, "expected memories carried over from session 1")
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_newborns_get_a_memory_namespace(self):
        from emergence.drives import DrivesConfig
        sim = make_simulation("guardian", config=SimulationConfig(seed=42),
                              drives=DrivesConfig(enabled=True, reproduction=True),
                              memory=True)
        sim.run()
        # Population grew; every living agent should be queryable in memory.
        for a in sim.agents:
            if a.alive:
                # Should not raise; returns a (possibly empty) list.
                self.assertIsInstance(sim.memory.recall(a.id, "born", k=1), list)
        sim.memory.close()


@requires_lib
class TestMemoryBackedLibrary(unittest.TestCase):
    """MemoryBackedLibrary implements TownLibrary's write/read/burn/__len__
    port over memory-agent's GameWorld, for cross-run persistence +
    supersession (#23) instead of the zero-dep default."""

    def test_offline_baseline_is_byte_identical(self):
        # The heuristic brain ignores the knowledge view either way, so
        # swapping in the memory-backed shelf must not perturb outcomes.
        cfg = SimulationConfig(seed=42)
        off = make_simulation("guardian", config=cfg)
        off.run()
        on = make_simulation("guardian", config=cfg, library=True, memory=True)
        on.run()
        self.assertEqual(off.metrics.as_dict(), on.metrics.as_dict())
        on.library.close(); on.memory.close()

    def test_write_then_read_round_trips(self):
        from emergence.memory_backend import MemoryBackedLibrary
        lib = MemoryBackedLibrary()
        lib.write(1, "a1", "Aria", "build a granary before the frost")
        got = lib.read("granary frost winter", k=3)
        self.assertTrue(got and "Aria" in got[0])
        lib.close()

    def test_identical_text_is_deduplicated_within_a_run(self):
        from emergence.memory_backend import MemoryBackedLibrary
        lib = MemoryBackedLibrary()
        first = lib.write(1, "a1", "Aria", "winter starves the careless")
        dup = lib.write(2, "a2", "Bao", "winter starves the careless")
        self.assertIsNotNone(first)
        self.assertIsNone(dup)
        lib.close()

    def test_len_reflects_the_persisted_active_memory_count(self):
        from emergence.memory_backend import MemoryBackedLibrary
        lib = MemoryBackedLibrary()
        self.assertEqual(len(lib), 0)
        lib.write(1, "a1", "Aria", "build a granary before the frost")
        lib.write(2, "a2", "Bao", "the market rewards patience")
        self.assertEqual(len(lib), 2)
        lib.close()

    def test_burn_reports_the_shelf_as_emptied(self):
        from emergence.memory_backend import MemoryBackedLibrary
        lib = MemoryBackedLibrary()
        lib.write(1, "a1", "Aria", "build a granary before the frost")
        self.assertEqual(lib.burn(), 1)
        lib.close()

    def test_library_survives_across_sessions(self):
        from emergence.memory_backend import MemoryBackedLibrary
        path = tempfile.mktemp(suffix=".db")
        try:
            lib1 = MemoryBackedLibrary(path=path)
            lib1.write(1, "a1", "Aria", "build a granary before the frost")
            self.assertEqual(len(lib1), 1)
            lib1.close()

            # A brand-new instance pointed at the same DB inherits the shelf.
            lib2 = MemoryBackedLibrary(path=path)
            self.assertEqual(len(lib2), 1)
            recalled = lib2.read("granary frost", k=3)
            self.assertTrue(recalled, "expected the book carried over from session 1")
            lib2.close()
        finally:
            if os.path.exists(path):
                os.remove(path)


if __name__ == "__main__":
    unittest.main()
