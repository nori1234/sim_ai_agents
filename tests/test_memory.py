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


if __name__ == "__main__":
    unittest.main()
