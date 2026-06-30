"""GroundingMonitor — periodic grounding probe over a training run.

The probe function is injected, so these tests are fast and deterministic without
torch or a real model: a fake probe returns a rising ``excess`` to stand in for a
brain that grounds more as it learns.
"""

import json
import os
import tempfile
import unittest

from emergence.grounding import GroundingResult
from emergence.grounding_monitor import GroundingMonitor


def _result(excess: float) -> GroundingResult:
    return GroundingResult(
        rule="demurrage", target="deposit", control_rate=0.2,
        counterfactual_rate=0.2 - excess, divergence=excess,
        floor_divergence=0.0, excess=excess,
        verdict="grounded (exceeds heuristic floor)" if excess > 0 else "baseline",
        days=20, n_agents=6)


class _RisingProbe:
    """A fake probe whose excess climbs each call — a brain that learns to ground."""

    def __init__(self):
        self.calls = 0

    def __call__(self, persona, *, rule, days, n_agents, seed, threshold,
                 brain_factory):
        self.calls += 1
        return _result(0.01 * self.calls)


class TestCadence(unittest.TestCase):
    def test_probes_only_on_due_epochs(self):
        probe = _RisingProbe()
        mon = GroundingMonitor(every=3, probe=probe)
        fired = [mon.maybe_probe(e, brain_factory=None) is not None
                 for e in range(7)]            # epochs 0..6
        self.assertEqual(fired, [True, False, False, True, False, False, True])
        self.assertEqual(probe.calls, 3)
        self.assertEqual([h["epoch"] for h in mon.history], [0, 3, 6])

    def test_every_must_be_positive(self):
        with self.assertRaises(ValueError):
            GroundingMonitor(every=0)


class TestRecordingAndTrend(unittest.TestCase):
    def test_history_carries_excess_and_verdict(self):
        mon = GroundingMonitor(probe=_RisingProbe())
        mon.probe(0, brain_factory=None)
        entry = mon.latest()
        self.assertEqual(entry["epoch"], 0)
        self.assertIn("excess", entry)
        self.assertIn("verdict", entry)

    def test_improving_detects_a_rising_excess(self):
        mon = GroundingMonitor(probe=_RisingProbe())
        for e in range(6):
            mon.probe(e, brain_factory=None)
        self.assertEqual(mon.excess_series(), [0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
        self.assertTrue(mon.improving())

    def test_not_improving_when_flat(self):
        mon = GroundingMonitor(probe=lambda *a, **k: _result(0.0))
        for e in range(4):
            mon.probe(e, brain_factory=None)
        self.assertFalse(mon.improving())

    def test_improving_needs_at_least_two_probes(self):
        mon = GroundingMonitor(probe=_RisingProbe())
        mon.probe(0, brain_factory=None)
        self.assertFalse(mon.improving())

    def test_on_result_callback_fires(self):
        seen = []
        mon = GroundingMonitor(probe=_RisingProbe(), on_result=seen.append)
        mon.probe(0, brain_factory=None)
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0]["excess"], 0.01)


class TestJsonl(unittest.TestCase):
    def test_writes_one_json_object_per_line(self):
        mon = GroundingMonitor(probe=_RisingProbe())
        for e in range(3):
            mon.probe(e, brain_factory=None)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "grounding.jsonl")
            mon.to_jsonl(path)
            with open(path) as fh:
                lines = [json.loads(ln) for ln in fh if ln.strip()]
        self.assertEqual(len(lines), 3)
        self.assertEqual([ln["epoch"] for ln in lines], [0, 1, 2])
        self.assertEqual(lines[2]["excess"], 0.03)


class TestUsesRealProbeByDefault(unittest.TestCase):
    """A smoke check that the default wiring calls the real grounding probe and
    produces a well-formed entry (heuristic floor → excess 0)."""

    def test_default_probe_runs_end_to_end(self):
        mon = GroundingMonitor("guardian", days=6, n_agents=5, seed=1)
        result = mon.probe(0, brain_factory=None)
        self.assertEqual(result.excess, 0.0)         # heuristic is its own floor
        self.assertEqual(mon.latest()["epoch"], 0)


if __name__ == "__main__":
    unittest.main()
