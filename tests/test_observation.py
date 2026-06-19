"""Guards for the lean per-other observation view.

`_observe` builds `obs.others` as a lean dict (not a full snapshot) for speed.
That is only safe if it still carries every field a brain reads — so pin the
set here. The four-society contract test is the deeper guarantee; this gives a
clear, fast signal if a needed field goes missing.
"""

import unittest

from emergence.agent import Agent
from emergence.scenario import make_simulation
from emergence.simulation import SimulationConfig

# Fields the heuristic brain reads off other agents (see brains/heuristic.py).
REQUIRED_OTHER_FIELDS = {
    "id", "distance", "trust", "money", "food", "materials",
    "hunger", "fatigue", "age_days", "reputation", "last_crime_day",
}


class TestLeanOthers(unittest.TestCase):
    def test_others_carry_every_field_the_brain_reads(self):
        sim = make_simulation("gemini", config=SimulationConfig(seed=42))
        a = sim.agents[0]
        obs = sim._observe(a)
        self.assertTrue(obs.others)
        for o in obs.others:
            missing = REQUIRED_OTHER_FIELDS - o.keys()
            self.assertEqual(missing, set(), f"lean other view dropped {missing}")

    def test_threshold_fields_keep_snapshot_rounding(self):
        # hunger/fatigue/reputation are compared to thresholds, so they must be
        # rounded exactly as snapshot() does (1 decimal) for byte-identical
        # decisions.
        sim = make_simulation("gemini", config=SimulationConfig(seed=42))
        a = sim.agents[0]
        other = sim.agents[1]
        other.hunger = 85.04   # would cross a <=85 threshold if left unrounded
        o = next(x for x in sim._observe(a).others if x["id"] == other.id)
        self.assertEqual(o["hunger"], round(other.hunger, 1))


if __name__ == "__main__":
    unittest.main()
