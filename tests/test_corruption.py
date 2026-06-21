"""Corruption as emergence (#38).

Enforcement is already a guard's *choice* and a bribe is an ordinary `give`, so
corruption needs no new mechanic — only that it be legible. A wanted offender
slipping money to a guard is named a "bribe" (a log + metric); the corruption is
the guard's own later choice not to arrest. Recognition is purely additive, so
the heuristic baseline is byte-identical (the contract test guards it).
"""

import unittest

from emergence.actions import Action, ActionType
from emergence.scenario import make_simulation
from emergence.simulation import Simulation, SimulationConfig


def _give_money(sim: Simulation, giver, recipient, amount=4):
    giver.add("money", amount)
    sim._move_items(giver, recipient, {"money": amount}, kind="give", consent=True)


class TestBriberyIsRecognised(unittest.TestCase):
    def _sim(self):
        return make_simulation("guardian", config=SimulationConfig(seed=1))

    def test_wanted_offender_paying_a_guard_is_a_bribe(self):
        sim = self._sim()
        guard = next(a for a in sim.agents if a.profession == "guard")
        briber = next(a for a in sim.agents if a is not guard)
        briber.last_crime_day = sim.world.day        # freshly wanted
        _give_money(sim, briber, guard)
        self.assertEqual(sim.metrics.bribes, 1)
        self.assertTrue(any(e["kind"] == "bribe" for e in sim.world.events))

    def test_an_honest_gift_to_a_guard_is_not_a_bribe(self):
        sim = self._sim()
        guard = next(a for a in sim.agents if a.profession == "guard")
        giver = next(a for a in sim.agents if a is not guard)
        giver.last_crime_day = None                  # not wanted → just a gift
        _give_money(sim, giver, guard)
        self.assertEqual(sim.metrics.bribes, 0)

    def test_paying_a_non_guard_is_not_a_bribe(self):
        sim = self._sim()
        a, b = sim.agents[0], sim.agents[1]
        a.profession, b.profession = "farmer", "merchant"
        a.last_crime_day = sim.world.day
        _give_money(sim, a, b)
        self.assertEqual(sim.metrics.bribes, 0)


if __name__ == "__main__":
    unittest.main()
