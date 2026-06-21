"""Principled fiscality under --economy.

Tax, fines and redistribution should obey conservation when the economy layer is
on — money moves to/from the treasury rather than vanishing or being conjured.
Offline (economy off) keeps the legacy behaviour, so the four-society baseline
stays byte-identical (the deeper guarantee is `test_baseline_contract`).
"""

import unittest

from emergence.scenario import make_simulation
from emergence.simulation import SimulationConfig


def _world(economy, law_text):
    sim = make_simulation("guardian", config=SimulationConfig(seed=1), economy=economy)
    sim.policy.enact(1, law_text, 1)   # parse the bill into its keyword effect
    return sim


class TestPrincipledFiscality(unittest.TestCase):
    def test_tax_conserves_money_into_the_treasury(self):
        sim = _world(True, "a wealth tax on the rich")
        for i, a in enumerate(sim.agents):
            a.money = 10 + i * 5
        before = sum(a.money for a in sim.agents) + sim.treasury
        food0 = sim.world.granary_food
        sim._apply_daily_policy()
        self.assertEqual(sum(a.money for a in sim.agents) + sim.treasury, before,
                         "tax must conserve money (no vanishing)")
        self.assertEqual(sim.world.granary_food, food0, "no food should be conjured")
        self.assertGreater(sim.treasury, 0, "the tax should land in the treasury")

    def test_offline_tax_keeps_the_legacy_behaviour(self):
        sim = _world(False, "a wealth tax on the rich")
        for i, a in enumerate(sim.agents):
            a.money = 10 + i * 5
        food0 = sim.world.granary_food
        sim._apply_daily_policy()
        self.assertGreater(sim.world.granary_food, food0, "legacy conjures granary food")
        self.assertEqual(sim.treasury, 0, "legacy doesn't use the treasury")

    def test_welfare_pays_from_the_treasury(self):
        sim = _world(True, "a food quota for the needy")   # FOOD_REDISTRIBUTION
        sim.treasury = 30
        for a in sim.agents:
            a.money = 0
        sim._apply_daily_policy()
        self.assertLess(sim.treasury, 30, "welfare is paid out of the treasury")
        self.assertGreater(sum(a.money for a in sim.agents), 0, "the poor receive money")

    def test_welfare_pays_nothing_when_the_treasury_is_empty(self):
        sim = _world(True, "a food quota for the needy")
        sim.treasury = 0
        for a in sim.agents:
            a.money = 0
        sim._apply_daily_policy()
        self.assertEqual(sum(a.money for a in sim.agents), 0,
                         "can't redistribute what the state doesn't hold")


if __name__ == "__main__":
    unittest.main()
