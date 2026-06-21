"""Audit cleanup (#44): trades/transfers route through the take/add primitive.

These institution-like money moves used to edit `.money` directly. They now go
through the same conserved primitive as any resource; net amounts are unchanged,
so the four-society baseline stays byte-identical (the contract test guards it).
"""

import unittest

from emergence.actions import Action, ActionType
from emergence.economy import apply_transfer as transfer
from emergence.scenario import make_simulation
from emergence.society import SocietyConfig
from emergence.simulation import SimulationConfig


class TestTransferHelper(unittest.TestCase):
    def test_money_moves_through_take_add_conserved(self):
        sim = make_simulation("guardian", config=SimulationConfig(seed=1))
        a, b = sim.agents[0], sim.agents[1]
        a.money, b.money = 10, 0
        ok, moved = transfer(a, b, "money", 4)
        self.assertTrue(ok)
        self.assertEqual((a.money, b.money, moved), (6, 4, 4))
        # Can't move more than is held (the take primitive caps it).
        ok, moved = transfer(a, b, "money", 99)
        self.assertEqual((moved, a.money, b.money), (6, 0, 10))


class TestDrugDealIsAConservedExchange(unittest.TestCase):
    def test_payment_is_conserved_not_minted(self):
        sim = make_simulation("grok", config=SimulationConfig(seed=1),
                              society=SocietyConfig(enabled=True))
        dealer, buyer = sim.agents[0], sim.agents[1]
        dealer.add("materials", 5)
        buyer.money, dealer.money = 10, 0
        buyer.x, buyer.y = dealer.x, dealer.y          # within deal range
        before = buyer.money + dealer.money
        sim._do_deal_drug(dealer, Action(ActionType.DEAL_DRUG, {"target": buyer.id}))
        self.assertEqual(sim.metrics.drug_deals, 1)
        self.assertEqual(buyer.money + dealer.money, before, "money must be conserved, not minted")
        self.assertEqual(dealer.money, 4)              # the price moved buyer → dealer
        self.assertEqual(buyer.money, 6)


if __name__ == "__main__":
    unittest.main()
