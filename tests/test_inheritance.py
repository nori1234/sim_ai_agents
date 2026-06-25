"""Inheritance & estates (相続): a death is settled, not a reset.

When an agent dies, its estate — coin, goods, and the bank-receipts it held —
passes to its **heirs** (living children, linked by `parent_ids`), split among
them; with **no heir** it **escheats to the treasury**. Everything moves by the
conserved transfer path (no money minted/vanished), so wealth can compound — and
concentrate — across generations. Opt-in under --economy, so the four-society
baseline is byte-identical. Guards:
  * heirs split the coin (conserved); goods & receipts go to the eldest heir;
  * with no heir, the coin escheats to the treasury and stray claims dissolve;
  * a dead depositor's receipt is reassigned to an heir (still redeemable);
  * off the economy layer, nothing is settled (estate untouched).
"""

import unittest

from emergence.market import Deposit
from emergence.scenario import make_simulation
from emergence.simulation import SimulationConfig


def _sim(economy=True):
    return make_simulation("guardian", n_agents=4,
                           config=SimulationConfig(seed=1), economy=economy)


def _claim(sim, holder, bank, amount):
    d = Deposit(id=sim._next_deposit_id, bank=bank.id, holder=holder.id, amount=amount)
    sim._next_deposit_id += 1
    sim.deposits.append(d)
    return d


class TestInheritance(unittest.TestCase):
    def test_heirs_split_the_coin_conserved(self):
        sim = _sim()
        dec, h1, h2 = sim.agents[0], sim.agents[1], sim.agents[2]
        dec.money = 11
        h1.money = h2.money = 0
        h1.parent_ids = (dec.id,)
        h2.parent_ids = (dec.id,)
        sim._settle_estate(dec)
        self.assertEqual(dec.money, 0, "the estate left the deceased")
        self.assertEqual(h1.money + h2.money, 11, "coin conserved across the heirs")
        self.assertEqual(sorted([h1.money, h2.money]), [5, 6], "split as evenly as possible")

    def test_goods_and_receipts_go_to_the_eldest_heir(self):
        sim = _sim()
        dec, young, old = sim.agents[0], sim.agents[1], sim.agents[2]
        dec.inventory["tools"] = 2
        for h in (young, old):
            h.parent_ids = (dec.id,)
            h.inventory["tools"] = 0
        young.age_days, old.age_days = 10, 40
        banker = sim.agents[3]
        _claim(sim, dec, banker, 7)
        sim._settle_estate(dec)
        self.assertEqual(old.inventory.get("tools", 0), 2, "the eldest takes the goods")
        self.assertEqual(young.inventory.get("tools", 0), 0)
        claim = next(d for d in sim.deposits if d.amount == 7)
        self.assertEqual(claim.holder, old.id, "the receipt is reassigned to the eldest heir")

    def test_no_heir_escheats_to_the_treasury(self):
        sim = _sim()
        dec, banker = sim.agents[0], sim.agents[1]
        dec.money = 9
        treasury_before = sim.treasury
        claim = _claim(sim, dec, banker, 4)
        sim._settle_estate(dec)              # no agent has dec as a parent
        self.assertEqual(dec.money, 0)
        self.assertEqual(sim.treasury, treasury_before + 9, "coin escheats to the state")
        self.assertEqual(claim.amount, 0, "an unclaimed receipt dissolves")

    def test_offline_baseline_estate_is_untouched(self):
        sim = _sim(economy=False)
        dec, heir = sim.agents[0], sim.agents[1]
        dec.money = 10
        heir.money = 0
        heir.parent_ids = (dec.id,)
        sim._settle_estate(dec)
        self.assertEqual(dec.money, 10, "economy off → no estate settlement")
        self.assertEqual(heir.money, 0)

    def test_inheritance_emerges_in_a_long_run(self):
        from emergence.drives import DrivesConfig
        sim = make_simulation("guardian", n_agents=10,
                              config=SimulationConfig(seed=0, days=100),
                              drives=DrivesConfig(enabled=True), economy=True)
        sim.run()
        settled = [e for e in sim.world.events if e.get("kind") in ("inheritance", "escheat")]
        self.assertTrue(settled, "elders' estates are settled as they pass away")


if __name__ == "__main__":
    unittest.main()
