"""Deposit interest: savings grow, which is what draws deposits in.

Banking was dormant because there was no reason to deposit — money just sat.
Now each day a bank pays interest *in coin* to its depositors, from its reserves
(conserved — no money minted). A bank funds this by lending those reserves at a
higher rate, so a banker earns the spread; one that over-lends can't cover the
interest, the first tremor before a run. And a capital-rich, secure agent will
*set up as a banker* to earn that spread — which is what brings deposits (and the
notes that ride on them) alive. Guards:
  * interest moves coin bank → depositor (conserved); the claim is unchanged;
  * a bank with no reserves pays nothing (it has over-extended);
  * a tiny deposit that rounds to zero earns nothing;
  * the rate is surfaced so savers know saving pays;
  * the heuristic: a secure capitalist mans a bank and lends its reserves;
  * everything is gated on --economy (offline baseline byte-identical).
"""

import unittest

from emergence.actions import ActionType
from emergence.brains.heuristic import BANKER_CAPITAL, HeuristicBrain
from emergence.market import DEPOSIT_INTEREST_PER_DAY, Deposit
from emergence.observation import Observation
from emergence.scenario import make_simulation
from emergence.simulation import SimulationConfig


def _sim(economy=True):
    return make_simulation("guardian", config=SimulationConfig(seed=1), economy=economy)


def _claim(sim, holder, banker, amount):
    dep = Deposit(id=sim._next_deposit_id, bank=banker.id, holder=holder.id, amount=amount)
    sim._next_deposit_id += 1
    sim.deposits.append(dep)
    return dep


class TestDepositInterest(unittest.TestCase):
    def test_interest_pays_coin_from_reserves_conserved(self):
        sim = _sim()
        banker, saver = sim.agents[0], sim.agents[1]
        banker.money, saver.money = 50, 0
        dep = _claim(sim, saver, banker, 20)
        total = banker.money + saver.money
        sim._pay_deposit_interest()
        due = round(20 * DEPOSIT_INTEREST_PER_DAY)
        self.assertEqual(saver.money, due, "the saver's money grows by the interest")
        self.assertEqual(banker.money, 50 - due, "the bank pays it from its reserves")
        self.assertEqual(banker.money + saver.money, total, "coin is conserved — none minted")
        self.assertEqual(dep.amount, 20, "the claim itself is unchanged")

    def test_a_bank_with_no_reserves_cannot_pay(self):
        sim = _sim()
        banker, saver = sim.agents[0], sim.agents[1]
        banker.money, saver.money = 0, 0           # lent everything out
        _claim(sim, saver, banker, 20)
        sim._pay_deposit_interest()
        self.assertEqual(saver.money, 0, "no reserves → no interest (a tremor before a run)")

    def test_tiny_deposit_rounding_to_zero_earns_nothing(self):
        sim = _sim()
        banker, saver = sim.agents[0], sim.agents[1]
        banker.money, saver.money = 50, 0
        _claim(sim, saver, banker, 3)              # 3 * 0.08 = 0.24 -> rounds to 0
        sim._pay_deposit_interest()
        self.assertEqual(saver.money, 0)
        self.assertEqual(banker.money, 50)

    def test_rate_is_surfaced_to_savers(self):
        sim = _sim()
        obs = sim._observe(sim.agents[0])
        self.assertEqual(obs.economy["deposit_rate"], DEPOSIT_INTEREST_PER_DAY)

    def test_offline_baseline_pays_no_interest(self):
        sim = _sim(economy=False)
        banker, saver = sim.agents[0], sim.agents[1]
        banker.money, saver.money = 50, 0
        _claim(sim, saver, banker, 20)
        sim._end_of_day(verbose=False)             # the daily hook runs but economy is off
        self.assertEqual(saver.money, 0)
        self.assertEqual(banker.money, 50)


class TestHeuristicBanker(unittest.TestCase):
    def _obs(self, *, bank_here=None, here_bank=False, near_bank=True,
             fear_level=0.0, offers=None):
        return Observation(
            day=1, tick=1, self_view={}, position=(0, 0),
            nearby_facilities=([{"type": "bank", "distance": 3}] if near_bank else []),
            here={"type": "bank", "name": "First Bank"} if here_bank else None,
            others=[], open_proposals=[], granary_food=0, recent_events=[],
            debts=[], fear_level=fear_level,
            economy={"enabled": True, "bank_here": bank_here,
                     "deposit_rate": DEPOSIT_INTEREST_PER_DAY},
            open_offers=offers or [])

    def _agent(self, money, *, energy=100.0):
        a = _sim().agents[0]
        a.money = money
        a.energy = energy
        return HeuristicBrain("guardian"), a

    def test_capitalist_at_a_bank_lends_its_reserves(self):
        brain, a = self._agent(BANKER_CAPITAL + 4)
        act = brain._bank_action(a, self._obs(here_bank=True))
        self.assertIsNotNone(act)
        self.assertEqual(act.type, ActionType.OFFER)
        self.assertTrue(act.params.get("loan"))

    def test_capitalist_heads_to_an_unmanned_bank(self):
        brain, a = self._agent(BANKER_CAPITAL + 4)
        act = brain._bank_action(a, self._obs(here_bank=False, near_bank=True))
        self.assertEqual(act.type, ActionType.MOVE)
        self.assertEqual(act.params["facility_type"], "bank")

    def test_does_not_set_up_where_a_banker_already_serves(self):
        # bank_here set → someone already mans it; be a customer, not a rival.
        brain, a = self._agent(BANKER_CAPITAL + 4)
        act = brain._bank_action(a, self._obs(bank_here="b1", here_bank=False))
        self.assertNotEqual(act.type if act else None, ActionType.MOVE)

    def test_the_poor_or_afraid_do_not_set_up(self):
        brain, a = self._agent(5)                       # too little capital
        self.assertIsNone(brain._bank_action(a, self._obs()))
        brain, a = self._agent(BANKER_CAPITAL + 4)
        self.assertIsNone(brain._bank_action(a, self._obs(fear_level=0.6)))


if __name__ == "__main__":
    unittest.main()
