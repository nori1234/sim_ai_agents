"""Banking on the economy layer: deposits as claim-items, withdrawals, runs.

A depositor hands money to a banker (an agent stationed at a BANK) and holds a
**deposit-receipt** — a claim the bank owes back. Money is conserved (the coin
sits in the banker's hands); the receipt is a promise, so if the banker spends
the funds a withdrawal comes up short — a run / embezzlement made visible.
Guards:
  * deposit moves coin to the banker (conserved) + records a claim;
  * banking requires a banker standing on a BANK facility;
  * withdrawal pays only from what the bank still holds (shortfall = run);
  * everything is gated on --economy (offline baseline byte-identical);
  * the heuristic's one rule: bank surplus / withdraw when short.
"""

import unittest

from emergence.actions import Action, ActionType
from emergence.brains.heuristic import HeuristicBrain
from emergence.market import Deposit
from emergence.observation import Observation
from emergence.scenario import make_simulation
from emergence.simulation import SimulationConfig
from emergence.world import FacilityType


def _sim(economy=True):
    return make_simulation("guardian", config=SimulationConfig(seed=1), economy=economy)


def _bank_facility(sim):
    return next(f for f in sim.world.facilities if f.ftype is FacilityType.BANK)


def _pair_at_bank(sim, *, banker_on_bank=True):
    banker, depositor = sim.agents[0], sim.agents[1]
    b = _bank_facility(sim)
    if banker_on_bank:
        banker.x, banker.y = b.x, b.y
    else:
        banker.x, banker.y = 0, 0
    depositor.x, depositor.y = banker.x, banker.y    # adjacent (same tile)
    return banker, depositor


def _deposit(sim, depositor, banker, amount):
    sim._do_deposit(depositor, Action(ActionType.DEPOSIT, {"bank": banker.id, "amount": amount}))


def _withdraw(sim, depositor, banker, amount):
    sim._do_withdraw(depositor, Action(ActionType.WITHDRAW, {"bank": banker.id, "amount": amount}))


class TestDeposit(unittest.TestCase):
    def test_deposit_moves_coin_and_records_a_claim(self):
        sim = _sim()
        banker, dep = _pair_at_bank(sim)
        banker.money, dep.money = 0, 20
        total = banker.money + dep.money
        _deposit(sim, dep, banker, 12)
        self.assertEqual(dep.money, 8, "the coin left the depositor")
        self.assertEqual(banker.money, 12, "the banker now holds it")
        self.assertEqual(banker.money + dep.money, total, "money conserved")
        claim = [d for d in sim.deposits if d.holder == dep.id and d.bank == banker.id]
        self.assertEqual(len(claim), 1)
        self.assertEqual(claim[0].amount, 12, "the depositor holds a 12 claim")

    def test_deposit_requires_a_bank_facility(self):
        sim = _sim()
        banker, dep = _pair_at_bank(sim, banker_on_bank=False)   # banker not at a BANK
        banker.money, dep.money = 0, 20
        _deposit(sim, dep, banker, 12)
        self.assertEqual(dep.money, 20, "no bank facility → no deposit")
        self.assertEqual(sim.deposits, [])

    def test_withdraw_pays_back_and_reduces_the_claim(self):
        sim = _sim()
        banker, dep = _pair_at_bank(sim)
        banker.money, dep.money = 0, 20
        _deposit(sim, dep, banker, 12)
        _withdraw(sim, dep, banker, 5)
        self.assertEqual(dep.money, 8 + 5)
        self.assertEqual(banker.money, 7)
        self.assertEqual(next(d for d in sim.deposits if d.holder == dep.id).amount, 7)

    def test_run_when_the_bank_spent_the_money(self):
        sim = _sim()
        banker, dep = _pair_at_bank(sim)
        banker.money, dep.money = 0, 20
        _deposit(sim, dep, banker, 12)
        banker.take("money", 12)             # the banker spent/embezzled the deposit
        before = dep.money
        _withdraw(sim, dep, banker, 12)
        self.assertEqual(dep.money, before, "nothing to pay out — a run")
        self.assertEqual(next(d for d in sim.deposits if d.holder == dep.id).amount, 12,
                         "the claim still stands, unhonoured")
        self.assertTrue(any(e.get("kind") == "bank_default" for e in sim.world.events))

    def test_partial_withdraw_is_a_partial_run(self):
        sim = _sim()
        banker, dep = _pair_at_bank(sim)
        banker.money, dep.money = 0, 20
        _deposit(sim, dep, banker, 12)
        banker.take("money", 9)              # only 3 left in the vault
        _withdraw(sim, dep, banker, 12)
        self.assertEqual(banker.money, 0)
        self.assertEqual(next(d for d in sim.deposits if d.holder == dep.id).amount, 9,
                         "got 3 back, 9 still owed")

    def test_offline_baseline_deposit_is_inert(self):
        sim = _sim(economy=False)
        banker, dep = _pair_at_bank(sim)
        banker.money, dep.money = 0, 20
        _deposit(sim, dep, banker, 12)
        self.assertEqual(dep.money, 20)
        self.assertEqual(sim.deposits, [])


class TestHeuristicBanking(unittest.TestCase):
    def _obs(self, *, bank_here=None, my_deposits=None):
        return Observation(
            day=1, tick=1, self_view={}, position=(0, 0), nearby_facilities=[],
            here=None, others=[], open_proposals=[], granary_food=0, recent_events=[],
            economy={"enabled": True, "bank_here": bank_here,
                     "my_deposits": my_deposits or []})

    def _agent(self, money):
        sim = _sim()
        a = sim.agents[1]; a.money = money; a.inventory["money"] = money
        return HeuristicBrain("guardian"), a

    def test_deposits_surplus_when_a_bank_is_open(self):
        brain, a = self._agent(20)
        act = brain._bank_action(a, self._obs(bank_here="b1"))
        self.assertIsNotNone(act)
        self.assertEqual(act.type, ActionType.DEPOSIT)
        self.assertEqual(act.params, {"bank": "b1", "amount": 12})   # keeps an 8 buffer

    def test_withdraws_when_short_and_a_claim_is_reachable(self):
        brain, a = self._agent(2)
        obs = self._obs(bank_here="b1", my_deposits=[{"id": 1, "bank": "b1", "amount": 9}])
        act = brain._bank_action(a, obs)
        self.assertEqual(act.type, ActionType.WITHDRAW)
        self.assertEqual(act.params, {"bank": "b1", "amount": 9})

    def test_nothing_without_a_bank_in_reach(self):
        brain, a = self._agent(20)
        self.assertIsNone(brain._bank_action(a, self._obs(bank_here=None)))


if __name__ == "__main__":
    unittest.main()
