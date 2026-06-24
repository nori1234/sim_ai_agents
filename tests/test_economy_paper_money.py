"""Deposit-receipts circulate: a bank-note can change hands and be redeemed.

A deposit-receipt (`market.Deposit`) is a *claim* the bank owes its holder. If
the holder can hand it to someone else — who can then redeem it (or pass it on)
— the receipt is a bearer note, and a trusted bank's notes can *become money*
emergently. This adds the keystone: `endorse` transfers a claim between agents,
conserved (the bank still owes the same total; only the holder changes). One
concrete use is wired: a coin-short debtor settles a loan by endorsing a note to
the creditor — a note functioning as money. Guards:
  * endorse moves the claim (holder changes), conserved; the bank's coin is untouched;
  * a partial endorse splits the claim; the recipient holds a redeemable note;
  * the recipient can withdraw the endorsed note (a note is as good as money);
  * you can't endorse more than you hold, to yourself, or out of reach;
  * repay falls back to settling a money debt with a note when coin is short;
  * everything is gated on --economy (offline baseline byte-identical).
"""

import unittest

from emergence.actions import Action, ActionType
from emergence.market import Deposit, Loan
from emergence.scenario import make_simulation
from emergence.simulation import SimulationConfig
from emergence.world import FacilityType


def _sim(economy=True):
    return make_simulation("guardian", config=SimulationConfig(seed=1), economy=economy)


def _bank_facility(sim):
    return next(f for f in sim.world.facilities if f.ftype is FacilityType.BANK)


def _claim(sim, holder, banker, amount):
    dep = Deposit(id=sim._next_deposit_id, bank=banker.id, holder=holder.id, amount=amount)
    sim._next_deposit_id += 1
    sim.deposits.append(dep)
    return dep


def _endorse(sim, frm, to, banker, amount):
    sim._do_endorse(frm, Action(ActionType.ENDORSE,
                    {"to": to.id, "bank": banker.id, "amount": amount}))


class TestEndorse(unittest.TestCase):
    def _trio(self):
        sim = _sim()
        banker, alice, bob = sim.agents[0], sim.agents[1], sim.agents[2]
        b = _bank_facility(sim)
        banker.x, banker.y = b.x, b.y
        alice.x, alice.y = 3, 3
        bob.x, bob.y = 3, 3            # alice & bob adjacent (same tile)
        return sim, banker, alice, bob

    def test_endorse_moves_the_claim_conserving_the_total(self):
        sim, banker, alice, bob = self._trio()
        banker.money = 12
        _claim(sim, alice, banker, 12)
        before_bank = banker.money
        _endorse(sim, alice, bob, banker, 12)
        a = next((d for d in sim.deposits if d.holder == alice.id), None)
        bclaim = next(d for d in sim.deposits if d.holder == bob.id)
        self.assertEqual(a.amount if a else 0, 0, "alice's claim is spent")
        self.assertEqual(bclaim.amount, 12, "bob now holds the note")
        self.assertEqual(banker.money, before_bank, "the bank's coin is untouched")

    def test_partial_endorse_splits_the_claim(self):
        sim, banker, alice, bob = self._trio()
        banker.money = 20
        _claim(sim, alice, banker, 20)
        _endorse(sim, alice, bob, banker, 8)
        self.assertEqual(next(d for d in sim.deposits if d.holder == alice.id).amount, 12)
        self.assertEqual(next(d for d in sim.deposits if d.holder == bob.id).amount, 8)

    def test_recipient_can_redeem_the_endorsed_note(self):
        sim, banker, alice, bob = self._trio()
        banker.money = 12
        _claim(sim, alice, banker, 12)
        _endorse(sim, alice, bob, banker, 10)
        # bob walks to the bank and redeems the note he was handed.
        bob.x, bob.y = banker.x, banker.y
        bob_before = bob.money
        sim._do_withdraw(bob, Action(ActionType.WITHDRAW, {"bank": banker.id, "amount": 10}))
        self.assertEqual(bob.money, bob_before + 10, "a note is as good as money at the bank")
        self.assertEqual(next(d for d in sim.deposits if d.holder == bob.id).amount, 0)

    def test_cannot_endorse_more_than_held(self):
        sim, banker, alice, bob = self._trio()
        _claim(sim, alice, banker, 5)
        _endorse(sim, alice, bob, banker, 9)
        self.assertEqual(next(d for d in sim.deposits if d.holder == alice.id).amount, 5)
        self.assertFalse(any(d.holder == bob.id for d in sim.deposits))

    def test_cannot_endorse_to_self_or_out_of_reach(self):
        sim, banker, alice, bob = self._trio()
        _claim(sim, alice, banker, 10)
        _endorse(sim, alice, alice, banker, 5)            # to self
        self.assertEqual(next(d for d in sim.deposits if d.holder == alice.id).amount, 10)
        bob.x, bob.y = 18, 18                              # far away
        _endorse(sim, alice, bob, banker, 5)
        self.assertFalse(any(d.holder == bob.id for d in sim.deposits))

    def test_offline_baseline_endorse_is_inert(self):
        sim = _sim(economy=False)
        banker, alice, bob = sim.agents[0], sim.agents[1], sim.agents[2]
        alice.x, alice.y = bob.x, bob.y = 3, 3
        _claim(sim, alice, banker, 10)
        _endorse(sim, alice, bob, banker, 5)
        self.assertFalse(any(d.holder == bob.id for d in sim.deposits))


class TestRepayWithNote(unittest.TestCase):
    def test_coin_short_debtor_settles_a_loan_with_a_note(self):
        sim = _sim()
        banker, creditor, debtor = sim.agents[0], sim.agents[1], sim.agents[2]
        creditor.x, creditor.y = 4, 4
        debtor.x, debtor.y = 4, 4                          # adjacent to hand the note
        debtor.money = 1                                   # short of coin
        _claim(sim, debtor, banker, 8)                     # but holds a note
        loan = Loan(id=sim._next_loan_id, creditor=creditor.id, debtor=debtor.id,
                    item="money", principal=5, repay=7, due_day=sim.world.day + 3)
        sim._next_loan_id += 1
        sim.loans.append(loan)
        sim._do_repay(debtor, Action(ActionType.REPAY, {"loan_id": loan.id}))
        self.assertTrue(loan.settled, "the debt is settled by endorsing a note")
        self.assertEqual(debtor.money, 1, "no coin moved — the note did")
        self.assertEqual(next(d for d in sim.deposits if d.holder == creditor.id).amount, 7,
                         "the creditor now holds the endorsed note")
        self.assertEqual(next(d for d in sim.deposits if d.holder == debtor.id).amount, 1)

    def test_no_coin_no_note_cannot_settle(self):
        sim = _sim()
        banker, creditor, debtor = sim.agents[0], sim.agents[1], sim.agents[2]
        creditor.x, creditor.y = debtor.x, debtor.y = 4, 4
        debtor.money = 1
        loan = Loan(id=sim._next_loan_id, creditor=creditor.id, debtor=debtor.id,
                    item="money", principal=5, repay=7, due_day=sim.world.day + 3)
        sim._next_loan_id += 1
        sim.loans.append(loan)
        sim._do_repay(debtor, Action(ActionType.REPAY, {"loan_id": loan.id}))
        self.assertFalse(loan.settled, "nothing to pay with → can't settle")


if __name__ == "__main__":
    unittest.main()
