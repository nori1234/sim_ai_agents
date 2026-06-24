"""Credit as a posted service: lending on the OFFER/ACCEPT order book.

A lender posts an *open loan* — "I lend N now, repay me M later" (M>N = interest)
— at a rate it picks, just like a goods or service offer. Whoever accepts takes
the principal now and owes the debt later (settled via REPAY). The lender must
hold the principal to post a credible offer; nothing is paid up front by the
borrower. The "price of money" emerges from which offers get taken. Guards:
  * a loan offer needs the maker to actually hold the principal;
  * accepting moves the principal lender->borrower and records a Loan (no upfront pay);
  * the posted loan settles through the existing REPAY path;
  * the accepted rate feeds the emergent interest reading;
  * everything is gated on --economy (offline baseline byte-identical);
  * the heuristic: a flush lender posts credit priced by temperament; a broke
    agent borrows the cheapest open loan in reach.
"""

import unittest

from emergence.actions import Action, ActionType
from emergence.brains.heuristic import HeuristicBrain
from emergence.market import Loan
from emergence.observation import Observation
from emergence.scenario import make_simulation
from emergence.simulation import SimulationConfig


def _sim(economy=True):
    return make_simulation("guardian", config=SimulationConfig(seed=1), economy=economy)


def _offer(sim, agent, **params):
    sim._do_offer(agent, Action(ActionType.OFFER, params))


def _accept(sim, agent, offer_id):
    sim._do_accept(agent, Action(ActionType.ACCEPT, {"offer_id": offer_id}))


class TestLoanOffer(unittest.TestCase):
    def _pair(self, sim):
        lender, borrower = sim.agents[0], sim.agents[1]
        lender.x, lender.y = borrower.x, borrower.y   # adjacent (same tile)
        return lender, borrower

    def test_loan_offer_shape(self):
        sim = _sim()
        lender, _ = self._pair(sim)
        lender.money = 10
        _offer(sim, lender, loan=True, item="money", principal=5, repay=7)
        self.assertEqual(len(sim.offers), 1)
        d = sim.offers[0].as_dict()
        self.assertEqual(d["loan"], True)
        self.assertEqual(d["item"], "money")
        self.assertEqual(d["principal"], 5)
        self.assertEqual(d["repay"], 7)

    def test_accept_moves_principal_records_loan_no_upfront_pay(self):
        sim = _sim()
        lender, borrower = self._pair(sim)
        lender.money, borrower.money = 10, 1
        total = lender.money + borrower.money
        _offer(sim, lender, loan=True, item="money", principal=5, repay=7)
        offer = sim.offers[0]
        _accept(sim, borrower, offer.id)
        self.assertEqual(borrower.money, 6, "borrower receives the principal now")
        self.assertEqual(lender.money, 5, "lender parted with the principal")
        self.assertEqual(lender.money + borrower.money, total, "money conserved")
        self.assertNotIn(offer, sim.offers, "the offer clears on accept")
        loan = next(l for l in sim.loans if l.debtor == borrower.id)
        self.assertEqual(loan.creditor, lender.id)
        self.assertEqual(loan.principal, 5)
        self.assertEqual(loan.repay, 7, "owes principal + interest")
        self.assertFalse(loan.settled)
        self.assertEqual(sim.metrics.loans_made, 1)

    def test_cannot_post_credit_without_the_principal(self):
        sim = _sim()
        lender, _ = self._pair(sim)
        lender.money = 3                       # can't back a 5 loan
        _offer(sim, lender, loan=True, item="money", principal=5, repay=7)
        self.assertEqual(sim.offers, [], "no principal in hand -> no credible offer")

    def test_posted_loan_settles_via_repay(self):
        sim = _sim()
        lender, borrower = self._pair(sim)
        lender.money, borrower.money = 10, 1
        _offer(sim, lender, loan=True, item="money", principal=5, repay=7)
        _accept(sim, borrower, sim.offers[0].id)
        loan = next(l for l in sim.loans if l.debtor == borrower.id)
        # borrower has 6 now; earns a bit more, then repays 7.
        borrower.money = 8
        sim._do_repay(borrower, Action(ActionType.REPAY, {"loan_id": loan.id}))
        self.assertTrue(loan.settled)
        self.assertEqual(borrower.money, 1)
        self.assertEqual(lender.money, 5 + 7, "lender gets principal back with interest")
        self.assertEqual(sim.metrics.loans_repaid, 1)

    def test_accepted_rate_feeds_emergent_interest(self):
        sim = _sim()
        lender, borrower = self._pair(sim)
        lender.money, borrower.money = 10, 1
        _offer(sim, lender, loan=True, item="money", principal=5, repay=7)
        _accept(sim, borrower, sim.offers[0].id)
        # the loan rate emerges as the accepted repay/principal ratio.
        self.assertAlmostEqual(sim.emergent_price("loan", "money"), 7 / 5)

    def test_offline_baseline_loan_offer_is_inert(self):
        sim = _sim(economy=False)
        lender, _ = self._pair(sim)
        lender.money = 10
        _offer(sim, lender, loan=True, item="money", principal=5, repay=7)
        self.assertEqual(sim.offers, [])


class TestHeuristicCredit(unittest.TestCase):
    def _obs(self, *, others=None, offers=None, debts=None):
        return Observation(
            day=1, tick=1, self_view={}, position=(0, 0), nearby_facilities=[],
            here=None, others=others or [], open_proposals=[], granary_food=0,
            recent_events=[], debts=debts or [],
            economy={"enabled": True}, open_offers=offers or [])

    def _agent(self, sim, money, *, food=0, materials=0, tools=0):
        a = sim.agents[0]
        a.profession = "smith"
        a.money = money
        a.inventory["food"] = food
        a.inventory["materials"] = materials
        a.inventory["tools"] = tools
        return a

    def test_flush_lender_posts_credit_priced_by_temperament(self):
        # A grasping (predator) lender charges more interest than a cooperative one.
        for persona, expect_repay in (("guardian", 6), ("predator", 8)):
            sim = _sim()
            brain = HeuristicBrain(persona)
            a = self._agent(sim, 16)
            act = brain._trade_action(a, self._obs())
            self.assertIsNotNone(act)
            self.assertEqual(act.type, ActionType.OFFER)
            self.assertTrue(act.params.get("loan"))
            self.assertEqual(act.params["principal"], 5)
            self.assertEqual(act.params["repay"], expect_repay)

    def test_broke_agent_borrows_cheapest_open_loan(self):
        sim = _sim()
        brain = HeuristicBrain("guardian")
        a = self._agent(sim, 1)
        offers = [
            {"id": 1, "maker": "x", "loan": True, "item": "money", "principal": 5, "repay": 9},
            {"id": 2, "maker": "y", "loan": True, "item": "money", "principal": 5, "repay": 6},
            {"id": 3, "maker": "z", "loan": True, "item": "money", "principal": 5, "repay": 7},
        ]
        act = brain._trade_action(a, self._obs(offers=offers))
        self.assertIsNotNone(act)
        self.assertEqual(act.type, ActionType.ACCEPT)
        self.assertEqual(act.params["offer_id"], 2, "the cheapest credit on offer")

    def test_does_not_borrow_its_own_offer(self):
        sim = _sim()
        brain = HeuristicBrain("guardian")
        a = self._agent(sim, 1)
        offers = [{"id": 1, "maker": a.id, "loan": True, "item": "money",
                   "principal": 5, "repay": 6}]
        # no one else's credit -> the broke-borrow rule finds nothing to take.
        act = brain._trade_action(a, self._obs(offers=offers))
        if act is not None:
            self.assertNotEqual(
                (act.type, act.params.get("offer_id")), (ActionType.ACCEPT, 1))


if __name__ == "__main__":
    unittest.main()
