import unittest

from emergence import market as MK
from emergence.actions import Action, ActionType
from emergence.agent import Agent
from emergence.scenario import make_simulation
from emergence.simulation import Simulation, SimulationConfig
from emergence.world import Facility, FacilityType, World


def _agent(**kw):
    base = dict(id="x", name="X", profession="t", persona="guardian", x=0, y=0)
    base.update(kw)
    return Agent(**base)


def _sim(agents, world=None):
    return Simulation(world=world or World(6, 6), agents=agents, brains={},
                      economy=True)


class TestPrimitivesDisabled(unittest.TestCase):
    def test_offer_noop_when_economy_off(self):
        a = _agent()
        a.inventory["food"] = 5
        sim = Simulation(world=World(4, 4), agents=[a], brains={})  # economy off
        sim._do_offer(a, Action(ActionType.OFFER, {"give_item": "food", "give_qty": 2,
                                                   "want_item": "money", "want_qty": 3}))
        self.assertEqual(sim.offers, [])


class TestOfferAccept(unittest.TestCase):
    def test_swap_conserves_and_executes(self):
        seller = _agent(id="s")
        seller.inventory["food"] = 5
        buyer = _agent(id="b", money=10)
        sim = _sim([seller, buyer])
        sim._do_offer(seller, Action(ActionType.OFFER,
                      {"give_item": "food", "give_qty": 2,
                       "want_item": "money", "want_qty": 3}))
        self.assertEqual(len(sim.offers), 1)
        oid = sim.offers[0].id
        sim._do_accept(buyer, Action(ActionType.ACCEPT, {"offer_id": oid}))
        # Goods moved both ways; nothing created or destroyed.
        self.assertEqual(seller.food(), 3)            # 5 - 2 sold
        self.assertEqual(buyer.food(), 5)             # default 3 + 2 bought
        self.assertEqual(seller.money, 23)   # 20 + 3
        self.assertEqual(buyer.money, 7)      # 10 - 3
        self.assertEqual(sim.metrics.trades, 1)
        self.assertEqual(sim.offers, [])

    def test_emergent_price_is_the_settled_ratio(self):
        s, b = _agent(id="s"), _agent(id="b", money=10)
        s.inventory["food"] = 5
        sim = _sim([s, b])
        sim._do_offer(s, Action(ActionType.OFFER, {"give_item": "food", "give_qty": 2,
                                                   "want_item": "money", "want_qty": 3}))
        sim._do_accept(b, Action(ActionType.ACCEPT, {"offer_id": sim.offers[0].id}))
        self.assertEqual(sim.emergent_price("food", "money"), 1.5)  # 3 money / 2 food

    def test_cannot_accept_own_offer(self):
        s = _agent(id="s")
        s.inventory["food"] = 5
        sim = _sim([s])
        sim._do_offer(s, Action(ActionType.OFFER, {"give_item": "food", "give_qty": 2,
                                                   "want_item": "money", "want_qty": 3}))
        sim._do_accept(s, Action(ActionType.ACCEPT, {"offer_id": sim.offers[0].id}))
        self.assertEqual(sim.metrics.trades, 0)

    def test_accept_fails_without_funds(self):
        s, b = _agent(id="s"), _agent(id="b", money=1)
        s.inventory["food"] = 5
        sim = _sim([s, b])
        sim._do_offer(s, Action(ActionType.OFFER, {"give_item": "food", "give_qty": 2,
                                                   "want_item": "money", "want_qty": 3}))
        sim._do_accept(b, Action(ActionType.ACCEPT, {"offer_id": sim.offers[0].id}))
        self.assertEqual(sim.metrics.trades, 0)
        self.assertEqual(len(sim.offers), 1)  # still open


class TestCraft(unittest.TestCase):
    def test_recipe_transforms_inputs(self):
        world = World(6, 6)
        world.add_facility(Facility("WS", FacilityType.WORKSHOP, 0, 0))
        a = _agent(x=0, y=0)
        a.inventory["materials"] = 3
        sim = _sim([a], world=world)
        sim._do_craft(a, Action(ActionType.CRAFT, {"item": "tools"}))
        self.assertEqual(a.inventory.get("tools", 0), 1)
        self.assertEqual(a.materials(), 1)   # 3 - 2 consumed
        self.assertEqual(sim.metrics.crafted, 1)

    def test_craft_needs_the_workshop(self):
        a = _agent(x=3, y=3)   # not at a workshop
        a.inventory["materials"] = 3
        sim = _sim([a])
        sim._do_craft(a, Action(ActionType.CRAFT, {"item": "tools"}))
        self.assertEqual(a.inventory.get("tools", 0), 0)


class TestCredit(unittest.TestCase):
    def test_lend_moves_principal_and_records_loan(self):
        a, b = _agent(id="a", money=20), _agent(id="b", money=1)
        sim = _sim([a, b])
        sim._do_lend(a, Action(ActionType.LEND, {"to": "b", "item": "money",
                                                 "qty": 4, "repay": 6, "due_in_days": 3}))
        self.assertEqual(a.money, 16)
        self.assertEqual(b.money, 5)          # got the principal now
        self.assertEqual(len(sim.loans), 1)
        self.assertEqual(sim.metrics.loans_made, 1)

    def test_repay_returns_with_interest_and_builds_trust(self):
        a, b = _agent(id="a", money=20), _agent(id="b", money=1)
        sim = _sim([a, b])
        sim._do_lend(a, Action(ActionType.LEND, {"to": "b", "item": "money",
                                                 "qty": 4, "repay": 6}))
        b.money += 10  # b earns enough to repay
        sim._do_repay(b, Action(ActionType.REPAY, {"loan_id": sim.loans[0].id}))
        self.assertEqual(sim.metrics.loans_repaid, 1)
        self.assertEqual(a.money, 16 + 6)     # principal out, repay (with interest) in
        self.assertGreater(a.trust_of("b"), 0)

    def test_cannot_repay_a_loan_you_dont_owe(self):
        a, b, c = _agent(id="a", money=20), _agent(id="b", money=1), _agent(id="c", money=20)
        sim = _sim([a, b, c])
        sim._do_lend(a, Action(ActionType.LEND, {"to": "b", "item": "money", "qty": 4,
                                                 "repay": 6}))
        sim._do_repay(c, Action(ActionType.REPAY, {"loan_id": sim.loans[0].id}))
        self.assertEqual(sim.metrics.loans_repaid, 0)

    def test_overdue_loan_defaults_and_damages_trust(self):
        a, b = _agent(id="a", money=20), _agent(id="b", money=1)
        sim = _sim([a, b])
        sim._do_lend(a, Action(ActionType.LEND, {"to": "b", "item": "money", "qty": 4,
                                                 "repay": 6, "due_in_days": 0}))
        sim.world.day = 2                     # past the due date (was day 1)
        sim._end_of_day(verbose=False)
        self.assertEqual(sim.metrics.loan_defaults, 1)
        self.assertLess(a.trust_of("b"), 0)   # creditor now distrusts the defaulter

    def test_cannot_lend_without_funds(self):
        a, b = _agent(id="a", money=2), _agent(id="b", money=1)
        sim = _sim([a, b])
        sim._do_lend(a, Action(ActionType.LEND, {"to": "b", "item": "money", "qty": 4,
                                                 "repay": 6}))
        self.assertEqual(len(sim.loans), 0)


class TestEndToEnd(unittest.TestCase):
    def test_market_comes_alive_when_enabled(self):
        sim = make_simulation("gemini", config=SimulationConfig(seed=42), economy=True)
        sim.run()
        self.assertGreater(sim.metrics.trades, 0)

    def test_cooperative_society_develops_credit(self):
        sim = make_simulation("guardian", config=SimulationConfig(seed=42), economy=True)
        sim.run()
        self.assertGreater(sim.metrics.loans_made, 0)
        self.assertGreater(sim.metrics.loans_repaid, 0)

    def test_off_is_byte_identical_baseline(self):
        sim = make_simulation("gemini", config=SimulationConfig(seed=42)); sim.run()
        self.assertEqual(sim.metrics.crimes_total, 133)
        self.assertEqual(sim.metrics.trades, 0)

    def test_deterministic(self):
        a = make_simulation("gemini", config=SimulationConfig(seed=3), economy=True)
        a.run()
        b = make_simulation("gemini", config=SimulationConfig(seed=3), economy=True)
        b.run()
        self.assertEqual(a.metrics.as_dict(), b.metrics.as_dict())


if __name__ == "__main__":
    unittest.main()
