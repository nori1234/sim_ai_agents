"""Conspicuous consumption: a feast as a service that buys honour.

A feast rides the same OFFER/ACCEPT service substrate as healing — but its
"effect" is *reputation*, not energy. A caterer (any agent) posts a feast offer
at a fee it picks; a host accepts, pays the caterer (conserved), and the lavish
outlay buys honour: reputation scales with the fee, so the price of honour
emerges from what hosts will spend. It only means anything where honour exists,
so the feast service is gated on BOTH --economy and --status:
  * without --status a feast can't even be offered (economy-only baseline has none);
  * accepting pays the caterer (conserved) and grants the host reputation by fee;
  * a dearer feast buys more honour (conspicuous);
  * the accepted fee feeds the emergent price of a feast;
  * the heuristic: a cash-poor agent caters; an esteem-hungry rich agent hosts
    the dearest feast it can reach.
"""

import unittest

from emergence.actions import Action, ActionType
from emergence.brains.heuristic import HeuristicBrain
from emergence.esteem import StatusConfig
from emergence.market import SERVICES, can_provide
from emergence.observation import Observation
from emergence.scenario import make_simulation
from emergence.simulation import SimulationConfig


def _sim(*, economy=True, status=True):
    return make_simulation(
        "guardian", config=SimulationConfig(seed=1), economy=economy,
        status=StatusConfig(enabled=status))


def _offer(sim, agent, **params):
    sim._do_offer(agent, Action(ActionType.OFFER, params))


def _accept(sim, agent, offer_id):
    sim._do_accept(agent, Action(ActionType.ACCEPT, {"offer_id": offer_id}))


class TestFeastRegistry(unittest.TestCase):
    def test_feast_is_anyones_to_offer(self):
        self.assertIn("feast", SERVICES)
        self.assertTrue(can_provide("feast", "smith"))
        self.assertTrue(can_provide("feast", "doctor"))

    def test_registry_and_effects_stay_in_sync(self):
        sim = _sim()
        self.assertEqual(sorted(SERVICES), sorted(sim._service_effects))


class TestFeastService(unittest.TestCase):
    def _pair(self, sim, *, adjacent=True):
        caterer, host = sim.agents[0], sim.agents[1]
        host.x, host.y = 0, 0
        caterer.x, caterer.y = (host.x, host.y) if adjacent else (9, 9)
        return caterer, host

    def test_host_pays_caterer_and_gains_honour_by_fee(self):
        sim = _sim()
        caterer, host = self._pair(sim)
        caterer.money, host.money = 0, 20
        host.reputation = 0.0
        total = caterer.money + host.money
        _offer(sim, caterer, service="feast", want_item="money", want_qty=8)
        _accept(sim, host, sim.offers[0].id)
        self.assertEqual(caterer.money, 8, "the caterer earns the fee it set")
        self.assertEqual(host.money, 12)
        self.assertEqual(caterer.money + host.money, total, "money conserved")
        self.assertAlmostEqual(host.reputation,
                               sim.status.rep_per_feast_coin * 8,
                               msg="honour scales with the outlay")

    def test_dearer_feast_buys_more_honour(self):
        cheap = _sim(); dear = _sim()
        for sim, fee in ((cheap, 4), (dear, 12)):
            caterer, host = self._pair(sim)
            caterer.money, host.money = 0, 30
            host.reputation = 0.0
            _offer(sim, caterer, service="feast", want_item="money", want_qty=fee)
            _accept(sim, host, sim.offers[0].id)
        host_cheap = cheap.agents[1]; host_dear = dear.agents[1]
        self.assertGreater(host_dear.reputation, host_cheap.reputation)

    def test_accepted_fee_feeds_emergent_feast_price(self):
        sim = _sim()
        caterer, host = self._pair(sim)
        host.money = 20
        _offer(sim, caterer, service="feast", want_item="money", want_qty=7)
        _accept(sim, host, sim.offers[0].id)
        self.assertEqual(sim.emergent_price("feast", "money"), 7.0)

    def test_out_of_reach_does_not_clear(self):
        sim = _sim()
        caterer, host = self._pair(sim, adjacent=False)
        host.money, host.reputation = 20, 0.0
        _offer(sim, caterer, service="feast", want_item="money", want_qty=5)
        _accept(sim, host, sim.offers[0].id)
        self.assertEqual(host.reputation, 0.0, "a feast is local — too far, no honour")
        self.assertEqual(host.money, 20)
        self.assertTrue(sim.offers, "the offer stays open for a host in reach")

    def test_cannot_afford_is_a_noop(self):
        sim = _sim()
        caterer, host = self._pair(sim)
        host.money, host.reputation = 3, 0.0
        _offer(sim, caterer, service="feast", want_item="money", want_qty=8)
        _accept(sim, host, sim.offers[0].id)
        self.assertEqual(host.reputation, 0.0)
        self.assertEqual(host.money, 3)

    def test_feast_not_offerable_without_status(self):
        sim = _sim(status=False)
        caterer = sim.agents[0]
        _offer(sim, caterer, service="feast", want_item="money", want_qty=5)
        self.assertEqual(sim.offers, [], "no honour layer → a feast can't be offered")

    def test_feast_hidden_from_services_without_status(self):
        on = _sim(status=True); off = _sim(status=False)
        self.assertIn("feast", on._observe(on.agents[0]).economy["services"])
        self.assertNotIn("feast", off._observe(off.agents[0]).economy["services"])

    def test_offline_baseline_feast_offer_is_inert(self):
        sim = _sim(economy=False, status=True)
        caterer = sim.agents[0]
        _offer(sim, caterer, service="feast", want_item="money", want_qty=5)
        self.assertEqual(sim.offers, [])


class TestHeuristicFeast(unittest.TestCase):
    def _obs(self, *, esteem_urge=0.0, others=None, offers=None, services=("feast",)):
        return Observation(
            day=1, tick=1, self_view={}, position=(0, 0), nearby_facilities=[],
            here=None, others=others or [], open_proposals=[], granary_food=0,
            recent_events=[], debts=[], esteem_urge=esteem_urge,
            economy={"enabled": True, "services": list(services)},
            open_offers=offers or [])

    def test_provisioned_agent_caters_priced_by_temperament(self):
        # Catering needs provisions on hand; a grasping (predator) caterer charges
        # more than a cooperative (guardian) one. No other surplus to sell first.
        for persona, expect_cheaper in (("guardian", True), ("predator", False)):
            sim = _sim()
            brain = HeuristicBrain(persona)
            a = sim.agents[0]
            a.money = 6
            a.inventory["food"] = 3           # provisions to host with
            a.inventory["materials"] = 0; a.inventory["tools"] = 0
            act = brain._trade_action(a, self._obs())
            self.assertIsNotNone(act)
            self.assertEqual(act.type, ActionType.OFFER)
            self.assertEqual(act.params.get("service"), "feast")
            if expect_cheaper:
                self.assertLessEqual(act.params["want_qty"], 5)
            else:
                self.assertGreaterEqual(act.params["want_qty"], 7)

    def test_rich_esteem_hungry_host_buys_dearest_reachable_feast(self):
        sim = _sim()
        brain = HeuristicBrain("guardian")
        host = sim.agents[1]
        host.money = 30
        offers = [
            {"id": 1, "maker": "c1", "service": "feast", "want": "4 money"},
            {"id": 2, "maker": "c2", "service": "feast", "want": "10 money"},  # dearest, reachable
            {"id": 3, "maker": "c3", "service": "feast", "want": "12 money"},  # dearer but far
        ]
        others = [{"id": "c1", "distance": 1}, {"id": "c2", "distance": 2},
                  {"id": "c3", "distance": 9}]
        act = brain._buy_feast_action(host, self._obs(others=others, offers=offers))
        self.assertIsNotNone(act)
        self.assertEqual(act.type, ActionType.ACCEPT)
        self.assertEqual(act.params["offer_id"], 2,
                         "the dearest feast in reach it can comfortably afford")

    def test_host_keeps_a_buffer(self):
        sim = _sim()
        brain = HeuristicBrain("guardian")
        host = sim.agents[1]
        host.money = 9                       # 9 - 8 fee = 1 left < 6 buffer
        offers = [{"id": 1, "maker": "c1", "service": "feast", "want": "8 money"}]
        others = [{"id": "c1", "distance": 1}]
        self.assertIsNone(brain._buy_feast_action(host, self._obs(others=others, offers=offers)))


if __name__ == "__main__":
    unittest.main()
