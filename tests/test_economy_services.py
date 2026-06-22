"""Economy Phase 2: services as labour — healing offered as a chosen service.

A service is the provider's *labour, offered for a price it picks* (0 = charity)
on the same OFFER/ACCEPT order book as goods. A doctor chooses to offer care; a
patient accepts (consent + pays); the fee moves to the doctor (conserved) and
the patient's energy is restored. Free / fair / gouging is the provider's
choice, and the price emerges from accepted fees. Guards:
  * a doctor can post a service offer; non-doctors can't provide healing;
  * accepting pays the doctor (conserved) and restores energy (more at a hospital);
  * services are local (provider must be within reach);
  * everything is gated on --economy (offline baseline byte-identical);
  * the heuristic: a doctor offers care priced by temperament; a depleted,
    food-less patient accepts the cheapest affordable offer in reach.
"""

import unittest

from emergence.actions import Action, ActionType
from emergence.brains.heuristic import HeuristicBrain, LOW_ENERGY
from emergence.market import SERVICES, can_provide
from emergence.observation import Observation
from emergence.scenario import make_simulation
from emergence.simulation import HEAL_ENERGY, SimulationConfig
from emergence.world import FacilityType


def _sim(economy=True):
    return make_simulation("guardian", config=SimulationConfig(seed=1), economy=economy)


def _offer(sim, agent, **params):
    sim._do_offer(agent, Action(ActionType.OFFER, params))


def _accept(sim, agent, offer_id):
    sim._do_accept(agent, Action(ActionType.ACCEPT, {"offer_id": offer_id}))


class TestServiceRegistry(unittest.TestCase):
    def test_capability(self):
        self.assertIn("healing", SERVICES)
        self.assertTrue(can_provide("healing", "doctor"))
        self.assertFalse(can_provide("healing", "smith"))
        self.assertFalse(can_provide("nonesuch", "doctor"))


class TestHealingService(unittest.TestCase):
    def _pair(self, sim, *, adjacent=True, on_hospital=False):
        doctor, patient = sim.agents[0], sim.agents[1]
        doctor.profession = "doctor"
        if on_hospital:
            h = next(f for f in sim.world.facilities if f.ftype is FacilityType.HOSPITAL)
            patient.x, patient.y = h.x, h.y
        else:
            patient.x, patient.y = 0, 0
        doctor.x, doctor.y = (patient.x, patient.y) if adjacent else (9, 9)
        return doctor, patient

    def test_doctor_offers_and_patient_is_healed_conserving_money(self):
        sim = _sim()
        doctor, patient = self._pair(sim)
        doctor.money, patient.money = 0, 10
        patient.energy = 30.0
        _offer(sim, doctor, service="healing", want_item="money", want_qty=4)
        offer = sim.offers[0]
        total = doctor.money + patient.money
        _accept(sim, patient, offer.id)
        self.assertEqual(patient.energy, 30.0 + HEAL_ENERGY)
        self.assertEqual(doctor.money, 4, "the doctor earns the fee it set")
        self.assertEqual(patient.money, 6)
        self.assertEqual(doctor.money + patient.money, total, "money conserved")
        self.assertNotIn(offer, sim.offers, "the offer clears on accept")

    def test_charity_is_a_zero_priced_service(self):
        sim = _sim()
        doctor, patient = self._pair(sim)
        patient.money = 0
        patient.energy = 20.0
        _offer(sim, doctor, service="healing", want_item="money", want_qty=0)
        _accept(sim, patient, sim.offers[0].id)
        self.assertEqual(patient.energy, 20.0 + HEAL_ENERGY, "free care still heals")
        self.assertEqual(patient.money, 0)

    def test_hospital_boosts_care(self):
        sim = _sim()
        doctor, patient = self._pair(sim, on_hospital=True)
        patient.money = 10
        patient.energy = 20.0
        _offer(sim, doctor, service="healing", want_item="money", want_qty=2)
        _accept(sim, patient, sim.offers[0].id)
        self.assertGreater(patient.energy, 20.0 + HEAL_ENERGY)

    def test_out_of_reach_does_not_clear(self):
        sim = _sim()
        doctor, patient = self._pair(sim, adjacent=False)
        patient.money, patient.energy = 10, 30.0
        _offer(sim, doctor, service="healing", want_item="money", want_qty=2)
        _accept(sim, patient, sim.offers[0].id)
        self.assertEqual(patient.energy, 30.0, "a service is local — too far, no care")
        self.assertEqual(patient.money, 10)
        self.assertTrue(sim.offers, "the offer stays open for someone in reach")

    def test_cannot_afford_is_a_noop(self):
        sim = _sim()
        doctor, patient = self._pair(sim)
        patient.money, patient.energy = 1, 30.0
        _offer(sim, doctor, service="healing", want_item="money", want_qty=4)
        _accept(sim, patient, sim.offers[0].id)
        self.assertEqual(patient.energy, 30.0)

    def test_non_doctor_cannot_offer_healing(self):
        sim = _sim()
        smith = sim.agents[0]
        smith.profession = "smith"
        _offer(sim, smith, service="healing", want_item="money", want_qty=2)
        self.assertEqual(sim.offers, [], "only a doctor can offer care")

    def test_price_emerges_from_accepted_fee(self):
        sim = _sim()
        doctor, patient = self._pair(sim)
        patient.money, patient.energy = 10, 30.0
        _offer(sim, doctor, service="healing", want_item="money", want_qty=3)
        _accept(sim, patient, sim.offers[0].id)
        self.assertEqual(sim.emergent_price("healing", "money"), 3.0)

    def test_offline_baseline_offer_is_inert(self):
        sim = _sim(economy=False)
        doctor = sim.agents[0]
        doctor.profession = "doctor"
        _offer(sim, doctor, service="healing", want_item="money", want_qty=2)
        self.assertEqual(sim.offers, [])


class TestHeuristicService(unittest.TestCase):
    def _obs(self, *, enabled=True, others=None, offers=None, here=None):
        return Observation(
            day=1, tick=1, self_view={}, position=(0, 0), nearby_facilities=[],
            here=here, others=others or [], open_proposals=[], granary_food=0,
            recent_events=[], debts=[],
            economy={"enabled": enabled} if enabled else {},
            open_offers=offers or [],
        )

    def test_doctor_offers_care_priced_by_temperament(self):
        # A cooperative (guardian) doctor asks little; a grasping (predator) doctor asks more.
        for persona, expect_cheaper in (("guardian", True), ("predator", False)):
            brain = HeuristicBrain(persona)
            sim = _sim()
            doc = sim.agents[0]
            doc.profession = "doctor"
            obs = self._obs(others=[{"id": "p1", "profession": "smith", "distance": 2}])
            act = brain._trade_action(doc, obs)
            self.assertIsNotNone(act)
            self.assertEqual(act.type, ActionType.OFFER)
            self.assertEqual(act.params["service"], "healing")
            if expect_cheaper:
                self.assertLessEqual(act.params["want_qty"], 2)
            else:
                self.assertGreaterEqual(act.params["want_qty"], 4)

    def test_patient_accepts_cheapest_reachable_affordable_offer(self):
        brain = HeuristicBrain("guardian")
        sim = _sim()
        patient = sim.agents[1]
        patient.profession = "smith"
        patient.money = 10
        patient.energy = LOW_ENERGY - 5
        patient.inventory["food"] = 0
        offers = [
            {"id": 1, "maker": "d1", "service": "healing", "want": "5 money"},
            {"id": 2, "maker": "d2", "service": "healing", "want": "2 money"},
            {"id": 3, "maker": "d3", "service": "healing", "want": "1 money"},  # cheap but far
        ]
        others = [{"id": "d1", "profession": "doctor", "distance": 1},
                  {"id": "d2", "profession": "doctor", "distance": 2},
                  {"id": "d3", "profession": "doctor", "distance": 9}]
        act = brain._buy_care_action(patient, self._obs(others=others, offers=offers))
        self.assertIsNotNone(act)
        self.assertEqual(act.type, ActionType.ACCEPT)
        self.assertEqual(act.params["offer_id"], 2, "cheapest *reachable* affordable offer")

    def test_patient_prefers_free_food(self):
        brain = HeuristicBrain("guardian")
        sim = _sim()
        patient = sim.agents[1]
        patient.money = 10
        patient.inventory["food"] = 2
        offers = [{"id": 1, "maker": "d1", "service": "healing", "want": "1 money"}]
        others = [{"id": "d1", "profession": "doctor", "distance": 1}]
        self.assertIsNone(brain._buy_care_action(patient, self._obs(others=others, offers=offers)))


if __name__ == "__main__":
    unittest.main()
