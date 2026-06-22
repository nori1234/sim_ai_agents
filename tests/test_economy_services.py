"""Economy Phase 2: services as labour — paid healing (money -> energy).

A depleted agent can pay a nearby doctor to restore energy. This gives money a
survival-grade consumption demand (you can buy energy), and gives the doctor a
wage. Guards:
  * conservation — money moves to the doctor, none is conjured or destroyed;
  * energy is restored, more so at a hospital;
  * the service is gated on --economy (offline baseline byte-identical);
  * the heuristic's one rule: depleted + no food + money + a doctor in reach buys care.
"""

import unittest

from emergence.actions import Action, ActionType
from emergence.affordances import FACILITY_AFFORDANCES
from emergence.brains.heuristic import HeuristicBrain, LOW_ENERGY
from emergence.observation import Observation
from emergence.scenario import make_simulation
from emergence.simulation import HEAL_FEE, HEAL_ENERGY, SimulationConfig
from emergence.world import FacilityType


def _sim(economy=True):
    return make_simulation("guardian", config=SimulationConfig(seed=1), economy=economy)


def _place(patient, doctor, sim, *, on_hospital=False):
    """Co-locate patient and doctor (adjacent), optionally on a hospital."""
    doctor.profession = "doctor"
    if on_hospital:
        h = next(f for f in sim.world.facilities if f.ftype is FacilityType.HOSPITAL)
        patient.x, patient.y = h.x, h.y
    else:
        # Somewhere with no facility-specific bonus.
        patient.x, patient.y = 0, 0
    doctor.x, doctor.y = patient.x, patient.y


class TestPaidHealing(unittest.TestCase):
    def test_conserves_money_and_restores_energy(self):
        sim = _sim()
        patient, doctor = sim.agents[0], sim.agents[1]
        _place(patient, doctor, sim)
        patient.money, doctor.money = 10, 5
        patient.energy = 30.0
        before_total = patient.money + doctor.money
        sim._do_treat(patient, Action(ActionType.TREAT, {"doctor": doctor.id}))
        self.assertEqual(patient.money, 10 - HEAL_FEE)
        self.assertEqual(doctor.money, 5 + HEAL_FEE, "the doctor earns the fee")
        self.assertEqual(patient.money + doctor.money, before_total, "money is conserved")
        self.assertEqual(patient.energy, 30.0 + HEAL_ENERGY)

    def test_hospital_boosts_the_care(self):
        sim = _sim()
        patient, doctor = sim.agents[0], sim.agents[1]
        _place(patient, doctor, sim, on_hospital=True)
        patient.money = 10
        patient.energy = 30.0
        sim._do_treat(patient, Action(ActionType.TREAT, {"doctor": doctor.id}))
        self.assertGreater(patient.energy, 30.0 + HEAL_ENERGY,
                           "a hospital should make care more effective")

    def test_auto_picks_a_doctor_in_reach(self):
        sim = _sim()
        patient, doctor = sim.agents[0], sim.agents[1]
        _place(patient, doctor, sim)
        patient.money, doctor.money = 10, 0
        patient.energy = 30.0
        sim._do_treat(patient, Action(ActionType.TREAT, {}))  # no doctor named
        self.assertEqual(patient.energy, 30.0 + HEAL_ENERGY, "auto-picked the doctor in reach")
        self.assertEqual(patient.money, 10 - HEAL_FEE)
        self.assertEqual(doctor.money, HEAL_FEE)

    def test_no_doctor_in_reach_is_a_noop(self):
        sim = _sim()
        patient, doctor = sim.agents[0], sim.agents[1]
        doctor.profession = "doctor"
        patient.x, patient.y = 0, 0
        doctor.x, doctor.y = 9, 9  # far away
        patient.money, patient.energy = 10, 30.0
        sim._do_treat(patient, Action(ActionType.TREAT, {}))
        self.assertEqual(patient.money, 10, "no doctor in reach -> no payment")
        self.assertEqual(patient.energy, 30.0)

    def test_cannot_afford_is_a_noop(self):
        sim = _sim()
        patient, doctor = sim.agents[0], sim.agents[1]
        _place(patient, doctor, sim)
        patient.money, patient.energy = HEAL_FEE - 1, 30.0
        sim._do_treat(patient, Action(ActionType.TREAT, {"doctor": doctor.id}))
        self.assertEqual(patient.energy, 30.0, "can't be treated without the fee")

    def test_offline_baseline_treat_is_inert(self):
        sim = _sim(economy=False)
        patient, doctor = sim.agents[0], sim.agents[1]
        _place(patient, doctor, sim)
        patient.money, patient.energy = 10, 30.0
        sim._do_treat(patient, Action(ActionType.TREAT, {"doctor": doctor.id}))
        self.assertEqual(patient.money, 10)
        self.assertEqual(patient.energy, 30.0)

    def test_hospital_affords_treat(self):
        self.assertIn(ActionType.TREAT, FACILITY_AFFORDANCES[FacilityType.HOSPITAL])


class TestHeuristicBuysCare(unittest.TestCase):
    def _obs(self, *, enabled=True, others=None):
        return Observation(
            day=1, tick=1, self_view={}, position=(0, 0), nearby_facilities=[],
            here=None, others=others or [], open_proposals=[], granary_food=0,
            recent_events=[],
            economy={"enabled": enabled, "care_fee": HEAL_FEE} if enabled else {},
        )

    def _patient(self):
        sim = _sim()
        a = sim.agents[0]
        a.profession = "smith"
        a.money = 10
        a.energy = LOW_ENERGY - 5
        a.inventory["food"] = 0
        return HeuristicBrain("guardian"), a

    def test_buys_care_when_depleted_no_food_doctor_in_reach(self):
        brain, patient = self._patient()
        others = [{"id": "d1", "profession": "doctor", "distance": 1}]
        act = brain._buy_care_action(patient, self._obs(others=others))
        self.assertIsNotNone(act)
        self.assertEqual(act.type, ActionType.TREAT)
        self.assertEqual(act.params["doctor"], "d1")

    def test_prefers_free_food_over_paying(self):
        brain, patient = self._patient()
        patient.inventory["food"] = 2  # has a free meal
        others = [{"id": "d1", "profession": "doctor", "distance": 1}]
        self.assertIsNone(brain._buy_care_action(patient, self._obs(others=others)))

    def test_no_care_without_a_doctor_or_money_or_economy(self):
        brain, patient = self._patient()
        far = [{"id": "d1", "profession": "doctor", "distance": 5}]
        self.assertIsNone(brain._buy_care_action(patient, self._obs(others=far)))
        near = [{"id": "d1", "profession": "doctor", "distance": 1}]
        patient.money = 0
        self.assertIsNone(brain._buy_care_action(patient, self._obs(others=near)))
        patient.money = 10
        self.assertIsNone(brain._buy_care_action(patient, self._obs(enabled=False, others=near)))


if __name__ == "__main__":
    unittest.main()
