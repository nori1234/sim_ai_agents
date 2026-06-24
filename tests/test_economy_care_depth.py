"""Care has depth: a doctor mends body, mind, and the addicted.

Healing on the service substrate used to only top up energy. A doctor should
also speed recovery from the engine's real debuffs — trauma (`fear`, psyche
layer) and the sickness of withdrawal (`addiction`, society drugs layer). Each
relief is gated on its layer, so without them this is exactly the old
energy-only care and the baseline is untouched. A hospital boosts all of it.
Guards:
  * healing relieves fear when --psyche is on (more at a hospital);
  * healing eases addiction when --society drugs is on;
  * with the layers off, fear/addiction are left untouched (energy-only care);
  * the heuristic: an afflicted agent (traumatised / in withdrawal) sees a
    doctor even when fed and rested — a meal mends neither.
"""

import unittest

from emergence.actions import Action, ActionType
from emergence.brains.heuristic import HeuristicBrain, SICK_ADDICTION
from emergence.observation import Observation
from emergence.psyche import PsycheConfig
from emergence.scenario import make_simulation
from emergence.simulation import (HEAL_ADDICTION_RELIEF, HEAL_FEAR_RELIEF,
                                  HOSPITAL_HEAL_BONUS, SimulationConfig)
from emergence.society import SocietyConfig
from emergence.world import FacilityType


def _sim(*, economy=True, psyche=False, society=False):
    return make_simulation(
        "guardian", config=SimulationConfig(seed=1), economy=economy,
        psyche=PsycheConfig(enabled=psyche),
        society=SocietyConfig(enabled=society, drugs=society))


def _heal(sim, doctor, patient, *, on_hospital=False):
    doctor.profession = "doctor"
    if on_hospital:
        h = next(f for f in sim.world.facilities if f.ftype is FacilityType.HOSPITAL)
        patient.x, patient.y = h.x, h.y
    else:
        patient.x, patient.y = 0, 0
    doctor.x, doctor.y = patient.x, patient.y
    sim._do_offer(doctor, Action(ActionType.OFFER,
                  {"service": "healing", "want_item": "money", "want_qty": 0}))
    sim._do_accept(patient, Action(ActionType.ACCEPT, {"offer_id": sim.offers[0].id}))


class TestCareRelievesTrauma(unittest.TestCase):
    def test_healing_calms_fear_under_psyche(self):
        sim = _sim(psyche=True)
        doctor, patient = sim.agents[0], sim.agents[1]
        patient.fear = 50.0
        _heal(sim, doctor, patient)
        self.assertAlmostEqual(patient.fear, 50.0 - HEAL_FEAR_RELIEF)

    def test_hospital_calms_more_fear(self):
        sim = _sim(psyche=True)
        doctor, patient = sim.agents[0], sim.agents[1]
        patient.fear = 80.0
        _heal(sim, doctor, patient, on_hospital=True)
        self.assertAlmostEqual(patient.fear, 80.0 - HEAL_FEAR_RELIEF * HOSPITAL_HEAL_BONUS)

    def test_fear_untouched_without_psyche(self):
        sim = _sim(psyche=False)
        doctor, patient = sim.agents[0], sim.agents[1]
        patient.fear = 50.0
        _heal(sim, doctor, patient)
        self.assertEqual(patient.fear, 50.0, "no psyche layer → care can't treat trauma")


class TestCareEasesWithdrawal(unittest.TestCase):
    def test_healing_eases_addiction_under_society(self):
        sim = _sim(society=True)
        doctor, patient = sim.agents[0], sim.agents[1]
        patient.addiction = 60.0
        _heal(sim, doctor, patient)
        self.assertAlmostEqual(patient.addiction, 60.0 - HEAL_ADDICTION_RELIEF)

    def test_addiction_untouched_without_society(self):
        sim = _sim(society=False)
        doctor, patient = sim.agents[0], sim.agents[1]
        patient.addiction = 60.0
        _heal(sim, doctor, patient)
        self.assertEqual(patient.addiction, 60.0, "no drugs layer → care can't detox")

    def test_relief_never_goes_negative(self):
        sim = _sim(psyche=True, society=True)
        doctor, patient = sim.agents[0], sim.agents[1]
        patient.fear = 5.0
        patient.addiction = 3.0
        _heal(sim, doctor, patient)
        self.assertEqual(patient.fear, 0.0)
        self.assertEqual(patient.addiction, 0.0)


class TestHeuristicAfflictedSeeksCare(unittest.TestCase):
    def _obs(self, *, fear_level=0.0, others=None, offers=None):
        return Observation(
            day=1, tick=1, self_view={}, position=(0, 0), nearby_facilities=[],
            here=None, others=others or [], open_proposals=[], granary_food=0,
            recent_events=[], debts=[], fear_level=fear_level,
            economy={"enabled": True}, open_offers=offers or [])

    def _setup(self):
        sim = _sim(psyche=True, society=True)
        brain = HeuristicBrain("guardian")
        patient = sim.agents[1]
        patient.energy = 100.0           # well-rested
        patient.inventory["food"] = 5    # well-fed: a meal won't help a wound
        patient.money = 10
        offers = [{"id": 1, "maker": "d1", "service": "healing", "want": "3 money"}]
        others = [{"id": "d1", "profession": "doctor", "distance": 1}]
        return brain, patient, offers, others

    def test_traumatised_agent_sees_a_doctor_though_fed(self):
        brain, patient, offers, others = self._setup()
        patient.fear = 80.0
        act = brain._survival_action(patient, self._obs(fear_level=0.5, others=others, offers=offers))
        self.assertIsNotNone(act)
        self.assertEqual(act.type, ActionType.ACCEPT)
        self.assertEqual(act.params["offer_id"], 1)

    def test_sick_agent_sees_a_doctor_though_fed(self):
        brain, patient, offers, others = self._setup()
        patient.addiction = SICK_ADDICTION + 5
        act = brain._survival_action(patient, self._obs(others=others, offers=offers))
        self.assertIsNotNone(act)
        self.assertEqual(act.type, ActionType.ACCEPT)
        self.assertEqual(act.params["offer_id"], 1)

    def test_well_and_unafflicted_agent_does_not_seek_care(self):
        brain, patient, offers, others = self._setup()
        # no fear, no addiction, fed and rested
        self.assertIsNone(brain._survival_action(patient, self._obs(others=others, offers=offers)))


if __name__ == "__main__":
    unittest.main()
