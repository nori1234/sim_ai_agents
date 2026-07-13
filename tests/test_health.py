"""Injury (怪我): a lingering wound, distinct from momentary energy loss.

Guards: the health.py helpers in isolation, that violence/rebellion inflict
injury only under the health layer, that it decays over time (faster under a
roof) and drains energy once severe, that a doctor mends it (more at a
hospital), that a badly hurt gatherer works at reduced capability, that the
heuristic seeks care and avoids picking fights while hurt, and that the
offline baseline (health off) stays byte-identical.
"""

import unittest

from emergence.agent import Agent
from emergence.brains.heuristic import SICK_INJURY, HeuristicBrain
from emergence.health import HealthConfig, capability_factor
from emergence.personas import get_persona
from emergence.scenario import make_simulation
from emergence.simulation import Simulation, SimulationConfig
from emergence.world import FacilityType, World


def _agent(**kw):
    base = dict(id="x", name="X", profession="tester", persona="guardian", x=0, y=0)
    base.update(kw)
    return Agent(**base)


def _sim(agents, world=None, **health_kw):
    return Simulation(world=world or World(12, 12), agents=agents, brains={},
                      health=HealthConfig(enabled=True, **health_kw))


def _farm(sim):
    return next(f for f in sim.world.facilities if f.ftype is FacilityType.FARM)


class TestCapabilityFactor(unittest.TestCase):
    def test_disabled_is_full_capability(self):
        self.assertEqual(capability_factor(_agent(injury=90), HealthConfig()), 1.0)

    def test_below_threshold_is_full_capability(self):
        cfg = HealthConfig(enabled=True, injury_severe_threshold=50)
        self.assertEqual(capability_factor(_agent(injury=30), cfg), 1.0)

    def test_worsens_with_injury(self):
        cfg = HealthConfig(enabled=True, injury_severe_threshold=50)
        light, heavy = _agent(injury=60), _agent(injury=100)
        light_f = capability_factor(light, cfg)
        heavy_f = capability_factor(heavy, cfg)
        self.assertLess(heavy_f, light_f)
        self.assertLess(light_f, 1.0)


class TestInjuryInfliction(unittest.TestCase):
    def test_strike_wounds_the_victim(self):
        off = _agent(id="off", x=0, y=0)
        vic = _agent(id="vic", x=1, y=0)
        sim = _sim([off, vic])
        sim._strike(off, victim=vic)
        self.assertGreater(vic.injury, 0)

    def test_disabled_by_default_no_injury_from_strikes(self):
        off = _agent(id="off", x=0, y=0)
        vic = _agent(id="vic", x=1, y=0)
        sim = Simulation(world=World(12, 12), agents=[off, vic], brains={})
        sim._strike(off, victim=vic)
        self.assertEqual(vic.injury, 0)

    def test_injury_is_capped_at_100(self):
        off = _agent(id="off", x=0, y=0)
        vic = _agent(id="vic", x=1, y=0, injury=95)
        sim = _sim([off, vic], injury_per_strike=30)
        sim._strike(off, victim=vic)
        self.assertEqual(vic.injury, 100.0)


class TestInjuryDecayAndDrain(unittest.TestCase):
    def test_decays_with_quiet_time(self):
        a = _agent(injury=50, x=11, y=11)
        sim = _sim([a], injury_decay_per_tick=3)
        sim._tick_upkeep(a)
        self.assertLess(a.injury, 50)

    def test_shelter_speeds_the_decay(self):
        exposed = _agent(id="e", injury=50, x=11, y=11)
        world = World(12, 12)
        from emergence.world import Facility
        world.add_facility(Facility("Home", FacilityType.HOUSE, 0, 0))
        sheltered = _agent(id="s", injury=50, x=0, y=0)
        sim = _sim([exposed, sheltered], world=world,
                   injury_decay_per_tick=1, injury_decay_shelter_bonus=5)
        sim._tick_upkeep(exposed)
        sim._tick_upkeep(sheltered)
        self.assertLess(sheltered.injury, exposed.injury)

    def test_a_severe_wound_steadily_drains_energy(self):
        a = _agent(injury=90, x=11, y=11)
        sim = _sim([a], injury_severe_threshold=50, injury_energy_penalty=4)
        before = a.energy
        sim._tick_upkeep(a)
        # Energy also pays the base per-tick decay, so just check the drop is
        # bigger than the base decay alone would cause.
        from emergence.simulation import ENERGY_DECAY_PER_TICK
        self.assertGreater(before - a.energy, ENERGY_DECAY_PER_TICK)


class TestHealingRelief(unittest.TestCase):
    def test_a_doctor_mends_the_wound(self):
        doc = _agent(id="doc", x=0, y=0)
        patient = _agent(id="pat", injury=60, x=0, y=0)
        sim = _sim([doc, patient])
        sim._serve_healing(doc, patient, fee=5)
        self.assertLess(patient.injury, 60)

    def test_hospital_boosts_the_relief(self):
        from emergence.world import Facility
        open_world = World(12, 12)
        hosp_world = World(12, 12)
        hosp_world.add_facility(Facility("Clinic", FacilityType.HOSPITAL, 0, 0))
        doc = _agent(id="doc", x=0, y=0)
        open_patient = _agent(id="op", injury=80, x=0, y=0)
        hosp_patient = _agent(id="hp", injury=80, x=0, y=0)
        sim_open = _sim([doc, open_patient], world=open_world)
        sim_hosp = _sim([doc, hosp_patient], world=hosp_world)
        sim_open._serve_healing(doc, open_patient, fee=5)
        sim_hosp._serve_healing(doc, hosp_patient, fee=5)
        self.assertLess(hosp_patient.injury, open_patient.injury)


class TestInjuredGathering(unittest.TestCase):
    def test_badly_hurt_gatherer_yields_less(self):
        sim = make_simulation("guardian", config=SimulationConfig(seed=1),
                              health=HealthConfig(enabled=True,
                                                   injury_severe_threshold=20,
                                                   injury_gather_penalty=0.6))
        farm = _farm(sim)
        healthy = sim.agents[0]
        healthy.x, healthy.y = farm.x, farm.y
        before = healthy.food()
        sim._harvest(healthy, farm)
        healthy_gain = healthy.food() - before

        hurt = sim.agents[1]
        hurt.injury = 90
        hurt.x, hurt.y = farm.x, farm.y
        before = hurt.food()
        sim._harvest(hurt, farm)
        hurt_gain = hurt.food() - before
        self.assertLess(hurt_gain, healthy_gain)
        self.assertGreaterEqual(hurt_gain, 1, "injury alone should never zero out a harvest")

    def test_offline_baseline_yield_unchanged(self):
        sim = make_simulation("guardian", config=SimulationConfig(seed=1))
        farm = _farm(sim)
        a = sim.agents[0]
        a.injury = 90  # meaningless without the health layer
        a.x, a.y = farm.x, farm.y
        before = a.food()
        sim._harvest(a, farm)
        self.assertEqual(a.food() - before, 3)  # the base yield, untouched


class TestHeuristicAvoidsFightsWhileHurt(unittest.TestCase):
    def _brain(self, persona_key="predator"):
        return HeuristicBrain(get_persona(persona_key))

    def test_a_badly_hurt_predator_does_not_initiate_aggression(self):
        # A predator persona has high aggression; force the RNG roll to always
        # "succeed" so the only thing that can stay its hand is the injury guard.
        import random

        class AlwaysZero(random.Random):
            def random(self):
                return 0.0

        brain = HeuristicBrain(get_persona("predator"), rng=AlwaysZero())
        agent = _agent(persona="predator", injury=90)  # full energy + food (defaults):
        # _survival_action bails out early, so the aggression checks are reached.
        target = {"id": "prey", "distance": 1, "trust": -0.9,
                  "money": 10, "food": 5, "materials": 2}
        from emergence.observation import Observation
        obs = Observation(
            day=1, tick=0,
            self_view=agent.snapshot(), position=(0, 0), nearby_facilities=[],
            here=None, others=[target], open_proposals=[], granary_food=0,
            recent_events=[], memory=[], knowledge=[],
        )
        action = brain.decide(agent, obs)
        self.assertNotIn(action.type.name, ("ATTACK", "STEAL"),
                         "a badly hurt agent should not pick a fight")

    def test_an_unhurt_predator_does_initiate_aggression(self):
        # Sanity check: the same forced-RNG setup, but healthy -- confirms the
        # guard above is actually gated on injury, not some other quirk of the
        # forced-random harness.
        import random

        class AlwaysZero(random.Random):
            def random(self):
                return 0.0

        brain = HeuristicBrain(get_persona("predator"), rng=AlwaysZero())
        agent = _agent(persona="predator", injury=0)
        target = {"id": "prey", "distance": 1, "trust": -0.9,
                  "money": 10, "food": 5, "materials": 2}
        from emergence.observation import Observation
        obs = Observation(
            day=1, tick=0,
            self_view=agent.snapshot(), position=(0, 0), nearby_facilities=[],
            here=None, others=[target], open_proposals=[], granary_food=0,
            recent_events=[], memory=[], knowledge=[],
        )
        action = brain.decide(agent, obs)
        self.assertIn(action.type.name, ("ATTACK", "STEAL"),
                     "an unhurt aggressive predator should still act on a target")


class TestOfflineBaselineUnchanged(unittest.TestCase):
    def test_health_off_is_byte_identical(self):
        cfg = SimulationConfig(seed=7)
        off = make_simulation("predator", config=cfg)
        off.run()
        also_off = make_simulation("predator", config=cfg, health=HealthConfig())
        also_off.run()
        self.assertEqual(off.metrics.as_dict(), also_off.metrics.as_dict())

    def test_no_default_cli_flag_means_disabled(self):
        self.assertFalse(make_simulation("guardian").health.enabled)


if __name__ == "__main__":
    unittest.main()
