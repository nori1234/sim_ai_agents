"""Illness & contagion (#86): an epidemic layer distinct from addiction's
withdrawal sickness. Opt-in via IllnessConfig.enabled (default False); with it
off nothing here fires and the four-society baseline is untouched.
"""

import unittest

from emergence.agent import Agent
from emergence.illness import IllnessConfig, capability_factor
from emergence.simulation import HEAL_ILLNESS_RELIEF, Simulation, SimulationConfig
from emergence.world import Facility, FacilityType, World


def _agent(**kw):
    base = dict(id="x", name="X", profession="tester", persona="predator", x=0, y=0)
    base.update(kw)
    return Agent(**base)


def _sim(agents, world=None, **ill_kw):
    return Simulation(world=world or World(8, 8), agents=agents, brains={},
                       illness=IllnessConfig(enabled=True, **ill_kw))


class TestCapabilityFactor(unittest.TestCase):
    def test_healthy_is_full_capability(self):
        a = _agent(illness=0)
        self.assertEqual(capability_factor(a, IllnessConfig(enabled=True)), 1.0)

    def test_mild_illness_below_threshold_is_full_capability(self):
        a = _agent(illness=30)  # below default severe_threshold=50
        self.assertEqual(capability_factor(a, IllnessConfig(enabled=True)), 1.0)

    def test_severe_illness_reduces_capability(self):
        a = _agent(illness=90)
        cfg = IllnessConfig(enabled=True)
        factor = capability_factor(a, cfg)
        self.assertLess(factor, 1.0)
        self.assertGreaterEqual(factor, 1.0 - cfg.gather_penalty)

    def test_disabled_config_is_always_full_capability(self):
        a = _agent(illness=90)
        self.assertEqual(capability_factor(a, IllnessConfig(enabled=False)), 1.0)


class TestOnset(unittest.TestCase):
    def test_daily_strike_can_infect_a_healthy_agent(self):
        a = _agent(illness=0)
        sim = _sim([a], daily_strike_chance=1.0)  # certain, deterministic
        sim._end_of_day(verbose=False)
        self.assertGreater(a.illness, 0)

    def test_no_onset_when_disabled(self):
        a = _agent(illness=0)
        sim = Simulation(world=World(8, 8), agents=[a], brains={})  # illness off
        sim._end_of_day(verbose=False)
        self.assertEqual(a.illness, 0)

    def test_already_ill_agent_does_not_restrike(self):
        a = _agent(illness=40)
        sim = _sim([a], daily_strike_chance=1.0)
        sim._end_of_day(verbose=False)
        self.assertEqual(a.illness, 40)  # unchanged by onset (only decay touches it)


class TestDecayAndShelter(unittest.TestCase):
    def test_illness_decays_per_tick(self):
        a = _agent(illness=40, x=5, y=5)
        sim = _sim([a])
        sim._tick_upkeep(a)
        self.assertLess(a.illness, 40)

    def test_shelter_speeds_recovery(self):
        world = World(8, 8)
        world.add_facility(Facility("H", FacilityType.HOUSE, 0, 0))
        outside = _agent(id="o", illness=40, x=5, y=5)
        inside = _agent(id="i", illness=40, x=0, y=0)
        sim = _sim([outside, inside], world=world)
        sim._tick_upkeep(outside)
        sim._tick_upkeep(inside)
        self.assertLess(inside.illness, outside.illness)

    def test_severe_illness_drains_energy(self):
        mild = _agent(id="m", illness=10, energy=80, x=5, y=5)
        severe = _agent(id="s", illness=90, energy=80, x=6, y=6)
        sim = _sim([mild, severe])
        sim._tick_upkeep(mild)
        sim._tick_upkeep(severe)
        self.assertLess(severe.energy, mild.energy)

    def test_never_ill_no_decay_penalty(self):
        a = _agent(illness=0, energy=80, x=5, y=5)
        sim = _sim([a])
        sim._tick_upkeep(a)
        self.assertEqual(a.energy, 80.0 - 5.0)  # base ENERGY_DECAY_PER_TICK only


class TestContagion(unittest.TestCase):
    def test_proximity_can_spread_illness(self):
        sick = _agent(id="s", illness=40, x=0, y=0)
        healthy = _agent(id="h", illness=0, x=1, y=1)
        sim = _sim([sick, healthy], contagion_chance=1.0, contagion_radius=2)
        sim._tick_upkeep(healthy)
        self.assertGreater(healthy.illness, 0)

    def test_out_of_radius_does_not_spread(self):
        sick = _agent(id="s", illness=40, x=0, y=0)
        healthy = _agent(id="h", illness=0, x=7, y=7)
        sim = _sim([sick, healthy], contagion_chance=1.0, contagion_radius=2)
        sim._tick_upkeep(healthy)
        self.assertEqual(healthy.illness, 0)

    def test_no_contagion_when_disabled(self):
        sick = _agent(id="s", illness=40, x=0, y=0)
        healthy = _agent(id="h", illness=0, x=1, y=1)
        sim = Simulation(world=World(8, 8), agents=[sick, healthy], brains={})  # off
        sim._tick_upkeep(healthy)
        self.assertEqual(healthy.illness, 0)


class TestHealing(unittest.TestCase):
    def test_doctor_care_relieves_illness(self):
        provider = _agent(id="doc", profession="doctor")
        taker = _agent(id="pat", illness=50)
        sim = _sim([provider, taker])
        sim._serve_healing(provider, taker, fee=5)
        self.assertEqual(taker.illness, 50 - HEAL_ILLNESS_RELIEF)

    def test_healing_floors_at_zero(self):
        provider = _agent(id="doc", profession="doctor")
        taker = _agent(id="pat", illness=5)
        sim = _sim([provider, taker])
        sim._serve_healing(provider, taker, fee=5)
        self.assertEqual(taker.illness, 0.0)

    def test_no_relief_when_disabled(self):
        provider = _agent(id="doc", profession="doctor")
        taker = _agent(id="pat", illness=50)
        sim = Simulation(world=World(8, 8), agents=[provider, taker], brains={})  # off
        sim._serve_healing(provider, taker, fee=5)
        self.assertEqual(taker.illness, 50)


class TestYieldPenalty(unittest.TestCase):
    def test_severe_illness_reduces_gather_yield(self):
        a = _agent(illness=90)
        sim = _sim([a])
        self.assertLess(sim._illness_scaled_yield(a, 10), 10)

    def test_yield_never_hits_zero(self):
        a = _agent(illness=100)
        sim = _sim([a], gather_penalty=0.9)
        self.assertGreaterEqual(sim._illness_scaled_yield(a, 1), 1)

    def test_no_penalty_when_disabled(self):
        a = _agent(illness=90)
        sim = Simulation(world=World(8, 8), agents=[a], brains={})  # off
        self.assertEqual(sim._illness_scaled_yield(a, 10), 10)

    def test_mild_illness_no_penalty(self):
        a = _agent(illness=10)
        sim = _sim([a])
        self.assertEqual(sim._illness_scaled_yield(a, 10), 10)


class TestBaselineUnaffected(unittest.TestCase):
    def test_illness_field_defaults_to_zero(self):
        a = _agent()
        self.assertEqual(a.illness, 0.0)

    def test_snapshot_includes_illness(self):
        a = _agent(illness=12.3)
        self.assertEqual(a.snapshot()["illness"], 12.3)


if __name__ == "__main__":
    unittest.main()
