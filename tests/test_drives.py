import unittest

from emergence.actions import Action, ActionType
from emergence.agent import Agent
from emergence.drives import DrivesConfig, can_reproduce
from emergence.scenario import make_simulation
from emergence.simulation import SimulationConfig
from emergence.world import Facility, FacilityType, World


def _agent(**kw):
    base = dict(id="x", name="X", profession="tester", persona="guardian", x=0, y=0)
    base.update(kw)
    return Agent(**base)


class TestDrivesDisabledByDefault(unittest.TestCase):
    def test_hunger_does_not_rise_when_disabled(self):
        sim = make_simulation("guardian", config=SimulationConfig(seed=1))
        sim.run()
        # With drives off, hunger/fatigue stay at their zero baseline.
        self.assertTrue(all(a.hunger == 0 and a.fatigue == 0 for a in sim.agents))

    def test_no_births_when_disabled(self):
        sim = make_simulation("guardian", config=SimulationConfig(seed=1))
        sim.run()
        self.assertEqual(sim.metrics.births, 0)


class TestDriveMechanics(unittest.TestCase):
    def _sim_with_drives(self, agent, **drive_kw):
        world = World(6, 6)
        drives = DrivesConfig(enabled=True, **drive_kw)
        return Simulation_with(agent, world, drives)

    def test_eat_relieves_hunger(self):
        from emergence.simulation import Simulation
        a = _agent(hunger=80.0)
        a.inventory["food"] = 3
        sim = Simulation(world=World(4, 4), agents=[a], brains={},
                         drives=DrivesConfig(enabled=True))
        sim._do_eat(a, Action(ActionType.EAT))
        self.assertLess(a.hunger, 80.0)

    def test_sleep_relieves_fatigue(self):
        from emergence.simulation import Simulation
        a = _agent(fatigue=90.0)
        sim = Simulation(world=World(4, 4), agents=[a], brains={},
                         drives=DrivesConfig(enabled=True))
        sim._do_sleep(a, Action(ActionType.SLEEP))
        self.assertLess(a.fatigue, 90.0)

    def test_drive_upkeep_raises_needs(self):
        from emergence.simulation import Simulation
        a = _agent()
        sim = Simulation(world=World(4, 4), agents=[a], brains={},
                         drives=DrivesConfig(enabled=True, hunger_per_tick=5,
                                             fatigue_per_tick=4))
        sim._drive_upkeep(a)
        self.assertEqual(a.hunger, 5.0)
        self.assertEqual(a.fatigue, 4.0)


class TestReproductionEligibility(unittest.TestCase):
    def test_too_young_cannot_reproduce(self):
        cfg = DrivesConfig(enabled=True, reproduction=True, maturity_age_days=2)
        a = _agent(age_days=1)
        self.assertFalse(can_reproduce(a, cfg, day=5))

    def test_too_hungry_cannot_reproduce(self):
        cfg = DrivesConfig(enabled=True, reproduction=True, repro_hunger_max=40)
        a = _agent(age_days=10, hunger=90)
        self.assertFalse(can_reproduce(a, cfg, day=5))

    def test_eligible_when_satisfied(self):
        cfg = DrivesConfig(enabled=True, reproduction=True)
        a = _agent(age_days=10, hunger=10, fatigue=10, energy=80)
        self.assertTrue(can_reproduce(a, cfg, day=5))

    def test_cooldown_blocks_reproduction(self):
        cfg = DrivesConfig(enabled=True, reproduction=True, repro_cooldown_days=3)
        a = _agent(age_days=10, hunger=10, fatigue=10, energy=80,
                   last_reproduced_day=4)
        self.assertFalse(can_reproduce(a, cfg, day=5))  # only 1 day passed
        self.assertTrue(can_reproduce(a, cfg, day=8))   # 4 days passed

    def test_disabled_reproduction_never_eligible(self):
        cfg = DrivesConfig(enabled=True, reproduction=False)
        a = _agent(age_days=10, hunger=0, fatigue=0, energy=100)
        self.assertFalse(can_reproduce(a, cfg, day=5))


class TestReproductionEndToEnd(unittest.TestCase):
    def _run(self, persona, repro=True):
        drives = DrivesConfig(enabled=True, reproduction=repro)
        sim = make_simulation(persona, config=SimulationConfig(seed=42),
                              drives=drives)
        sim.run()
        return sim

    def test_guardian_population_grows(self):
        sim = self._run("guardian")
        self.assertGreater(sim.metrics.births, 0)
        self.assertGreater(sim.metrics.survivors, sim.metrics.population)

    def test_predator_does_not_reproduce(self):
        # Violence destroys the trust needed to pair off.
        sim = self._run("predator")
        self.assertEqual(sim.metrics.births, 0)

    def test_children_have_parents_recorded(self):
        sim = self._run("guardian")
        children = [a for a in sim.agents if a.parent_ids]
        self.assertTrue(children)
        for c in children:
            self.assertEqual(len(c.parent_ids), 2)

    def test_population_cap_respected(self):
        drives = DrivesConfig(enabled=True, reproduction=True, max_population=15)
        sim = make_simulation("guardian", config=SimulationConfig(seed=42),
                              drives=drives)
        sim.run()
        living = sum(1 for a in sim.agents if a.alive)
        self.assertLessEqual(living, 15)

    def test_reproduction_run_is_deterministic(self):
        a = self._run("guardian")
        b = self._run("guardian")
        self.assertEqual(a.metrics.as_dict(), b.metrics.as_dict())


def Simulation_with(agent, world, drives):
    from emergence.simulation import Simulation
    return Simulation(world=world, agents=[agent], brains={}, drives=drives)


if __name__ == "__main__":
    unittest.main()
