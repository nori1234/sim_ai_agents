"""Innovation & productivity growth (#97): human capital and discovery, not a
fixed tech tree. Skill (learning-by-doing) rises with repeated gather/craft
and scales yield up; an experienced crafter occasionally discovers a better
recipe, drawn from a small predefined pool -- never generated -- which then
diffuses to every crafter in the town (and, where a library exists, is
written down). Opt-in via InnovationConfig.enabled (default False); off,
Agent.skill sits at 0 and Simulation.recipes is an untouched copy of
market.RECIPES, so the four-society baseline and plain --economy runs are
byte-identical.
"""

import unittest

from emergence import market as MK
from emergence.actions import Action, ActionType
from emergence.agent import Agent
from emergence.innovation import DISCOVERY_POOL, InnovationConfig, skill_yield_mult
from emergence.library import TownLibrary
from emergence.simulation import Simulation, SimulationConfig
from emergence.world import Facility, FacilityType, World


def _agent(**kw):
    base = dict(id="x", name="X", profession="t", persona="guardian", x=0, y=0)
    base.update(kw)
    return Agent(**base)


def _sim(agents, world=None, **inn_kw):
    return Simulation(world=world or World(6, 6), agents=agents, brains={},
                      economy=True, innovation=InnovationConfig(enabled=True, **inn_kw))


class TestSkillYieldMult(unittest.TestCase):
    def test_no_skill_is_no_bonus(self):
        cfg = InnovationConfig(enabled=True)
        self.assertEqual(skill_yield_mult(0.0, cfg), 1.0)

    def test_full_skill_reaches_max_bonus(self):
        cfg = InnovationConfig(enabled=True, skill_yield_bonus=0.5, skill_cap=1.0)
        self.assertAlmostEqual(skill_yield_mult(1.0, cfg), 1.5)

    def test_disabled_is_always_one(self):
        cfg = InnovationConfig(enabled=False)
        self.assertEqual(skill_yield_mult(1.0, cfg), 1.0)


class TestSkillFromGathering(unittest.TestCase):
    def test_gathering_raises_skill(self):
        world = World(6, 6)
        world.add_facility(Facility("F", FacilityType.FOREST, 0, 0))
        a = _agent(x=0, y=0)
        sim = _sim([a], world=world, skill_gain_per_use=0.1)
        sim._do_gather(a, Action(ActionType.GATHER))
        self.assertGreater(a.skill, 0.0)

    def test_skill_saturates_at_cap(self):
        world = World(6, 6)
        world.add_facility(Facility("F", FacilityType.FOREST, 0, 0))
        a = _agent(x=0, y=0, skill=0.99)
        sim = _sim([a], world=world, skill_gain_per_use=0.5, skill_cap=1.0)
        sim._do_gather(a, Action(ActionType.GATHER))
        self.assertLessEqual(a.skill, 1.0)

    def test_high_skill_raises_gather_yield(self):
        world = World(6, 6)
        world.add_facility(Facility("F1", FacilityType.FOREST, 0, 0))
        world.add_facility(Facility("F2", FacilityType.FOREST, 1, 1))
        novice = _agent(id="n", x=0, y=0, skill=0.0)
        expert = _agent(id="e", x=1, y=1, skill=1.0)
        sim = _sim([novice, expert], world=world, skill_yield_bonus=1.0)
        sim._do_gather(novice, Action(ActionType.GATHER))
        sim._do_gather(expert, Action(ActionType.GATHER))
        self.assertGreater(expert.materials(), novice.materials())

    def test_no_skill_effect_off_the_layer(self):
        world = World(6, 6)
        world.add_facility(Facility("F", FacilityType.FOREST, 0, 0))
        a = _agent(x=0, y=0, skill=1.0)
        sim = Simulation(world=world, agents=[a], brains={}, economy=True)  # off
        sim._do_gather(a, Action(ActionType.GATHER))
        self.assertEqual(a.skill, 1.0, "not touched off the layer")


class TestSkillFromCrafting(unittest.TestCase):
    def _workshop_sim(self, **inn_kw):
        world = World(6, 6)
        world.add_facility(Facility("WS", FacilityType.WORKSHOP, 0, 0))
        a = _agent(x=0, y=0)
        a.inventory["materials"] = 10
        sim = _sim([a], world=world, **inn_kw)
        return sim, a

    def test_crafting_raises_skill(self):
        sim, a = self._workshop_sim(skill_gain_per_use=0.1)
        sim._do_craft(a, Action(ActionType.CRAFT, {"item": "tools"}))
        self.assertGreater(a.skill, 0.0)

    def test_high_skill_raises_craft_output(self):
        sim, a = self._workshop_sim(skill_yield_bonus=1.0)
        a.skill = 1.0
        sim._do_craft(a, Action(ActionType.CRAFT, {"item": "tools"}))
        self.assertGreater(a.inventory.get("tools", 0), 1)

    def test_default_layer_off_matches_the_original_craft(self):
        world = World(6, 6)
        world.add_facility(Facility("WS", FacilityType.WORKSHOP, 0, 0))
        a = _agent(x=0, y=0)
        a.inventory["materials"] = 3
        sim = Simulation(world=world, agents=[a], brains={}, economy=True)  # off
        sim._do_craft(a, Action(ActionType.CRAFT, {"item": "tools"}))
        self.assertEqual(a.inventory.get("tools", 0), 1)
        self.assertEqual(a.materials(), 1)


class TestDiscovery(unittest.TestCase):
    def _workshop_sim(self, **inn_kw):
        world = World(6, 6)
        world.add_facility(Facility("WS", FacilityType.WORKSHOP, 0, 0))
        a = _agent(x=0, y=0, skill=0.9)
        a.inventory["materials"] = 10
        sim = _sim([a], world=world, **inn_kw)
        return sim, a

    def test_skilled_crafter_can_discover_a_cheaper_recipe(self):
        sim, a = self._workshop_sim(discovery_chance=1.0, discovery_skill_min=0.6)
        sim._do_craft(a, Action(ActionType.CRAFT, {"item": "tools"}))
        self.assertEqual(sim.recipes["tools"], DISCOVERY_POOL["tools"][0])
        self.assertEqual(sim.metrics.inventions, 1)

    def test_discovery_never_touches_the_shared_module_default(self):
        sim, a = self._workshop_sim(discovery_chance=1.0, discovery_skill_min=0.6)
        sim._do_craft(a, Action(ActionType.CRAFT, {"item": "tools"}))
        self.assertEqual(MK.RECIPES["tools"], ({"materials": 2}, "workshop"))

    def test_discovered_recipe_is_cheaper_to_use(self):
        sim, a = self._workshop_sim(discovery_chance=1.0, discovery_skill_min=0.6)
        sim._do_craft(a, Action(ActionType.CRAFT, {"item": "tools"}))
        before = a.materials()
        sim._do_craft(a, Action(ActionType.CRAFT, {"item": "tools"}))
        self.assertEqual(before - a.materials(), 1, "the discovered recipe costs 1 material")

    def test_unskilled_crafter_never_discovers(self):
        sim, a = self._workshop_sim(discovery_chance=1.0, discovery_skill_min=0.6)
        a.skill = 0.1
        sim._do_craft(a, Action(ActionType.CRAFT, {"item": "tools"}))
        self.assertEqual(sim.metrics.inventions, 0)

    def test_default_discovery_chance_never_fires(self):
        sim, a = self._workshop_sim()  # discovery_chance default 0.03, skill 0.9
        for _ in range(1):
            sim._do_craft(a, Action(ActionType.CRAFT, {"item": "tools"}))
        # Deterministic RNG seed 0 shared across the suite would make this
        # flaky at 3% -- instead force it off explicitly.
        sim2, a2 = self._workshop_sim(discovery_chance=0.0)
        sim2._do_craft(a2, Action(ActionType.CRAFT, {"item": "tools"}))
        self.assertEqual(sim2.metrics.inventions, 0)

    def test_discovery_writes_to_the_library(self):
        world = World(6, 6)
        world.add_facility(Facility("WS", FacilityType.WORKSHOP, 0, 0))
        a = _agent(x=0, y=0, skill=0.9)
        a.inventory["materials"] = 10
        sim = Simulation(world=world, agents=[a], brains={}, economy=True,
                         innovation=InnovationConfig(enabled=True, discovery_chance=1.0,
                                                    discovery_skill_min=0.6),
                         library=TownLibrary())
        sim._do_craft(a, Action(ActionType.CRAFT, {"item": "tools"}))
        self.assertGreater(len(sim.library), 0)


class TestRecipesSurfacedInObservation(unittest.TestCase):
    def test_observation_reflects_a_discovery(self):
        world = World(6, 6)
        world.add_facility(Facility("WS", FacilityType.WORKSHOP, 0, 0))
        a = _agent(x=0, y=0, skill=0.9)
        a.inventory["materials"] = 10
        sim = _sim([a], world=world, discovery_chance=1.0, discovery_skill_min=0.6)
        sim._do_craft(a, Action(ActionType.CRAFT, {"item": "tools"}))
        obs = sim._observe(a)
        self.assertEqual(obs.economy["recipes"]["tools"], {"materials": 1})


class TestBaselineUntouched(unittest.TestCase):
    def test_skill_field_defaults_to_zero(self):
        a = _agent()
        self.assertEqual(a.skill, 0.0)

    def test_snapshot_includes_skill(self):
        a = _agent(skill=0.42)
        self.assertEqual(a.snapshot()["skill"], 0.42)

    def test_recipes_default_to_an_identical_copy(self):
        sim = Simulation(world=World(6, 6), agents=[], brains={}, economy=True)
        self.assertEqual(sim.recipes, dict(MK.RECIPES))
        self.assertIsNot(sim.recipes, MK.RECIPES)


if __name__ == "__main__":
    unittest.main()
