"""Agriculture depth (#105): follow-ups deferred from the first agriculture
slice (#95) -- seed & storage, crop failure, and a soil/fallow benefit.

Each knob defaults to inert (seed_cost=0, crop_failure_chance=0,
soil_depletion_per_harvest=0/soil_regen_per_fallow_day=0), so with agriculture
on but these left at their defaults the cycle is byte-identical to #95's
original behaviour (see tests/test_agriculture.py, which stays green
unmodified). Deliberately out of scope here (per the issue's own text):
per-crop variety, and a dedicated farmer routine (which needs a new
"move to a specific facility" affordance beyond MOVE {facility_type}).
"""

import unittest

from emergence.actions import Action, ActionType
from emergence.environment import Environment, EnvironmentConfig
from emergence.scenario import make_simulation
from emergence.simulation import SimulationConfig
from emergence.world import FacilityType


def _env(**kw):
    cfg = EnvironmentConfig(enabled=True, agriculture=True, weather=False,
                            disasters=False, **kw)
    sim = make_simulation("guardian", config=SimulationConfig(seed=1), environment=cfg)
    farm = next(f for f in sim.world.facilities if f.ftype is FacilityType.FARM)
    return sim, sim.environment, farm


class TestSeedCost(unittest.TestCase):
    def test_default_seed_cost_is_free(self):
        sim, env, farm = _env()
        env.grow.clear(); env.ripe.clear()
        a = sim.agents[0]; a.x, a.y = farm.x, farm.y; a.inventory["food"] = 0
        sim._do_sow(a, Action(ActionType.SOW, {}))
        self.assertEqual(env.crop_state(farm), "growing")

    def test_seed_cost_is_consumed_on_success(self):
        sim, env, farm = _env(seed_cost=2)
        env.grow.clear(); env.ripe.clear()
        a = sim.agents[0]; a.x, a.y = farm.x, farm.y; a.inventory["food"] = 5
        sim._do_sow(a, Action(ActionType.SOW, {}))
        self.assertEqual(env.crop_state(farm), "growing")
        self.assertEqual(a.food(), 3)

    def test_cannot_sow_without_enough_seed(self):
        sim, env, farm = _env(seed_cost=2)
        env.grow.clear(); env.ripe.clear()
        a = sim.agents[0]; a.x, a.y = farm.x, farm.y; a.inventory["food"] = 1
        sim._do_sow(a, Action(ActionType.SOW, {}))
        self.assertEqual(env.crop_state(farm), "empty", "not enough seed-corn")
        self.assertEqual(a.food(), 1, "nothing spent on a failed sow")


class TestCropFailure(unittest.TestCase):
    def _fix_weather(self, env, condition):
        # weather=False in the fixture forces "clear" on every _advance_weather()
        # call, which would overwrite a manually-set condition before the
        # agriculture step reads it -- freeze it for this tick instead.
        from emergence import environment as E
        env.weather = E.Weather(season="summer", condition=condition)
        env._advance_weather = lambda: None

    def test_severe_weather_can_wipe_a_growing_crop(self):
        sim, env, farm = _env(crop_grow_days=5, crop_failure_chance=1.0)
        env.grow.clear(); env.ripe.clear()
        env.sow(farm)
        self._fix_weather(env, "storm")
        sim._end_of_day(verbose=False)
        self.assertEqual(env.crop_state(farm), "empty", "wiped out, needs re-sowing")
        self.assertTrue(any(e.get("kind") == "crop_failure" for e in sim.world.events))

    def test_default_chance_never_fails(self):
        sim, env, farm = _env(crop_grow_days=5)
        env.grow.clear(); env.ripe.clear()
        env.sow(farm)
        self._fix_weather(env, "storm")
        sim._end_of_day(verbose=False)
        self.assertEqual(env.crop_state(farm), "growing", "0 chance (default) never fails")

    def test_mild_weather_never_triggers_failure(self):
        sim, env, farm = _env(crop_grow_days=5, crop_failure_chance=1.0)
        env.grow.clear(); env.ripe.clear()
        env.sow(farm)
        self._fix_weather(env, "clear")
        sim._end_of_day(verbose=False)
        self.assertEqual(env.crop_state(farm), "growing", "clear weather doesn't wipe crops")


class TestSoilFertility(unittest.TestCase):
    def test_default_soil_never_depletes(self):
        sim, env, farm = _env(crop_harvests=3, crop_yield=4)
        env.grow.clear(); env.ripe[farm.name] = 3
        got = [env.harvest_crop(farm) for _ in range(3)]
        self.assertEqual(got, [4, 4, 4], "0 depletion (default) leaves yield unchanged")

    def test_repeated_harvests_deplete_soil_and_lower_yield(self):
        sim, env, farm = _env(crop_harvests=4, crop_yield=10,
                              soil_depletion_per_harvest=0.2, soil_min_fertility=0.3)
        env.grow.clear(); env.ripe[farm.name] = 4
        got = [env.harvest_crop(farm) for _ in range(4)]
        self.assertEqual(got[0], 10, "first harvest at full fertility")
        self.assertLess(got[-1], got[0], "yield falls as the field is worked")

    def test_soil_floors_at_min_fertility(self):
        sim, env, farm = _env(crop_yield=10, soil_depletion_per_harvest=0.5,
                              soil_min_fertility=0.4)
        env.soil[farm.name] = 1.0
        for _ in range(5):
            env.ripe[farm.name] = 1
            env.harvest_crop(farm)
        self.assertGreaterEqual(env.soil[farm.name], 0.4)

    def test_fallow_rest_restores_fertility(self):
        sim, env, farm = _env(soil_regen_per_fallow_day=0.25)
        env.grow.clear(); env.ripe.clear()          # fully fallow (empty)
        env.soil[farm.name] = 0.5
        sim._end_of_day(verbose=False)
        self.assertGreater(env.soil[farm.name], 0.5)

    def test_growing_field_does_not_regen_soil(self):
        sim, env, farm = _env(crop_grow_days=5, soil_regen_per_fallow_day=0.25)
        env.grow.clear(); env.ripe.clear()
        env.soil[farm.name] = 0.5
        env.sow(farm)                                 # now growing, not fallow
        sim._end_of_day(verbose=False)
        self.assertEqual(env.soil[farm.name], 0.5, "a growing field isn't resting")


class TestBaselineUntouched(unittest.TestCase):
    def test_defaults_reproduce_the_original_agriculture_cycle(self):
        # Same shape as test_agriculture.py's own cycle test, with every #105
        # knob left at its inert default.
        sim, env, farm = _env(crop_grow_days=2, crop_yield=4)
        env.grow.clear(); env.ripe.clear()
        a = sim.agents[0]; a.x, a.y = farm.x, farm.y; a.inventory["food"] = 0
        sim._do_sow(a, Action(ActionType.SOW, {}))
        self.assertEqual(env.crop_state(farm), "growing")
        for _ in range(2):
            sim._end_of_day(verbose=False)
        self.assertEqual(env.crop_state(farm), "ripe")
        sim._do_gather(a, Action(ActionType.GATHER, {}))
        self.assertEqual(a.food(), 4)


if __name__ == "__main__":
    unittest.main()
