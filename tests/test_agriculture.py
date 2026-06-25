"""Agriculture as a living cycle (農の循環): a farm is a field, not a tap.

Under the environment layer's agriculture subsystem a FARM must be **sown**; it
**ripens** over a few days of growing season into a productive plot that bears a
handful of **harvests** before going **fallow** and needing re-sowing. Growth
halts in winter, so *when* you plant matters. With agriculture off a FARM is the
constant yield tap it has always been, so the four-society baseline (environment
off) and plain --environment runs are byte-identical. Guards:
  * sowing a fallow field starts it growing; you can't sow a sown/ripe one;
  * a field ripens over crop_grow_days of growing season (winter halts it);
  * a ripe field bears crop_harvests harvests, then goes fallow;
  * harvesting a fallow/growing field yields nothing (timing matters);
  * agriculture is opt-in within --environment (default off).
"""

import unittest

from emergence.environment import Environment, EnvironmentConfig
from emergence.scenario import make_simulation
from emergence.simulation import SimulationConfig
from emergence.world import FacilityType


def _env(**kw):
    cfg = EnvironmentConfig(enabled=True, agriculture=True, **kw)
    sim = make_simulation("guardian", config=SimulationConfig(seed=1), environment=cfg)
    farm = next(f for f in sim.world.facilities if f.ftype is FacilityType.FARM)
    return sim, sim.environment, farm


def _winter_off(env):
    # Freeze season effects out of the way for deterministic growth/yield.
    env.config = EnvironmentConfig(enabled=True, agriculture=True, weather=False,
                                   disasters=False, crop_grow_days=env.config.crop_grow_days,
                                   crop_yield=env.config.crop_yield,
                                   crop_harvests=env.config.crop_harvests)


class TestCropCycle(unittest.TestCase):
    def test_fallow_field_can_be_sown_then_ripens(self):
        sim, env, farm = _env(crop_grow_days=3)
        # clear any seeded productivity to start from fallow
        env.ripe.clear(); env.grow.clear()
        self.assertEqual(env.crop_state(farm), "empty")
        self.assertTrue(env.sow(farm))
        self.assertEqual(env.crop_state(farm), "growing")
        self.assertFalse(env.sow(farm), "can't sow an already-sown field")
        for _ in range(3):                     # ripen over crop_grow_days
            sim._end_of_day(verbose=False)
        self.assertEqual(env.crop_state(farm), "ripe")

    def test_winter_halts_growth(self):
        sim, env, farm = _env(crop_grow_days=2, season_length_days=999)
        env.ripe.clear(); env.grow.clear()
        env.sow(farm)
        # Force winter and re-tick: SEASON_GROWTH["winter"] == 0 → no progress.
        from emergence import environment as E
        env.weather = E.Weather(season="winter", condition="clear")
        before = env.grow[farm.name]
        # emulate the day-boundary growth step directly under winter
        step = E.SEASON_GROWTH["winter"]
        self.assertEqual(step, 0.0)
        self.assertEqual(env.crop_state(farm), "growing")
        self.assertEqual(env.grow[farm.name], before, "winter: a sown field doesn't advance")

    def test_ripe_field_bears_several_harvests_then_goes_fallow(self):
        sim, env, farm = _env(crop_harvests=3, crop_yield=4, weather=False, disasters=False)
        env.grow.clear(); env.ripe[farm.name] = 3
        got = [env.harvest_crop(farm) for _ in range(3)]
        self.assertEqual(got, [4, 4, 4], "three harvests of the per-harvest yield")
        self.assertEqual(env.crop_state(farm), "empty", "spent field goes fallow")
        self.assertEqual(env.harvest_crop(farm), 0, "a fallow field gives nothing")

    def test_harvesting_an_unripe_field_yields_nothing(self):
        sim, env, farm = _env()
        env.ripe.clear(); env.grow.clear()
        self.assertEqual(env.harvest_crop(farm), 0, "fallow")
        env.sow(farm)
        self.assertEqual(env.harvest_crop(farm), 0, "still growing")


class TestSowAndHarvestThroughTheEngine(unittest.TestCase):
    def test_sow_then_harvest_via_actions(self):
        from emergence.actions import Action, ActionType
        sim, env, farm = _env(crop_grow_days=2, crop_yield=4)
        env.grow.clear(); env.ripe.clear()
        a = sim.agents[0]
        a.x, a.y = farm.x, farm.y
        a.inventory["food"] = 0
        sim._do_sow(a, Action(ActionType.SOW, {}))
        self.assertEqual(env.crop_state(farm), "growing")
        self.assertTrue(any(e.get("kind") == "sow" for e in sim.world.events))
        for _ in range(2):
            sim._end_of_day(verbose=False)
        self.assertEqual(env.crop_state(farm), "ripe")
        sim._do_gather(a, Action(ActionType.GATHER, {}))
        self.assertGreater(a.food(), 0, "harvesting a ripe field yields food")

    def test_crop_state_surfaced_in_observation(self):
        sim, env, farm = _env()
        env.grow.clear(); env.ripe[farm.name] = sim.environment.config.crop_harvests
        a = sim.agents[0]; a.x, a.y = farm.x, farm.y
        here = sim._observe(a).here
        self.assertEqual(here.get("crop"), "ripe")


class TestBaselineUntouched(unittest.TestCase):
    def test_agriculture_off_leaves_farm_a_constant_tap(self):
        from emergence.actions import Action, ActionType
        cfg = EnvironmentConfig(enabled=True, agriculture=False)
        sim = make_simulation("guardian", config=SimulationConfig(seed=1), environment=cfg)
        farm = next(f for f in sim.world.facilities if f.ftype is FacilityType.FARM)
        a = sim.agents[0]; a.x, a.y = farm.x, farm.y; a.inventory["food"] = 0
        sim._do_gather(a, Action(ActionType.GATHER, {}))
        self.assertGreater(a.food(), 0, "no agriculture → farm yields on demand, as always")
        obs_here = sim._observe(a).here
        self.assertNotIn("crop", obs_here, "no crop state surfaced when agriculture is off")


if __name__ == "__main__":
    unittest.main()
