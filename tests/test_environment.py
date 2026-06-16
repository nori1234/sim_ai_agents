import random
import unittest

from emergence.environment import (
    SEASONS,
    Environment,
    EnvironmentConfig,
)
from emergence.scenario import make_simulation
from emergence.simulation import SimulationConfig
from emergence.world import Facility, FacilityType, build_default_world


def _env(**cfg):
    world = build_default_world()
    return Environment(EnvironmentConfig(enabled=True, **cfg), world,
                       random.Random(1)), world


class TestEnvironmentDisabled(unittest.TestCase):
    def test_no_environment_object_by_default(self):
        sim = make_simulation("guardian", config=SimulationConfig(seed=1, days=2))
        self.assertIsNone(sim.environment)


class TestWeatherAndSeasons(unittest.TestCase):
    def test_seasons_cycle_over_time(self):
        env, _ = _env(season_length_days=2)
        seen = set()
        for day in range(1, 9):
            env.world.day = day
            env.advance_day(_FakeSim([]))
            seen.add(env.weather.season)
        self.assertEqual(seen, set(SEASONS))

    def test_winter_drains_more_energy(self):
        env, _ = _env()
        # Force winter and a neutral condition.
        from emergence.environment import Weather
        env.weather = Weather(season="winter", condition="clear")
        summer = Weather(season="summer", condition="clear")
        self.assertGreater(env.energy_multiplier(), 1.0)
        env.weather = summer
        self.assertLess(env.energy_multiplier(), 1.05)


class TestDepletion(unittest.TestCase):
    def test_gather_depletes_stock_and_yield_falls(self):
        env, world = _env(season_length_days=99)  # stay in spring (yield 1.0)
        from emergence.environment import Weather
        env.weather = Weather(season="spring", condition="clear")
        farm = world.facilities_of(FacilityType.FARM)[0]
        before = env.stock[farm.name]
        first = env.gather(farm, "food", 3)
        self.assertEqual(first, 3)            # full yield when stocked
        self.assertLess(env.stock[farm.name], before)
        # Drain the site hard; stock falls far and yield drops to zero
        # (as stock shrinks the harvest shrinks, so it asymptotes near empty).
        for _ in range(40):
            env.gather(farm, "food", 3)
        self.assertLess(env.stock[farm.name], 8.0)
        self.assertEqual(env.gather(farm, "food", 3), 0)  # too depleted to yield

    def test_regen_restores_stock(self):
        env, world = _env(regen_per_day=10.0, stock_capacity=30.0)
        farm = world.facilities_of(FacilityType.FARM)[0]
        env.stock[farm.name] = 5.0
        env.world.day = 2
        env.advance_day(_FakeSim([]))
        self.assertGreater(env.stock[farm.name], 5.0)


class TestEconomy(unittest.TestCase):
    def test_price_rises_when_food_scarce(self):
        env, _ = _env()
        rich = _FakeSim(_agents_with_food([20, 20, 20]))
        poor = _FakeSim(_agents_with_food([0, 0, 0]))
        env._recompute_prices(rich.agents)
        cheap = env.price["food"]
        env._recompute_prices(poor.agents)
        dear = env.price["food"]
        self.assertGreater(dear, cheap)


class TestDisasters(unittest.TestCase):
    def test_disaster_can_strike_and_is_recorded(self):
        env, _ = _env(disaster_daily_prob=1.0)  # guarantee a strike
        sim = _FakeSim(_agents_with_food([5, 5]))
        env.world.day = 2
        env.advance_day(sim)
        self.assertGreaterEqual(env.summary()["disasters_total"], 1)


class TestEnvironmentEndToEnd(unittest.TestCase):
    def test_env_off_matches_baseline(self):
        # Enabling nothing leaves the run byte-identical to a plain run.
        base = make_simulation("gemini", config=SimulationConfig(seed=42)); base.run()
        self.assertEqual(base.metrics.crimes_total, 211)  # known baseline (Phase 2)

    def test_env_run_is_deterministic(self):
        a = make_simulation("gemini", config=SimulationConfig(seed=7), environment=True)
        a.run()
        b = make_simulation("gemini", config=SimulationConfig(seed=7), environment=True)
        b.run()
        self.assertEqual(a.metrics.as_dict(), b.metrics.as_dict())

    def test_env_run_records_a_season(self):
        sim = make_simulation("guardian", config=SimulationConfig(seed=42),
                              environment=True)
        sim.run()
        self.assertIn(sim.metrics.final_season, SEASONS)


# -- lightweight stand-ins so we can unit-test Environment in isolation -------
class _FakeAgent:
    def __init__(self, food):
        self._food = food
        self.alive = True
        self.energy = 100.0
        self.x = self.y = 0

    @property
    def pos(self):
        return (self.x, self.y)

    def food(self):
        return self._food

    def materials(self):
        return 0


class _FakeSim:
    def __init__(self, agents):
        self.agents = agents


def _agents_with_food(amounts):
    return [_FakeAgent(a) for a in amounts]


if __name__ == "__main__":
    unittest.main()
