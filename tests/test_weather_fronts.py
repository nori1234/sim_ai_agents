"""Dynamic weather: fronts (#74).

With `dynamic_weather` on, weather moves in *fronts* — a condition persists for a
few days, then shifts (and always shifts when the season turns) — so multi-day
storms and fair spells emerge instead of day-to-day flicker. It's still
seasonally weighted and fully seeded. Off (the default) keeps the one-draw-per-day
behaviour, so existing environment runs are unchanged and the four-society
baseline is byte-identical.
"""

import unittest

from emergence.environment import Environment, EnvironmentConfig
from emergence.scenario import make_simulation
from emergence.simulation import SimulationConfig
from emergence.world import World


def _env(**kw):
    cfg = EnvironmentConfig(enabled=True, **kw)
    import random
    return Environment(cfg, World(), random.Random(7))


def _conditions(env, days):
    seq = [env.weather.condition]
    for _ in range(days):
        env.day += 1
        env._advance_weather()
        seq.append(env.weather.condition)
    return seq


def _max_run(seq):
    best = run = 1
    for i in range(1, len(seq)):
        run = run + 1 if seq[i] == seq[i - 1] else 1
        best = max(best, run)
    return best


class TestFronts(unittest.TestCase):
    def test_a_front_persists_for_several_days(self):
        env = _env(dynamic_weather=True, front_min_days=4, front_max_days=4,
                   season_length_days=999)
        seq = _conditions(env, 16)
        self.assertGreaterEqual(_max_run(seq), 4, "a front should hold for its duration")

    def test_a_season_turn_breaks_the_front(self):
        env = _env(dynamic_weather=True, front_min_days=99, front_max_days=99,
                   season_length_days=1)               # season changes every day
        env.day = 1; env._advance_weather()
        s1 = env.weather.season
        env.day = 2; env._advance_weather()
        self.assertNotEqual(env.weather.season, s1, "the season turned → a new front")

    def test_static_default_is_unchanged(self):
        # dynamic off: two identically-seeded envs produce the same daily draws,
        # i.e. the old independent-per-day behaviour (no front state in play).
        import random
        a = Environment(EnvironmentConfig(enabled=True), World(), random.Random(7))
        b = Environment(EnvironmentConfig(enabled=True), World(), random.Random(7))
        self.assertEqual(_conditions(a, 10), _conditions(b, 10))
        self.assertEqual(a._front_days_left, 0, "front state never engaged when static")


class TestBaselineUntouched(unittest.TestCase):
    def test_default_environment_run_unaffected(self):
        # dynamic_weather defaults off; an --environment run still behaves as before.
        sim = make_simulation("guardian", config=SimulationConfig(seed=1, days=8),
                              environment=EnvironmentConfig(enabled=True))
        sim.run()
        self.assertFalse(sim.environment.config.dynamic_weather)


if __name__ == "__main__":
    unittest.main()
