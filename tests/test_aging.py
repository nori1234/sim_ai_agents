"""Aging & natural death (老化・寿命): a lifecycle, not only violent/starvation death.

While the drives layer advances `age_days`, the body ages: past a senescence age
it tires faster (a gentle energy drain), and past a mortality onset it faces a
daily death hazard that rises with age and worsens with frailty (low energy) —
so elders pass away of `old_age`. Gated on the drives layer (which is what makes
age_days advance), so the four-society baseline (drives off) is byte-identical.
Guards:
  * an old agent loses extra energy each decay tick (senescence);
  * past the onset age, an agent can die of old_age (hazard rises with age);
  * a young agent never dies of age;
  * frailty (low energy) raises the hazard;
  * off the drives layer, aging is wholly inert.
"""

import unittest

from emergence.drives import DrivesConfig
from emergence.scenario import make_simulation
from emergence.simulation import SimulationConfig


def _sim(**drive_kw):
    cfg = DrivesConfig(enabled=drive_kw.pop("enabled", True), **drive_kw)
    return make_simulation("guardian", n_agents=4,
                           config=SimulationConfig(seed=1), drives=cfg)


class TestSenescence(unittest.TestCase):
    def test_the_old_lose_extra_energy_each_tick(self):
        sim = _sim(senescence_age_days=10, senescence_energy_penalty=5.0)
        young, old = sim.agents[0], sim.agents[1]
        for a in (young, old):
            a.hunger = a.fatigue = 0.0   # isolate: no hunger/fatigue penalties
            a.energy = 80.0
        young.age_days, old.age_days = 5, 40
        sim._tick_upkeep(young)
        sim._tick_upkeep(old)
        self.assertAlmostEqual(young.energy - old.energy, 5.0,
                               msg="the elder loses an extra senescence_energy_penalty")

    def test_no_senescence_drain_off_the_drives_layer(self):
        # Drives off: only the base metabolic decay applies, equally to young and
        # old — no extra senescence drain, so age makes no difference at all.
        sim = _sim(enabled=False, senescence_age_days=10, senescence_energy_penalty=5.0)
        young, old = sim.agents[0], sim.agents[1]
        for a in (young, old):
            a.hunger = a.fatigue = 0.0
            a.energy = 80.0
        young.age_days, old.age_days = 5, 200
        sim._tick_upkeep(young)
        sim._tick_upkeep(old)
        self.assertEqual(young.energy, old.energy, "drives off → age is inert")


class TestNaturalDeath(unittest.TestCase):
    def test_an_elder_dies_of_old_age(self):
        sim = _sim(mortality_onset_days=10, mortality_hazard_per_day=1.0)
        a = sim.agents[0]
        a.age_days = 20                  # well past onset; hazard saturates to 1
        sim._maybe_die_of_age(a)
        self.assertFalse(a.alive)
        self.assertEqual(a.cause_of_death, "old_age")

    def test_the_young_never_die_of_age(self):
        sim = _sim(mortality_onset_days=100, mortality_hazard_per_day=1.0)
        a = sim.agents[0]
        a.age_days = 30                  # below onset
        sim._maybe_die_of_age(a)
        self.assertTrue(a.alive, "not old enough to die of age")

    def test_frailty_raises_the_hazard(self):
        # base hazard 0 → never dies; frailty multiplies 0, still 0. Use a base
        # that's 0.4: a hale elder (mult 1) may live, a frail one (mult 5) dies
        # deterministically for a draw between 0.4 and 2.0. Force it via seed.
        sim = _sim(mortality_onset_days=10, mortality_hazard_per_day=0.4,
                   mortality_frailty_energy=30.0, mortality_frailty_mult=5.0)
        # Make the RNG draw a fixed value we can reason about.
        import random
        sim.rng = random.Random(0)
        draw = random.Random(0).random()          # the value _maybe_die_of_age will see
        a = sim.agents[0]
        a.age_days = 11                            # over=1 → hazard ~0.44 (hale) vs ~2.2 (frail)
        a.energy = 10.0                            # frail → hazard *5 → ~2.2 >= any draw
        sim._maybe_die_of_age(a)
        self.assertFalse(a.alive, f"a frail elder dies (draw was {draw:.3f})")
        self.assertEqual(a.cause_of_death, "old_age")

    def test_natural_death_emerges_in_a_long_run_but_not_a_short_one(self):
        short = make_simulation("guardian", n_agents=10,
                                config=SimulationConfig(seed=0, days=30),
                                drives=DrivesConfig(enabled=True))
        short.run()
        self.assertFalse(any(e.get("cause") == "old_age" for e in short.world.events),
                         "a 30-day run sees no old-age death (founders still hale)")
        long = make_simulation("guardian", n_agents=10,
                               config=SimulationConfig(seed=0, days=100),
                               drives=DrivesConfig(enabled=True))
        long.run()
        self.assertTrue(any(e.get("cause") == "old_age" for e in long.world.events),
                        "a long run sees elders pass away")

    def test_baseline_has_no_old_age_death(self):
        # drives off entirely: age never advances, nobody dies of old age.
        sim = make_simulation("guardian", n_agents=10,
                              config=SimulationConfig(seed=0, days=100))
        sim.run()
        self.assertFalse(any(e.get("cause") == "old_age" for e in sim.world.events))


if __name__ == "__main__":
    unittest.main()
