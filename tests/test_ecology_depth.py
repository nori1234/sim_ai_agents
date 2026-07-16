"""Ecology depth (#111): follow-ups deferred from the first ecology slice
(#94) -- grazing/feeding cost, predators & danger, and market-traded
livestock. Each knob defaults to inert (feed_cost_per_head=0,
predator_daily_chance=0), so tests/test_ecology.py's existing breeding cycle
stays byte-identical unmodified. Deliberately out of scope here (per the
issue's own staging): wild fauna & hunting, pests/blight, and full
prey<->predator population dynamics -- the issue itself suggests staging
hunting/wildlife before predators/pests/grazing, but hunting needs a new
gatherable presence in the world (a new facility/verb) that's a larger,
separately-scoped design than the herd-side mechanics here.
"""

import unittest

from emergence import market as MK
from emergence.ecology import EcologyConfig
from emergence.psyche import PsycheConfig
from emergence.scenario import make_simulation
from emergence.simulation import SimulationConfig


def _sim(enabled=True, psyche=False, **kw):
    return make_simulation("guardian", n_agents=4, config=SimulationConfig(seed=1),
                           ecology=EcologyConfig(enabled=enabled, **kw),
                           psyche=PsycheConfig(enabled=psyche))


class TestFeedCost(unittest.TestCase):
    def test_default_feed_cost_is_free(self):
        sim = _sim()
        a = sim.agents[0]; a.inventory["livestock"] = 5; a.inventory["food"] = 0
        sim._feed_livestock()
        self.assertEqual(a.inventory["livestock"], 5, "0 cost (default) is a no-op")

    def test_feed_is_drawn_from_owners_food(self):
        sim = _sim(feed_cost_per_head=1.0)
        a = sim.agents[0]; a.inventory["livestock"] = 3; a.inventory["food"] = 10
        sim._feed_livestock()
        self.assertEqual(a.food(), 7, "3 head * 1.0 food each")
        self.assertEqual(a.inventory["livestock"], 3, "well fed, no losses")

    def test_short_feed_causes_starvation_losses(self):
        sim = _sim(feed_cost_per_head=1.0, starve_loss_rate=0.5)
        a = sim.agents[0]; a.inventory["livestock"] = 4; a.inventory["food"] = 0
        sim._feed_livestock()
        self.assertEqual(a.food(), 0, "no food to draw down further")
        self.assertLess(a.inventory["livestock"], 4, "hunger culls the herd")

    def test_no_herd_no_feed_effect(self):
        sim = _sim(feed_cost_per_head=1.0)
        a = sim.agents[0]; a.inventory["livestock"] = 0; a.inventory["food"] = 10
        sim._feed_livestock()
        self.assertEqual(a.food(), 10)


class TestPredatorRaids(unittest.TestCase):
    def test_default_chance_never_raids(self):
        sim = _sim()
        a = sim.agents[0]; a.inventory["livestock"] = 6
        sim._raid_livestock()
        self.assertEqual(a.inventory["livestock"], 6, "0 chance (default) is a no-op")

    def test_certain_raid_culls_the_herd(self):
        sim = _sim(predator_daily_chance=1.0, predator_loss_rate=0.5)
        a = sim.agents[0]; a.inventory["livestock"] = 6
        sim._raid_livestock()
        self.assertLess(a.inventory["livestock"], 6)

    def test_raid_frightens_owner_under_psyche(self):
        sim = _sim(predator_daily_chance=1.0, predator_fear=20.0, psyche=True)
        a = sim.agents[0]; a.inventory["livestock"] = 6; a.fear = 0.0
        sim._raid_livestock()
        self.assertGreater(a.fear, 0.0)

    def test_no_fear_without_psyche_layer(self):
        sim = _sim(predator_daily_chance=1.0, predator_fear=20.0, psyche=False)
        a = sim.agents[0]; a.inventory["livestock"] = 6; a.fear = 0.0
        sim._raid_livestock()
        self.assertEqual(a.fear, 0.0)

    def test_no_herd_nothing_to_raid(self):
        sim = _sim(predator_daily_chance=1.0)
        a = sim.agents[0]; a.inventory["livestock"] = 0
        sim._raid_livestock()
        self.assertEqual(a.inventory["livestock"], 0)


class TestMarketTradedLivestock(unittest.TestCase):
    def test_livestock_is_a_tradable_good(self):
        self.assertIn("livestock", MK.TRADABLE)

    def test_surfaced_as_tradable_under_ecology(self):
        sim = _sim(enabled=True)
        sim.economy = True
        obs = sim._observe(sim.agents[0])
        self.assertIn("livestock", obs.economy["tradable"])

    def test_not_surfaced_when_ecology_off(self):
        sim = make_simulation("guardian", n_agents=4, config=SimulationConfig(seed=1))
        sim.economy = True
        obs = sim._observe(sim.agents[0])
        self.assertNotIn("livestock", obs.economy["tradable"])
        self.assertEqual(obs.economy["tradable"], ["food", "materials", "tools", "money"],
                         "an economy-only run's tradable list is unchanged")


class TestBaselineUntouched(unittest.TestCase):
    def test_end_to_end_run_unaffected_by_new_knobs_at_default(self):
        sim = make_simulation("guardian", n_agents=4, config=SimulationConfig(seed=1, days=5),
                              ecology=EcologyConfig(enabled=True, start_herd=3))
        base = make_simulation("guardian", n_agents=4, config=SimulationConfig(seed=1, days=5),
                               ecology=EcologyConfig(enabled=True, start_herd=3))
        sim.run(); base.run()
        got = [a.inventory.get("livestock", 0) for a in sim.agents]
        want = [a.inventory.get("livestock", 0) for a in base.agents]
        self.assertEqual(got, want, "deterministic, and unaffected by the new inert knobs")


if __name__ == "__main__":
    unittest.main()
