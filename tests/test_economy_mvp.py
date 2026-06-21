"""Economy MVP: specialisation of supply creates demand for money.

Each profession produces one good well and is inefficient at off-specialty
self-supply, so buying from a specialist becomes the sensible path — that is what
gives food/materials a *structural* demand. Guards:
  * the specialisation multipliers,
  * `_harvest` applies them only under `--economy` (offline baseline untouched),
  * survival is preserved (off-specialty yield is low but never zero),
  * the heuristic's one rule: a poor food-gatherer buys rather than farms.
"""

import unittest

from emergence.affordances import gather_multiplier, gather_specialty
from emergence.brains.heuristic import HeuristicBrain
from emergence.observation import Observation
from emergence.scenario import make_simulation
from emergence.simulation import SimulationConfig
from emergence.world import FacilityType
from emergence.actions import ActionType


def _farm(sim):
    return next(f for f in sim.world.facilities if f.ftype is FacilityType.FARM)


class TestSpecialisation(unittest.TestCase):
    def test_multipliers(self):
        self.assertEqual(gather_multiplier("farmer", "food"), 2.0)
        self.assertEqual(gather_multiplier("miner", "materials"), 2.0)
        self.assertEqual(gather_multiplier("smith", "food"), 0.5)   # off-specialty
        self.assertEqual(gather_multiplier("farmer", "materials"), 0.5)
        self.assertEqual(gather_multiplier("smith", "money"), 1.0)  # unspecialised
        self.assertEqual(gather_specialty("farmer"), "food")
        self.assertIsNone(gather_specialty("guard"))


class TestHarvest(unittest.TestCase):
    def _harvest_food(self, economy, profession):
        sim = make_simulation("guardian", config=SimulationConfig(seed=1),
                              economy=economy)
        a = sim.agents[0]
        a.profession = profession
        farm = _farm(sim)
        a.x, a.y = farm.x, farm.y
        before = a.food()
        sim._harvest(a, farm)
        return a.food() - before

    def test_offline_baseline_unchanged(self):
        # economy off: profession does not matter, yield is the base (3).
        self.assertEqual(self._harvest_food(False, "farmer"), 3)
        self.assertEqual(self._harvest_food(False, "smith"), 3)

    def test_specialist_outproduces_generalist_when_economy_on(self):
        farmer = self._harvest_food(True, "farmer")
        smith = self._harvest_food(True, "smith")
        self.assertGreater(farmer, smith, "a farmer should out-farm a smith")
        self.assertGreaterEqual(smith, 1, "off-specialty must never be zero (no starvation lock)")


class TestHeuristicBuysFood(unittest.TestCase):
    def _obs(self, *, enabled=True, offers=None):
        return Observation(
            day=1, tick=1, self_view={}, position=(0, 0), nearby_facilities=[],
            here=None, others=[], open_proposals=[], granary_food=0,
            recent_events=[], economy={"enabled": enabled} if enabled else {},
            open_offers=offers or [],
        )

    def _smith(self):
        sim = make_simulation("guardian", config=SimulationConfig(seed=1), economy=True)
        a = sim.agents[0]
        a.profession = "smith"
        a.money = 10
        return HeuristicBrain("guardian"), a

    def test_poor_gatherer_buys_when_affordable(self):
        brain, smith = self._smith()
        offer = [{"id": 7, "give": "1 food", "want": "2 money", "maker": "a99"}]
        act = brain._buy_food_action(smith, self._obs(offers=offer))
        self.assertIsNotNone(act)
        self.assertEqual(act.type, ActionType.ACCEPT)
        self.assertEqual(act.params["offer_id"], 7)

    def test_farmer_does_not_buy(self):
        brain, smith = self._smith()
        farmer = smith
        farmer.profession = "farmer"
        offer = [{"id": 7, "give": "1 food", "want": "2 money", "maker": "a99"}]
        self.assertIsNone(brain._buy_food_action(farmer, self._obs(offers=offer)))

    def test_no_buy_when_unaffordable_or_economy_off(self):
        brain, smith = self._smith()
        dear = [{"id": 7, "give": "1 food", "want": "50 money", "maker": "a99"}]
        self.assertIsNone(brain._buy_food_action(smith, self._obs(offers=dear)))
        cheap = [{"id": 7, "give": "1 food", "want": "2 money", "maker": "a99"}]
        self.assertIsNone(brain._buy_food_action(smith, self._obs(enabled=False, offers=cheap)))


if __name__ == "__main__":
    unittest.main()
