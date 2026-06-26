"""Home as a refuge from the weather (#73, first slice).

Foul weather (a storm, a cold snap, deep winter) drains energy faster; a roof
keeps the *extra* bite off, so harsh weather is a real reason to head indoors.
Only the environment layer carries weather, so with it off nothing changes (the
four-society baseline is byte-identical). Guards:
  * indoors, harsh-weather drain is reduced by shelter_weather_relief;
  * mild weather (no extra drain) shelters nothing;
  * off the environment layer there's no weather and no shelter effect;
  * the heuristic: a tiring body heads indoors / rests when the weather is harsh.

(Indoor safety from street crime, and burglary as its counter, are a follow-up.)
"""

import unittest

from emergence.actions import ActionType
from emergence.brains.heuristic import HeuristicBrain, LOW_ENERGY
from emergence.environment import EnvironmentConfig, Weather
from emergence.observation import Observation
from emergence.scenario import make_simulation
from emergence.simulation import SimulationConfig
from emergence.world import FacilityType


def _sim(**env):
    cfg = EnvironmentConfig(enabled=True, **env)
    return make_simulation("guardian", config=SimulationConfig(seed=1), environment=cfg)


class TestWeatherShelter(unittest.TestCase):
    def _drop(self, sim, indoors):
        a = sim.agents[0]
        if indoors:
            h = next(f for f in sim.world.facilities if f.ftype is FacilityType.HOUSE)
            a.x, a.y = h.x, h.y
        else:
            # a tile with no facility
            a.x, a.y = 0, 0
            assert sim.world.facility_at(a.pos) is None
        a.energy = 80.0
        sim._tick_upkeep(a)
        return 80.0 - a.energy

    def test_indoors_dampens_harsh_weather_drain(self):
        sim = _sim(shelter_weather_relief=0.7)
        sim.environment.weather = Weather(season="winter", condition="cold snap")
        outside = self._drop(sim, indoors=False)
        inside = self._drop(sim, indoors=True)
        self.assertLess(inside, outside, "a roof keeps the cold's extra bite off")

    def test_mild_weather_shelters_nothing(self):
        sim = _sim(shelter_weather_relief=0.7)
        sim.environment.weather = Weather(season="spring", condition="clear")  # mult <= 1
        outside = self._drop(sim, indoors=False)
        inside = self._drop(sim, indoors=True)
        self.assertAlmostEqual(inside, outside, msg="no extra drain → nothing to shelter from")

    def test_no_environment_no_shelter_effect(self):
        sim = make_simulation("guardian", config=SimulationConfig(seed=1))  # env off
        a = sim.agents[0]
        h = next(f for f in sim.world.facilities if f.ftype is FacilityType.HOUSE)
        a.x, a.y = h.x, h.y
        a.energy = 80.0
        sim._tick_upkeep(a)
        # base decay only; nothing weather-related
        self.assertEqual(a.energy, 80.0 - 5.0)  # ENERGY_DECAY_PER_TICK


class TestHeuristicSeeksShelter(unittest.TestCase):
    def _obs(self, *, weather="clear", season="spring", here=None):
        return Observation(
            day=1, tick=1, self_view={}, position=(0, 0), nearby_facilities=[],
            here={"type": here, "name": here} if here else None,
            others=[], open_proposals=[], granary_food=0, recent_events=[],
            environment={"weather": weather, "season": season})

    def _agent(self, energy):
        a = make_simulation("guardian", config=SimulationConfig(seed=1)).agents[0]
        a.energy = energy
        return HeuristicBrain("guardian"), a

    def test_heads_home_in_a_storm_when_tiring(self):
        brain, a = self._agent(LOW_ENERGY)            # wearing down
        act = brain._shelter_action(a, self._obs(weather="storm"))
        self.assertIsNotNone(act)
        self.assertEqual(act.type, ActionType.MOVE)
        self.assertEqual(act.params["facility_type"], "house")

    def test_rests_when_already_indoors(self):
        brain, a = self._agent(LOW_ENERGY)
        act = brain._shelter_action(a, self._obs(weather="cold snap", here="house"))
        self.assertEqual(act.type, ActionType.REST)

    def test_no_shelter_in_mild_weather(self):
        brain, a = self._agent(LOW_ENERGY)
        self.assertIsNone(brain._shelter_action(a, self._obs(weather="clear", season="spring")))

    def test_no_shelter_when_hale(self):
        brain, a = self._agent(100.0)                 # plenty of energy
        self.assertIsNone(brain._shelter_action(a, self._obs(weather="storm")))

    def test_inert_without_environment(self):
        brain, a = self._agent(LOW_ENERGY)
        obs = Observation(day=1, tick=1, self_view={}, position=(0, 0),
                          nearby_facilities=[], here=None, others=[], open_proposals=[],
                          granary_food=0, recent_events=[], environment={})
        self.assertIsNone(brain._shelter_action(a, obs))


if __name__ == "__main__":
    unittest.main()
