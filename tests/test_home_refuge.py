"""Home as a refuge (#73/#113).

Two halves. First, from the *weather* (#73): foul weather (a storm, a cold
snap, deep winter) drains energy faster; a roof keeps the extra bite off, so
harsh weather is a real reason to head indoors. Only the environment layer
carries weather, so with it off nothing changes.

Second, from *street crime* (#113): being sheltered at home makes an agent a
poor mark for opportunistic theft/violence -- surfaced only under the psyche
layer, so heuristic targeting is unchanged without it. Burglary is the
counter: breaking into an owned home (reusing strike(building), keyed off
the owner from #93/#102) is a costlier, rarer crime than street theft, gated
on --society so a plain --economy town still reads every house-strike as
arson.

Either way the four-society baseline is byte-identical off every relevant
layer.
"""

import unittest

from emergence.actions import Action, ActionType
from emergence.brains.heuristic import HeuristicBrain, LOW_ENERGY
from emergence.environment import EnvironmentConfig, Weather
from emergence.observation import Observation
from emergence.psyche import PsycheConfig
from emergence.scenario import make_simulation
from emergence.simulation import SimulationConfig
from emergence.society import SocietyConfig
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


def _build(sim, agent, name="Cabin", ftype="house"):
    agent.inventory["materials"] = 2
    sim._do_build(agent, Action(ActionType.BUILD, {"facility_type": ftype, "name": name}))
    return next(f for f in sim.world.facilities if f.name == name)


class TestHomeSafety(unittest.TestCase):
    def test_sheltered_others_are_flagged_only_under_psyche(self):
        sim = make_simulation("guardian", config=SimulationConfig(seed=1),
                              psyche=PsycheConfig(enabled=True))
        a, b = sim.agents[0], sim.agents[1]
        home = next(f for f in sim.world.facilities if f.ftype is FacilityType.HOUSE)
        b.x, b.y = home.x, home.y
        obs = sim._observe(a)
        other = next(o for o in obs.others if o["id"] == b.id)
        self.assertTrue(other["sheltered"])

    def test_no_sheltered_key_without_psyche(self):
        sim = make_simulation("guardian", config=SimulationConfig(seed=1))
        a, b = sim.agents[0], sim.agents[1]
        home = next(f for f in sim.world.facilities if f.ftype is FacilityType.HOUSE)
        b.x, b.y = home.x, home.y
        obs = sim._observe(a)
        other = next(o for o in obs.others if o["id"] == b.id)
        self.assertNotIn("sheltered", other)

    def test_heuristic_skips_a_sheltered_mark_for_theft(self):
        brain = HeuristicBrain("guardian")
        rich_but_sheltered = {"id": "r", "distance": 1, "money": 100, "food": 10,
                              "materials": 10, "sheltered": True}
        poor_and_exposed = {"id": "p", "distance": 2, "money": 1, "food": 0,
                            "materials": 0, "sheltered": False}
        obs = Observation(day=1, tick=1, self_view={}, position=(0, 0),
                          nearby_facilities=[], here=None,
                          others=[rich_but_sheltered, poor_and_exposed],
                          open_proposals=[], granary_food=0, recent_events=[])
        target = brain._nearby_target(obs)
        self.assertEqual(target["id"], "p", "the richer mark is skipped because it's sheltered")

    def test_heuristic_skips_a_sheltered_foe_for_retaliation(self):
        brain = HeuristicBrain("guardian")
        sheltered_foe = {"id": "f", "distance": 1, "trust": -0.9, "sheltered": True}
        obs = Observation(day=1, tick=1, self_view={}, position=(0, 0),
                          nearby_facilities=[], here=None, others=[sheltered_foe],
                          open_proposals=[], granary_food=0, recent_events=[])
        self.assertIsNone(brain._nearby_foe(obs))


class TestBurglary(unittest.TestCase):
    def _sim(self, economy=True, society=True):
        return make_simulation(
            "guardian", n_agents=4, config=SimulationConfig(seed=1),
            economy=economy, society=SocietyConfig(enabled=society))

    def test_striking_an_owned_house_under_society_is_burglary_not_arson(self):
        sim = self._sim()
        owner, thief = sim.agents[0], sim.agents[1]
        f = _build(sim, owner, name="Homestead")
        owner.money = 50
        thief.x, thief.y = f.x, f.y
        before = thief.money
        sim._strike(thief, facility=f)
        self.assertGreater(thief.money, before, "burglary loots the absent owner")
        self.assertEqual(sim.metrics.crimes_by_type.get("burglary"), 1)
        self.assertIsNone(sim.metrics.crimes_by_type.get("arson"))

    def test_loot_is_conserved_from_the_owner(self):
        sim = self._sim()
        owner, thief = sim.agents[0], sim.agents[1]
        f = _build(sim, owner, name="Homestead")
        owner.money = 50
        thief.x, thief.y = f.x, f.y
        before_owner, before_thief = owner.money, thief.money
        sim._strike(thief, facility=f)
        gained = thief.money - before_thief
        self.assertGreater(gained, 0)
        self.assertEqual(owner.money, before_owner - gained)

    def test_owner_cannot_burgle_their_own_house(self):
        sim = self._sim()
        owner = sim.agents[0]
        f = _build(sim, owner, name="Homestead")
        owner.x, owner.y = f.x, f.y
        before = sim.metrics.crimes_total
        sim._strike(owner, facility=f)
        self.assertEqual(sim.metrics.crimes_by_type.get("burglary"), None)
        self.assertEqual(sim.metrics.crimes_by_type.get("arson", 0), 1)

    def test_unowned_commons_house_strike_is_still_arson(self):
        sim = self._sim()
        thief = sim.agents[0]
        commons_house = next(f for f in sim.world.facilities if f.ftype is FacilityType.HOUSE)
        self.assertIsNone(commons_house.owner)
        thief.x, thief.y = commons_house.x, commons_house.y
        sim._strike(thief, facility=commons_house)
        self.assertEqual(sim.metrics.crimes_by_type.get("burglary"), None)
        self.assertEqual(sim.metrics.crimes_by_type.get("arson"), 1)

    def test_without_society_an_owned_house_strike_is_still_arson(self):
        sim = self._sim(economy=True, society=False)
        owner, thief = sim.agents[0], sim.agents[1]
        f = _build(sim, owner, name="Homestead")
        thief.x, thief.y = f.x, f.y
        sim._strike(thief, facility=f)
        self.assertEqual(sim.metrics.crimes_by_type.get("burglary"), None)
        self.assertEqual(sim.metrics.crimes_by_type.get("arson"), 1)

    def test_burglary_costs_more_energy_than_a_plain_strike(self):
        sim = self._sim()
        owner, thief = sim.agents[0], sim.agents[1]
        f = _build(sim, owner, name="Homestead")
        thief.x, thief.y = f.x, f.y
        thief.energy = 90.0
        sim._strike(thief, facility=f)
        from emergence.simulation import BURGLARY_ENERGY_COST
        self.assertEqual(thief.energy, 90.0 - BURGLARY_ENERGY_COST)

    def test_offline_baseline_never_burgles(self):
        sim = make_simulation("guardian", config=SimulationConfig(seed=1))
        thief = sim.agents[0]
        house = next(f for f in sim.world.facilities if f.ftype is FacilityType.HOUSE)
        thief.x, thief.y = house.x, house.y
        sim._strike(thief, facility=house)
        self.assertIsNone(sim.metrics.crimes_by_type.get("burglary"))


if __name__ == "__main__":
    unittest.main()
