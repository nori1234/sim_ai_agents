"""Corruption as emergence (#38).

Enforcement is already a guard's *choice* and a bribe is an ordinary `give`, so
corruption needs no new mechanic — only that it be legible. A wanted offender
slipping money to a guard is named a "bribe" (a log + metric); the corruption is
the guard's own later choice not to arrest. Recognition is purely additive, so
the heuristic baseline is byte-identical (the contract test guards it).
"""

import unittest

from emergence.actions import Action, ActionType
from emergence.brains.heuristic import BRIBE_PRICE, HeuristicBrain
from emergence.observation import Observation
from emergence.scenario import make_simulation
from emergence.simulation import Simulation, SimulationConfig
from emergence.society import SocietyConfig


def _wanted(**kw):
    o = {"id": "x", "distance": 2, "last_crime_day": 1, "trust": -0.2,
         "money": 0, "profession": "thief", "gang": None}
    o.update(kw)
    return o


def _obs(others, *, society=True, day=1):
    return Observation(
        day=day, tick=1, self_view={}, position=(0, 0), nearby_facilities=[],
        here=None, others=others, open_proposals=[], granary_food=0,
        recent_events=[], society={"active": society, "gangs": society})


def _give_money(sim: Simulation, giver, recipient, amount=4):
    giver.add("money", amount)
    sim._move_items(giver, recipient, {"money": amount}, kind="give", consent=True)


class TestBriberyIsRecognised(unittest.TestCase):
    def _sim(self):
        return make_simulation("guardian", config=SimulationConfig(seed=1))

    def test_wanted_offender_paying_a_guard_is_a_bribe(self):
        sim = self._sim()
        guard = next(a for a in sim.agents if a.profession == "guard")
        briber = next(a for a in sim.agents if a is not guard)
        briber.last_crime_day = sim.world.day        # freshly wanted
        _give_money(sim, briber, guard)
        self.assertEqual(sim.metrics.bribes, 1)
        self.assertTrue(any(e["kind"] == "bribe" for e in sim.world.events))

    def test_an_honest_gift_to_a_guard_is_not_a_bribe(self):
        sim = self._sim()
        guard = next(a for a in sim.agents if a.profession == "guard")
        giver = next(a for a in sim.agents if a is not guard)
        giver.last_crime_day = None                  # not wanted → just a gift
        _give_money(sim, giver, guard)
        self.assertEqual(sim.metrics.bribes, 0)

    def test_paying_a_non_guard_is_not_a_bribe(self):
        sim = self._sim()
        a, b = sim.agents[0], sim.agents[1]
        a.profession, b.profession = "farmer", "merchant"
        a.last_crime_day = sim.world.day
        _give_money(sim, a, b)
        self.assertEqual(sim.metrics.bribes, 0)


class TestGangSurfaced(unittest.TestCase):
    def _sim(self, society=True):
        return make_simulation("predator", n_agents=6, config=SimulationConfig(seed=1),
                               society=SocietyConfig(enabled=society))

    def test_gang_is_in_the_others_view_only_with_gangs_live(self):
        on = self._sim(society=True)
        for a in on.agents:
            a.gang_id = "g1"
        self.assertIn("gang", on._observe(on.agents[0]).others[0])
        off = self._sim(society=False)
        self.assertNotIn("gang", off._observe(off.agents[0]).others[0])


class TestVenalGuard(unittest.TestCase):
    def _guard(self, *, gang=None):
        sim = make_simulation("predator", n_agents=6, config=SimulationConfig(seed=1),
                              society=SocietyConfig(enabled=True))
        g = sim.agents[0]
        g.profession = "guard"
        g.gang_id = gang
        return HeuristicBrain("predator"), g

    def test_spares_a_trusted_ally(self):
        brain, g = self._guard()
        self.assertIsNone(brain._enforce_action(g, _obs([_wanted(id="ally", trust=0.5)])))

    def test_spares_a_crew_mate(self):
        brain, g = self._guard(gang="g1")
        act = brain._enforce_action(g, _obs([_wanted(id="crew", trust=-0.3, gang="g1")]))
        self.assertIsNone(act, "you don't arrest your own crew")

    def test_takes_a_bribe_from_the_wealthy(self):
        brain, g = self._guard()
        act = brain._enforce_action(g, _obs([_wanted(id="rich", money=BRIBE_PRICE + 5)]))
        self.assertIsNone(act, "look the other way for a fat purse")

    def test_still_collars_a_poor_outsider(self):
        brain, g = self._guard(gang="g1")
        others = [_wanted(id="rich", money=99), _wanted(id="crew", gang="g1"),
                  _wanted(id="poor", money=0, trust=-0.2, distance=3)]
        act = brain._enforce_action(g, _obs(others))
        self.assertIsNotNone(act)
        self.assertEqual(act.params["target"], "poor", "the unprotected, broke outsider")

    def test_off_the_society_layer_a_predator_guard_enforces_straight(self):
        brain, g = self._guard()
        act = brain._enforce_action(g, _obs([_wanted(id="rich", money=99)], society=False))
        self.assertIsNotNone(act, "no society layer → straight enforcement (baseline)")
        self.assertEqual(act.params["target"], "rich")


class TestHonestGuard(unittest.TestCase):
    def test_arrests_the_nearest_regardless_of_purse_or_ties(self):
        sim = make_simulation("guardian", n_agents=6, config=SimulationConfig(seed=1),
                              society=SocietyConfig(enabled=True))
        g = sim.agents[0]; g.profession = "guard"
        brain = HeuristicBrain("guardian")
        act = brain._enforce_action(g, _obs([_wanted(id="rich", money=99, trust=0.9)]))
        self.assertIsNotNone(act, "an honest guard is not for sale")
        self.assertEqual(act.params["target"], "rich")


class TestOffenderBribes(unittest.TestCase):
    def _offender(self, *, money=BRIBE_PRICE + 2, crime_day=1):
        sim = make_simulation("predator", n_agents=6, config=SimulationConfig(seed=1),
                              society=SocietyConfig(enabled=True))
        a = sim.agents[0]
        a.money = money
        a.last_crime_day = crime_day
        return HeuristicBrain("predator"), a

    def _guard_near(self, dist=2):
        return [{"id": "g", "profession": "guard", "distance": dist,
                 "last_crime_day": None, "trust": 0.0, "money": 0, "gang": None}]

    def test_wanted_solvent_schemer_bribes_a_nearby_guard(self):
        brain, a = self._offender()
        act = brain._bribe_action(a, _obs(self._guard_near(), day=2))
        self.assertIsNotNone(act)
        self.assertEqual(act.type, ActionType.TRANSFER)
        self.assertEqual(act.params["target"], "g")
        self.assertEqual(act.params["amount"], BRIBE_PRICE)

    def test_no_bribe_when_not_wanted(self):
        brain, a = self._offender(crime_day=None)
        self.assertIsNone(brain._bribe_action(a, _obs(self._guard_near(), day=2)))

    def test_no_bribe_when_broke(self):
        brain, a = self._offender(money=1)
        self.assertIsNone(brain._bribe_action(a, _obs(self._guard_near(), day=2)))

    def test_no_bribe_off_the_society_layer(self):
        brain, a = self._offender()
        self.assertIsNone(brain._bribe_action(a, _obs(self._guard_near(), society=False, day=2)))

    def test_an_honest_persona_does_not_bribe(self):
        sim = make_simulation("idealist", n_agents=6, config=SimulationConfig(seed=1),
                              society=SocietyConfig(enabled=True))
        a = sim.agents[0]; a.money = 20; a.last_crime_day = 1
        brain = HeuristicBrain("idealist")          # deception 0.05 < 0.3
        self.assertIsNone(brain._bribe_action(a, _obs(self._guard_near(), day=2)))


class TestEmbezzlement(unittest.TestCase):
    """The third form of corruption (#38): a collector pockets the take. Under
    --economy the tax is a conserved take into the treasury; a venal mayor (the
    collector) skims part of it into its own purse. Honest state / no mayor /
    offline → straight collection, baseline byte-identical."""

    def _sim_with_tax(self, economy=True):
        from emergence.governance import Mayor
        sim = make_simulation("guardian", n_agents=6,
                              config=SimulationConfig(seed=1), economy=economy)
        sim.policy.enact(1, "tax the wealthy", day=1)        # a TAX law is in force
        self.assertTrue(sim.policy.has_tax())
        rich, mayor = sim.agents[0], sim.agents[1]
        for a in sim.agents:
            a.money = 1
        rich.money = 100                                     # the top taxpayer
        mayor.money = 1
        sim.mayor = Mayor(agent_id=mayor.id, elected_day=0, term_ends_day=99)
        return sim, rich, mayor

    def test_a_venal_mayor_skims_the_take(self):
        sim, rich, mayor = self._sim_with_tax()
        mayor.persona = "predator"                            # cold/scheming → corrupt
        before_treasury = sim.treasury
        sim._apply_daily_policy()
        self.assertGreater(mayor.money, 1, "the mayor pocketed part of the take")
        self.assertGreater(sim.metrics.embezzled, 0)
        self.assertTrue(any(e.get("kind") == "embezzle" for e in sim.world.events))
        # conservation: what left the rich = treasury gain + the skim
        paid = 100 - rich.money
        self.assertEqual(paid, (sim.treasury - before_treasury) + (mayor.money - 1))

    def test_an_honest_mayor_collects_straight(self):
        sim, rich, mayor = self._sim_with_tax()
        mayor.persona = "guardian"                            # cooperative → honest
        before_treasury = sim.treasury
        sim._apply_daily_policy()
        self.assertEqual(mayor.money, 1, "an honest mayor takes nothing for itself")
        self.assertEqual(sim.metrics.embezzled, 0)
        self.assertEqual(sim.treasury - before_treasury, 100 - rich.money)

    def test_no_mayor_no_embezzlement(self):
        sim, rich, mayor = self._sim_with_tax()
        sim.mayor = None
        sim._apply_daily_policy()
        self.assertEqual(sim.metrics.embezzled, 0)

    def test_offline_baseline_has_no_embezzlement(self):
        sim, rich, mayor = self._sim_with_tax(economy=False)
        mayor.persona = "predator"
        sim._apply_daily_policy()
        self.assertEqual(sim.metrics.embezzled, 0, "no conserved tax offline → nothing to skim")


if __name__ == "__main__":
    unittest.main()
