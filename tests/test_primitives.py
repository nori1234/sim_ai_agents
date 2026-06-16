"""Physical primitives + the interpretation layer (verb-primitive slice 1).

The thesis: the same physics (move items between holders) is read as a
different institution depending on context. A consent-less take from a person
is theft; a consensual give to a person is a gift. Macros (steal, transfer)
lower to these primitives, and their effects come out of the interpretation
layer rather than being welded into the verb.
"""

import unittest

from emergence.actions import Action, ActionType
from emergence.agent import Agent
from emergence.simulation import Simulation
from emergence.world import World


def _agent(**kw):
    base = dict(id="x", name="X", profession="t", persona="guardian", x=0, y=0)
    base.update(kw)
    return Agent(**base)


def _sim(agents):
    return Simulation(world=World(4, 4), agents=agents, brains={})


class TestInterpretation(unittest.TestCase):
    def test_consentless_take_is_theft(self):
        a, b = _agent(id="a"), _agent(id="b", money=10)
        b.inventory["food"] = 5
        sim = _sim([a, b])
        sim._do_take(a, Action(ActionType.TAKE,
                     {"from": "b", "items": {"money": 4, "food": 2},
                      "consent": False}))
        # Items moved (conserved)...
        self.assertEqual(a.money, 24)         # 20 + 4
        self.assertEqual(a.food(), 5)         # 3 + 2
        self.assertEqual(b.money, 6)          # 10 - 4
        # ...and the act was interpreted as a crime.
        self.assertEqual(sim.metrics.crimes_total, 1)
        self.assertEqual(b.times_victimized, 1)

    def test_consensual_give_is_a_gift(self):
        a, b = _agent(id="a"), _agent(id="b")
        a.inventory["food"] = 5
        sim = _sim([a, b])
        sim._do_give(a, Action(ActionType.GIVE,
                     {"to": "b", "items": {"food": 3}, "consent": True}))
        self.assertEqual(b.food(), 6)         # 3 + 3
        self.assertEqual(a.food(), 2)         # 5 - 3
        self.assertEqual(sim.metrics.transfers, 1)
        self.assertGreater(b.trust_of("a"), 0)   # a gift builds trust
        self.assertEqual(sim.metrics.crimes_total, 0)  # not a crime

    def test_conservation_nothing_created(self):
        a, b = _agent(id="a", money=7), _agent(id="b", money=0)
        sim = _sim([a, b])
        before = a.money + b.money
        sim._do_give(a, Action(ActionType.GIVE,
                     {"to": "b", "items": {"money": 5}, "consent": True}))
        self.assertEqual(a.money + b.money, before)   # conserved


class TestMacrosLowerToPrimitives(unittest.TestCase):
    def test_steal_macro_still_loots_and_is_a_crime(self):
        a, b = _agent(id="a"), _agent(id="b", money=10)
        b.inventory["food"] = 5
        sim = _sim([a, b])
        sim._do_steal(a, Action(ActionType.STEAL, {"target": "b"}))
        self.assertEqual(a.money, 25)         # 20 + 5 looted
        self.assertEqual(a.food(), 5)         # 3 + 2 looted
        self.assertEqual(sim.metrics.crimes_total, 1)

    def test_transfer_macro_still_gifts_and_builds_trust(self):
        a, b = _agent(id="a"), _agent(id="b")
        a.inventory["food"] = 5
        sim = _sim([a, b])
        sim._do_transfer(a, Action(ActionType.TRANSFER,
                         {"target": "b", "resource": "food", "amount": 3}))
        self.assertEqual(b.food(), 6)
        self.assertEqual(sim.metrics.transfers, 1)
        self.assertGreater(b.trust_of("a"), 0)


if __name__ == "__main__":
    unittest.main()
