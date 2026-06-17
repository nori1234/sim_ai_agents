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


class TestUsePrimitive(unittest.TestCase):
    def test_use_food_restores_energy_and_consumes_it(self):
        a = _agent(id="a")
        a.inventory["food"] = 3
        a.energy = 50.0
        sim = _sim([a])
        sim._do_use(a, Action(ActionType.USE, {"item": "food", "qty": 2}))
        self.assertEqual(a.food(), 1)         # 3 - 2 consumed
        self.assertGreater(a.energy, 50.0)    # energy restored

    def test_eat_macro_matches_use(self):
        # The eat macro is exactly "use 2 food on self".
        a = _agent(id="a"); a.inventory["food"] = 5; a.energy = 40.0
        b = _agent(id="b"); b.inventory["food"] = 5; b.energy = 40.0
        sim = _sim([a, b])
        sim._do_eat(a, Action(ActionType.EAT))
        sim._do_use(b, Action(ActionType.USE, {"item": "food", "qty": 2}))
        self.assertEqual(a.food(), b.food())
        self.assertEqual(a.energy, b.energy)


class TestStrikePrimitive(unittest.TestCase):
    def test_strike_a_person_is_violence(self):
        a, b = _agent(id="a"), _agent(id="b", money=10)
        b.energy = 100.0
        sim = _sim([a, b])
        sim._do_strike(a, Action(ActionType.STRIKE, {"target": "b"}))
        self.assertLess(b.energy, 100.0)               # harmed
        self.assertEqual(a.money, 23)                  # 20 + 3 robbed
        self.assertEqual(sim.metrics.crimes_by_type.get("violence", 0), 1)

    def test_strike_a_structure_is_arson(self):
        from emergence.world import Facility, FacilityType, World
        world = World(6, 6)
        world.add_facility(Facility("Barn", FacilityType.GRANARY, 0, 0))
        world.granary_food = 10
        a = _agent(id="a", x=0, y=0)
        sim = Simulation(world=world, agents=[a], brains={})
        sim._do_strike(a, Action(ActionType.STRIKE, {"facility_name": "Barn"}))
        self.assertEqual(sim.metrics.crimes_by_type.get("arson", 0), 1)
        self.assertEqual(world.granary_food, 5)        # commons spilled

    def test_attack_macro_matches_strike(self):
        a = _agent(id="a"); b = _agent(id="b", money=10); b.energy = 100.0
        c = _agent(id="c"); d = _agent(id="d", money=10); d.energy = 100.0
        s1 = _sim([a, b]); s1._do_attack(a, Action(ActionType.ATTACK, {"target": "b"}))
        s2 = _sim([c, d]); s2._do_strike(c, Action(ActionType.STRIKE, {"target": "d"}))
        self.assertEqual(b.energy, d.energy)
        self.assertEqual(a.money, c.money)


class TestSayPrimitive(unittest.TestCase):
    def test_say_logs_a_public_statement(self):
        a = _agent(id="a")
        sim = _sim([a])
        before = len(sim.world.events)
        sim._do_say(a, Action(ActionType.SAY, {"text": "hello town"}))
        self.assertEqual(len(sim.world.events), before + 1)
        self.assertEqual(sim.world.events[-1]["kind"], "speech")

    def test_speak_macro_is_a_say(self):
        a = _agent(id="a")
        sim = _sim([a])
        sim._do_speak(a, Action(ActionType.SPEAK, {"text": "x"}))
        self.assertEqual(sim.world.events[-1]["kind"], "speech")


class TestBondPrimitive(unittest.TestCase):
    def test_bond_with_an_agent_is_a_mutual_pact(self):
        a, b = _agent(id="a"), _agent(id="b")
        sim = _sim([a, b])
        sim._do_bond(a, Action(ActionType.BOND, {"with": "b"}))
        self.assertGreater(a.trust_of("b"), 0)   # allegiance is mutual
        self.assertGreater(b.trust_of("a"), 0)

    def test_bond_to_a_proposal_casts_a_vote(self):
        a, b = _agent(id="a"), _agent(id="b")
        sim = _sim([a, b])
        p = sim.legislature.propose(b.id, "Plant more farms", sim.world.day)
        sim._do_bond(a, Action(ActionType.BOND,
                     {"proposal_id": p.id, "support": True}))
        self.assertEqual(a.votes_cast, 1)
        self.assertIn(a.id, p.votes)


class TestSayIntentPraise(unittest.TestCase):
    def test_praise_macro_lowers_to_say_and_grants_esteem(self):
        from emergence.esteem import StatusConfig
        a, b = _agent(id="a"), _agent(id="b", esteem=80.0)
        sim = Simulation(world=World(6, 6), agents=[a, b], brains={},
                         status=StatusConfig(enabled=True))
        sim._do_praise(a, Action(ActionType.PRAISE, {"target": "b"}))
        self.assertEqual(sim.metrics.total_praise, 1)
        self.assertEqual(b.praise_received, 1)
        self.assertLess(b.esteem, 80.0)            # esteem relieved
        self.assertGreater(b.reputation, 0)        # honour granted
        self.assertGreater(b.trust_of("a"), 0)     # bond warmed

    def test_praise_is_noop_when_esteem_layer_off(self):
        a, b = _agent(id="a"), _agent(id="b", esteem=80.0)
        sim = _sim([a, b])                          # status off
        sim._do_praise(a, Action(ActionType.PRAISE, {"target": "b"}))
        self.assertEqual(sim.metrics.total_praise, 0)
        self.assertEqual(b.esteem, 80.0)


class TestReligionFoldings(unittest.TestCase):
    def test_preach_macro_lowers_to_say_and_founds_a_faith(self):
        from emergence.society import SocietyConfig
        from emergence.world import Facility, FacilityType, World
        world = World(8, 8)
        world.add_facility(Facility("Plaza", FacilityType.PLAZA, 0, 0))
        a = _agent(id="a", x=0, y=0, reputation=10.0)
        sim = Simulation(world=world, agents=[a], brains={},
                         society=SocietyConfig(enabled=True))
        sim._do_preach(a, Action(ActionType.PREACH, {}))
        self.assertIsNotNone(a.faith)                  # founded a faith
        self.assertEqual(sim.metrics.religions_founded, 1)

    def test_worship_macro_lowers_to_bond_and_relieves_fear(self):
        from emergence.society import SocietyConfig
        from emergence.world import Facility, FacilityType, World
        world = World(8, 8)
        temple = Facility("Temple", FacilityType.PLAZA, 0, 0)
        temple.add_role("temple")
        world.add_facility(temple)
        a = _agent(id="a", x=0, y=0)
        a.faith = "r1"
        a.fear = 50.0
        sim = Simulation(world=world, agents=[a], brains={},
                         society=SocietyConfig(enabled=True))
        sim._do_worship(a, Action(ActionType.WORSHIP, {}))
        self.assertEqual(sim.metrics.acts_of_worship, 1)
        self.assertLess(a.fear, 50.0)                  # prayer eases fear


class TestUnderworldFoldings(unittest.TestCase):
    def test_take_drug_lowers_to_use_and_doses(self):
        from emergence.society import SocietyConfig
        a = _agent(id="a")
        a.inventory["materials"] = 2
        sim = Simulation(world=World(6, 6), agents=[a], brains={},
                         society=SocietyConfig(enabled=True))
        sim._do_take_drug(a, Action(ActionType.TAKE_DRUG, {}))
        self.assertEqual(sim.metrics.doses_taken, 1)
        self.assertGreater(a.addiction, 0)         # the dose hooks

    def test_craft_weapon_lowers_to_make(self):
        from emergence.society import SocietyConfig
        from emergence.world import Facility, FacilityType, World
        world = World(6, 6)
        world.add_facility(Facility("Forge", FacilityType.WORKSHOP, 0, 0))
        a = _agent(id="a", x=0, y=0)
        a.inventory["materials"] = 2
        sim = Simulation(world=world, agents=[a], brains={},
                         society=SocietyConfig(enabled=True))
        sim._do_craft_weapon(a, Action(ActionType.CRAFT_WEAPON, {}))
        self.assertEqual(a.weapons, 1)
        self.assertEqual(sim.metrics.weapons_crafted, 1)

    def test_join_gang_lowers_to_bond(self):
        from emergence.society import SocietyConfig
        a = _agent(id="a")
        sim = Simulation(world=World(6, 6), agents=[a], brains={},
                         society=SocietyConfig(enabled=True))
        sim._do_join_gang(a, Action(ActionType.JOIN_GANG, {}))
        self.assertIsNotNone(a.gang_id)            # founded/joined a crew
        self.assertEqual(sim.metrics.gangs_formed, 1)


class TestBuildFolding(unittest.TestCase):
    def test_build_macro_lowers_to_make_a_structure(self):
        from emergence.esteem import StatusConfig
        from emergence.world import FacilityType, World
        a = _agent(id="a", x=2, y=2)
        a.inventory["materials"] = 3
        sim = Simulation(world=World(6, 6), agents=[a], brains={},
                         status=StatusConfig(enabled=True))
        sim._do_build(a, Action(ActionType.BUILD,
                      {"name": "Spire", "facility_type": "monument"}))
        built = [f for f in sim.world.facilities if f.name == "Spire"]
        self.assertEqual(len(built), 1)               # structure raised
        self.assertEqual(sim.metrics.monuments_built, 1)
        self.assertGreater(a.reputation, 0)           # a monument earns honour
        self.assertEqual(a.materials(), 1)            # 3 - 2 spent


class TestGatherFolding(unittest.TestCase):
    def test_gather_macro_takes_yield_from_a_world_node(self):
        from emergence.world import Facility, FacilityType, World
        world = World(6, 6)
        world.add_facility(Facility("Field", FacilityType.FARM, 0, 0))
        a = _agent(id="a", x=0, y=0)
        before = a.food()
        sim = Simulation(world=world, agents=[a], brains={})
        sim._do_gather(a, Action(ActionType.GATHER, {}))
        self.assertGreater(a.food(), before)       # yield flowed in from the node

    def test_gather_is_noop_off_a_node(self):
        a = _agent(id="a", x=3, y=3)               # standing on nothing
        before = (a.food(), a.materials())
        sim = _sim([a])
        sim._do_gather(a, Action(ActionType.GATHER, {}))
        self.assertEqual((a.food(), a.materials()), before)


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
