import unittest

from emergence.actions import Action, ActionType
from emergence.agent import Agent
from emergence.governance import GovernanceConfig, GovernanceForm
from emergence.scenario import make_simulation
from emergence.simulation import Simulation, SimulationConfig
from emergence.society import SocietyConfig, discontent
from emergence.world import Facility, FacilityType, World


def _agent(**kw):
    base = dict(id="x", name="X", profession="tester", persona="predator", x=0, y=0)
    base.update(kw)
    return Agent(**base)


def _sim(agents, world=None, gov=None, **soc_kw):
    s = Simulation(world=world or World(8, 8), agents=agents, brains={},
                   society=SocietyConfig(enabled=True, **soc_kw))
    if gov is not None:
        from emergence.governance import Legislature, PolicyEngine
        s.legislature = Legislature(gov)
        s.policy = PolicyEngine(gov)
    return s


class TestWeapons(unittest.TestCase):
    def test_craft_weapon_at_workshop(self):
        world = World(6, 6)
        world.add_facility(Facility("WS", FacilityType.WORKSHOP, 0, 0))
        a = _agent()
        a.inventory["materials"] = 3
        sim = _sim([a], world=world, weapon_material_cost=1)
        sim._do_craft_weapon(a, Action(ActionType.CRAFT_WEAPON))
        self.assertEqual(a.weapons, 1)
        self.assertIn("weapons_factory", world.facility_at((0, 0)).roles)

    def test_craft_needs_a_workshop(self):
        a = _agent(x=2, y=2)
        a.inventory["materials"] = 3
        sim = _sim([a])
        sim._do_craft_weapon(a, Action(ActionType.CRAFT_WEAPON))
        self.assertEqual(a.weapons, 0)

    def test_armed_attacker_deals_more_damage(self):
        off = _agent(id="o", x=0, y=0)
        vic = _agent(id="v", x=1, y=0, energy=100)
        unarmed_off = _agent(id="o2", x=0, y=0)
        vic2 = _agent(id="v2", x=1, y=0, energy=100)
        off.weapons = 1
        sim = _sim([off, vic, unarmed_off, vic2], world=World(8, 8),
                   weapon_attack_bonus=14)
        sim._do_attack(off, Action(ActionType.ATTACK, {"target": "v"}))
        sim._do_attack(unarmed_off, Action(ActionType.ATTACK, {"target": "v2"}))
        self.assertLess(vic.energy, vic2.energy)  # armed hurt more


class TestDrugs(unittest.TestCase):
    def test_dose_spikes_energy_and_addiction(self):
        a = _agent(energy=40, addiction=0)
        sim = _sim([a])
        sim._dose(a)
        self.assertGreater(a.energy, 40)
        self.assertGreater(a.addiction, 0)
        self.assertGreater(a.pleasure, 0)

    def test_dealing_hooks_the_buyer(self):
        dealer = _agent(id="d", x=0, y=0)
        dealer.inventory["materials"] = 2
        buyer = _agent(id="b", x=1, y=0, money=10)
        sim = _sim([dealer, buyer], world=World(6, 6), drug_price=4)
        sim._do_deal_drug(dealer, Action(ActionType.DEAL_DRUG, {"target": "b"}))
        self.assertGreater(buyer.addiction, 0)
        self.assertEqual(dealer.money, 24)   # 20 + 4
        self.assertEqual(sim.metrics.drug_deals, 1)

    def test_withdrawal_drains_energy(self):
        clean = _agent(id="c", energy=80, addiction=0)
        hooked = _agent(id="h", energy=80, addiction=90)
        sim = _sim([clean, hooked], withdrawal_threshold=45)
        sim._tick_upkeep(clean)
        sim._tick_upkeep(hooked)
        self.assertLess(hooked.energy, clean.energy)


class TestGangs(unittest.TestCase):
    def test_join_forms_a_gang(self):
        a = _agent()
        sim = _sim([a], world=World(8, 8))
        sim._do_join_gang(a, Action(ActionType.JOIN_GANG))
        self.assertIsNotNone(a.gang_id)
        self.assertEqual(len(sim.gangs), 1)
        self.assertEqual(sim.metrics.gangs_formed, 1)

    def test_second_member_joins_existing_gang(self):
        a = _agent(id="a", x=0, y=0)
        b = _agent(id="b", x=1, y=0)
        sim = _sim([a, b], world=World(8, 8), gang_join_radius=6)
        sim._do_join_gang(a, Action(ActionType.JOIN_GANG))
        sim._do_join_gang(b, Action(ActionType.JOIN_GANG))
        self.assertEqual(len(sim.gangs), 1)
        self.assertEqual(a.gang_id, b.gang_id)

    def test_gang_members_trust_each_other(self):
        a = _agent(id="a", x=0, y=0)
        b = _agent(id="b", x=1, y=0)
        sim = _sim([a, b], world=World(8, 8), gang_loyalty=0.4)
        sim._do_join_gang(a, Action(ActionType.JOIN_GANG))
        sim._do_join_gang(b, Action(ActionType.JOIN_GANG))
        self.assertGreater(a.trust_of("b"), 0)


class TestRebellion(unittest.TestCase):
    def test_armed_malcontents_depose_the_mayor(self):
        from emergence.governance import Mayor
        rebels = [_agent(id=f"r{i}", x=i, y=0, weapons=1, fear=100, reputation=0)
                  for i in range(4)]
        sim = _sim(rebels, world=World(10, 10),
                   rebellion_discontent=40, rebellion_min_rebels=3)
        sim.mayor = Mayor(agent_id="r0", elected_day=1, term_ends_day=6)
        sim._do_rebel(rebels[1], Action(ActionType.REBEL))
        self.assertEqual(sim.metrics.rebellions, 1)
        self.assertIsNone(sim.mayor)

    def test_unarmed_cannot_rebel(self):
        a = _agent(weapons=0, fear=100)
        sim = _sim([a])
        sim._do_rebel(a, Action(ActionType.REBEL))
        self.assertEqual(sim.metrics.rebellions, 0)

    def test_discontent_rises_under_oligarchy(self):
        a = _agent(fear=20, reputation=0)
        free = discontent(a, oppressed=False)
        ruled = discontent(a, oppressed=True)
        self.assertGreater(ruled, free)


class TestReligion(unittest.TestCase):
    def test_found_and_consecrate_temple(self):
        world = World(8, 8)
        world.add_facility(Facility("Plaza", FacilityType.PLAZA, 0, 0))
        a = _agent(persona="guardian", reputation=10)
        sim = _sim([a], world=world, faith_min_reputation=5)
        sim._do_preach(a, Action(ActionType.PREACH))
        self.assertIsNotNone(a.faith)
        self.assertEqual(sim.metrics.religions_founded, 1)
        self.assertIn("temple", world.facility_at((0, 0)).roles)

    def test_low_standing_cannot_found_faith(self):
        a = _agent(reputation=1)
        sim = _sim([a], faith_min_reputation=5)
        sim._do_preach(a, Action(ActionType.PREACH))
        self.assertIsNone(a.faith)

    def test_preaching_converts_the_trusting(self):
        prophet = _agent(id="p", x=0, y=0, faith="r1")
        follower = _agent(id="f", x=1, y=0)
        follower.adjust_trust("p", 0.5)
        from emergence.society import Religion
        sim = _sim([prophet, follower], world=World(8, 8), conversion_radius=5)
        sim.religions.append(Religion(id="r1", name="Faith", prophet="p",
                                      members=["p"]))
        sim._do_preach(prophet, Action(ActionType.PREACH))
        self.assertEqual(follower.faith, "r1")
        self.assertEqual(sim.metrics.conversions, 1)

    def test_worship_eases_fear(self):
        world = World(8, 8)
        temple = Facility("T", FacilityType.PLAZA, 0, 0)
        temple.add_role("temple")
        world.add_facility(temple)
        a = _agent(x=0, y=0, faith="r1", fear=60)
        sim = _sim([a], world=world, worship_fear_relief=28)
        sim._do_worship(a, Action(ActionType.WORSHIP))
        self.assertLess(a.fear, 60)
        self.assertEqual(sim.metrics.acts_of_worship, 1)


class TestSocietyEndToEnd(unittest.TestCase):
    def _run(self, persona, **soc):
        sim = make_simulation(persona, config=SimulationConfig(seed=42),
                              society=SocietyConfig(enabled=True), **soc)
        sim.run()
        return sim

    def test_baseline_unaffected_when_disabled(self):
        sim = make_simulation("guardian", config=SimulationConfig(seed=42))
        sim.run()
        self.assertEqual(sim.metrics.gangs_formed, 0)
        self.assertEqual(sim.metrics.weapons_crafted, 0)
        self.assertEqual(sim.metrics.religions_founded, 0)
        self.assertEqual(sim.metrics.crimes_total, 0)

    def test_violent_society_breeds_gangs(self):
        sim = self._run("grok")
        self.assertGreater(sim.metrics.gangs_formed, 0)

    def test_deterministic(self):
        a = self._run("gemini").metrics.as_dict()
        b = self._run("gemini").metrics.as_dict()
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
