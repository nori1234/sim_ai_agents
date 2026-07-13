"""Land & property ownership (所有権).

A facility can be *owned*: building one under the economy layer makes the builder
its owner; seeded town facilities (and everything offline) are unowned commons.
Ownership is a transferable claim that **passes at death** through the estate
(#92) — to the eldest heir, or reverting to commons with no heir — and can now
be *acted on* (#102): an owner can charge rent (the service-offer substrate),
exclude non-owners (using an owned facility without consent reads as trespass),
and sell the claim outright (the OFFER/ACCEPT book). Burglary (breaking into a
home) is deliberately deferred to #113. All of it is gated behind owner being
set, which itself only happens under --economy, so the offline baseline stays
byte-identical.
"""

import unittest

from emergence.actions import Action, ActionType
from emergence.scenario import make_simulation
from emergence.simulation import SimulationConfig
from emergence.world import FacilityType


def _sim(economy=True):
    return make_simulation("guardian", n_agents=4,
                           config=SimulationConfig(seed=1), economy=economy)


def _build(sim, agent, name="Cabin", ftype="house"):
    agent.inventory["materials"] = 2
    sim._do_build(agent, Action(ActionType.BUILD,
                  {"facility_type": ftype, "name": name}))
    return next(f for f in sim.world.facilities if f.name == name)


class TestOwnershipClaim(unittest.TestCase):
    def test_you_own_what_you_build_under_economy(self):
        sim = _sim()
        a = sim.agents[0]
        f = _build(sim, a)
        self.assertEqual(f.owner, a.id, "the builder owns the new structure")

    def test_building_offline_is_unowned_commons(self):
        sim = _sim(economy=False)
        a = sim.agents[0]
        f = _build(sim, a)
        self.assertIsNone(f.owner, "no economy → commons, no owner")

    def test_seeded_town_facilities_are_unowned(self):
        sim = _sim()
        self.assertTrue(all(f.owner is None for f in sim.world.facilities),
                        "the founding town is commons until someone builds")

    def test_owner_is_surfaced_only_when_owned(self):
        sim = _sim()
        a = sim.agents[0]
        _build(sim, a, name="Manor")
        obs = sim._observe(a)
        manor = next(v for v in obs.nearby_facilities if v["name"] == "Manor")
        self.assertEqual(manor["owner"], a.id)
        commons = next(v for v in obs.nearby_facilities if "owner" not in v)
        self.assertNotIn("owner", commons, "commons carry no owner key")


class TestLandInheritance(unittest.TestCase):
    def test_land_passes_to_the_eldest_heir(self):
        sim = _sim()
        dec, young, old = sim.agents[0], sim.agents[1], sim.agents[2]
        f = _build(sim, dec, name="Homestead")
        for h in (young, old):
            h.parent_ids = (dec.id,)
        young.age_days, old.age_days = 10, 40
        sim._settle_estate(dec)
        self.assertEqual(f.owner, old.id, "title passes to the eldest heir")

    def test_land_escheats_to_commons_with_no_heir(self):
        sim = _sim()
        dec = sim.agents[0]
        f = _build(sim, dec, name="Lonely Hut")
        sim._settle_estate(dec)                # no children
        self.assertIsNone(f.owner, "with no heir the land reverts to commons")


class TestRentAffordance(unittest.TestCase):
    def test_only_the_owner_can_offer_it_for_rent(self):
        sim = _sim()
        owner, stranger = sim.agents[0], sim.agents[1]
        f = _build(sim, owner, name="Cabin")
        stranger.x, stranger.y = f.x, f.y
        sim._do_offer(stranger, Action(ActionType.OFFER,
                      {"service": "rent", "want_item": "money", "want_qty": 3}))
        self.assertEqual(len(sim.offers), 0, "a non-owner cannot rent out someone else's place")

    def test_owner_can_offer_and_a_taker_pays_for_paid_use(self):
        sim = _sim()
        owner, taker = sim.agents[0], sim.agents[1]
        f = _build(sim, owner, name="Cabin")
        owner.x, owner.y = f.x, f.y
        sim._do_offer(owner, Action(ActionType.OFFER,
                      {"service": "rent", "want_item": "money", "want_qty": 5}))
        self.assertEqual(len(sim.offers), 1)
        taker.x, taker.y = f.x, f.y
        taker.money = 20
        before_owner_money, before_taker_energy = owner.money, taker.energy
        taker.energy = 50  # below MAX_ENERGY so the rent benefit is visible
        sim._do_accept(taker, Action(ActionType.ACCEPT, {"offer_id": sim.offers[0].id}))
        self.assertEqual(owner.money, before_owner_money + 5, "the fee is conserved, paid to the owner")
        self.assertEqual(taker.money, 15)
        self.assertGreater(taker.energy, 50, "paid use of the property restores energy")
        self.assertEqual(len(sim.offers), 0)


class TestExclusionTrespass(unittest.TestCase):
    def test_working_someone_elses_property_registers_as_trespass(self):
        sim = _sim()
        owner, trespasser = sim.agents[0], sim.agents[1]
        f = _build(sim, owner, name="Workshop", ftype="workshop")
        trespasser.x, trespasser.y = f.x, f.y
        before = sim.metrics.crimes_total
        sim._do_work(trespasser, Action(ActionType.WORK))
        self.assertEqual(sim.metrics.crimes_total, before + 1)
        self.assertEqual(sim.metrics.crimes_by_type.get("trespass"), 1)

    def test_the_owner_working_their_own_property_is_not_trespass(self):
        sim = _sim()
        owner = sim.agents[0]
        f = _build(sim, owner, name="Workshop", ftype="workshop")
        owner.x, owner.y = f.x, f.y
        before = sim.metrics.crimes_total
        sim._do_work(owner, Action(ActionType.WORK))
        self.assertEqual(sim.metrics.crimes_total, before)

    def test_working_unowned_commons_is_not_trespass(self):
        sim = _sim()
        agent = sim.agents[0]
        commons = next(f for f in sim.world.facilities if f.is_workplace())
        agent.x, agent.y = commons.x, commons.y
        before = sim.metrics.crimes_total
        sim._do_work(agent, Action(ActionType.WORK))
        self.assertEqual(sim.metrics.crimes_total, before)

    def test_work_still_succeeds_despite_being_trespass(self):
        # Enforcement is a choice, not a hard block -- the act itself still
        # happens (and pays), it's just also legally exposed.
        sim = _sim()
        owner, trespasser = sim.agents[0], sim.agents[1]
        f = _build(sim, owner, name="Workshop", ftype="workshop")
        trespasser.x, trespasser.y = f.x, f.y
        before_money = trespasser.money
        sim._do_work(trespasser, Action(ActionType.WORK))
        self.assertGreater(trespasser.money, before_money)

    def test_offline_baseline_has_no_trespass(self):
        sim = _sim(economy=False)
        agent = sim.agents[0]
        workshop = next(f for f in sim.world.facilities if f.is_workplace())
        agent.x, agent.y = workshop.x, workshop.y
        before = sim.metrics.crimes_total
        sim._do_work(agent, Action(ActionType.WORK))
        self.assertEqual(sim.metrics.crimes_total, before, "owner is always None offline")


class TestPropertySale(unittest.TestCase):
    def test_only_the_owner_can_list_it_for_sale(self):
        sim = _sim()
        owner, stranger = sim.agents[0], sim.agents[1]
        f = _build(sim, owner, name="Cabin")
        sim._do_offer(stranger, Action(ActionType.OFFER,
                      {"give_facility": f.name, "want_item": "money", "want_qty": 10}))
        self.assertEqual(len(sim.offers), 0)

    def test_selling_transfers_ownership_for_the_agreed_price(self):
        sim = _sim()
        seller, buyer = sim.agents[0], sim.agents[1]
        f = _build(sim, seller, name="Cabin")
        buyer.money = 20
        sim._do_offer(seller, Action(ActionType.OFFER,
                      {"give_facility": f.name, "want_item": "money", "want_qty": 12}))
        self.assertEqual(len(sim.offers), 1)
        before_seller_money = seller.money
        sim._do_accept(buyer, Action(ActionType.ACCEPT, {"offer_id": sim.offers[0].id}))
        self.assertEqual(f.owner, buyer.id, "ownership transfers to the buyer")
        self.assertEqual(buyer.money, 8)
        self.assertEqual(seller.money, before_seller_money + 12, "conserved, paid to the seller")
        self.assertEqual(len(sim.offers), 0)

    def test_offline_baseline_cannot_sell_commons(self):
        sim = _sim(economy=False)
        agent = sim.agents[0]
        f = sim.world.facilities[0]  # unowned commons offline
        sim._do_offer(agent, Action(ActionType.OFFER,
                      {"give_facility": f.name, "want_item": "money", "want_qty": 5}))
        self.assertEqual(len(sim.offers), 0, "economy is off, so _do_offer is a no-op entirely")


if __name__ == "__main__":
    unittest.main()
