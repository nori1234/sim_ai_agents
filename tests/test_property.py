"""Land & property ownership (所有権) — minimal slice.

A facility can be *owned*: building one under the economy layer makes the builder
its owner; seeded town facilities (and everything offline) are unowned commons.
Ownership is a transferable claim that **passes at death** through the estate
(#92) — to the eldest heir, or reverting to commons with no heir. This slice only
*records and inherits* ownership; rent / exclusion / trespass are deliberately
deferred to follow-up issues, so the owner field is inert (nothing reads it to
gate behaviour) and the four-society baseline stays byte-identical.
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


if __name__ == "__main__":
    unittest.main()
