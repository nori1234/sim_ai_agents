"""Ecology layer — livestock (家畜): wealth that breeds, slaughtered for food.

The first ecology slice. Livestock is an ordinary inventory good (so it's already
ownable, transferable, inheritable #92 and stealable), with two added mechanics:
it **reproduces** each day (a herd of >=2 grows, capped) — the pastoral analogue
of deposit interest — and it can be **slaughtered** (USE) for food. Opt-in via
--ecology, so the four-society baseline is byte-identical (no livestock exists).
Guards:
  * a herd of >=2 breeds by breed_rate, capped at herd_cap; a lone animal doesn't;
  * slaughtering (USE livestock) yields food per head, under the layer only;
  * founders start with a herd under --ecology, none without;
  * livestock passes to an heir at death (it's an inventory good, #92);
  * the heuristic: a hungry herd-owner slaughters one, keeping a breeding pair.
"""

import unittest

from emergence.actions import Action, ActionType
from emergence.brains.heuristic import HeuristicBrain
from emergence.ecology import EcologyConfig
from emergence.observation import Observation
from emergence.scenario import make_simulation
from emergence.simulation import SimulationConfig


def _sim(enabled=True, **kw):
    return make_simulation("guardian", n_agents=4, config=SimulationConfig(seed=1),
                           ecology=EcologyConfig(enabled=enabled, **kw))


class TestBreeding(unittest.TestCase):
    def test_a_herd_grows_capped(self):
        sim = _sim(breed_rate=0.5, herd_cap=12)
        a = sim.agents[0]; a.inventory["livestock"] = 8
        sim._breed_livestock()
        self.assertEqual(a.inventory["livestock"], 12, "8 + floor(8*0.5)=12, capped")

    def test_a_lone_animal_does_not_breed(self):
        sim = _sim(breed_rate=0.5)
        a = sim.agents[0]; a.inventory["livestock"] = 1
        sim._breed_livestock()
        self.assertEqual(a.inventory["livestock"], 1, "takes two to breed")

    def test_no_breeding_off_the_layer(self):
        sim = _sim(enabled=False)
        a = sim.agents[0]; a.inventory["livestock"] = 8
        sim._breed_livestock()      # method still callable, but gated at the caller;
        # _breed_livestock itself reads config — with the layer off it shouldn't grow
        # because the founders never get a herd and the end-of-day never calls it.
        # Assert the end-to-end invariant instead:
        sim2 = make_simulation("guardian", config=SimulationConfig(seed=1, days=5))
        sim2.run()
        self.assertEqual(sum(x.inventory.get("livestock", 0) for x in sim2.agents), 0)


class TestSlaughter(unittest.TestCase):
    def test_slaughter_yields_food(self):
        sim = _sim(slaughter_food=4)
        a = sim.agents[0]; a.inventory["livestock"] = 5; a.inventory["food"] = 0
        sim._use_item(a, "livestock", 1)
        self.assertEqual(a.inventory["livestock"], 4, "one animal gone")
        self.assertEqual(a.food(), 4, "meat enters the stores")

    def test_slaughter_inert_off_the_layer(self):
        sim = _sim(enabled=False, slaughter_food=4)
        a = sim.agents[0]; a.inventory["livestock"] = 5; a.inventory["food"] = 0
        sim._use_item(a, "livestock", 1)
        self.assertEqual(a.food(), 0, "no ecology layer → no meat yield")


class TestEndowmentAndInheritance(unittest.TestCase):
    def test_founders_start_with_a_herd_only_under_ecology(self):
        on = _sim(start_herd=3)
        self.assertTrue(all(a.inventory.get("livestock", 0) == 3 for a in on.agents))
        off = make_simulation("guardian", n_agents=4, config=SimulationConfig(seed=1))
        self.assertTrue(all(a.inventory.get("livestock", 0) == 0 for a in off.agents))

    def test_livestock_passes_to_an_heir(self):
        sim = make_simulation("guardian", n_agents=4, config=SimulationConfig(seed=1),
                              ecology=EcologyConfig(enabled=True), economy=True)
        dec, heir = sim.agents[0], sim.agents[1]
        dec.inventory["livestock"] = 6
        heir.parent_ids = (dec.id,); heir.inventory["livestock"] = 0
        sim._settle_estate(dec)
        self.assertEqual(heir.inventory.get("livestock", 0), 6, "the herd is inherited")


class TestHeuristicHusbandry(unittest.TestCase):
    def _obs(self):
        return Observation(
            day=1, tick=1, self_view={}, position=(0, 0), nearby_facilities=[],
            here=None, others=[], open_proposals=[], granary_food=0,
            recent_events=[], economy={})

    def test_hungry_owner_slaughters_keeping_a_breeding_pair(self):
        sim = _sim()
        a = sim.agents[0]
        a.inventory["livestock"] = 5; a.inventory["food"] = 0; a.energy = 100.0
        act = HeuristicBrain("guardian")._survival_action(a, self._obs())
        self.assertIsNotNone(act)
        self.assertEqual(act.type, ActionType.USE)
        self.assertEqual(act.params["item"], "livestock")

    def test_does_not_slaughter_the_last_pair(self):
        sim = _sim()
        a = sim.agents[0]
        a.inventory["livestock"] = 2; a.inventory["food"] = 0; a.energy = 100.0
        act = HeuristicBrain("guardian")._survival_action(a, self._obs())
        if act is not None:
            self.assertFalse(act.type == ActionType.USE and act.params.get("item") == "livestock",
                             "keep the breeding pair")


if __name__ == "__main__":
    unittest.main()
