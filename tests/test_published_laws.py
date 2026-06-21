"""Every enacted law is published to agents — so an LLM can act on even novel
legislation the engine has no built-in mechanism for. A law's force is emergent
(an agent choosing to honour/enforce it), not engine code. The heuristic brain
ignores this view, so offline outcomes are unchanged (the contract test guards it).
"""

import unittest

from emergence.scenario import make_simulation
from emergence.simulation import SimulationConfig


class TestPublishedLaws(unittest.TestCase):
    def _sim(self):
        return make_simulation("guardian", config=SimulationConfig(seed=1))

    def test_novel_law_is_published_even_with_no_engine_mechanism(self):
        sim = self._sim()
        sim.policy.enact(1, "a ban on theft", 1)            # a known keyword effect
        sim.policy.enact(2, "plant a tree each spring", 2)  # NOVEL — no engine mechanism
        obs = sim._observe(sim.agents[0])
        texts = [l["text"] for l in obs.laws]
        self.assertIn("plant a tree each spring", texts, "a novel law must reach agents")
        self.assertIn("a ban on theft", texts)
        novel = next(l for l in obs.laws if l["text"] == "plant a tree each spring")
        self.assertEqual(novel["effects"], [], "no built-in effect is fine — force is emergent")

    def test_repassed_bills_are_deduplicated(self):
        sim = self._sim()
        for i in range(5):
            sim.policy.enact(i, "a ban on theft", 1)
        obs = sim._observe(sim.agents[0])
        self.assertEqual(sum(1 for l in obs.laws if l["text"] == "a ban on theft"), 1)

    def test_no_laws_means_empty_view(self):
        sim = self._sim()
        self.assertEqual(sim._observe(sim.agents[0]).laws, [])


if __name__ == "__main__":
    unittest.main()
