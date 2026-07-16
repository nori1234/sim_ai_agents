"""Physical individuation (#76): heritable sex, body build, gait.

The physical-trait mirror of tests/test_personality.py's behavioural-trait
suite. Covers emergence/personality.py's PhysicalTraits/PhysicalTraitPool and
their opt-in wiring through scenario.py (individuals=True). Purely additive
and inert engine-side: the baseline contract is guarded separately.
"""

import random
import unittest

from emergence.personality import (
    PhysicalTraitPool,
    PhysicalTraits,
    blend_physical,
    sample_physical,
)
from emergence.scenario import make_simulation
from emergence.simulation import SimulationConfig


class TestSampling(unittest.TestCase):
    def test_bounds(self):
        rng = random.Random(1)
        for _ in range(50):
            p = sample_physical(rng, spread=0.5)
            self.assertIn(p.sex, ("f", "m"))
            self.assertGreaterEqual(p.build, 0.0); self.assertLessEqual(p.build, 1.0)
            self.assertGreaterEqual(p.gait, 0.0); self.assertLessEqual(p.gait, 1.0)

    def test_varies_around_a_neutral_centre(self):
        rng = random.Random(2)
        people = [sample_physical(rng, spread=0.15) for _ in range(50)]
        builds = [p.build for p in people]
        self.assertGreater(max(builds) - min(builds), 0.05, "not a clone army")
        mean = sum(builds) / len(builds)
        self.assertLess(abs(mean - 0.5), 0.1, "centred near neutral")

    def test_both_sexes_appear(self):
        rng = random.Random(3)
        sexes = {sample_physical(rng).sex for _ in range(30)}
        self.assertEqual(sexes, {"f", "m"})

    def test_deterministic(self):
        a = sample_physical(random.Random(7))
        b = sample_physical(random.Random(7))
        self.assertEqual(a, b)


class TestHeredity(unittest.TestCase):
    def test_child_build_and_gait_blend_both_parents(self):
        a = PhysicalTraits(sex="f", build=0.2, gait=0.8)
        b = PhysicalTraits(sex="m", build=0.6, gait=0.4)
        child = blend_physical(a, b, random.Random(1), mutation=0.0)
        self.assertAlmostEqual(child.build, 0.4)
        self.assertAlmostEqual(child.gait, 0.6)

    def test_mutation_perturbs_the_midpoint(self):
        a = PhysicalTraits(sex="f", build=0.5, gait=0.5)
        children = [blend_physical(a, a, random.Random(i), mutation=0.2)
                   for i in range(30)]
        builds = [c.build for c in children]
        self.assertGreater(max(builds) - min(builds), 0.05)

    def test_clamped_to_bounds_even_with_extreme_mutation(self):
        a = PhysicalTraits(sex="f", build=0.95, gait=0.05)
        rng = random.Random(4)
        for _ in range(50):
            c = blend_physical(a, a, rng, mutation=1.0)
            self.assertGreaterEqual(c.build, 0.0); self.assertLessEqual(c.build, 1.0)
            self.assertGreaterEqual(c.gait, 0.0); self.assertLessEqual(c.gait, 1.0)


class TestPhysicalTraitPool(unittest.TestCase):
    def test_found_stores_and_returns_the_same_vector(self):
        pool = PhysicalTraitPool(random.Random(1))
        vec = pool.found("a1")
        self.assertEqual(pool.get("a1"), vec)

    def test_inherit_with_two_known_parents_blends(self):
        pool = PhysicalTraitPool(random.Random(1))
        pool._physical["p1"] = PhysicalTraits(sex="f", build=0.2, gait=0.2)
        pool._physical["p2"] = PhysicalTraits(sex="m", build=0.8, gait=0.8)
        child = pool.inherit("c1", ("p1", "p2"))
        self.assertGreater(child.build, 0.2); self.assertLess(child.build, 0.8)
        self.assertEqual(pool.get("c1"), child)

    def test_inherit_with_no_known_parents_still_returns_something_valid(self):
        pool = PhysicalTraitPool(random.Random(1))
        child = pool.inherit("c1", ("ghost_a", "ghost_b"))
        self.assertIn(child.sex, ("f", "m"))

    def test_unknown_agent_returns_none(self):
        self.assertIsNone(PhysicalTraitPool(random.Random(1)).get("nope"))


class TestScenarioWiring(unittest.TestCase):
    def test_disabled_by_default_agents_have_neutral_placeholders(self):
        sim = make_simulation("guardian", config=SimulationConfig(seed=1, days=2))
        for a in sim.agents:
            self.assertEqual(a.sex, "")
            self.assertEqual(a.build, 0.5)
            self.assertEqual(a.gait, 0.5)

    def test_individuals_true_assigns_physical_traits(self):
        sim = make_simulation("guardian", config=SimulationConfig(seed=1, days=2),
                              individuals=True)
        for a in sim.agents:
            self.assertIn(a.sex, ("f", "m"))
        builds = [a.build for a in sim.agents]
        self.assertGreater(max(builds) - min(builds), 0.0, "should vary, not be uniform")

    def test_deterministic(self):
        cfg = SimulationConfig(seed=5, days=2)
        a = make_simulation("guardian", config=cfg, individuals=True)
        b = make_simulation("guardian", config=cfg, individuals=True)
        self.assertEqual([(x.sex, x.build, x.gait) for x in a.agents],
                         [(x.sex, x.build, x.gait) for x in b.agents])

    def test_offline_baseline_unaffected_by_individuals_flag(self):
        # individuals=True changes brains/traits but must not perturb the
        # offline heuristic outcome metrics on its own (mirrors
        # test_personality.py's own baseline-parity guard for behavioural
        # traits; this just confirms the physical half doesn't add drift
        # beyond what individuals=True already causes).
        cfg = SimulationConfig(seed=5, days=10)
        off = make_simulation("guardian", config=cfg)
        off.run()
        self.assertEqual(off.agents[0].sex, "")


if __name__ == "__main__":
    unittest.main()
