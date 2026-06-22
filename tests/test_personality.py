"""Vertical/genetic inheritance: individuals, heredity, the developmental window.

Covers emergence/personality.py and its opt-in wiring through scenario.py.
The baseline contract (every agent the exact preset Persona) is guarded
separately by tests/test_baseline_contract.py; here we check that the *opt-in*
path behaves and that it does not perturb the default path.
"""

import random
import unittest

from emergence.personality import (
    TRAIT_FIELDS,
    TraitPool,
    blend,
    matured,
    sample_individual,
)
from emergence.personas import GUARDIAN, PREDATOR, get_persona
from emergence.scenario import make_simulation
from emergence.simulation import SimulationConfig


def _knobs(p):
    return {f: getattr(p, f) for f in TRAIT_FIELDS}


class TestSampling(unittest.TestCase):
    def test_keeps_identity_and_bounds(self):
        rng = random.Random(1)
        p = sample_individual(GUARDIAN, rng, spread=0.5)
        # Identity (the coarse culture label) is preserved; only knobs vary.
        self.assertEqual(p.key, GUARDIAN.key)
        self.assertEqual(p.label, GUARDIAN.label)
        for f in TRAIT_FIELDS:
            self.assertGreaterEqual(getattr(p, f), 0.0)
            self.assertLessEqual(getattr(p, f), 1.0)

    def test_varies_around_centre(self):
        rng = random.Random(2)
        people = [sample_individual(GUARDIAN, rng, spread=0.15) for _ in range(50)]
        # The sample is not a clone army...
        coop = [p.cooperation for p in people]
        self.assertGreater(max(coop) - min(coop), 0.05)
        # ...but it is centred near the culture (mean within a knob of GUARDIAN).
        mean = sum(coop) / len(coop)
        self.assertLess(abs(mean - GUARDIAN.cooperation), 0.1)

    def test_deterministic(self):
        a = sample_individual(GUARDIAN, random.Random(7))
        b = sample_individual(GUARDIAN, random.Random(7))
        self.assertEqual(_knobs(a), _knobs(b))


class TestHeredity(unittest.TestCase):
    def test_child_blends_both_parents(self):
        rng = random.Random(3)
        child = blend(GUARDIAN, PREDATOR, rng, mutation=0.0)
        # With no mutation a child sits at the parents' midpoint on every knob.
        for f in TRAIT_FIELDS:
            mid = (getattr(GUARDIAN, f) + getattr(PREDATOR, f)) / 2.0
            self.assertAlmostEqual(getattr(child, f), mid, places=6)

    def test_mutation_perturbs_but_stays_bounded(self):
        rng = random.Random(4)
        child = blend(GUARDIAN, PREDATOR, rng, mutation=0.1)
        differs = any(
            abs(getattr(child, f) - (getattr(GUARDIAN, f) + getattr(PREDATOR, f)) / 2)
            > 1e-9
            for f in TRAIT_FIELDS
        )
        self.assertTrue(differs)
        for f in TRAIT_FIELDS:
            self.assertGreaterEqual(getattr(child, f), 0.0)
            self.assertLessEqual(getattr(child, f), 1.0)

    def test_pool_founders_then_inherit(self):
        pool = TraitPool(random.Random(5))
        a = pool.found("a1", GUARDIAN)
        b = pool.found("a2", PREDATOR)
        child = pool.inherit("a3", ("a1", "a2"), GUARDIAN)
        # Stored and recoverable.
        self.assertEqual(_knobs(pool.get("a3")), _knobs(child))
        # The child lies between its parents on a knob where they differ.
        lo, hi = sorted((a.aggression, b.aggression))
        self.assertGreaterEqual(child.aggression, max(0.0, lo - 0.4))
        self.assertLessEqual(child.aggression, min(1.0, hi + 0.4))

    def test_pool_single_known_parent_samples_around_it(self):
        pool = TraitPool(random.Random(6))
        pool.found("a1", GUARDIAN)
        child = pool.inherit("a3", ("a1", "ghost"), PREDATOR)
        self.assertIsNotNone(child)
        self.assertEqual(child.key, GUARDIAN.key)


class TestDevelopmentalWindow(unittest.TestCase):
    def test_newborn_is_plastic_then_fixed(self):
        adult = get_persona("predator")  # high aggression, far from neutral 0.5
        born = matured(adult, age_days=0, window_days=2)
        grown = matured(adult, age_days=2, window_days=2)
        older = matured(adult, age_days=99, window_days=2)
        # At birth the effective vector sits at the neutral baseline...
        self.assertAlmostEqual(born.aggression, 0.5, places=6)
        # ...and by the end of the window it equals the inherited adult vector,
        # then stays fixed.
        self.assertEqual(_knobs(grown), _knobs(adult))
        self.assertEqual(_knobs(older), _knobs(adult))

    def test_matures_monotonically_toward_adult(self):
        adult = get_persona("predator")
        gap0 = abs(matured(adult, 0, 4).aggression - adult.aggression)
        gap1 = abs(matured(adult, 1, 4).aggression - adult.aggression)
        gap2 = abs(matured(adult, 2, 4).aggression - adult.aggression)
        self.assertGreater(gap0, gap1)
        self.assertGreater(gap1, gap2)

    def test_zero_window_is_immediate(self):
        adult = get_persona("guardian")
        self.assertEqual(_knobs(matured(adult, 0, window_days=0)), _knobs(adult))


class TestScenarioWiring(unittest.TestCase):
    def test_default_path_unchanged_by_the_flag(self):
        # Opt-in OFF must reproduce the existing run byte-for-byte. We compare a
        # representative metric bundle to a fresh default run.
        base = make_simulation("guardian", config=SimulationConfig(seed=42))
        base.run()
        again = make_simulation("guardian", config=SimulationConfig(seed=42),
                                individuals=False)
        again.run()
        for f in ("survivors", "population", "crimes_total", "frauds",
                  "collaborations", "births"):
            self.assertEqual(getattr(base.metrics, f), getattr(again.metrics, f), f)

    def test_individuals_run_is_deterministic_and_alive(self):
        a = make_simulation("guardian", config=SimulationConfig(seed=42),
                            individuals=True)
        a.run()
        b = make_simulation("guardian", config=SimulationConfig(seed=42),
                            individuals=True)
        b.run()
        self.assertEqual(a.metrics.survivors, b.metrics.survivors)
        self.assertEqual(a.metrics.crimes_total, b.metrics.crimes_total)
        # A guardian town should still hold together when individuated.
        self.assertGreater(a.metrics.survivors, 0)


if __name__ == "__main__":
    unittest.main()
