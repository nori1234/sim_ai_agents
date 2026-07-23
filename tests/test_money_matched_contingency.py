"""G2 (money-matched contingency) — the reflex-proof grounding metric.

Pins the two properties that make G2 worth adopting over G1 (norm_contingency):
its ==0-for-any-memoryless-policy invariant (fairness: the blind floor cannot
fake it) and its >0-for-history-dependent-suppression sensitivity (it detects the
real thing). See docs/runs/metric-trajectory-confound-1.
"""

import unittest

from emergence.grounding import (
    money_matched_contingency,
    measure_money_matched_contingency,
    DEFAULT_MONEY_BINS,
)


class TestG2PureFunction(unittest.TestCase):
    def test_memoryless_identical_bins_give_zero(self):
        # A pure function of money fires at the SAME rate per bin in both regimes;
        # only the bin populations differ (trajectory divergence). G2 must be 0.
        # control: heavy in the rich bins; cf: crushed into the poor bin. Same
        # fire-rate (100%) everywhere.
        control = [(14.0, True)] * 200 + [(22.0, True)] * 150
        cf = [(14.0, True)] * 180 + [(22.0, True)] * 5
        res = money_matched_contingency(control, cf)
        self.assertAlmostEqual(res.g2, 0.0, places=9)
        # ...even though the raw (G1) rates would look wildly asymmetric by count.
        self.assertGreater(res.n_decisions_control, res.n_decisions_counterfactual)

    def test_partial_firing_still_zero_if_rate_matched(self):
        # 50% fire in a bin in BOTH regimes -> matched -> 0, regardless of counts.
        control = [(14.0, i % 2 == 0) for i in range(100)]
        cf = [(14.0, i % 2 == 0) for i in range(40)]
        res = money_matched_contingency(control, cf)
        self.assertAlmostEqual(res.g2, 0.0, places=9)

    def test_matched_wealth_suppression_is_positive(self):
        # SAME money bin, but cf fires less (regime detected -> deposit suppressed
        # at wealth it would have banked in control). This is genuine grounding.
        control = [(14.0, True)] * 100          # 100% in control
        cf = [(14.0, i < 30) for i in range(100)]  # 30% at the same wealth
        res = money_matched_contingency(control, cf)
        self.assertAlmostEqual(res.g2, 0.70, places=6)

    def test_single_sided_bin_excluded(self):
        # A bin seen only in control carries no MATCHED signal -> excluded from G2
        # (that asymmetry is G1's job). Only the shared bin counts.
        control = [(14.0, True)] * 50 + [(50.0, True)] * 50  # rich bin ctl-only
        cf = [(14.0, True)] * 50                              # matched -> gap 0
        res = money_matched_contingency(control, cf)
        self.assertAlmostEqual(res.g2, 0.0, places=9)
        # the rich bin recorded samples but contributed nothing (cf_n == 0 there)
        rich = res.per_bin[(40, 10**9)]
        self.assertEqual(rich[2], 0)   # cf_n == 0

    def test_sign_is_control_minus_cf(self):
        # Depositing MORE in cf at matched wealth (anti-grounded) is negative.
        control = [(14.0, i < 30) for i in range(100)]
        cf = [(14.0, True)] * 100
        res = money_matched_contingency(control, cf)
        self.assertLess(res.g2, 0.0)


class TestG2Floor(unittest.TestCase):
    def test_blind_floor_is_not_positive(self):
        # The fairness contract: the reflex floor (deposit iff money>=12) cannot
        # produce matched-wealth regime SUPPRESSION -- its G2 sits at/below ~0.
        # It is not exactly 0 because HeuristicBrain.decide() gates banking behind
        # survival/energy priorities that vary with the (regime-divergent)
        # trajectory, leaking a small non-money residual into the money-matched
        # gap. The contract is: small in magnitude, an order below its G1 (+0.5),
        # and NOT positive (a reflex never grounds).
        from emergence.brains.heuristic import HeuristicBrain

        def factory(agent, persona, rng):
            return HeuristicBrain(persona, rng)

        res = measure_money_matched_contingency(
            "claude", seeds=tuple(range(42, 45)), days=12, n_agents=4,
            brain_factory=factory)
        self.assertGreater(res.n_decisions_control, 0)
        self.assertGreater(res.n_decisions_counterfactual, 0)
        self.assertLess(res.g2, 0.05)          # never a grounding signal
        self.assertGreater(res.g2, -0.20)      # and small (residual, not a swing)

    def test_raising_threshold_does_not_raise_g2(self):
        # The discriminator that makes G2 worth adopting over G1: a memoryless
        # policy can INFLATE G1 by depositing only when very rich (a higher
        # threshold -> larger raw regime rate gap), with zero regime knowledge.
        # G2 must be immune to that game -- a pure-money threshold has ~matched
        # within-bin rates whatever the threshold, so G2 stays ~0 as G1 climbs.
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
        from threshold_landscape import _threshold_brain_class

        def measure(T):
            Brain = _threshold_brain_class(float(T), None)
            return measure_money_matched_contingency(
                "claude", seeds=tuple(range(42, 48)), days=20, n_agents=6,
                brain_factory=lambda a, p, r, _B=Brain: _B(p, r))

        # Both thresholds are memoryless money rules; whatever G1 they reach
        # (measured elsewhere: +0.52 at T=12, +0.64 at T=20), neither manufactures
        # a positive matched-wealth signal -- G2 stays sub-grounding for both.
        self.assertLess(measure(12).g2, 0.05)
        self.assertLess(measure(20).g2, 0.05)


class TestC1StableIncome(unittest.TestCase):
    def test_stable_income_kills_the_floors_handle(self):
        # C1: with spendable money fixed each day, the blind wealth-threshold
        # reflex loses its regime handle -> its money-matched G2 (already ~0) and,
        # more tellingly, its G1 collapse toward 0. Here we check G2 stays ~0 and
        # that the decision population is no longer regime-skewed (the baseline
        # task crushes cf decisions low; C1 balances them).
        from emergence.brains.heuristic import HeuristicBrain

        def factory(agent, persona, rng):
            return HeuristicBrain(persona, rng)

        base = measure_money_matched_contingency(
            "claude", seeds=tuple(range(42, 45)), days=16, n_agents=5,
            demurrage_per_day=0.25, stable_income=0, brain_factory=factory)
        c1 = measure_money_matched_contingency(
            "claude", seeds=tuple(range(42, 45)), days=16, n_agents=5,
            demurrage_per_day=0.25, stable_income=20, brain_factory=factory)
        # both memoryless -> G2 not positive either way
        self.assertLess(c1.g2, 0.05)
        # the tell: baseline crushes cf decisions far below control; C1 balances
        # them (regime no longer skews the decision population).
        base_skew = base.n_decisions_control / max(1, base.n_decisions_counterfactual)
        c1_skew = c1.n_decisions_control / max(1, c1.n_decisions_counterfactual)
        self.assertGreater(base_skew, 1.5)      # baseline: control >> cf decisions
        self.assertLess(c1_skew, 1.3)           # C1: roughly balanced


if __name__ == "__main__":
    unittest.main()
