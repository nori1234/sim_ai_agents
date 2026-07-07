"""Pure-stdlib paired statistics (emergence.grounding_stats).

These back the battery's paired-test read-out (a harder-to-Goodhart
alternative to the hard-threshold `fraction_grounded`) and the floor-
regression diagnostic. Verified against hand-computable small cases since
there is no scipy in this codebase to compare against.
"""

import unittest

from emergence.grounding_stats import (
    linear_regression,
    paired_bootstrap_ci,
    regression_slope_bootstrap_ci,
    sign_test_p,
    wilcoxon_signed_rank_p,
)


class TestSignTest(unittest.TestCase):
    def test_all_positive_is_significant(self):
        # 5/5 positive: P(X>=5) under Binomial(5, 0.5) = 1/32.
        self.assertAlmostEqual(sign_test_p([0.1, 0.2, 0.3, 0.05, 0.4]), 1 / 32)

    def test_all_negative_is_not_significant(self):
        self.assertAlmostEqual(sign_test_p([-0.1, -0.2, -0.3]), 1.0)

    def test_mixed_evenly_is_not_significant(self):
        # 1/2 positive: P(X>=1) under Binomial(2, 0.5) = 3/4.
        self.assertAlmostEqual(sign_test_p([0.1, -0.1]), 3 / 4)

    def test_empty_returns_one(self):
        self.assertEqual(sign_test_p([]), 1.0)

    def test_zeros_are_dropped_not_counted_either_way(self):
        # Only the two nonzero values matter; the zero cannot help H1.
        self.assertAlmostEqual(sign_test_p([0.1, 0.2, 0.0]), 1 / 4)


class TestWilcoxonSignedRank(unittest.TestCase):
    def test_all_positive_is_significant(self):
        # n=4, all positive => W+ = total rank sum = 10, the single most
        # extreme outcome out of 2**4=16 sign assignments => p = 1/16.
        p = wilcoxon_signed_rank_p([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(p, 1 / 16)

    def test_all_negative_is_not_significant(self):
        p = wilcoxon_signed_rank_p([-1.0, -2.0, -3.0, -4.0])
        self.assertAlmostEqual(p, 1.0)

    def test_symmetric_mix_gives_a_mid_p_value(self):
        # +1,+2,-3,+4 has W+ = 1+2+4 = 7 out of total 10; some but not
        # overwhelming evidence, so p should land comfortably mid-range.
        p = wilcoxon_signed_rank_p([1.0, 2.0, -3.0, 4.0])
        self.assertGreater(p, 0.1)
        self.assertLess(p, 0.9)

    def test_empty_returns_one(self):
        self.assertEqual(wilcoxon_signed_rank_p([]), 1.0)

    def test_larger_sample_falls_back_to_normal_approx_without_crashing(self):
        values = [0.05 * (i - 10) for i in range(1, 31)]  # 30 values, mostly positive
        p = wilcoxon_signed_rank_p(values)
        self.assertGreaterEqual(p, 0.0)
        self.assertLessEqual(p, 1.0)


class TestBootstrapCI(unittest.TestCase):
    def test_ci_brackets_the_mean_for_tight_data(self):
        values = [0.20, 0.21, 0.19, 0.20, 0.22]
        lo, hi = paired_bootstrap_ci(values, n_boot=2000, seed=1)
        mean = sum(values) / len(values)
        self.assertLessEqual(lo, mean)
        self.assertGreaterEqual(hi, mean)

    def test_is_reproducible_for_a_fixed_seed(self):
        values = [0.1, -0.2, 0.3, 0.0, 0.4]
        a = paired_bootstrap_ci(values, n_boot=500, seed=7)
        b = paired_bootstrap_ci(values, n_boot=500, seed=7)
        self.assertEqual(a, b)

    def test_single_value_returns_a_degenerate_interval(self):
        self.assertEqual(paired_bootstrap_ci([0.5]), (0.5, 0.5))

    def test_empty_returns_zero_interval(self):
        self.assertEqual(paired_bootstrap_ci([]), (0.0, 0.0))


class TestLinearRegression(unittest.TestCase):
    def test_recovers_an_exact_line(self):
        xs = [0.0, 1.0, 2.0, 3.0]
        ys = [1.0, 3.0, 5.0, 7.0]          # y = 2x + 1
        slope, intercept = linear_regression(xs, ys)
        self.assertAlmostEqual(slope, 2.0)
        self.assertAlmostEqual(intercept, 1.0)

    def test_degenerate_x_returns_zero_slope_and_mean_y(self):
        slope, intercept = linear_regression([5.0, 5.0, 5.0], [1.0, 2.0, 3.0])
        self.assertAlmostEqual(slope, 0.0)
        self.assertAlmostEqual(intercept, 2.0)

    def test_too_few_points_returns_zero_slope(self):
        self.assertEqual(linear_regression([], []), (0.0, 0.0))
        slope, intercept = linear_regression([1.0], [4.0])
        self.assertAlmostEqual(slope, 0.0)
        self.assertAlmostEqual(intercept, 4.0)


class TestRegressionSlopeBootstrapCI(unittest.TestCase):
    def test_a_clean_line_gives_a_tight_ci_around_the_true_slope(self):
        xs = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        ys = [1.0, 3.0, 5.0, 7.0, 9.0, 11.0]      # exactly y = 2x + 1, no noise
        lo, hi = regression_slope_bootstrap_ci(xs, ys, n_boot=2000, seed=1)
        self.assertLessEqual(lo, 2.0)
        self.assertGreaterEqual(hi, 2.0)
        self.assertLess(hi - lo, 0.5, "a noiseless line should bootstrap to a tight CI")

    def test_a_handful_of_scattered_points_gives_a_wide_ci(self):
        xs = [0.0, 1.0, 2.0]
        ys = [5.0, -3.0, 8.0]                     # no real linear relationship
        lo, hi = regression_slope_bootstrap_ci(xs, ys, n_boot=2000, seed=1)
        self.assertGreater(hi - lo, 1.0, "noisy, tiny-n data should bootstrap wide")

    def test_is_reproducible_for_a_fixed_seed(self):
        xs, ys = [0.0, 1.0, 2.0, 3.0], [1.0, 2.5, 2.0, 4.0]
        a = regression_slope_bootstrap_ci(xs, ys, n_boot=500, seed=3)
        b = regression_slope_bootstrap_ci(xs, ys, n_boot=500, seed=3)
        self.assertEqual(a, b)

    def test_too_few_points_returns_zero_interval(self):
        self.assertEqual(regression_slope_bootstrap_ci([], []), (0.0, 0.0))
        self.assertEqual(regression_slope_bootstrap_ci([1.0], [2.0]), (0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
