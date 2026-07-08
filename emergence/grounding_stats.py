"""Pure-stdlib paired statistics for the grounding battery.

``fraction_grounded`` (worlds where ``excess`` clears a hard threshold) is a
brittle statistic: one borderline world flipping across the line swings it by
``1/n_worlds``, and it is blind to *how much* a world misses by. These
functions give the battery a real paired hypothesis test instead — is the
excess across worlds distinguishable from zero? — without adding a numpy/scipy
dependency (the engine is stdlib-only by design; torch/llm_model_agi are the
neural *brain's* optional extras, not the engine's — see CLAUDE.md).

Every p-value here is **one-sided** for H1: the population value is *positive*
(a grounded brain diverges *more* than the floor, or more than a floor-slope
regression predicts) — the only direction "grounded" ever claims. None of this
replaces ``fraction_grounded``/``min_excess`` (still reported, still what
``replay_inexplicable`` gates on) — it's an additional, harder-to-Goodhart
read of the same per-world numbers.
"""

from __future__ import annotations

import math
import random
from statistics import NormalDist


def sign_test_p(values: list) -> float:
    """One-sided exact binomial sign test for H1: median(values) > 0.

    Exact zeros are dropped before counting (the usual sign-test convention);
    that only removes support for the alternative, so it never makes the test
    anti-conservative. Returns 1.0 (cannot reject) if nothing is left."""
    nonzero = [v for v in values if v != 0.0]
    n = len(nonzero)
    if n == 0:
        return 1.0
    k = sum(1 for v in nonzero if v > 0.0)
    return sum(math.comb(n, i) for i in range(k, n + 1)) / (2 ** n)


def _signed_ranks(values: list) -> tuple:
    """Rank |values| (average rank on ties), return (ranks, signs) aligned to
    the non-zero subset of ``values``."""
    nonzero = [v for v in values if v != 0.0]
    n = len(nonzero)
    order = sorted(range(n), key=lambda i: abs(nonzero[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(nonzero[order[j + 1]]) == abs(nonzero[order[i]]):
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks, nonzero


def wilcoxon_signed_rank_p(values: list) -> float:
    """One-sided Wilcoxon signed-rank p-value for H1: median(values) > 0.

    Exact (via a subset-sum DP over rank sums — O(n * total_rank), trivial at
    battery scale) when there are no tied ranks and n is small enough to
    enumerate; a normal approximation with continuity correction otherwise.
    Exact zeros are dropped first, per the standard Wilcoxon convention.
    Returns 1.0 (cannot reject) if nothing is left after dropping zeros."""
    ranks, nonzero = _signed_ranks(values)
    n = len(nonzero)
    if n == 0:
        return 1.0

    w_plus = sum(r for r, v in zip(ranks, nonzero) if v > 0.0)
    total = n * (n + 1) / 2.0
    ranks_are_whole = all(r == int(r) for r in ranks)

    if n <= 25 and ranks_are_whole:
        max_sum = int(total)
        counts = [1] + [0] * max_sum
        filled = 0
        for r in ranks:
            r = int(r)
            filled += r
            new_counts = [0] * (filled + 1)
            for s in range(filled - r + 1):
                if counts[s] == 0:
                    continue
                new_counts[s] += counts[s]        # rank r assigned '-'
                new_counts[s + r] += counts[s]     # rank r assigned '+'
            counts = new_counts
        total_paths = 2 ** n
        target = round(w_plus)
        ge = sum(c for s, c in enumerate(counts) if s >= target)
        return ge / total_paths

    mean = total / 2.0
    var = n * (n + 1) * (2 * n + 1) / 24.0
    if var <= 0:
        return 1.0
    z = (w_plus - mean - 0.5) / math.sqrt(var)
    return 1.0 - NormalDist().cdf(z)


def paired_bootstrap_ci(values: list, *, n_boot: int = 10000, alpha: float = 0.05,
                        seed: int = 0) -> tuple:
    """A percentile bootstrap CI for the mean of ``values`` (per-world excess,
    or floor-regression residuals). Resamples worlds with replacement; the
    only randomness is which worlds get resampled, seeded for reproducibility
    so a report is exactly re-derivable from the same per-world numbers."""
    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    if n == 1:
        return (values[0], values[0])
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[max(0, int((alpha / 2) * n_boot))]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return (lo, hi)


def linear_regression(xs: list, ys: list) -> tuple:
    """OLS ``(slope, intercept)`` for ``y = slope*x + intercept``. Degenerate
    input (all-equal x, or < 2 points) returns slope 0.0 and the mean of ys —
    i.e. "no floor-dependence detectable", not a crash."""
    n = len(xs)
    if n < 2:
        return (0.0, (sum(ys) / n) if n else 0.0)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx == 0.0:
        return (0.0, mean_y)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    return (slope, mean_y - slope * mean_x)


def regression_slope_bootstrap_ci(xs: list, ys: list, *, n_boot: int = 2000,
                                  alpha: float = 0.05, seed: int = 0) -> tuple:
    """A percentile bootstrap CI for the OLS slope of ``y`` on ``x``, resampling
    ``(x, y)`` pairs jointly with replacement. This is how a caller checks
    whether a fitted slope is actually identified rather than an artifact of a
    handful of points — a wide or degenerate CI (e.g. spanning both signs, or
    collapsing to a point because the resampled ``x`` values keep landing on
    the same value) is the signal that a regression-based verdict from this
    fit shouldn't be trusted, however small the point-estimate residual test's
    p-value looks."""
    n = len(xs)
    if n < 2:
        return (0.0, 0.0)
    rng = random.Random(seed)
    slopes = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        slope, _ = linear_regression([xs[i] for i in idx], [ys[i] for i in idx])
        slopes.append(slope)
    slopes.sort()
    lo = slopes[max(0, int((alpha / 2) * n_boot))]
    hi = slopes[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return (lo, hi)
