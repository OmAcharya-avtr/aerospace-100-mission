"""Known-answer tests. Every expected value is derived by hand in a comment.

The headline one is the exact finite-sample coverage of split conformal, which
is the only statement in this package that is exactly true rather than
measured.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from conformalband.bounds import (
    clopper_pearson,
    effective_sample_size,
    split_conformal_coverage_bound,
)
from conformalband.conformal import (
    MondrianConformal,
    SplitConformal,
    WeightedConformal,
    conformal_rank,
    weighted_quantile,
)
from conformalband.physics import leg_energy, level_flight_power
from conformalband.shift import (
    CALIBRATION_HEADWIND_MEAN,
    CALIBRATION_HEADWIND_SD,
    CALIBRATION_MASS_MEAN,
    CALIBRATION_MASS_SD,
    CovariateShift,
)

# ---------------------------------------------------------------------------
# 1. The finite-sample coverage bound, by hand
# ---------------------------------------------------------------------------
# k = ceil((n + 1)(1 - alpha)) and the exact coverage is k / (n + 1).
#
#   n = 19, alpha = 0.1 : (19+1)(0.9) = 18 exactly  -> k = 18, 18/20 = 0.9 exactly
#   n = 18, alpha = 0.1 : (18+1)(0.9) = 17.1        -> k = 18, 18/19 = 0.947368421052631...
#   n =  9, alpha = 0.1 : (9+1)(0.9)  = 9 exactly   -> k =  9,  9/10 = 0.9 exactly
#   n = 100, alpha = 0.1: (100+1)(0.9) = 90.9       -> k = 91, 91/101 = 0.900990099009901...
#   n = 99, alpha = 0.05: (99+1)(0.95) = 95 exactly -> k = 95, 95/100 = 0.95 exactly
HAND_BOUNDS = [
    (19, 0.1, 18, 18 / 20),
    (18, 0.1, 18, 18 / 19),
    (9, 0.1, 9, 9 / 10),
    (100, 0.1, 91, 91 / 101),
    (99, 0.05, 95, 95 / 100),
]


@pytest.mark.parametrize(("n", "alpha", "rank", "exact"), HAND_BOUNDS)
def test_coverage_bound_matches_hand_arithmetic(n, alpha, rank, exact):
    bound = split_conformal_coverage_bound(n, alpha)
    assert bound.rank == rank
    assert bound.exact == pytest.approx(exact, rel=1e-15)


@pytest.mark.parametrize(("n", "alpha", "rank", "exact"), HAND_BOUNDS)
def test_coverage_bound_lies_in_the_published_window(n, alpha, rank, exact):
    bound = split_conformal_coverage_bound(n, alpha)
    assert bound.lower <= bound.exact <= bound.upper + 1e-15
    assert bound.upper == pytest.approx(1.0 - alpha + 1.0 / (n + 1), rel=1e-15)


def test_rank_guards_against_the_floating_point_ceiling():
    # (24 + 1)(1 - 0.44) = 25 * 0.56 = 14 exactly in exact arithmetic, but
    # 14.000000000000002 in binary floating point, so an unguarded ceil
    # returns 15. The guard in conformal_rank rounds to ten decimals first.
    assert (24 + 1) * (1.0 - 0.44) > 14.0
    assert math.ceil((24 + 1) * (1.0 - 0.44)) == 15
    assert conformal_rank(24, 0.44) == 14


def test_rank_is_exact_for_the_common_levels():
    # alpha = 0.1 and 0.05 are not affected: 20 * 0.9 and 100 * 0.95 are both
    # exact in binary floating point.
    assert (19 + 1) * (1.0 - 0.1) == 18.0
    assert conformal_rank(19, 0.1) == 18
    assert (99 + 1) * (1.0 - 0.05) == 95.0
    assert conformal_rank(99, 0.05) == 95


def test_weighted_quantile_guard_matches_the_rank_guard():
    # The same hazard inside the weighted quantile: equal weights at
    # n = 24, alpha = 0.44 must select the 14th order statistic, not the 15th.
    values = np.arange(1.0, 25.0)
    got = weighted_quantile(values, np.ones(24), 1.0 - 0.44, tail_weight=1.0)
    assert got == pytest.approx(14.0, rel=0.0)


def test_exact_coverage_can_sit_one_ulp_below_one_minus_alpha():
    # n = 1249, alpha = 0.18. In exact arithmetic (1249+1)(1-0.18) = 1250 * 0.82
    # = 1025, so k = 1025 and the exact coverage is 1025/1250 = 0.82 = 1 - alpha:
    # the bound is attained with equality. In binary floating point 1 - 0.18
    # evaluates to 0.8200000000000001, one ulp above 0.82, so `exact < lower`
    # by one ulp. Found by Hypothesis, kept as a known answer rather than
    # papered over by redefining `lower`.
    bound = split_conformal_coverage_bound(1249, 0.18)
    assert bound.rank == 1025
    assert bound.exact == 1025 / 1250
    assert bound.exact == 0.82
    assert bound.lower == 1.0 - 0.18
    assert bound.lower > bound.exact
    assert bound.lower - bound.exact < 1e-15
    assert bound.exact == pytest.approx(bound.lower, abs=1e-12)


def test_bound_conservatism_is_the_discretisation_gap():
    # n = 100, alpha = 0.1: 91/101 - 0.9 = 0.000990099009900...
    bound = split_conformal_coverage_bound(100, 0.1)
    assert bound.conservatism == pytest.approx(91 / 101 - 0.9, rel=1e-14)


# ---------------------------------------------------------------------------
# 2. Monte-Carlo confirmation of the exact coverage on exchangeable data
# ---------------------------------------------------------------------------
# With n calibration scores and one test score drawn i.i.d. from any
# continuous distribution, the n + 1 scores are exchangeable and
# P(V_{n+1} <= V_(k)) = k / (n + 1) exactly, independently of the
# distribution. Uniform scores are used because the result must not depend on
# the law; a lognormal set is checked alongside for the same reason.
#
# Standard error at 200 000 replicates is sqrt(0.9 * 0.1 / 200000) = 6.7e-4,
# so a tolerance of 4e-3 is six standard errors and distinguishes 18/19 =
# 0.9474 from 18/20 = 0.9000 by a factor of twelve.
MONTE_CARLO_REPLICATES = 200_000
MONTE_CARLO_TOLERANCE = 4e-3


def _empirical_split_coverage(n: int, alpha: float, replicates: int, seed: int, law: str) -> float:
    rng = np.random.default_rng(seed)
    if law == "uniform":
        scores = rng.random((replicates, n + 1))
    elif law == "lognormal":
        scores = rng.lognormal(0.0, 1.0, (replicates, n + 1))
    else:  # pragma: no cover - guarded by the parametrisation
        raise AssertionError(law)
    rank = conformal_rank(n, alpha)
    calibration = scores[:, :n]
    quantile = np.partition(calibration, rank - 1, axis=1)[:, rank - 1]
    return float(np.mean(scores[:, n] <= quantile))


@pytest.mark.parametrize(("n", "alpha", "exact"), [(19, 0.1, 18 / 20), (18, 0.1, 18 / 19)])
@pytest.mark.parametrize("law", ["uniform", "lognormal"])
def test_exact_finite_sample_coverage_monte_carlo(n, alpha, exact, law):
    measured = _empirical_split_coverage(n, alpha, MONTE_CARLO_REPLICATES, 57101, law)
    assert measured == pytest.approx(exact, abs=MONTE_CARLO_TOLERANCE)


def test_monte_carlo_separates_the_two_calibration_sizes():
    # The whole point of the exact statement: dropping one calibration point
    # raises coverage from 0.9000 to 0.9474, and the measurement sees it.
    at_19 = _empirical_split_coverage(19, 0.1, MONTE_CARLO_REPLICATES, 57102, "uniform")
    at_18 = _empirical_split_coverage(18, 0.1, MONTE_CARLO_REPLICATES, 57102, "uniform")
    assert at_18 - at_19 == pytest.approx(18 / 19 - 18 / 20, abs=2 * MONTE_CARLO_TOLERANCE)


def test_coverage_is_above_nominal_not_merely_near_it():
    measured = _empirical_split_coverage(100, 0.1, MONTE_CARLO_REPLICATES, 57103, "uniform")
    assert measured > 0.9 - MONTE_CARLO_TOLERANCE
    assert measured == pytest.approx(91 / 101, abs=MONTE_CARLO_TOLERANCE)


# ---------------------------------------------------------------------------
# 3. The weighted quantile reduces to the order statistic
# ---------------------------------------------------------------------------
def test_weighted_quantile_with_unit_weights_is_the_kth_order_statistic():
    # Equal weights and tail_weight = 1 reproduce split conformal exactly:
    # total = n + 1, threshold = (1 - alpha)(n + 1), so the selected index is
    # ceil((n+1)(1-alpha)) - 1 in 0-based terms.
    values = np.array([5.0, 1.0, 4.0, 2.0, 3.0])
    rank = conformal_rank(5, 0.5)  # ceil(6 * 0.5) = 3
    assert rank == 3
    got = weighted_quantile(values, np.ones(5), 0.5, tail_weight=1.0)
    assert got == pytest.approx(np.sort(values)[rank - 1], rel=0.0)


def test_weighted_quantile_scale_invariance():
    values = np.arange(1.0, 11.0)
    a = weighted_quantile(values, np.ones(10), 0.9, tail_weight=1.0)
    b = weighted_quantile(values, 7.5 * np.ones(10), 0.9, tail_weight=7.5)
    assert a == pytest.approx(b, rel=0.0)


def test_weighted_quantile_is_infinite_when_the_tail_mass_is_too_large():
    # tail_weight / (sum w + tail_weight) > alpha makes the level unreachable:
    # here 10 / (1 + 10) = 0.909 > 0.1.
    assert math.isinf(weighted_quantile(np.array([1.0]), np.array([1.0]), 0.9, tail_weight=10.0))


def test_weighted_quantile_picks_the_heavy_point():
    # All mass on the largest value forces the quantile up to it.
    values = np.array([1.0, 2.0, 100.0])
    weights = np.array([1e-6, 1e-6, 1.0])
    assert weighted_quantile(values, weights, 0.9) == pytest.approx(100.0, rel=0.0)


# ---------------------------------------------------------------------------
# 4. Likelihood ratio closed form against the densities themselves
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("severity", [0.5, 1.0, 2.0, 3.0, 5.0])
def test_likelihood_ratio_matches_the_gaussian_density_ratio(severity):
    shift = CovariateShift(severity=severity)
    rng = np.random.default_rng(57104)
    mass = rng.normal(CALIBRATION_MASS_MEAN, CALIBRATION_MASS_SD, 500)
    headwind = rng.normal(CALIBRATION_HEADWIND_MEAN, CALIBRATION_HEADWIND_SD, 500)
    numerator = stats.norm.pdf(mass, shift.mass_mean, CALIBRATION_MASS_SD) * stats.norm.pdf(
        headwind, shift.headwind_mean, CALIBRATION_HEADWIND_SD
    )
    denominator = stats.norm.pdf(
        mass, CALIBRATION_MASS_MEAN, CALIBRATION_MASS_SD
    ) * stats.norm.pdf(headwind, CALIBRATION_HEADWIND_MEAN, CALIBRATION_HEADWIND_SD)
    assert np.allclose(shift.likelihood_ratio(mass, headwind), numerator / denominator, rtol=1e-12)


def test_likelihood_ratio_is_one_without_a_shift():
    shift = CovariateShift(severity=0.0)
    assert np.allclose(shift.likelihood_ratio(np.linspace(4.0, 8.0, 7), np.linspace(-4, 4, 7)), 1.0)


def test_likelihood_ratio_at_the_calibration_mean_is_exp_minus_half_mahalanobis_squared():
    # At x = mu_cal, log w = -sum delta_j^2 / (2 sigma_j^2) = -mahalanobis^2 / 2.
    shift = CovariateShift(severity=2.0)
    got = float(shift.likelihood_ratio(CALIBRATION_MASS_MEAN, CALIBRATION_HEADWIND_MEAN))
    assert got == pytest.approx(math.exp(-0.5 * shift.mahalanobis**2), rel=1e-14)


def test_mahalanobis_for_severity_one():
    # delta_mass / sd = 0.18 / 0.60 = 0.3, delta_headwind / sd = 0.60 / 2.00 = 0.3,
    # so the displacement is sqrt(0.3^2 + 0.3^2) = 0.3 sqrt(2) = 0.424264068711929.
    assert CovariateShift(severity=1.0).mahalanobis == pytest.approx(0.3 * math.sqrt(2), rel=1e-14)


# ---------------------------------------------------------------------------
# 5. Energy model, by hand
# ---------------------------------------------------------------------------
def test_level_flight_power_hand_computation():
    # V = 20 m/s, m = 6 kg, rho = 1.18 kg/m^3, shipped airframe, constant eta.
    #   P_parasite = 0.5 * 1.18 * 20^3 * 0.30 * 0.035          = 49.560000000000 W
    #   W  = 6 * 9.80665                                       = 58.839900000000 N
    #   W^2                                                    = 3462.133832010000 N^2
    #   denominator = 1.18 * 20 * pi * 1.20^2 * 0.85           = 90.749302028656
    #   P_induced   = 2 * 3462.133832010000 / 90.749302028656  = 76.301056969380 W
    #   P_total     = (49.56 + 76.301056969380) / 0.62 + 12    = 215.001704789323 W
    got = float(level_flight_power(20.0, 6.0, 1.18, constant_efficiency=True))
    assert got == pytest.approx(215.001704789323, rel=1e-12)


def test_leg_energy_hand_computation():
    # Ground speed 20 m/s over 1000 m is 50 s, so
    #   E = 215.001704789323 * 50 / 3600 = 2.986134788741 Wh
    got = float(leg_energy(20.0, 6.0, 1.18, 0.0, 1000.0, constant_efficiency=True))
    assert got == pytest.approx(2.986134788741, rel=1e-12)


# ---------------------------------------------------------------------------
# 6. Degenerate cases of the three conformal variants agree
# ---------------------------------------------------------------------------
def test_mondrian_with_one_bin_equals_split():
    rng = np.random.default_rng(57105)
    y = rng.normal(3.0, 0.2, 400)
    p = rng.normal(3.0, 0.2, 400)
    split = SplitConformal(0.1).calibrate(y, p)
    mondrian = MondrianConformal(0.1).calibrate(y, p, np.zeros(400, dtype=int))
    assert mondrian.quantiles[0] == pytest.approx(split.quantile, rel=0.0)


def test_weighted_with_unit_weights_equals_split():
    rng = np.random.default_rng(57106)
    y = rng.normal(3.0, 0.2, 400)
    p = rng.normal(3.0, 0.2, 400)
    split = SplitConformal(0.1).calibrate(y, p)
    weighted = WeightedConformal(0.1).calibrate(y, p, np.ones(400))
    got = weighted.quantiles(np.ones(5))
    assert np.allclose(got, split.quantile, rtol=0.0)


def test_weighted_under_zero_shift_equals_split_on_real_weights():
    rng = np.random.default_rng(57107)
    y = rng.normal(3.0, 0.2, 300)
    p = rng.normal(3.0, 0.2, 300)
    shift = CovariateShift(severity=0.0)
    mass = rng.normal(CALIBRATION_MASS_MEAN, CALIBRATION_MASS_SD, 300)
    wind = rng.normal(CALIBRATION_HEADWIND_MEAN, CALIBRATION_HEADWIND_SD, 300)
    weighted = WeightedConformal(0.1).calibrate(y, p, shift.likelihood_ratio(mass, wind))
    split = SplitConformal(0.1).calibrate(y, p)
    assert weighted.quantiles(np.ones(3))[0] == pytest.approx(split.quantile, rel=0.0)


# ---------------------------------------------------------------------------
# 7. Diagnostics, by hand
# ---------------------------------------------------------------------------
def test_effective_sample_size_of_equal_weights_is_n():
    # (n * w)^2 / (n * w^2) = n for any w > 0.
    assert effective_sample_size(np.full(37, 2.5)) == pytest.approx(37.0, rel=1e-14)


def test_effective_sample_size_of_a_single_dominant_weight_is_one():
    weights = np.zeros(50)
    weights[7] = 3.0
    assert effective_sample_size(weights) == pytest.approx(1.0, rel=1e-14)


def test_effective_sample_size_two_weights_one_and_three():
    # (1 + 3)^2 / (1 + 9) = 16 / 10 = 1.6 exactly.
    assert effective_sample_size(np.array([1.0, 3.0])) == pytest.approx(1.6, rel=1e-15)


def test_clopper_pearson_zero_successes_has_zero_lower_limit():
    low, high = clopper_pearson(0, 20)
    assert low == 0.0
    # 1 - (0.025)^(1/20) = 0.16843...  for the upper limit at 95 % two-sided.
    assert high == pytest.approx(1.0 - 0.025 ** (1.0 / 20.0), rel=1e-10)


def test_clopper_pearson_all_successes_has_unit_upper_limit():
    low, high = clopper_pearson(20, 20)
    assert high == 1.0
    assert low == pytest.approx(0.025 ** (1.0 / 20.0), rel=1e-10)


def test_clopper_pearson_brackets_the_point_estimate():
    low, high = clopper_pearson(900, 1000)
    assert low < 0.9 < high
