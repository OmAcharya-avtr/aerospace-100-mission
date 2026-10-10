"""Property-based tests with Hypothesis, for the identities that are algebraic."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from conformalband.bounds import effective_sample_size, split_conformal_coverage_bound
from conformalband.conformal import (
    MondrianConformal,
    SplitConformal,
    WeightedConformal,
    conformal_rank,
    weighted_quantile,
)
from conformalband.physics import leg_energy, level_flight_power
from conformalband.shift import CovariateShift

SETTINGS = settings(max_examples=60, deadline=None, derandomize=True)
"""``derandomize=True`` so the gate re-runs the same examples every time.

Without it this suite found a real floating-point artefact on one run in
several and passed on the others, which is the worst of both worlds: see
``test_known_answers.test_exact_coverage_can_sit_one_ulp_below_one_minus_alpha``.
"""

BOUND_TOLERANCE = 1e-12

scores = st.lists(
    st.floats(min_value=0.0, max_value=1e4, allow_nan=False, allow_infinity=False),
    min_size=9,
    max_size=60,
)
positive_weights = st.lists(
    st.floats(min_value=1e-6, max_value=1e6, allow_nan=False, allow_infinity=False),
    min_size=9,
    max_size=60,
)


@SETTINGS
@given(n=st.integers(min_value=9, max_value=5000), alpha=st.floats(0.001, 0.5))
def test_rank_never_exceeds_n_when_the_bound_is_defined(n, alpha):
    try:
        bound = split_conformal_coverage_bound(n, alpha)
    except ValueError:
        assume(False)
        return
    assert 1 <= bound.rank <= n
    # The tolerance is on both sides on purpose; see BOUND_TOLERANCE above.
    assert bound.lower - BOUND_TOLERANCE <= bound.exact <= bound.upper + BOUND_TOLERANCE


@SETTINGS
@given(n=st.integers(min_value=9, max_value=5000), alpha=st.floats(0.001, 0.5))
def test_rank_is_monotone_non_decreasing_in_n(n, alpha):
    assume(conformal_rank(n, alpha) <= n)
    assume(conformal_rank(n + 1, alpha) <= n + 1)
    assert conformal_rank(n + 1, alpha) >= conformal_rank(n, alpha)


@SETTINGS
@given(values=scores, level=st.floats(0.01, 0.99))
def test_weighted_quantile_with_equal_weights_is_an_order_statistic(values, level):
    array = np.asarray(values, dtype=float)
    got = weighted_quantile(array, np.ones(array.size), level)
    assert math.isinf(got) or got in set(array.tolist())


@SETTINGS
@given(values=scores, weights=positive_weights, level=st.floats(0.01, 0.99))
def test_weighted_quantile_is_scale_invariant_in_the_weights(values, weights, level):
    n = min(len(values), len(weights))
    v = np.asarray(values[:n], dtype=float)
    w = np.asarray(weights[:n], dtype=float)
    a = weighted_quantile(v, w, level)
    b = weighted_quantile(v, 123.0 * w, level)
    assert (math.isinf(a) and math.isinf(b)) or a == b


@SETTINGS
@given(values=scores, weights=positive_weights, level=st.floats(0.01, 0.95))
def test_weighted_quantile_is_monotone_in_level(values, weights, level):
    n = min(len(values), len(weights))
    v = np.asarray(values[:n], dtype=float)
    w = np.asarray(weights[:n], dtype=float)
    low = weighted_quantile(v, w, level)
    high = weighted_quantile(v, w, min(level + 0.04, 0.99))
    assert math.isinf(high) or high >= low


@SETTINGS
@given(values=scores, weights=positive_weights, level=st.floats(0.01, 0.99))
def test_weighted_quantile_lies_in_the_value_range(values, weights, level):
    n = min(len(values), len(weights))
    v = np.asarray(values[:n], dtype=float)
    w = np.asarray(weights[:n], dtype=float)
    got = weighted_quantile(v, w, level)
    assert math.isinf(got) or (v.min() <= got <= v.max())


@SETTINGS
@given(values=scores, tail=st.floats(1e-6, 1e3))
def test_a_larger_tail_weight_never_narrows_the_quantile(values, tail):
    array = np.asarray(values, dtype=float)
    small = weighted_quantile(array, np.ones(array.size), 0.9, tail_weight=tail)
    large = weighted_quantile(array, np.ones(array.size), 0.9, tail_weight=10.0 * tail)
    assert math.isinf(large) or large >= small


@SETTINGS
@given(weights=positive_weights)
def test_effective_sample_size_is_bounded(weights):
    w = np.asarray(weights, dtype=float)
    assert 1.0 - 1e-9 <= effective_sample_size(w) <= w.size + 1e-9


@SETTINGS
@given(weights=positive_weights, factor=st.floats(1e-3, 1e3))
def test_effective_sample_size_is_scale_invariant(weights, factor):
    w = np.asarray(weights, dtype=float)
    assert effective_sample_size(w) == pytest.approx(effective_sample_size(factor * w), rel=1e-9)


@SETTINGS
@given(
    severity=st.floats(0.0, 6.0),
    mass=st.floats(3.0, 9.0),
    headwind=st.floats(-8.0, 8.0),
)
def test_likelihood_ratio_is_positive_and_log_consistent(severity, mass, headwind):
    shift = CovariateShift(severity=severity)
    ratio = float(shift.likelihood_ratio(mass, headwind))
    log_ratio = float(shift.log_likelihood_ratio(mass, headwind))
    assert ratio > 0.0
    assert math.log(ratio) == pytest.approx(log_ratio, rel=1e-9) or abs(log_ratio) < 1e-12


@SETTINGS
@given(severity=st.floats(0.0, 6.0), fraction=st.floats(0.0, 2.0))
def test_scaled_severity_is_multiplicative(severity, fraction):
    shift = CovariateShift(severity=severity).scaled(fraction)
    assert shift.severity == pytest.approx(severity * fraction, rel=1e-12)


@SETTINGS
@given(
    airspeed=st.floats(14.0, 26.0),
    mass_low=st.floats(3.0, 6.0),
    extra=st.floats(0.1, 3.0),
    density=st.floats(1.0, 1.3),
)
def test_power_is_strictly_increasing_in_mass(airspeed, mass_low, extra, density):
    low = float(level_flight_power(airspeed, mass_low, density))
    high = float(level_flight_power(airspeed, mass_low + extra, density))
    assert high > low


@SETTINGS
@given(
    airspeed=st.floats(14.0, 26.0),
    mass=st.floats(3.0, 9.0),
    density=st.floats(1.0, 1.3),
    headwind=st.floats(-8.0, 8.0),
    distance=st.floats(200.0, 4000.0),
)
def test_energy_is_positive_and_scales_with_distance(airspeed, mass, density, headwind, distance):
    assume(airspeed - headwind > 1.0)
    one = float(leg_energy(airspeed, mass, density, headwind, distance))
    two = float(leg_energy(airspeed, mass, density, headwind, 2.0 * distance))
    assert one > 0.0
    assert two == pytest.approx(2.0 * one, rel=1e-12)


@SETTINGS
@given(
    residual_scale=st.floats(0.01, 10.0),
    n=st.integers(min_value=20, max_value=400),
    alpha=st.floats(0.05, 0.4),
)
def test_split_quantile_scales_with_the_residuals(residual_scale, n, alpha):
    rng = np.random.default_rng(n)
    base = rng.normal(0.0, 1.0, n)
    plain = SplitConformal(alpha).calibrate(base, np.zeros(n)).quantile
    scaled = SplitConformal(alpha).calibrate(residual_scale * base, np.zeros(n)).quantile
    assert scaled == pytest.approx(residual_scale * plain, rel=1e-9)


@SETTINGS
@given(n=st.integers(min_value=20, max_value=300), alpha=st.floats(0.05, 0.4))
def test_mondrian_single_bin_matches_split(n, alpha):
    rng = np.random.default_rng(n + 7)
    y = rng.normal(0.0, 1.0, n)
    p = np.zeros(n)
    split = SplitConformal(alpha).calibrate(y, p).quantile
    mondrian = MondrianConformal(alpha).calibrate(y, p, np.zeros(n, dtype=int)).quantiles[0]
    assert mondrian == pytest.approx(split, rel=0.0)


@SETTINGS
@given(n=st.integers(min_value=20, max_value=300), alpha=st.floats(0.05, 0.4))
def test_weighted_unit_weights_match_split(n, alpha):
    rng = np.random.default_rng(n + 13)
    y = rng.normal(0.0, 1.0, n)
    p = np.zeros(n)
    split = SplitConformal(alpha).calibrate(y, p).quantile
    weighted = WeightedConformal(alpha).calibrate(y, p, np.ones(n)).quantiles(np.ones(1))[0]
    assert weighted == pytest.approx(split, rel=0.0)
