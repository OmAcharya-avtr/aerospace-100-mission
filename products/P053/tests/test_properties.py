"""Hypothesis property tests for the algebraic identities in the statistics.

Each property below is an exact identity, not an approximation, so the
tolerances are at round-off level. Where an identity only holds up to
accumulated floating-point error the tolerance is stated relative to the
magnitude of the statistic.
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import array_shapes, arrays

from twininvalidate import (
    arl0_from_rate,
    cusum_statistic,
    ewma_lambda1_threshold,
    ewma_statistic,
    first_alarm,
    glr_statistic,
    glr_window1_threshold,
    rate_from_arl0,
    variance_cusum_statistic,
    window_features,
    zoh_discretise,
)

residuals = arrays(
    dtype=np.float64,
    shape=array_shapes(min_dims=2, max_dims=2, min_side=1, max_side=12),
    elements=st.floats(min_value=-8.0, max_value=8.0, allow_nan=False, allow_infinity=False),
)

positive_scales = st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False)


@settings(max_examples=60, deadline=None)
@given(residuals)
def test_every_statistic_is_non_negative(z):
    for stat in (
        cusum_statistic(z),
        ewma_statistic(z),
        glr_statistic(z, window=4),
        variance_cusum_statistic(z),
    ):
        assert stat.min() >= 0.0
        assert stat.shape == z.shape


@settings(max_examples=60, deadline=None)
@given(residuals, positive_scales)
def test_glr_is_quadratically_homogeneous(z, a):
    # g_k(a z) = max_n (a S_n)^2 / (2n) = a^2 g_k(z), exactly.
    lhs = glr_statistic(a * z, window=5)
    rhs = (a**2) * glr_statistic(z, window=5)
    assert np.allclose(lhs, rhs, rtol=1e-11, atol=1e-13)


@settings(max_examples=60, deadline=None)
@given(residuals, positive_scales)
def test_cusum_is_homogeneous_when_the_reference_scales_too(z, a):
    # S+_k(a z; a k) = a S+_k(z; k), because max(0, .) commutes with
    # multiplication by a positive scalar.
    lhs = cusum_statistic(a * z, reference=a * 0.25)
    rhs = a * cusum_statistic(z, reference=0.25)
    assert np.allclose(lhs, rhs, rtol=1e-11, atol=1e-13)


@settings(max_examples=60, deadline=None)
@given(residuals, positive_scales)
def test_ewma_is_absolutely_homogeneous(z, a):
    lhs = ewma_statistic(a * z, lam=0.1)
    rhs = a * ewma_statistic(z, lam=0.1)
    assert np.allclose(lhs, rhs, rtol=1e-11, atol=1e-13)


@settings(max_examples=60, deadline=None)
@given(residuals)
def test_ewma_lambda_one_is_the_absolute_value(z):
    assert np.allclose(ewma_statistic(z, lam=1.0), np.abs(z), rtol=0.0, atol=1e-13)


@settings(max_examples=60, deadline=None)
@given(residuals)
def test_glr_window_one_is_half_the_square(z):
    assert np.allclose(glr_statistic(z, window=1), z**2 / 2.0, rtol=1e-13, atol=1e-15)


@settings(max_examples=60, deadline=None)
@given(residuals)
def test_glr_is_monotone_in_the_window(z):
    # A larger window maximises over a superset, so the statistic cannot fall.
    small = glr_statistic(z, window=2)
    large = glr_statistic(z, window=6)
    assert np.all(large >= small - 1e-12)


@settings(max_examples=60, deadline=None)
@given(residuals, st.floats(min_value=0.05, max_value=2.0))
def test_cusum_is_monotone_decreasing_in_the_reference(z, extra):
    low = cusum_statistic(z, reference=0.25)
    high = cusum_statistic(z, reference=0.25 + extra)
    assert np.all(high <= low + 1e-12)


@settings(max_examples=60, deadline=None)
@given(residuals)
def test_cusum_is_bounded_by_the_largest_subinterval_sum(z):
    # S+_k = max over onsets j of (sum_{i=j..k} z_i - (k-j+1) k)^+, which is at
    # most the largest sum over any sub-interval; the same bound applies to the
    # lower arm with z negated. With C the cumulative sum prefixed by 0, that
    # largest absolute sub-interval sum is max(C) - min(C).
    cum = np.concatenate([np.zeros((z.shape[0], 1)), np.cumsum(z, axis=1)], axis=1)
    bound = cum.max(axis=1) - cum.min(axis=1)
    assert np.all(cusum_statistic(z).max(axis=1) <= bound + 1e-9)


@settings(max_examples=60, deadline=None)
@given(residuals, st.floats(min_value=0.0, max_value=5.0))
def test_first_alarm_is_consistent_with_the_statistic(z, threshold):
    stat = cusum_statistic(z)
    idx = first_alarm(stat, threshold)
    for r, i in enumerate(idx):
        if i < 0:
            assert np.all(stat[r] <= threshold)
        else:
            assert stat[r, i] > threshold
            assert np.all(stat[r, :i] <= threshold)


@settings(max_examples=40, deadline=None)
@given(
    st.floats(min_value=1.0001, max_value=1e8, allow_nan=False, allow_infinity=False),
)
def test_the_two_closed_form_thresholds_agree(target):
    # GLR(window=1) alarms iff |z| > sqrt(2h); EWMA(lam=1) iff |z| > L.
    # Therefore h = L^2 / 2 for every target ARL0.
    assert glr_window1_threshold(target) == np.float64(
        ewma_lambda1_threshold(target) ** 2 / 2.0
    )


@settings(max_examples=40, deadline=None)
@given(
    st.floats(min_value=1e-6, max_value=1e6, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.01, max_value=1000.0, allow_nan=False, allow_infinity=False),
)
def test_rate_and_arl0_are_mutual_inverses(rate, hz):
    arl0 = arl0_from_rate(rate, hz)
    assert rate_from_arl0(arl0, hz) == np.float64(rate) or abs(
        rate_from_arl0(arl0, hz) - rate
    ) <= 1e-9 * rate


@settings(max_examples=40, deadline=None)
@given(
    st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False),
    st.integers(min_value=4, max_value=12),
)
def test_constant_stream_features_are_exact(value, window):
    f = window_features(np.full((1, window), float(value)), window)[0, 0]
    scale = max(1.0, abs(value))
    assert abs(f[0] - value) < 1e-12 * scale  # mean
    assert f[1] == 0.0 or abs(f[1]) < 1e-12  # std
    assert abs(f[2] - abs(value)) < 1e-12  # mean |z|
    assert abs(f[3] - abs(value)) < 1e-12  # max |z|
    assert f[4] == 0.0  # autocorrelation of a constant is defined as 0
    assert f[5] == 0.0  # kurtosis likewise
    assert abs(f[6]) < 1e-9  # no trend
    assert abs(f[8] - value**2) < 1e-12  # mean z^2


@settings(max_examples=40, deadline=None)
@given(
    st.floats(min_value=-3.0, max_value=3.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.01, max_value=0.5, allow_nan=False, allow_infinity=False),
)
def test_zoh_of_a_scalar_matches_its_closed_form(a, dt):
    # xdot = a x + u: A = exp(a dt); B = (exp(a dt) - 1)/a, or dt when a = 0.
    # The closed form for B is written with expm1 so it stays accurate as
    # a -> 0, where (exp(a dt) - 1)/a cancels catastrophically; Hypothesis
    # found a = 1e-12 and the naive form was wrong by 5 %, not the code.
    a_d, b_d = zoh_discretise(np.array([[a]]), np.array([[1.0]]), dt)
    assert abs(a_d[0, 0] - np.exp(a * dt)) < 1e-12
    x = a * dt
    expected_b = dt if x == 0.0 else dt * np.expm1(x) / x
    assert abs(b_d[0, 0] - expected_b) < 1e-11 * max(1.0, abs(expected_b))


@settings(max_examples=40, deadline=None)
@given(
    st.floats(min_value=-3.0, max_value=3.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.01, max_value=0.5, allow_nan=False, allow_infinity=False),
    st.integers(min_value=2, max_value=6),
)
def test_zoh_composes_over_time(a, dt, n):
    # Discretising once over n dt must equal the n-th power of discretising
    # over dt, because the matrix exponential is a one-parameter group.
    a_1, _ = zoh_discretise(np.array([[a]]), np.array([[1.0]]), dt)
    a_n, _ = zoh_discretise(np.array([[a]]), np.array([[1.0]]), dt * n)
    assert abs(a_n[0, 0] - a_1[0, 0] ** n) < 1e-10 * max(1.0, abs(a_n[0, 0]))
