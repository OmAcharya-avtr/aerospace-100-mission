"""Property-based tests. These exist because hand-written cases miss defects.

On product P071 of this mission, Hypothesis found six equation defects that the
hand-written tests did not. The properties below are the algebraic identities
of the estimators, stated as identities and checked over generated inputs.

Floating point is handled per the mission's rule: a statistic is compared with
a tolerance on both sides, and no boolean produced by a strict threshold
comparison is asserted to be invariant. Where an identity is exact in integer
arithmetic that is stated and asserted exactly; where the implementation
converts to ``float64`` the claim is agreement to within rounding.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from timetransfer.allan import (
    equivalent_degrees_of_freedom,
    fit_log_slope,
    overlapping_allan_variance,
)
from timetransfer.exchange import Exchange, ExchangeSet
from timetransfer.skew import KalmanConfig, fit_least_squares, kalman_skew
from timetransfer.timebase import NS_PER_S, VirtualTimebase, offset_quantisation_uncertainty
from timetransfer.twoway import asymmetry_bias_s, asymmetry_from_bias_s, symmetric_offsets
from timetransfer.uncertainty import (
    AsymmetrySource,
    effective_sample_size,
    measure_coverage,
    offset_uncertainty,
)

SLOW = settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)

ns = st.integers(min_value=0, max_value=10**12)
small_ns = st.integers(min_value=0, max_value=10**9)
theta_ns = st.integers(min_value=-10**9, max_value=10**9)


def _build(t1: int, d_ab: int, turn: int, d_ba: int, theta: int) -> ExchangeSet:
    """One exchange built from true delays, so the truth is known by construction."""
    t2 = t1 + d_ab + theta
    t3 = t2 + turn
    t4 = t1 + d_ab + turn + d_ba
    return ExchangeSet(
        t1_ns=np.asarray([t1], dtype=np.int64),
        t2_ns=np.asarray([t2], dtype=np.int64),
        t3_ns=np.asarray([t3], dtype=np.int64),
        t4_ns=np.asarray([t4], dtype=np.int64),
        seq=np.asarray([0], dtype=np.int64),
    )


@given(t1=ns, d_ab=small_ns, turn=small_ns, d_ba=small_ns, theta=theta_ns)
@settings(max_examples=300, deadline=None)
def test_offset_plus_delay_is_the_forward_dwell(
    t1: int, d_ab: int, turn: int, d_ba: int, theta: int
) -> None:
    """``delay + offset == (t2 - t1)`` and ``delay - offset == (t4 - t3)``.

    Exact in exact arithmetic; the implementation divides by ``NS_PER_S`` once,
    so the claim is agreement to within rounding.
    """
    exchanges = _build(t1, d_ab, turn, d_ba, theta)
    offset, delay = symmetric_offsets(exchanges)
    fwd = float(exchanges.forward_dwell_ns()[0]) / NS_PER_S
    rev = float(exchanges.reverse_dwell_ns()[0]) / NS_PER_S
    assert delay[0] + offset[0] == pytest.approx(fwd, rel=1e-12, abs=1e-15)
    assert delay[0] - offset[0] == pytest.approx(rev, rel=1e-12, abs=1e-15)


@given(t1=ns, d=small_ns, turn=small_ns, theta=theta_ns)
@settings(max_examples=300, deadline=None)
def test_symmetric_delay_recovers_theta(t1: int, d: int, turn: int, theta: int) -> None:
    """With ``d_ab == d_ba`` the estimator returns ``theta`` with no bias term."""
    offset, delay = symmetric_offsets(_build(t1, d, turn, d, theta))
    assert offset[0] == pytest.approx(theta / NS_PER_S, rel=1e-12, abs=1e-15)
    assert delay[0] == pytest.approx(d / NS_PER_S, rel=1e-12, abs=1e-15)


@given(t1=ns, d_ab=small_ns, turn=small_ns, d_ba=small_ns, theta=theta_ns)
@settings(max_examples=300, deadline=None)
def test_bias_is_half_the_constructed_asymmetry(
    t1: int, d_ab: int, turn: int, d_ba: int, theta: int
) -> None:
    """``offset - theta == (d_ab - d_ba) / 2``, equation (5)."""
    offset, _ = symmetric_offsets(_build(t1, d_ab, turn, d_ba, theta))
    expected = (theta + 0.5 * (d_ab - d_ba)) / NS_PER_S
    assert offset[0] == pytest.approx(expected, rel=1e-12, abs=1e-15)


@given(t1=ns, d_ab=small_ns, turn=small_ns, d_ba=small_ns, theta=theta_ns, shift=theta_ns)
@settings(max_examples=200, deadline=None)
def test_a_remote_clock_shift_moves_the_offset_and_not_the_delay(
    t1: int, d_ab: int, turn: int, d_ba: int, theta: int, shift: int
) -> None:
    """Adding ``c`` to ``t2`` and ``t3`` adds ``c`` to the offset and nothing to the delay."""
    base = _build(t1, d_ab, turn, d_ba, theta)
    shifted = _build(t1, d_ab, turn, d_ba, theta + shift)
    o0, d0 = symmetric_offsets(base)
    o1, d1 = symmetric_offsets(shifted)
    assert o1[0] - o0[0] == pytest.approx(shift / NS_PER_S, rel=1e-9, abs=1e-15)
    assert d1[0] == pytest.approx(d0[0], rel=1e-12, abs=1e-15)


@given(t1=ns, d_ab=small_ns, turn=small_ns, d_ba=small_ns, theta=theta_ns)
@settings(max_examples=300, deadline=None)
def test_round_trip_is_the_exact_integer_sum_of_the_dwells(
    t1: int, d_ab: int, turn: int, d_ba: int, theta: int
) -> None:
    """``(t4-t1) - (t3-t2) == (t2-t1) + (t4-t3)`` exactly, in integers."""
    exchanges = _build(t1, d_ab, turn, d_ba, theta)
    rtt = int(exchanges.round_trip_ns()[0])
    assert rtt == int(exchanges.forward_dwell_ns()[0]) + int(exchanges.reverse_dwell_ns()[0])
    assert rtt == d_ab + d_ba


@given(
    asymmetry=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False)
)
def test_bias_and_its_inverse_are_mutually_inverse(asymmetry: float) -> None:
    """``asymmetry_from_bias_s(asymmetry_bias_s(a)) == a`` to within rounding."""
    back = asymmetry_from_bias_s(asymmetry_bias_s(asymmetry))
    assert back == pytest.approx(asymmetry, rel=1e-15, abs=1e-300)


@given(
    t1=ns,
    d_ab=small_ns,
    turn=small_ns,
    d_ba=small_ns,
    theta=theta_ns,
)
@settings(max_examples=200, deadline=None)
def test_exchange_derived_quantities_match_the_set(
    t1: int, d_ab: int, turn: int, d_ba: int, theta: int
) -> None:
    """A single :class:`Exchange` and a one-row set agree exactly."""
    exchanges = _build(t1, d_ab, turn, d_ba, theta)
    single = Exchange(
        seq=0,
        t1_ns=int(exchanges.t1_ns[0]),
        t2_ns=int(exchanges.t2_ns[0]),
        t3_ns=int(exchanges.t3_ns[0]),
        t4_ns=int(exchanges.t4_ns[0]),
    )
    assert single.forward_dwell_ns == int(exchanges.forward_dwell_ns()[0])
    assert single.reverse_dwell_ns == int(exchanges.reverse_dwell_ns()[0])
    assert single.round_trip_ns == int(exchanges.round_trip_ns()[0])
    assert single.midpoint_ns == int(exchanges.midpoint_ns()[0])


phase = st.lists(
    st.floats(min_value=-1e-3, max_value=1e-3, allow_nan=False, allow_infinity=False),
    min_size=12,
    max_size=200,
)


@given(x=phase, m=st.integers(min_value=1, max_value=4))
@SLOW
def test_allan_variance_is_non_negative(x: list[float], m: int) -> None:
    assume(2 * m <= len(x) - 1)
    avar, n_terms = overlapping_allan_variance(np.asarray(x), 1.0, m)
    assert avar >= 0.0
    assert n_terms == len(x) - 2 * m


@given(
    x=phase,
    m=st.integers(min_value=1, max_value=4),
    shift=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
@SLOW
def test_allan_variance_is_invariant_under_a_phase_offset(
    x: list[float], m: int, shift: float
) -> None:
    """A constant phase offset cancels in the second difference."""
    assume(2 * m <= len(x) - 1)
    arr = np.asarray(x)
    a, _ = overlapping_allan_variance(arr, 1.0, m)
    b, _ = overlapping_allan_variance(arr + shift, 1.0, m)
    assert b == pytest.approx(a, rel=1e-6, abs=1e-24)


@given(
    x=phase,
    m=st.integers(min_value=1, max_value=4),
    scale=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
)
@SLOW
def test_allan_variance_scales_quadratically(x: list[float], m: int, scale: float) -> None:
    assume(2 * m <= len(x) - 1)
    arr = np.asarray(x)
    a, _ = overlapping_allan_variance(arr, 1.0, m)
    b, _ = overlapping_allan_variance(scale * arr, 1.0, m)
    assert b == pytest.approx(scale * scale * a, rel=1e-9, abs=1e-30)


@given(
    n=st.integers(min_value=20, max_value=500),
    slope=st.floats(min_value=-1e-3, max_value=1e-3, allow_nan=False, allow_infinity=False),
    intercept=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    m=st.integers(min_value=1, max_value=4),
)
@SLOW
def test_allan_variance_of_a_linear_phase_is_zero_to_rounding(
    n: int, slope: float, intercept: float, m: int
) -> None:
    """A pure frequency offset has vanishing second differences."""
    assume(2 * m <= n - 1)
    x = intercept + slope * np.arange(n, dtype=np.float64)
    avar, _ = overlapping_allan_variance(x, 1.0, m)
    assert avar == pytest.approx(0.0, abs=1e-24 + 1e-18 * slope * slope)


@given(
    kind=st.sampled_from(["white_pm", "white_fm", "flicker_fm", "random_walk_fm"]),
    n=st.integers(min_value=10, max_value=5000),
    m=st.integers(min_value=1, max_value=50),
)
def test_edf_is_at_least_one_and_finite(kind: str, n: int, m: int) -> None:
    assume(2 * m <= n - 1)
    edf = equivalent_degrees_of_freedom(kind, n, m)
    assert edf >= 1.0
    assert np.isfinite(edf)


@given(
    a=st.floats(min_value=-1e-2, max_value=1e-2, allow_nan=False, allow_infinity=False),
    b=st.floats(min_value=-1e-5, max_value=1e-5, allow_nan=False, allow_infinity=False),
    n=st.integers(min_value=8, max_value=300),
)
@SLOW
def test_least_squares_recovers_an_exact_line(a: float, b: float, n: int) -> None:
    """OLS on noiseless data returns the generating coefficients."""
    t = np.arange(n, dtype=np.float64)
    fit = fit_least_squares(t, a + b * t, order=1, t_ref_s=0.0)
    assert fit.offset0_s == pytest.approx(a, rel=1e-8, abs=1e-14)
    assert fit.skew_s_per_s == pytest.approx(b, rel=1e-6, abs=1e-16)
    assert fit.rss == pytest.approx(0.0, abs=1e-20)


@given(
    values=st.lists(
        st.floats(min_value=-1e-3, max_value=1e-3, allow_nan=False, allow_infinity=False),
        min_size=8,
        max_size=200,
    )
)
@SLOW
def test_least_squares_residuals_sum_to_zero(values: list[float]) -> None:
    """An OLS fit with an intercept has zero-sum residuals, to within rounding."""
    t = np.arange(len(values), dtype=np.float64)
    fit = fit_least_squares(t, np.asarray(values), order=1)
    scale = max(1e-12, float(np.max(np.abs(fit.residuals))))
    assert float(np.sum(fit.residuals)) == pytest.approx(0.0, abs=1e-9 * scale * len(values))


@given(
    values=st.lists(
        st.floats(min_value=-1e-3, max_value=1e-3, allow_nan=False, allow_infinity=False),
        min_size=8,
        max_size=200,
    )
)
@SLOW
def test_effective_sample_size_is_bounded_by_n(values: list[float]) -> None:
    ess = effective_sample_size(np.asarray(values))
    assert 1.0 <= ess.n_effective <= float(len(values))
    assert ess.inflation >= 1.0 - 1e-12
    assert -1.0 <= ess.lag1_autocorrelation <= 1.0


@given(
    n=st.integers(min_value=10, max_value=200),
    sigma=st.floats(min_value=1e-9, max_value=1e-5, allow_nan=False, allow_infinity=False),
    bound=st.floats(min_value=1e-9, max_value=1e-3, allow_nan=False, allow_infinity=False),
    resolution=st.integers(min_value=1, max_value=10**6),
)
@SLOW
def test_combined_uncertainty_dominates_every_component(
    n: int, sigma: float, bound: float, resolution: int
) -> None:
    rng = np.random.default_rng(n * 7 + resolution)
    t = np.arange(n, dtype=np.float64)
    fit = fit_least_squares(t, rng.normal(0.0, sigma, size=n), order=1)
    unc = offset_uncertainty(
        fit,
        resolution_ns=resolution,
        asymmetry_source=AsymmetrySource.DECLARED_BOUND,
        asymmetry_bound_s=bound,
    )
    assert unc.u_combined_s >= unc.u_statistical_s - 1e-30
    assert unc.u_combined_s >= unc.u_quantisation_s - 1e-30
    assert unc.u_combined_s >= unc.u_asymmetry_s - 1e-30
    assert unc.u_asymmetry_s == pytest.approx((bound / 2.0) / np.sqrt(3.0), rel=1e-12)
    assert unc.u_quantisation_s == pytest.approx(
        offset_quantisation_uncertainty(resolution) / np.sqrt(unc.n_effective), rel=1e-12
    )


@given(
    errors=st.lists(
        st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        min_size=4,
        max_size=200,
    ),
    k=st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False),
)
@SLOW
def test_coverage_is_a_fraction_inside_its_own_interval(errors: list[float], k: float) -> None:
    arr = np.asarray(errors)
    result = measure_coverage(arr, np.ones(arr.size), coverage_factor=k)
    assert 0.0 <= result.measured <= 1.0
    assert result.lower <= result.measured + 1e-12
    assert result.measured <= result.upper + 1e-12
    assert 0.0 < result.nominal < 1.0


@given(
    n=st.integers(min_value=6, max_value=200),
    sigma=st.floats(min_value=1e-9, max_value=1e-4, allow_nan=False, allow_infinity=False),
)
@SLOW
def test_kalman_covariance_is_symmetric_and_positive_semidefinite(n: int, sigma: float) -> None:
    rng = np.random.default_rng(n * 13 + 1)
    t = np.arange(n, dtype=np.float64)
    y = rng.normal(0.0, sigma, size=n)
    result = kalman_skew(
        t, y, config=KalmanConfig(q1_s2_per_s=1e-20, q2_per_s=1e-26)
    )
    final = result.covariances[-1]
    assert final == pytest.approx(final.T, abs=1e-30)
    eigenvalues = np.linalg.eigvalsh(final)
    assert float(np.min(eigenvalues)) > -1e-28
    assert np.all(result.innovation_variances > 0.0)


@given(
    scale=st.floats(min_value=1e-9, max_value=1e3, allow_nan=False, allow_infinity=False),
    exponent=st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False),
)
def test_log_slope_recovers_an_exact_power_law(scale: float, exponent: float) -> None:
    tau = np.asarray([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    fit = fit_log_slope(tau, scale * tau**exponent)
    assert fit.slope == pytest.approx(exponent, abs=1e-9)


@given(
    resolution=st.integers(min_value=1, max_value=10**6),
    steps=st.lists(st.integers(min_value=0, max_value=10**9), min_size=1, max_size=40),
)
def test_virtual_timebase_is_monotone_non_decreasing(resolution: int, steps: list[int]) -> None:
    tb = VirtualTimebase(resolution_ns=resolution)
    previous = tb.now_ns()
    for step in steps:
        current = tb.advance_ns(step)
        assert current >= previous
        assert current % resolution == 0
        previous = current
