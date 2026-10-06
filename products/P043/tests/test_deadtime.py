"""Dead time: hand-computable known answers, the non-monotonicity, both branches."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from photoncount import deadtime as dt

TAU = 1e-6


def test_nonparalyzable_known_answer():
    # tau = 1 us, n = 1e5 /s -> n*tau = 0.1
    # m = 1e5 / 1.1 = 90909.090909...
    assert float(dt.nonparalyzable_observed(1e5, TAU)[0]) == pytest.approx(
        90909.0909090909, rel=1e-12
    )


def test_paralyzable_known_answer():
    # tau = 1 us, n = 1e5 /s -> m = 1e5 * exp(-0.1) = 90483.74180359596
    assert float(dt.paralyzable_observed(1e5, TAU)[0]) == pytest.approx(
        90483.74180359596, rel=1e-12
    )


def test_paralyzable_maximum_known_answer():
    # n_max = 1/tau = 1e6; m_max = 1/(e tau) = 367879.4411714423
    n_max, m_max = dt.paralyzable_maximum(TAU)
    assert n_max == pytest.approx(1e6, rel=1e-15)
    assert m_max == pytest.approx(367879.4411714423, rel=1e-12)
    # The maximum really is a maximum of the forward map.
    grid = np.linspace(0.1e6, 3e6, 4001)
    assert float(dt.paralyzable_observed(grid, TAU).max()) <= m_max * (1 + 1e-12)


def test_paralyzable_forward_map_is_non_monotonic():
    """The trap: observed rate falls as illumination rises past 1/tau."""
    below = float(dt.paralyzable_observed(0.5e6, TAU)[0])
    at = float(dt.paralyzable_observed(1.0e6, TAU)[0])
    above = float(dt.paralyzable_observed(2.0e6, TAU)[0])
    assert below < at
    assert above < at
    assert above < below


def test_paralyzable_inverse_is_two_valued():
    m = 3.0e5
    lo = float(dt.paralyzable_true(m, TAU, "lower")[0])
    hi = float(dt.paralyzable_true(m, TAU, "upper")[0])
    assert lo < 1.0 / TAU < hi
    assert float(dt.paralyzable_observed(lo, TAU)[0]) == pytest.approx(m, rel=1e-10)
    assert float(dt.paralyzable_observed(hi, TAU)[0]) == pytest.approx(m, rel=1e-10)


def test_paralyzable_true_requires_an_explicit_branch():
    with pytest.raises(TypeError):
        dt.paralyzable_true(1e5, TAU)  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        dt.paralyzable_true(1e5, TAU, "middle")  # type: ignore[arg-type]


def test_paralyzable_inverse_rejects_unreachable_rate():
    _, m_max = dt.paralyzable_maximum(TAU)
    with pytest.raises(ValueError, match="paralyzable maximum"):
        dt.paralyzable_true(m_max * 1.01, TAU, "lower")


def test_nonparalyzable_inverse_rejects_saturated_rate():
    with pytest.raises(ValueError, match="must be < 1"):
        dt.nonparalyzable_true(1.0 / TAU, TAU)


def test_loss_fractions_known_answers():
    # At n*tau = 1: non-paralyzable loses 0.5, paralyzable loses 1 - 1/e = 0.6321206
    assert float(dt.dead_time_loss_fraction(1e6, TAU, "nonparalyzable")[0]) == pytest.approx(0.5)
    assert float(dt.dead_time_loss_fraction(1e6, TAU, "paralyzable")[0]) == pytest.approx(
        1.0 - np.exp(-1.0), rel=1e-12
    )


def test_live_time_fractions_differ_by_model():
    # Non-paralyzable live time is 1 - m*tau = 1 - 0.5 = 0.5 at n*tau = 1.
    # Paralyzable live time is exp(-n*tau) = 0.3678794.
    assert float(dt.live_time_fraction(1e6, TAU, "nonparalyzable")[0]) == pytest.approx(0.5)
    assert float(dt.live_time_fraction(1e6, TAU, "paralyzable")[0]) == pytest.approx(
        np.exp(-1.0), rel=1e-12
    )


def test_is_observable_boundaries():
    _, m_max = dt.paralyzable_maximum(TAU)
    assert dt.is_observable([m_max * 0.99, m_max * 1.01], TAU, "paralyzable").tolist() == [
        True,
        False,
    ]
    assert dt.is_observable([0.99 / TAU, 1.01 / TAU], TAU, "nonparalyzable").tolist() == [
        True,
        False,
    ]


def test_zero_rate_is_fixed_point():
    assert float(dt.nonparalyzable_observed(0.0, TAU)[0]) == 0.0
    assert float(dt.paralyzable_observed(0.0, TAU)[0]) == 0.0
    assert float(dt.dead_time_loss_fraction(0.0, TAU, "paralyzable")[0]) == 0.0


def test_models_agree_to_first_order_at_low_loading():
    """Both reduce to m ~ n(1 - n tau) for n tau << 1; they must agree there."""
    n = 1e3  # n*tau = 1e-3
    a = float(dt.nonparalyzable_observed(n, TAU)[0])
    b = float(dt.paralyzable_observed(n, TAU)[0])
    assert abs(a - b) / a < 1e-6


@pytest.mark.parametrize("model", ["paralyzable", "nonparalyzable"])
def test_dispatch_matches_direct_call(model):
    n = 2.5e5
    direct = (
        dt.paralyzable_observed(n, TAU)
        if model == "paralyzable"
        else dt.nonparalyzable_observed(n, TAU)
    )
    assert np.allclose(dt.observed_rate(n, TAU, model), direct)


def test_unknown_model_raises():
    for func in (dt.observed_rate, dt.true_rate, dt.is_observable, dt.dead_time_loss_fraction):
        with pytest.raises(ValueError, match="model must be"):
            func(1e5, TAU, "extending")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="model must be"):
        dt.live_time_fraction(1e5, TAU, "extending")  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_tau", [0.0, -1e-6, float("inf"), float("nan")])
def test_bad_dead_time_raises(bad_tau):
    with pytest.raises(ValueError):
        dt.nonparalyzable_observed(1e5, bad_tau)


def test_negative_rate_raises():
    with pytest.raises(ValueError):
        dt.paralyzable_observed(-1.0, TAU)


@given(
    st.floats(min_value=1e-9, max_value=0.9),
    st.floats(min_value=1e-9, max_value=1e-3),
)
@settings(max_examples=60, deadline=None)
def test_nonparalyzable_round_trip(x, tau):
    """(D1) and (D2) are exact inverses for every n tau."""
    n = x / tau
    m = dt.nonparalyzable_observed(n, tau)
    back = dt.nonparalyzable_true(m, tau)
    assert float(back[0]) == pytest.approx(n, rel=1e-9)


@given(
    st.floats(min_value=1e-6, max_value=0.999),
    st.floats(min_value=1e-9, max_value=1e-3),
)
@settings(max_examples=60, deadline=None)
def test_paralyzable_lower_branch_round_trip(x, tau):
    n = x / tau
    m = dt.paralyzable_observed(n, tau)
    back = dt.paralyzable_true(m, tau, "lower")
    assert float(back[0]) == pytest.approx(n, rel=1e-7)


@given(
    st.floats(min_value=1.001, max_value=20.0),
    st.floats(min_value=1e-9, max_value=1e-3),
)
@settings(max_examples=60, deadline=None)
def test_paralyzable_upper_branch_round_trip(x, tau):
    n = x / tau
    m = dt.paralyzable_observed(n, tau)
    back = dt.paralyzable_true(m, tau, "upper")
    assert float(back[0]) == pytest.approx(n, rel=1e-6)


@given(
    st.floats(min_value=1e-6, max_value=10.0),
    st.floats(min_value=1e-9, max_value=1e-3),
)
@settings(max_examples=60, deadline=None)
def test_paralyzable_loses_more_than_nonparalyzable(x, tau):
    """exp(-x) <= 1/(1+x) for x >= 0: an extending dead time always loses more."""
    n = x / tau
    assert float(dt.paralyzable_observed(n, tau)[0]) <= float(
        dt.nonparalyzable_observed(n, tau)[0]
    ) * (1 + 1e-12)
