"""Limit specification, margin arithmetic and crossing detection."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from degradepredict.limits import (
    LimitSpec,
    crossing_status,
    first_crossing_time,
    has_crossed,
    margin,
)


def test_limitspec_rejects_non_finite_value():
    with pytest.raises(ValueError, match="finite"):
        LimitSpec(value=float("nan"))


def test_limitspec_rejects_unknown_direction():
    with pytest.raises(ValueError, match="direction"):
        LimitSpec(value=1.0, direction="sideways")  # type: ignore[arg-type]


def test_limitspec_rejects_negative_atol():
    with pytest.raises(ValueError, match="atol"):
        LimitSpec(value=1.0, atol=-1e-3)


def test_margin_below_direction_hand_calculation():
    # Hand calculation: metric 85 against a lower limit of 80 leaves 5 units of
    # margin; metric 78 is 2 units past it.
    lim = LimitSpec(value=80.0, direction="below")
    assert margin(85.0, lim) == pytest.approx(5.0)
    assert margin(78.0, lim) == pytest.approx(-2.0)


def test_margin_above_direction_hand_calculation():
    # Upper limit at 0.020 N*m; a drag torque of 0.015 leaves 0.005 of margin.
    lim = LimitSpec(value=0.020, direction="above", unit="N*m")
    assert margin(0.015, lim) == pytest.approx(0.005)
    assert margin(0.023, lim) == pytest.approx(-0.003)


def test_crossing_status_is_three_valued_inside_the_band():
    lim = LimitSpec(value=80.0, atol=0.01)
    assert crossing_status(80.005, lim) == "within_tolerance"
    assert crossing_status(79.995, lim) == "within_tolerance"
    assert crossing_status(80.5, lim) == "not_crossed"
    assert crossing_status(79.5, lim) == "crossed"


def test_crossing_status_rejects_non_finite_value():
    with pytest.raises(ValueError, match="finite"):
        crossing_status(float("inf"), LimitSpec(value=80.0))


def test_has_crossed_matches_status_outside_the_tolerance_band():
    lim = LimitSpec(value=80.0, atol=0.01)
    for value in (70.0, 79.0, 81.0, 95.0):
        expected = crossing_status(value, lim) == "crossed"
        assert has_crossed(value, lim) is expected


def test_first_crossing_time_known_answer():
    # y = 100 - 3 t crosses 80 at t = 20/3 = 6.6667 d exactly in exact arithmetic.
    lim = LimitSpec(value=80.0, atol=1e-9)
    t = np.linspace(0.0, 10.0, 101)
    y = 100.0 - 3.0 * t
    assert first_crossing_time(t, y, lim) == pytest.approx(20.0 / 3.0, rel=1e-9)


def test_first_crossing_time_returns_none_when_never_below():
    lim = LimitSpec(value=80.0)
    t = np.linspace(0.0, 10.0, 51)
    assert first_crossing_time(t, 100.0 - 0.5 * t, lim) is None


def test_first_crossing_time_handles_already_crossed_at_first_sample():
    lim = LimitSpec(value=80.0)
    t = np.array([0.0, 1.0, 2.0])
    y = np.array([79.0, 78.0, 77.0])
    assert first_crossing_time(t, y, lim) == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("times", "values", "match"),
    [
        (np.array([0.0]), np.array([1.0]), "at least two"),
        (np.array([0.0, 1.0]), np.array([1.0]), "same shape"),
        (np.array([0.0, 0.0]), np.array([1.0, 2.0]), "strictly increasing"),
        (np.array([0.0, np.nan]), np.array([1.0, 2.0]), "finite"),
    ],
)
def test_first_crossing_time_input_validation(times, values, match):
    with pytest.raises(ValueError, match=match):
        first_crossing_time(times, values, LimitSpec(value=0.5))


def test_interpolation_error_shrinks_quadratically_with_the_grid():
    # y = 100 exp(-t/10) crosses 80 at t = 10 ln(1.25) = 2.231435513 d.
    exact = 10.0 * np.log(1.25)
    lim = LimitSpec(value=80.0)
    errs = []
    for n in (21, 41, 81):
        t = np.linspace(0.0, 10.0, n)
        errs.append(abs(first_crossing_time(t, 100.0 * np.exp(-t / 10.0), lim) - exact))
    # Halving the step should cut the error by roughly four; allow a factor of
    # two of slack on each ratio because the crossing moves within its cell.
    assert errs[0] / errs[1] > 2.0
    assert errs[1] / errs[2] > 2.0


@settings(max_examples=120, deadline=None)
@given(
    scale=st.floats(min_value=0.05, max_value=20.0, allow_nan=False),
    offset=st.floats(min_value=-50.0, max_value=50.0, allow_nan=False),
    value=st.floats(min_value=60.0, max_value=120.0, allow_nan=False),
)
def test_margin_is_equivariant_under_affine_rescaling(scale, offset, value):
    """Rescaling metric and limit together scales the margin and must not flip it.

    What is guaranteed: the two margins agree to within rounding, relative to
    the magnitudes involved. What is NOT guaranteed is the crossing boolean,
    because it is a strict comparison on a rounded quantity. The boolean is
    therefore only required to agree when the margin sits outside a tolerance
    band scaled to the arithmetic; inside the band the tri-state result is
    ``within_tolerance`` on at least one side and no claim is made.
    """
    base = LimitSpec(value=80.0, atol=0.0)
    m_base = float(margin(value, base))
    tol = 1e-9 * max(1.0, abs(value), abs(base.value))
    scaled_limit = LimitSpec(value=scale * 80.0 + offset, atol=0.0)
    m_scaled = float(margin(scale * value + offset, scaled_limit))
    assert m_scaled == pytest.approx(scale * m_base, abs=scale * tol * 10.0)
    if abs(m_base) > tol * 1e3:
        assert has_crossed(value, base) is has_crossed(
            scale * value + offset, scaled_limit
        )
