"""Property-based tests, where an algebraic identity is genuinely at stake.

Every property here is an identity in exact arithmetic, so every assertion
carries a tolerance on both sides and none of them asserts bit-exact equality.
Where a property is only true for physically possible inputs -- rail power is
positive, timestamps do not go backwards -- the strategy says so rather than the
assertion tolerating the difference.

Any falsifying example Hypothesis finds here is pinned as a named literal
regression test in ``tests/test_regressions.py``, because ``.hypothesis/`` is
gitignored and a counterexample found by one run is lost on a fresh clone.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from powerenvelope.bounds import (
    EPS,
    Bracket,
    extreme_value_bracket,
    riemann_bracket,
    slack_from_conditioning,
)
from powerenvelope.decomposition import decompose
from powerenvelope.energy import (
    GAP_TOLERANCE_S,
    clip_series,
    merge_intervals,
    trapezoid_energy_mj,
)
from powerenvelope.predictor import target_quantisation_sd
from powerenvelope.schema import (
    DECISION_COLUMNS,
    MONITOR_COLUMNS,
    POWER_COLUMNS,
    SchemaError,
    validate_decision_header,
    validate_monitor_header,
    validate_power_header,
)

#: Rail power is positive and of the order of a watt or two on this board. The
#: strategy stays in that range: a property about an energy integral of a rail
#: trace does not need to hold for a power of 1e300.
POWER = st.floats(min_value=1.0, max_value=1e5, allow_nan=False, allow_infinity=False)

#: Sample spacings. Strictly positive, so the grid is strictly increasing.
SPACING = st.floats(min_value=1e-4, max_value=1.0, allow_nan=False, allow_infinity=False)

#: Positive mixing coefficients. Both positive, because the identity being
#: tested is linearity of an integral over POSITIVE integrands: allowing a
#: negative coefficient would test catastrophic cancellation instead, which is
#: a different question and is covered by the bracket checks.
COEFFICIENT = st.floats(min_value=1e-3, max_value=1e3, allow_nan=False, allow_infinity=False)


@st.composite
def series(draw, min_size: int = 2, max_size: int = 60):
    """A strictly increasing timestamp grid with a positive power at each point."""
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    spacings = draw(st.lists(SPACING, min_size=size - 1, max_size=size - 1))
    start = draw(st.floats(min_value=0.0, max_value=1e5, allow_nan=False))
    t = np.concatenate(([start], start + np.cumsum(spacings)))
    power = np.asarray(draw(st.lists(POWER, min_size=size, max_size=size)))
    return t, power


@settings(max_examples=200, deadline=None)
@given(series(), COEFFICIENT, COEFFICIENT, st.data())
def test_integral_is_linear_in_power(grid, alpha, beta, data):
    t, first = grid
    second = np.asarray(
        data.draw(st.lists(POWER, min_size=t.size, max_size=t.size))
    )
    mixed = alpha * first + beta * second
    combined = alpha * trapezoid_energy_mj(t, first) + beta * trapezoid_energy_mj(t, second)
    assert trapezoid_energy_mj(t, mixed) == pytest.approx(combined, rel=1e-9, abs=0.0)


@settings(max_examples=200, deadline=None)
@given(series(), COEFFICIENT)
def test_integral_is_homogeneous_in_power(grid, alpha):
    t, power = grid
    assert trapezoid_energy_mj(t, alpha * power) == pytest.approx(
        alpha * trapezoid_energy_mj(t, power), rel=1e-12, abs=0.0
    )


@settings(max_examples=200, deadline=None)
@given(series(), st.floats(min_value=-1e4, max_value=1e4, allow_nan=False))
def test_integral_time_shift_error_is_bounded_by_the_timestamp_resolution(grid, shift):
    """A time shift changes the integral, by an amount this bound explains.

    The naive property -- that the integral is invariant under a shift of the
    timestamps -- is FALSE in floating point, and Hypothesis falsified it on the
    first run of this file at ``t = [0.0, 1e-4]``, ``shift = 2048.0``, where the
    relative change is 2.0e-9. The counterexample is pinned as a literal
    regression test in ``tests/test_regressions.py``.

    The reason matters for this product. A timestamp of magnitude ``T`` is
    represented to about ``EPS * T``, so a spacing ``dt`` on a grid offset by
    ``T`` carries a relative error of up to ``EPS * T / dt``, and the integral
    inherits it. The campaign's ``t_mono`` is the board's uptime, of order 1e5
    to 1e6 seconds, with 0.05 s spacings, which puts the floor on every energy
    figure in this repository at about 1e-8 relative. That is ten orders of
    magnitude below the 40 mW current-quantisation floor, so it changes no
    reported figure; it is recorded in README.md under Limitations because it is
    a real property of the measurement and not of this code.
    """
    t, power = grid
    shifted = t + shift
    assume(np.all(np.diff(shifted) > 0.0))
    bound = 4.0 * EPS * float(np.max(np.abs(shifted))) / float(np.min(np.diff(t)))
    assert trapezoid_energy_mj(shifted, power) == pytest.approx(
        trapezoid_energy_mj(t, power), rel=max(bound, 1e-12), abs=0.0
    )


@settings(max_examples=300, deadline=None)
@given(series(min_size=3), st.floats(min_value=0.05, max_value=0.95, allow_nan=False))
def test_integral_is_additive_over_adjacent_windows(grid, split):
    t, power = grid
    low, high = float(t[0]), float(t[-1])
    middle = low + split * (high - low)
    assume(middle > low and middle < high)
    whole = trapezoid_energy_mj(*clip_series(t, power, low, high))
    left_t, left_p = clip_series(t, power, low, middle)
    right_t, right_p = clip_series(t, power, middle, high)
    assume(left_t.size >= 2 and right_t.size >= 2)
    parts = trapezoid_energy_mj(left_t, left_p) + trapezoid_energy_mj(right_t, right_p)
    assert parts == pytest.approx(whole, rel=1e-9, abs=0.0)


@settings(max_examples=200, deadline=None)
@given(series())
def test_trapezoid_lies_inside_both_independent_brackets(grid):
    t, power = grid
    value = trapezoid_energy_mj(t, power)
    assert riemann_bracket(t, power).contains(value)
    assert extreme_value_bracket(t, power).contains(value)


@settings(max_examples=300, deadline=None)
@given(
    st.lists(
        st.tuples(
            st.floats(min_value=0.0, max_value=1e3, allow_nan=False),
            st.floats(min_value=0.0, max_value=10.0, allow_nan=False),
        ),
        min_size=1,
        max_size=40,
    )
)
def test_merged_windows_are_sorted_and_separated_by_more_than_the_tolerance(intervals):
    starts = np.asarray([start for start, _width in intervals])
    ends = starts + np.asarray([width for _start, width in intervals])
    lo, hi = merge_intervals(starts, ends, GAP_TOLERANCE_S)
    assert np.all(np.diff(lo) > 0.0) or lo.size == 1
    assert np.all(hi >= lo)
    if lo.size > 1:
        assert np.all(lo[1:] - hi[:-1] > GAP_TOLERANCE_S)
    # Every input interval lies inside exactly one merged window.
    for start, end in zip(starts, ends, strict=True):
        covered = np.any((lo <= start) & (hi >= end))
        assert covered


@settings(max_examples=200, deadline=None)
@given(
    st.lists(
        st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        min_size=0,
        max_size=40,
    )
)
def test_slack_is_non_negative_and_grows_with_magnitude(terms):
    array = np.asarray(terms, dtype=np.float64)
    slack = slack_from_conditioning(array)
    assert slack >= 0.0
    bigger = slack_from_conditioning(array * 2.0)
    assert bigger >= slack - 1e-300


@settings(max_examples=200, deadline=None)
@given(
    st.floats(min_value=-1e6, max_value=1e6, allow_nan=False),
    st.floats(min_value=0.0, max_value=1e6, allow_nan=False),
    st.floats(min_value=0.0, max_value=1e3, allow_nan=False),
    st.floats(min_value=-1e7, max_value=1e7, allow_nan=False),
)
def test_bracket_excess_is_zero_exactly_when_it_contains(low, width, slack, value):
    bracket = Bracket(low=low, high=low + width, slack=slack, source="property")
    assert (bracket.excess(value) == 0.0) == bracket.contains(value)


@settings(max_examples=200, deadline=None)
@given(
    st.floats(min_value=1e-6, max_value=1e4, allow_nan=False),
    st.integers(min_value=1, max_value=10**6),
    st.integers(min_value=1, max_value=10**6),
)
def test_quantisation_noise_falls_with_more_samples(step, first, second):
    low, high = min(first, second), max(first, second)
    assert float(target_quantisation_sd(step, high)) <= float(
        target_quantisation_sd(step, low)
    ) * (1.0 + 1e-12)


@settings(max_examples=120, deadline=None)
@given(
    st.sampled_from(POWER_COLUMNS),
    st.sampled_from(POWER_COLUMNS),
)
def test_loader_rejects_every_power_header_with_a_renamed_column(target, replacement):
    assume(target != replacement)
    header = tuple(replacement if name == target else name for name in POWER_COLUMNS)
    assume(header != POWER_COLUMNS)
    with pytest.raises(SchemaError):
        validate_power_header(header)


@settings(max_examples=120, deadline=None)
@given(st.integers(min_value=0, max_value=len(MONITOR_COLUMNS) - 1))
def test_loader_rejects_every_monitor_header_with_one_column_removed(index):
    header = tuple(name for position, name in enumerate(MONITOR_COLUMNS) if position != index)
    with pytest.raises(SchemaError):
        validate_monitor_header(header)


@settings(max_examples=120, deadline=None)
@given(st.permutations(list(DECISION_COLUMNS)))
def test_loader_rejects_every_decision_header_permutation_but_the_right_one(order):
    header = tuple(order)
    if header == DECISION_COLUMNS:
        validate_decision_header(header)
        return
    with pytest.raises(SchemaError, match="wrong order"):
        validate_decision_header(header)


@settings(max_examples=40, deadline=None)
@given(
    st.sampled_from(
        [
            "w48_d010",
            "w48_d025",
            "w48_d050",
            "w48_d075",
            "w48_d100",
            "w64_d010",
            "w64_d025",
            "w64_d050",
            "w64_d075",
            "w64_d100",
            "w96_d010",
            "w96_d025",
            "w96_d050",
            "w96_d075",
            "w96_d100",
            "w64_d050_rep2",
            "w64_d050_rep3",
            "w64_d050_rep4",
            "w64_d050_permissive",
        ]
    ),
    st.sampled_from(["measured-active", "nominal-rail"]),
)
def test_decomposition_factors_multiply_to_the_factor(campaign, label, variant):
    result = decompose(campaign.require(label), variant=variant)
    assert result.product_of_named_factors() == pytest.approx(
        result.factor, rel=1e-12, abs=0.0
    )
    assert abs(result.closure_error) < 1e-12
