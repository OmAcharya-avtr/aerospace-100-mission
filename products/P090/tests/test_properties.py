"""Property-based tests for the algebraic identities this package relies on.

Every identity here is stated as what is actually guaranteed. Several of them
are exact only in exact arithmetic, and those are compared with a tolerance on
both sides rather than for equality; two products of this mission shipped
assertions of exact floating-point identities that a release gate then
falsified, so the distinction is made explicitly in each docstring.

``.hypothesis/`` is gitignored, so a falsifying example found by one run is lost
on a fresh clone. Every counterexample Hypothesis found while this package was
being built is pinned as a literal in ``tests/test_regressions.py``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from tlecheck.benchmark import (
    area_under_curve,
    operating_characteristic,
    per_class,
    recall_at_fpr,
)
from tlecheck.consequence import CLASSES, position_errors_km
from tlecheck.dataset import kozai_mean_motion, unkozai_mean_motion
from tlecheck.elements import checksum, format_tle, parse_tle
from tlecheck.epoch import epoch_julian_date
from tlecheck.history import modified_z, robust_scale, theil_sen, unwrap_degrees
from tlecheck.propagate import propagate, time_grid

FAST = settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

finite_floats = st.floats(
    min_value=-1.0e6, max_value=1.0e6, allow_nan=False, allow_infinity=False
)


@FAST
@given(st.text(alphabet="ABCXYZ.+ ", min_size=1, max_size=20))
def test_checksum_ignores_non_digit_non_minus_characters(text):
    """The checksum rule counts digits and minus signs only."""
    assert checksum(text) == 0


@FAST
@given(
    st.lists(st.integers(min_value=0, max_value=9), min_size=1, max_size=40),
    st.integers(min_value=0, max_value=9),
)
def test_checksum_is_the_digit_sum_modulo_ten(digits, extra):
    """Appending a digit adds its value to the sum, modulo 10. Exact in integers."""
    text = "".join(str(digit) for digit in digits)
    assert checksum(text + str(extra)) == (checksum(text) + extra) % 10


@FAST
@given(st.integers(min_value=0, max_value=60), st.integers(min_value=1, max_value=9))
def test_changing_one_digit_shifts_the_checksum_by_that_much(column, delta):
    """Exact in integer arithmetic: the checksum is linear in each digit."""
    line = "1" + "2" * 67 + "0"
    index = column
    assume(line[index].isdigit())
    original = int(line[index])
    replacement = (original + delta) % 10
    changed = line[:index] + str(replacement) + line[index + 1 :]
    assert checksum(changed) == (checksum(line) + replacement - original) % 10


@FAST
@given(
    st.floats(min_value=0.0001, max_value=179.9999),
    st.floats(min_value=0.0, max_value=359.9999),
    st.floats(min_value=0.0, max_value=0.9),
    st.floats(min_value=0.6, max_value=16.9),
)
def test_format_parse_round_trip_to_format_precision(
    inclination, angle, eccentricity, mean_motion
):
    """Round trip to the precision the text format carries, not to exactness.

    The format holds four decimals on the angles, seven implied decimals on the
    eccentricity and eight on the mean motion, so the round trip is guaranteed
    only to half a count of each field. That is what is asserted.
    """
    line1, line2 = format_tle(
        satnum=12345,
        epoch_year=2026,
        epoch_day=200.5,
        inclination_deg=inclination,
        raan_deg=angle,
        eccentricity=eccentricity,
        arg_perigee_deg=angle,
        mean_anomaly_deg=angle,
        mean_motion_rev_per_day=mean_motion,
    )
    parsed = parse_tle(line1, line2)
    assert parsed.faults == ()
    assert parsed.inclination_deg == pytest.approx(inclination, rel=0.0, abs=5.0e-5)
    assert parsed.raan_deg == pytest.approx(angle, rel=0.0, abs=5.0e-5)
    assert parsed.eccentricity == pytest.approx(eccentricity, rel=0.0, abs=5.0e-8)
    assert parsed.mean_motion_rev_per_day == pytest.approx(
        mean_motion, rel=0.0, abs=5.0e-9
    )


@FAST
@given(st.lists(finite_floats, min_size=1, max_size=30), finite_floats)
def test_robust_scale_is_shift_invariant(values, shift):
    """Exact in exact arithmetic; compared to a relative tolerance here."""
    array = np.array(values, dtype=float)
    assert robust_scale(array + shift) == pytest.approx(
        robust_scale(array), rel=1.0e-9, abs=1.0e-9
    )


@FAST
@given(
    st.lists(finite_floats, min_size=1, max_size=30),
    st.floats(min_value=0.01, max_value=100.0),
)
def test_robust_scale_is_positively_homogeneous(values, factor):
    """``s(c x) = |c| s(x)``, exact in exact arithmetic."""
    array = np.array(values, dtype=float)
    assert robust_scale(array * factor) == pytest.approx(
        factor * robust_scale(array), rel=1.0e-9, abs=1.0e-9
    )


@FAST
@given(st.lists(finite_floats, min_size=1, max_size=30))
def test_robust_scale_is_non_negative(values):
    scale = robust_scale(np.array(values, dtype=float))
    assert scale >= 0.0 or math.isnan(scale)


@FAST
@given(
    st.lists(finite_floats, min_size=3, max_size=30),
    finite_floats,
    st.floats(min_value=0.5, max_value=20.0),
)
def test_modified_z_is_affine_invariant(values, shift, factor):
    """``z`` is unchanged by an affine rescaling, **where the shift is representable**.

    The identity is exact in exact arithmetic and Hypothesis falsified the
    unconditional version of this test immediately. The falsifying case is
    ``values=[0.0, 1.0, 6.663237840215803e-40]`` with ``shift=1.0``: the robust
    scale of that sample is 9.88e-40, and adding 1.0 to every element collapses
    the subnormal element onto 1.0 exactly, so the shifted sample is constant,
    its scale is 0 and ``z`` becomes an infinity where it was 1.01e39.

    So what is guaranteed is the conditional identity: the invariance holds when
    the sample's robust scale is representable beside the shifted magnitude. The
    guard below states that condition, and the falsifying case itself is pinned
    as a literal in ``tests/test_regressions.py`` because ``.hypothesis/`` does
    not survive a fresh clone.
    """
    array = np.array(values, dtype=float)
    scale = robust_scale(array)
    magnitude = 1.0 + abs(shift) + float(np.max(np.abs(array)))
    assume(scale > 1.0e-9 * magnitude)
    candidate = float(array[0]) + 1.0
    first = modified_z(candidate, array)
    second = modified_z(candidate * factor + shift, array * factor + shift)
    if math.isinf(first) or math.isinf(second):
        assert math.isinf(first) and math.isinf(second)
    else:
        assert second == pytest.approx(first, rel=1.0e-6, abs=1.0e-9)


@FAST
@given(
    st.floats(min_value=-100.0, max_value=100.0),
    st.floats(min_value=-100.0, max_value=100.0),
    st.integers(min_value=3, max_value=20),
)
def test_theil_sen_recovers_an_exact_line(slope, intercept, count):
    """Exact in exact arithmetic; a relative tolerance is used on both sides."""
    times = np.arange(float(count))
    values = intercept + slope * times
    estimated_slope, estimated_intercept = theil_sen(times, values)
    assert estimated_slope == pytest.approx(slope, rel=1.0e-9, abs=1.0e-9)
    assert estimated_intercept == pytest.approx(intercept, rel=1.0e-9, abs=1.0e-9)


@FAST
@given(st.lists(st.floats(min_value=0.0, max_value=359.999), min_size=1, max_size=20))
def test_unwrap_preserves_the_angle_modulo_three_sixty(values):
    """Unwrapping adds a multiple of 360 and nothing else."""
    array = np.array(values, dtype=float)
    unwrapped = unwrap_degrees(array)
    difference = unwrapped - array
    assert np.allclose(difference / 360.0, np.round(difference / 360.0), rtol=0.0, atol=1.0e-9)


@FAST
@given(
    st.floats(min_value=0.6, max_value=16.5),
    st.floats(min_value=0.0, max_value=0.8),
    st.floats(min_value=0.0, max_value=179.0),
)
def test_kozai_round_trip(mean_motion_rev_day, eccentricity, inclination_deg):
    """Equation (18) and its fixed-point inverse, to a relative tolerance."""
    kozai = 2.0 * math.pi * mean_motion_rev_day / 1440.0
    inclination = math.radians(inclination_deg)
    brouwer = unkozai_mean_motion(kozai, eccentricity, inclination)
    assert kozai_mean_motion(brouwer, eccentricity, inclination) == pytest.approx(
        kozai, rel=1.0e-12, abs=0.0
    )


@FAST
@given(
    st.floats(min_value=0.0001, max_value=0.2),
    st.floats(min_value=1.0e-6, max_value=1.0e-3),
)
def test_position_error_is_symmetric(epoch_shift, bstar):
    """Equation (11) is a norm of a difference and is symmetric exactly."""
    fields = {
        "satnum": 30000,
        "epoch_year": 2026,
        "epoch_day": 200.5,
        "inclination_deg": 51.6,
        "raan_deg": 120.0,
        "eccentricity": 0.001,
        "arg_perigee_deg": 40.0,
        "mean_anomaly_deg": 200.0,
        "mean_motion_rev_per_day": 15.5,
        "bstar": bstar,
    }
    first = parse_tle(*format_tle(**fields))
    second = parse_tle(*format_tle(**{**fields, "bstar": -bstar}))
    grid = time_grid(epoch_julian_date(first) + epoch_shift, 1.0, 5)
    forward = position_errors_km(propagate(first, grid), propagate(second, grid))
    backward = position_errors_km(propagate(second, grid), propagate(first, grid))
    assert np.allclose(forward, backward, rtol=0.0, atol=0.0)


@FAST
@given(
    st.lists(
        st.tuples(st.floats(min_value=0.0, max_value=1.0), st.booleans()),
        min_size=4,
        max_size=40,
    )
)
def test_operating_characteristic_is_monotone_in_the_threshold(pairs):
    """Lowering the threshold can only flag more, so recall and the rate rise."""
    scores = np.array([score for score, _ in pairs], dtype=float)
    positive = np.array([flag for _, flag in pairs], dtype=bool)
    assume(positive.any() and (~positive).any())
    points = operating_characteristic(scores, positive)
    recalls = [point.recall for point in points]
    rates = [point.false_positive_rate for point in points]
    assert recalls == sorted(recalls)
    assert rates == sorted(rates)


@FAST
@given(
    st.lists(
        st.tuples(st.floats(min_value=0.0, max_value=1.0), st.booleans()),
        min_size=4,
        max_size=40,
    )
)
def test_area_under_curve_is_in_the_unit_interval(pairs):
    scores = np.array([score for score, _ in pairs], dtype=float)
    positive = np.array([flag for _, flag in pairs], dtype=bool)
    assume(positive.any() and (~positive).any())
    area = area_under_curve(scores, positive)
    assert -1.0e-12 <= area <= 1.0 + 1.0e-12


@FAST
@given(
    st.lists(
        st.tuples(st.floats(min_value=0.0, max_value=1.0), st.booleans()),
        min_size=4,
        max_size=40,
    ),
    st.floats(min_value=0.0, max_value=1.0),
)
def test_recall_at_fpr_never_exceeds_its_target(pairs, target):
    scores = np.array([score for score, _ in pairs], dtype=float)
    positive = np.array([flag for _, flag in pairs], dtype=bool)
    assume(positive.any() and (~positive).any())
    point = recall_at_fpr(scores, positive, target)
    assert point.false_positive_rate <= target + 1.0e-12


@FAST
@given(
    st.lists(
        st.tuples(st.sampled_from(CLASSES), st.sampled_from(CLASSES)),
        min_size=1,
        max_size=40,
    )
)
def test_per_class_rates_are_rates(pairs):
    truth = np.array([a for a, _ in pairs], dtype=object)
    predicted = np.array([b for _, b in pairs], dtype=object)
    for score in per_class(truth, predicted).values():
        for value in (score.precision, score.recall):
            assert math.isnan(value) or 0.0 <= value <= 1.0
        assert score.true_positive + score.false_negative == score.support
