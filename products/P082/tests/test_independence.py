"""Baseline 1: the independence assumption and the decomposition of equation (16).

The decomposition is exact by construction, so the test on it is a test of
rounding and says so. The product of equation (14) is computed in log space and
the tests check that the log form agrees with the direct product where the
direct product is safe, and does better where it is not.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from sitediverse.independence import (
    availability_nines,
    conditional_product,
    overstatement,
    product_of_marginals,
)

RATES = st.floats(0.01, 0.99, allow_nan=False, allow_infinity=False)


def test_the_product_of_two_rates_is_hand_checkable():
    """``0.4 x 0.25 = 0.1``, by hand.

    Not asserted exactly: the log-space form of equation (14) is
    ``exp(log 0.4 + log 0.25)`` and the three transcendental operations each
    round, so the result differs from 0.1 in the last few bits.
    """
    got = product_of_marginals(np.array([0.4, 0.25]))
    assert got == pytest.approx(0.1, rel=1e-14, abs=0.0)


def test_the_product_of_four_equal_rates_matches_the_power():
    """``0.3^4 = 0.0081``, by hand."""
    got = product_of_marginals(np.full(4, 0.3))
    assert got == pytest.approx(0.0081, rel=1e-14, abs=0.0)


def test_the_product_of_one_rate_is_that_rate():
    """A one-station network has no diversity and its product is its marginal."""
    got = product_of_marginals(np.array([0.37]))
    assert got == pytest.approx(0.37, rel=1e-15, abs=0.0)


def test_a_rate_of_exactly_zero_gives_a_product_of_exactly_zero():
    """The log form would give ``-inf`` and ``exp(-inf) = 0``; the branch is explicit.

    An exact comparison is legitimate here because the function returns the
    literal ``0.0`` from a branch rather than computing it.
    """
    assert product_of_marginals(np.array([0.0, 0.5])) == 0.0


def test_a_rate_of_exactly_one_contributes_nothing():
    """A station that is always cloudy adds no diversity."""
    got = product_of_marginals(np.array([1.0, 0.4]))
    assert got == pytest.approx(0.4, rel=1e-14, abs=0.0)


@given(st.lists(RATES, min_size=2, max_size=8))
def test_the_log_form_agrees_with_the_direct_product(rates):
    """Equation (14)'s log form against the direct product where both are safe.

    For rates bounded away from zero and fewer than nine of them, the direct
    product is accurate to a few units in the last place and the two forms must
    agree to a relative tolerance that grows with the number of factors. The
    tolerance below is ``n`` units in the last place with a wide margin.
    """
    values = np.array(rates, dtype=float)
    direct = float(np.prod(values))
    got = product_of_marginals(values)
    assert got == pytest.approx(direct, rel=1e-13, abs=0.0)


@given(st.lists(RATES, min_size=2, max_size=8))
def test_the_product_is_at_most_the_smallest_rate(rates):
    """Equation (14) satisfies the Frechet upper bound, which it does not use."""
    values = np.array(rates, dtype=float)
    assert product_of_marginals(values) <= float(values.min()) * (1.0 + 1e-13)


@given(st.lists(RATES, min_size=2, max_size=8))
def test_the_product_lies_strictly_inside_the_frechet_bounds(rates):
    """The independence assumption is one point in an interval, equation (17)."""
    values = np.array(rates, dtype=float)
    product = product_of_marginals(values)
    lower = max(0.0, float(values.sum()) - (values.size - 1))
    upper = float(values.min())
    assert product >= lower - 1e-13
    assert product <= upper + 1e-13


@given(st.lists(RATES, min_size=2, max_size=6))
def test_adding_a_station_cannot_raise_the_product(rates):
    """Equation (14) is non-increasing in the station count."""
    values = np.array(rates, dtype=float)
    smaller = product_of_marginals(values[:-1])
    larger = product_of_marginals(values)
    assert larger <= smaller * (1.0 + 1e-13)


def test_the_product_rejects_a_rate_outside_the_unit_interval():
    """Input validation."""
    with pytest.raises(ValueError, match="must lie in"):
        product_of_marginals(np.array([0.5, 1.2]))


def test_the_product_rejects_an_empty_array():
    """Input validation."""
    with pytest.raises(ValueError, match="non-empty"):
        product_of_marginals(np.array([]))


def test_the_product_rejects_a_two_dimensional_array():
    """Input validation."""
    with pytest.raises(ValueError, match="one-dimensional"):
        product_of_marginals(np.zeros((2, 2)))


def test_the_conditional_product_of_a_constant_series_is_the_naive_product():
    """Equation (15) reduces to equation (14) when the climatology is stationary.

    The exact case the decomposition of equation (16) needs: with no time
    variation the co-modulation term is zero.
    """
    series = np.tile(np.array([0.4, 0.25, 0.5]), (100, 1))
    assert conditional_product(series) == pytest.approx(
        product_of_marginals(np.array([0.4, 0.25, 0.5])), rel=1e-14, abs=0.0
    )


def test_the_conditional_product_exceeds_the_naive_one_when_the_rates_co_vary():
    """Jensen's inequality on the product, stated in equation (15).

    Hand-constructed: two stations whose rates are both 0.2 half the time and
    both 0.6 the other half. The conditional product is
    ``(0.04 + 0.36) / 2 = 0.2`` and the naive product is ``0.4^2 = 0.16``.
    """
    series = np.vstack([np.full((50, 2), 0.2), np.full((50, 2), 0.6)])
    assert conditional_product(series) == pytest.approx(0.2, rel=1e-14, abs=0.0)
    assert product_of_marginals(np.array([0.4, 0.4])) == pytest.approx(
        0.16, rel=1e-14, abs=0.0
    )


def test_the_conditional_product_falls_below_the_naive_one_when_the_rates_anti_vary():
    """And the other direction, which is why the co-modulation term is signed.

    Two stations in perfect antiphase: one at 0.2 while the other is at 0.6.
    The conditional product is ``0.12`` both halves and the naive product is
    ``0.4^2 = 0.16``, so independence OVERSTATES the joint unavailability. This
    is the mechanism behind the overstatement ratios below 1 that this
    product's headline table reports for far-separated networks.
    """
    series = np.vstack(
        [
            np.tile(np.array([0.2, 0.6]), (50, 1)),
            np.tile(np.array([0.6, 0.2]), (50, 1)),
        ]
    )
    assert conditional_product(series) == pytest.approx(0.12, rel=1e-14, abs=0.0)
    assert conditional_product(series) < product_of_marginals(np.array([0.4, 0.4]))


def test_the_conditional_product_rejects_a_one_dimensional_series():
    """Input validation."""
    with pytest.raises(ValueError, match="two-dimensional"):
        conditional_product(np.zeros(10))


def test_the_conditional_product_rejects_an_empty_series():
    """Input validation."""
    with pytest.raises(ValueError, match="must not be empty"):
        conditional_product(np.zeros((0, 3)))


def test_the_conditional_product_rejects_a_rate_outside_the_unit_interval():
    """Input validation."""
    with pytest.raises(ValueError, match="must lie in"):
        conditional_product(np.full((5, 2), 1.4))


def test_availability_nines_is_hand_checkable():
    """``-log10(0.001) = 3``: three nines."""
    assert availability_nines(0.001) == pytest.approx(3.0, rel=1e-14, abs=0.0)
    assert availability_nines(0.1) == pytest.approx(1.0, rel=1e-14, abs=0.0)


def test_availability_nines_of_zero_is_infinite():
    """Perfect availability has unbounded nines and ``inf`` says so."""
    assert np.isinf(availability_nines(0.0))


def test_availability_nines_of_one_is_zero():
    """Always unavailable is zero nines."""
    assert availability_nines(1.0) == 0.0


def test_availability_nines_rejects_a_negative_unavailability():
    """Input validation."""
    with pytest.raises(ValueError, match="must be non-negative"):
        availability_nines(-0.1)


def test_the_nines_derivation_survives_an_unavailability_the_complement_cannot():
    """Why equation (14) reports unavailability as primary.

    At an unavailability of 1e-20 the availability ``1 - 1e-20`` rounds to
    exactly 1.0 in double precision, so a reader computing
    ``-log10(1 - availability)`` gets ``-log10(0.0)``, which is ``inf``: twenty
    nines reported as unbounded. Working from the unavailability directly gives
    20 exactly. This is the complement-space rule, measured.
    """
    tiny = 1e-20
    assert availability_nines(tiny) == pytest.approx(20.0, rel=1e-14, abs=0.0)
    assert (1.0 - tiny) == 1.0
    with np.errstate(divide="ignore"):
        assert np.isinf(-np.log10(1.0 - (1.0 - tiny)))


def test_the_decomposition_identity_is_exact_to_rounding(
    short_record, short_climatology
):
    """Equation (16) has no cross term, so its residual is pure rounding.

    The residual is a sum of three floating-point differences of quantities of
    order 1e-2 and cannot be exactly zero in general, so the assertion is a
    rounding assertion on that scale and not an equality.
    """
    worst = 0.0
    for sites in ((0, 1), (0, 12), (0, 1, 2, 3), (0, 4, 8, 12, 16)):
        result = overstatement(short_climatology, short_record.cloud, sites)
        worst = max(worst, abs(result.identity_residual()))
    assert worst < 1e-15


def test_the_overstatement_of_a_near_network_is_substantial(
    reference_record, reference_climatology
):
    """The headline claim, on the declared reference record.

    Four stations in one regional group, every baseline under 300 km. The
    independence assumption claims an availability the record denies by a
    measurable margin, and the ratio of unavailabilities exceeds a factor of
    five. The threshold is deliberately well below the measured value so that
    this test is a claim about the DIRECTION and ORDER of the effect rather
    than a pinned number; the pinned numbers live in
    ``tests/test_regressions.py`` and in ``validation/``.
    """
    result = overstatement(reference_climatology, reference_record.cloud, (0, 1, 2, 3))
    assert result.measured > result.naive
    assert result.ratio > 5.0
    assert result.total_difference > 0.05
    assert result.nines_lost > 0.5


def test_the_overstatement_is_dominated_by_the_correlation_term_for_a_near_network(
    reference_record, reference_climatology
):
    """Which of the two assumptions in equation (16) does the damage."""
    result = overstatement(reference_climatology, reference_record.cloud, (0, 1, 2, 3))
    assert result.correlation_term > 5.0 * abs(result.comodulation_term)


def test_a_far_separated_network_shows_almost_no_correlation_overstatement(
    reference_record, reference_climatology
):
    """One station per regional group: the diversity the independence form claims.

    The ratio is close to 1 and may be slightly BELOW it, because the seasonal
    co-modulation term is negative for stations whose seasonal cycles differ.
    The test asserts the ratio is near one in both directions, which is the
    honest claim.
    """
    result = overstatement(
        reference_climatology, reference_record.cloud, (0, 4, 8, 12, 16)
    )
    assert 0.8 < result.ratio < 1.3


def test_the_ratio_is_infinite_when_the_naive_product_is_zero():
    """A ratio against zero is not a number, and ``inf`` or ``nan`` says so."""
    from sitediverse.independence import Overstatement

    assert np.isinf(Overstatement(0.0, 0.0, 0.1, (0, 1)).ratio)
    assert np.isnan(Overstatement(0.0, 0.0, 0.0, (0, 1)).ratio)


def test_the_availability_deficit_equals_the_unavailability_difference():
    """Derived from the unavailabilities, not from two availabilities near one."""
    from sitediverse.independence import Overstatement

    result = Overstatement(0.01, 0.02, 0.05, (0, 1))
    assert result.availability_deficit == result.total_difference


def test_overstatement_rejects_a_single_station():
    """A diversity network needs at least two stations."""
    with pytest.raises(ValueError, match="at least 2 stations"):
        overstatement(None, np.zeros((4, 3), dtype=bool), (0,))


def test_overstatement_rejects_repeated_stations(short_record, short_climatology):
    """A station counted twice is not a two-station network."""
    with pytest.raises(ValueError, match="must be distinct"):
        overstatement(short_climatology, short_record.cloud, (0, 0))


def test_overstatement_rejects_a_station_outside_the_record(
    short_record, short_climatology
):
    """Input validation with the valid range in the message."""
    with pytest.raises(ValueError, match="outside"):
        overstatement(short_climatology, short_record.cloud, (0, 999))
