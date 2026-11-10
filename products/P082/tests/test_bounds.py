"""Distribution-free bounds, and the conditioning rule of equation (19).

Two things are tested that nothing else in this package tests. First, that the
bounds hold for every route by which this package computes a joint
unavailability -- which is a check against mathematics the implementations do
not use. Second, that the slack of equation (19) FOLLOWS the conditioning
rather than being a constant, which is the defect class a predecessor in this
mission shipped: a slack floored at an absolute constant made a bracket of
width 4.5e-23 vacuous.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from sitediverse.bounds import (
    MACHINE_EPSILON,
    SLACK_FACTOR,
    bonferroni_bracket,
    bound_condition_number,
    bound_slack,
    frechet_hoeffding,
)
from sitediverse.empirical import joint_rate, pair_clear_matrix
from sitediverse.independence import overstatement, product_of_marginals

RATES = st.floats(0.01, 0.99, allow_nan=False, allow_infinity=False)


def test_the_frechet_bounds_are_hand_checkable():
    """Two stations at 0.6 and 0.7: lower ``0.6 + 0.7 - 1 = 0.3``, upper ``0.6``."""
    bounds = frechet_hoeffding(np.array([0.6, 0.7]))
    assert bounds.lower == pytest.approx(0.3, rel=1e-14, abs=0.0)
    assert bounds.upper == pytest.approx(0.6, rel=1e-15, abs=0.0)
    assert bounds.width == pytest.approx(0.3, rel=1e-14, abs=0.0)


def test_the_lower_bound_clips_at_zero_when_the_union_bound_is_vacuous():
    """Two stations at 0.3 and 0.4: ``0.3 + 0.4 - 1 < 0``, so the lower bound is 0."""
    bounds = frechet_hoeffding(np.array([0.3, 0.4]))
    assert bounds.lower == 0.0
    assert bounds.upper == pytest.approx(0.3, rel=1e-15, abs=0.0)


def test_three_stations_give_the_declared_lower_bound():
    """``0.9 + 0.9 + 0.9 - 2 = 0.7``, by hand."""
    bounds = frechet_hoeffding(np.full(3, 0.9))
    assert bounds.lower == pytest.approx(0.7, rel=1e-13, abs=0.0)


@given(st.lists(RATES, min_size=2, max_size=8))
def test_the_independence_product_lies_inside_the_frechet_bounds(rates):
    """Equation (14) against equation (17), which it does not use."""
    values = np.array(rates, dtype=float)
    bounds = frechet_hoeffding(values)
    assert bounds.contains(product_of_marginals(values))


@given(st.lists(RATES, min_size=2, max_size=8))
def test_the_bounds_are_ordered_and_both_lie_in_the_unit_interval(rates):
    """Structural properties of equation (17)."""
    bounds = frechet_hoeffding(np.array(rates, dtype=float))
    assert 0.0 <= bounds.lower <= bounds.upper <= 1.0


@given(st.lists(RATES, min_size=2, max_size=8))
def test_the_bounds_are_invariant_under_reordering_the_stations(rates):
    """Equation (17) is a function of the multiset of marginals.

    ``min`` is order-invariant exactly; the sum is not bit-invariant under
    reordering, so the lower bound is compared with a rounding tolerance.
    """
    values = np.array(rates, dtype=float)
    forward = frechet_hoeffding(values)
    backward = frechet_hoeffding(values[::-1])
    assert forward.upper == backward.upper
    assert forward.lower == pytest.approx(backward.lower, rel=1e-13, abs=1e-18)


def test_the_bounds_hold_for_every_route_on_the_reference_record(
    reference_record, reference_climatology
):
    """The bound check this product ships, over five declared networks.

    Three routes to a joint unavailability -- the product of marginals, the
    epoch-wise conditional product and the observed joint rate -- against a
    bound none of them uses. Any one of the three could be wrong in a way the
    other two share, and the bound would still catch it.
    """
    networks = ((0, 1), (0, 12), (0, 1, 2, 3), (0, 1, 2, 4, 5, 6), (0, 4, 8, 12, 16))
    for sites in networks:
        rates = reference_climatology.long_term_rate[list(sites)]
        bounds = frechet_hoeffding(rates)
        result = overstatement(reference_climatology, reference_record.cloud, sites)
        for value in (result.naive, result.conditional, result.measured):
            assert bounds.contains(value), (sites, value, bounds)


def test_the_bounds_hold_for_every_pair_of_the_declared_network(
    reference_record, reference_climatology, all_pairs
):
    """All 190 pairs, every route. The widest check in this file."""
    violations = 0
    for site_i, site_j in all_pairs:
        rates = reference_climatology.long_term_rate[[site_i, site_j]]
        bounds = frechet_hoeffding(rates)
        measured = joint_rate(reference_record.cloud, (site_i, site_j))
        if not bounds.contains(measured):
            violations += 1
        if not bounds.contains(product_of_marginals(rates)):
            violations += 1
    assert violations == 0


def test_some_pair_has_a_strictly_positive_lower_bound(reference_climatology, all_pairs):
    """The lower bound of equation (17) is not vacuous everywhere on this network.

    A check against a bound that is identically zero is worth little, so the
    test establishes that the declared network contains pairs whose marginals
    sum above one and for which the lower bound therefore bites.
    """
    positive = [
        pair
        for pair in all_pairs
        if frechet_hoeffding(reference_climatology.long_term_rate[list(pair)]).lower
        > 0.0
    ]
    assert len(positive) > 0


def test_the_relative_slack_grows_without_limit_as_the_bound_expression_cancels():
    """Equation (19): the RELATIVE slack follows the conditioning.

    Two stations whose marginals sum to ``1 + delta`` give a lower bound of
    ``delta``, and the relative condition number of that expression is about
    ``1 / delta``, which diverges as ``delta -> 0``. The ratio of slack to bound
    value must grow by the same factor the conditioning does, over six decades
    of ``delta``.
    """
    ratios = []
    for delta in (1e-2, 1e-4, 1e-6, 1e-8):
        rates = np.array([0.5, 0.5 + delta])
        ratios.append(bound_slack(rates, delta) / delta)
    for earlier, later in zip(ratios[:-1], ratios[1:], strict=True):
        assert later > 50.0 * earlier


def test_the_condition_number_tracks_the_reciprocal_of_the_bound():
    """``kappa = sum|q| / |f(q)|``, checked against its own definition."""
    for delta in (1e-2, 1e-5, 1e-9):
        rates = np.array([0.5, 0.5 + delta])
        expected = float(np.sum(rates)) / delta
        assert bound_condition_number(rates, delta) == pytest.approx(
            expected, rel=1e-12, abs=0.0
        )


def test_the_slack_is_not_a_relative_slack_in_disguise():
    """The error this product made during its own build, pinned.

    The first version of equation (19) computed ``eps * kappa * |f|``, which is
    algebraically ``eps * sum|q|`` and therefore independent of ``f``
    altogether: it looked conditioned and was not. The slack must depend on the
    SUMMANDS and change only weakly with the bound value, so an implementation
    that multiplied by ``|f|`` would make the slack vanish as the bound does --
    the opposite of what is wanted.
    """
    rates = np.array([0.5, 0.5000001])
    tiny = bound_slack(rates, 1e-7)
    assert tiny > 1e-17
    assert tiny == pytest.approx(
        SLACK_FACTOR * MACHINE_EPSILON * (float(np.sum(rates)) + 1e-7),
        rel=1e-12,
        abs=0.0,
    )


def test_the_slack_at_a_well_conditioned_bound_is_a_few_units_of_rounding():
    """And at the other end it is tight, which is what makes the check a check."""
    rates = np.array([0.9, 0.9])
    slack = bound_slack(rates, 0.8)
    assert slack < SLACK_FACTOR * MACHINE_EPSILON * 10.0
    assert slack > 0.0


def test_the_slack_is_never_zero():
    """A zero slack would make the comparison a bit-exact comparison."""
    for rates, value in (
        (np.array([0.5, 0.5]), 0.0),
        (np.array([0.9, 0.95]), 0.85),
        (np.array([0.1, 0.1]), -0.8),
    ):
        assert bound_slack(rates, value) > 0.0


def test_an_ill_conditioned_lower_bound_is_reported_as_uninformative():
    """The consequence of equation (19) the implementation must carry.

    Where the bound value falls below its own slack the check on that side
    admits anything, and ``lower_informative`` says so rather than letting a
    report claim a pass. Constructed at a ``delta`` of 1e-17, which is below the
    rounding scale of a sum of two numbers of order 0.5.
    """
    well = frechet_hoeffding(np.array([0.95, 0.95]))
    assert well.lower_informative
    assert well.lower > 0.0
    rates = np.array([0.5, 0.5 + 1e-17])
    ill = frechet_hoeffding(rates)
    assert not ill.lower_informative


def test_the_upper_bound_is_informative_whenever_a_station_is_ever_clear():
    """``min q_i < 1`` excludes the comonotone extreme, so the side bites."""
    assert frechet_hoeffding(np.array([0.3, 0.4])).upper_informative
    assert not frechet_hoeffding(np.array([1.0, 1.0])).upper_informative


def test_the_condition_number_is_infinite_where_the_bound_vanishes_identically():
    """``sum q - (n - 1) == 0`` has no finite relative conditioning."""
    bounds = frechet_hoeffding(np.array([0.5, 0.5]))
    assert np.isinf(bounds.condition_number)


def test_a_violation_is_reported_as_a_magnitude_and_not_a_boolean():
    """So that a report can say how badly a bound failed."""
    bounds = frechet_hoeffding(np.array([0.6, 0.7]))
    assert bounds.violation(0.45) == 0.0
    assert bounds.violation(0.2) == pytest.approx(0.1, rel=1e-6, abs=1e-12)
    assert bounds.violation(0.9) == pytest.approx(0.3, rel=1e-6, abs=1e-12)


def test_frechet_rejects_a_single_marginal():
    """A joint bound needs at least two events."""
    with pytest.raises(ValueError, match="at least 2 marginals"):
        frechet_hoeffding(np.array([0.5]))


def test_frechet_rejects_a_rate_outside_the_unit_interval():
    """Input validation."""
    with pytest.raises(ValueError, match="must lie in"):
        frechet_hoeffding(np.array([0.5, -0.1]))


def test_frechet_rejects_a_two_dimensional_input():
    """Input validation."""
    with pytest.raises(ValueError, match="one-dimensional"):
        frechet_hoeffding(np.zeros((2, 2)))


def test_bound_slack_rejects_an_empty_rate_array():
    """Input validation."""
    with pytest.raises(ValueError, match="non-empty"):
        bound_slack(np.array([]), 0.1)


def test_the_bonferroni_upper_bound_is_an_exact_identity_for_two_stations(
    reference_record, reference_climatology
):
    """For ``n = 2`` inclusion-exclusion is exact, so equation (18) must close.

    ``P(both cloudy) = 1 - P(i clear) - P(j clear) + P(both clear)`` exactly.
    The right-hand side is computed from CLEAR-rate statistics and the
    left-hand side by counting cloudy epochs, so the two are different
    computations of the same number and the residual is rounding. This is the
    sharpest independent-route check in this package.
    """
    clear = pair_clear_matrix(reference_record.cloud)
    worst = 0.0
    for site_i, site_j in ((0, 1), (0, 12), (4, 5), (3, 19), (8, 16)):
        rates = reference_climatology.long_term_rate[[site_i, site_j]]
        bracket = bonferroni_bracket(
            rates, clear[np.ix_([site_i, site_j], [site_i, site_j])]
        )
        measured = joint_rate(reference_record.cloud, (site_i, site_j))
        worst = max(worst, abs(bracket.upper - measured))
    assert worst < 1e-14


def test_the_bonferroni_bracket_contains_the_measured_rate_for_larger_networks(
    reference_record, reference_climatology
):
    """A bound rather than an identity for ``n > 2``, and it must hold."""
    clear = pair_clear_matrix(reference_record.cloud)
    for sites in ((0, 1, 2), (0, 1, 2, 3), (0, 1, 2, 4, 5, 6)):
        rates = reference_climatology.long_term_rate[list(sites)]
        bracket = bonferroni_bracket(rates, clear[np.ix_(sites, sites)])
        measured = joint_rate(reference_record.cloud, sites)
        assert bracket.contains(measured)


def test_the_bonferroni_bracket_is_vacuous_for_a_large_network_and_says_so(
    reference_record, reference_climatology
):
    """Reported rather than hidden: the bracket leaves the unit interval.

    With six stations at cloud rates near 0.3 the sum of the clear rates
    exceeds 4, so the lower bound ``1 - sum u_i`` is below -3 and the bracket
    carries no information. A check that is vacuous is worth knowing about and
    the CLI prints it; this test pins the fact.
    """
    sites = (0, 1, 2, 4, 5, 6)
    clear = pair_clear_matrix(reference_record.cloud)
    rates = reference_climatology.long_term_rate[list(sites)]
    bracket = bonferroni_bracket(rates, clear[np.ix_(sites, sites)])
    assert bracket.lower < 0.0
    assert bracket.upper > 1.0


def test_the_bonferroni_lower_bound_matches_the_frechet_one_before_clipping():
    """Equation (18)'s lower bound IS equation (17)'s, which is worth stating."""
    rates = np.array([0.9, 0.95, 0.92])
    clear = np.zeros((3, 3))
    bracket = bonferroni_bracket(rates, clear)
    expected = 1.0 - float(np.sum(1.0 - rates))
    assert bracket.lower == pytest.approx(expected, rel=1e-13, abs=1e-18)
    assert bracket.lower == pytest.approx(
        frechet_hoeffding(rates).lower, rel=1e-13, abs=1e-18
    )


def test_bonferroni_rejects_a_mismatched_pair_matrix():
    """Input validation."""
    with pytest.raises(ValueError, match="pair_clear must be"):
        bonferroni_bracket(np.array([0.5, 0.5]), np.eye(3))


def test_bonferroni_rejects_a_pair_probability_outside_the_unit_interval():
    """Input validation."""
    with pytest.raises(ValueError, match="pairwise clear probability"):
        bonferroni_bracket(np.array([0.5, 0.5]), np.full((2, 2), 1.5))


def test_bonferroni_rejects_a_single_marginal():
    """Input validation."""
    with pytest.raises(ValueError, match="at least 2 marginals"):
        bonferroni_bracket(np.array([0.5]), np.eye(1))
