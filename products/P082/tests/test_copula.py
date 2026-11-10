"""The copula: equations (7) to (11), and the cancellation they exist to avoid.

The hardest claim here is the one in equation (9): the correlation excess is
computed as an integral of a positive integrand, so it keeps its relative
precision where the subtraction ``Phi_2 - Phi(h)Phi(k)`` loses all of it. That
claim is tested directly, at correlations small enough for the subtraction to
fail, and the failure of the subtraction is measured rather than asserted.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from scipy.stats import multivariate_normal, norm

from sitediverse.copula import (
    EPOCH_BINS,
    NODES,
    RHO_MAX,
    RHO_MIN,
    SEGMENTS,
    bivariate_density,
    bivariate_orthant,
    correlation_excess,
    epoch_nodes,
    implied_correlation,
    network_joint_monte_carlo,
    pair_excess_exact,
    pair_joint_exact,
)

LEVELS = st.floats(-3.2, 1.0, allow_nan=False, allow_infinity=False)
CORRELATIONS = st.floats(-0.9, 0.98, allow_nan=False, allow_infinity=False)


def test_the_declared_quadrature_is_the_one_documented():
    """32 nodes over 4 segments, as equation (9) says."""
    assert NODES == 32
    assert SEGMENTS == 4


def test_at_zero_correlation_the_orthant_is_the_product_of_the_margins():
    """``Phi_2(h, k; 0) = Phi(h) Phi(k)``, which is baseline 1 for a pair.

    The excess at ``rho = 0`` is exactly zero: the integral of equation (9) is
    over an interval of zero length, which the quadrature evaluates as a sum of
    terms each multiplied by a half-width of exactly 0.0. So this one is an
    exact comparison and the docstring says why.
    """
    for h, k in ((-1.0, -1.0), (-2.5, 0.5), (0.0, 0.0)):
        assert float(correlation_excess(h, k, 0.0)) == 0.0
        assert float(bivariate_orthant(h, k, 0.0)) == pytest.approx(
            float(norm.cdf(h) * norm.cdf(k)), rel=1e-15, abs=0.0
        )


def test_the_orthant_agrees_with_scipys_independent_implementation():
    """Equation (9) against SciPy's Genz-algorithm multivariate normal CDF.

    This is a cross-check between two implementations of the same quantity by
    DIFFERENT mathematics: equation (9) integrates the Plackett derivative
    identity by Gauss-Legendre quadrature, and SciPy uses the Genz lattice
    method on the integral itself. The agreement is absolute, at the 1e-16
    level, which is below SciPy's own stated accuracy; the relative agreement
    is limited by SciPy and is reported here rather than asserted tightly.
    """
    rng = np.random.default_rng(777)
    levels_h = -rng.uniform(0.1, 3.0, 150)
    levels_k = -rng.uniform(0.1, 3.0, 150)
    rho = rng.uniform(-0.9, 0.98, 150)
    mine = bivariate_orthant(levels_h, levels_k, rho)
    theirs = np.array(
        [
            multivariate_normal.cdf([h, k], mean=[0.0, 0.0], cov=[[1.0, r], [r, 1.0]])
            for h, k, r in zip(levels_h, levels_k, rho, strict=True)
        ]
    )
    assert float(np.max(np.abs(mine - theirs))) < 1e-14


def test_at_perfect_positive_correlation_the_orthant_tends_to_the_minimum():
    """The comonotone limit is the Frechet upper bound ``min(q_i, q_j)``.

    Approached, not attained: the quadrature stops at ``RHO_MAX`` because the
    integrand has an integrable singularity at ``|rho| = 1``. The tolerance is
    the distance the declared bracket end leaves.
    """
    h, k = -1.0, -1.5
    got = float(bivariate_orthant(h, k, RHO_MAX))
    expected = float(min(norm.cdf(h), norm.cdf(k)))
    assert got < expected
    assert got == pytest.approx(expected, rel=0.05, abs=0.0)


def test_at_perfect_negative_correlation_the_orthant_tends_to_the_lower_bound():
    """The countermonotone limit is ``max(0, q_i + q_j - 1)``."""
    h, k = -1.0, -1.5
    got = float(bivariate_orthant(h, k, RHO_MIN))
    expected = max(0.0, float(norm.cdf(h) + norm.cdf(k) - 1.0))
    assert got >= expected - 1e-12
    assert got < 0.05


@given(LEVELS, LEVELS, CORRELATIONS)
def test_the_orthant_is_a_probability(h, k, rho):
    """Equation (8) returns a value in ``[0, 1]`` for every admissible input."""
    got = float(bivariate_orthant(h, k, rho))
    assert -1e-15 <= got <= 1.0 + 1e-15


@given(LEVELS, LEVELS, CORRELATIONS)
def test_the_orthant_lies_inside_the_frechet_bounds(h, k, rho):
    """Equation (8) against equation (17), which it does not use.

    A bound check, not an agreement check: the Frechet bounds come from the
    marginals alone and the orthant comes from a quadrature on a density. The
    slack is a rounding slack on quantities of order one.
    """
    left = float(norm.cdf(h))
    right = float(norm.cdf(k))
    got = float(bivariate_orthant(h, k, rho))
    assert got <= min(left, right) + 1e-13
    assert got >= max(0.0, left + right - 1.0) - 1e-13


@given(LEVELS, LEVELS)
def test_the_orthant_is_monotone_increasing_in_the_correlation(h, k):
    """The derivative is the density of equation (9), which is strictly positive."""
    grid = np.linspace(-0.9, 0.95, 25)
    values = bivariate_orthant(np.full(25, h), np.full(25, k), grid)
    assert np.all(np.diff(values) > -1e-15)


@given(LEVELS, LEVELS, CORRELATIONS)
def test_the_orthant_is_symmetric_in_its_two_levels(h, k, rho):
    """``Phi_2(h, k; rho) == Phi_2(k, h; rho)`` to rounding, not bit-exactly.

    The quadratic form of equation (9) is symmetric in ``h`` and ``k`` in exact
    arithmetic; the two evaluations round independently.
    """
    forward = float(bivariate_orthant(h, k, rho))
    backward = float(bivariate_orthant(k, h, rho))
    assert forward == pytest.approx(backward, rel=1e-13, abs=1e-18)


@given(CORRELATIONS)
def test_the_density_integrates_to_the_orthant_difference(rho):
    """A hand-rolled trapezoid on the same integrand reproduces the quadrature.

    An independent numerical route to the SAME integral, at much lower order.
    It catches a sign error or a missing Jacobian, which the high-order
    quadrature could hide behind its own accuracy. The tolerance is the
    trapezoid's own truncation error at 20001 points.
    """
    h, k = -1.2, -0.8
    grid = np.linspace(0.0, rho, 20001)
    integrand = bivariate_density(h, k, grid)
    trapezoid = float(np.trapezoid(integrand, grid))
    assert trapezoid == pytest.approx(
        float(correlation_excess(h, k, rho)), rel=1e-7, abs=1e-15
    )


def test_the_excess_keeps_relative_precision_where_the_subtraction_loses_it():
    """Equation (9)'s whole purpose, measured rather than asserted.

    At ``rho = 1e-16`` the excess is of order 1e-21 while both probabilities are
    of order 2e-3, so the subtraction has about five significant digits of room
    and needs twenty. The integral form is correct to full relative precision,
    which is checked against the first-order expansion ``rho phi_2(h, k; 0)``
    -- an independent closed form valid to ``O(rho^3)``.
    """
    h = k = -3.0
    exact_first_order = 1e-16 * float(bivariate_density(h, k, 0.0))
    integral = float(correlation_excess(h, k, 1e-16))
    naive = float(bivariate_orthant(h, k, 1e-16)) - float(
        norm.cdf(h) * norm.cdf(k)
    )
    assert integral == pytest.approx(exact_first_order, rel=1e-12, abs=0.0)
    # The subtraction is WRONG, by a measurable amount, and that is the point.
    assert abs(naive - exact_first_order) / exact_first_order > 0.01


def test_the_subtraction_failure_grows_as_the_correlation_shrinks():
    """A sweep showing where the subtraction stops being usable."""
    h = k = -3.0
    worst = 0.0
    for rho in (1e-6, 1e-9, 1e-12, 1e-15):
        first_order = rho * float(bivariate_density(h, k, 0.0))
        naive = float(bivariate_orthant(h, k, rho)) - float(
            norm.cdf(h) * norm.cdf(k)
        )
        worst = max(worst, abs(naive - first_order) / first_order)
    assert worst > 0.001


@given(LEVELS, LEVELS)
def test_the_first_order_expansion_matches_at_small_correlation(h, k):
    """``excess = rho phi_2(h, k; 0) + O(rho^2)``, an independent closed form.

    The remainder is second order and not third: the second derivative of
    ``Phi_2`` in ``rho`` is ``h k phi_2 / ...`` and vanishes only at ``h k =
    0``. Hypothesis found the discrepancy at ``h = k = 1.0``, where the
    measured relative error at ``rho = 1e-5`` is 5.0e-06 -- exactly first order
    in ``rho``, as it should be. The test therefore evaluates at ``rho = 1e-9``,
    where the same remainder is of order 1e-10, and the literal case is pinned
    in ``tests/test_regressions.py``.
    """
    rho = 1e-9
    expected = rho * float(bivariate_density(h, k, 0.0))
    got = float(correlation_excess(h, k, rho))
    assert got == pytest.approx(expected, rel=1e-7, abs=1e-300)


@given(LEVELS, LEVELS, CORRELATIONS)
def test_the_excess_has_the_sign_of_the_correlation(h, k, rho):
    """Positive dependence raises the joint probability and negative lowers it."""
    got = float(correlation_excess(h, k, rho))
    if rho > 0.0:
        assert got > 0.0
    elif rho < 0.0:
        assert got < 0.0


def test_the_density_rejects_a_correlation_of_magnitude_one():
    """``1 - rho^2`` is zero there and the density is not defined."""
    with pytest.raises(ValueError, match=r"\|rho\| < 1"):
        bivariate_density(-1.0, -1.0, 1.0)


def test_the_excess_rejects_a_correlation_of_magnitude_one():
    """The same, through the integral."""
    with pytest.raises(ValueError, match=r"\|rho\| < 1"):
        correlation_excess(-1.0, -1.0, -1.0)


def test_epoch_nodes_reproduce_the_exact_epoch_average(short_record, short_climatology):
    """The binning of ``EpochNodes``, measured against the exact mean.

    This is the approximation that makes the fitting loops affordable, and the
    error is reported rather than assumed, and the node of a cell is the MEAN
    of the epochs in it rather than the cell centre, which is worth nearly
    three orders of magnitude. The tolerance below is four orders of magnitude
    under the block-bootstrap standard error of an empirical joint rate on this
    record, which is of order 5e-3.
    """
    threshold = short_climatology.threshold_series()
    worst = 0.0
    for i, j in ((0, 1), (0, 12), (4, 5), (8, 16), (3, 19)):
        for rho in (0.0, 0.3, 0.7, 0.9):
            nodes = epoch_nodes(threshold[:, i], threshold[:, j])
            binned = nodes.joint(rho)
            exact = pair_joint_exact(threshold[:, i], threshold[:, j], rho)
            worst = max(worst, abs(binned - exact))
    assert worst < 1e-5


def test_epoch_nodes_weights_sum_to_one(short_climatology):
    """Every epoch lands in exactly one cell."""
    threshold = short_climatology.threshold_series()
    nodes = epoch_nodes(threshold[:, 0], threshold[:, 1])
    assert float(nodes.weight.sum()) == pytest.approx(1.0, rel=1e-14, abs=0.0)
    assert nodes.node_count <= EPOCH_BINS**2
    assert nodes.epochs == threshold.shape[0]


def test_more_bins_reduce_the_binning_error(short_climatology):
    """A convergence check on the binning, not an assertion of a magic number."""
    threshold = short_climatology.threshold_series()
    exact = pair_joint_exact(threshold[:, 0], threshold[:, 1], 0.7)
    coarse = abs(
        epoch_nodes(threshold[:, 0], threshold[:, 1], bins=6).joint(0.7) - exact
    )
    fine = abs(
        epoch_nodes(threshold[:, 0], threshold[:, 1], bins=96).joint(0.7) - exact
    )
    assert fine < coarse


def test_epoch_nodes_rejects_mismatched_series():
    """Input validation."""
    with pytest.raises(ValueError, match="same shape"):
        epoch_nodes(np.zeros(5), np.zeros(6))


def test_epoch_nodes_rejects_an_empty_series():
    """Input validation."""
    with pytest.raises(ValueError, match="must not be empty"):
        epoch_nodes(np.zeros(0), np.zeros(0))


def test_epoch_nodes_rejects_fewer_than_two_bins():
    """Input validation."""
    with pytest.raises(ValueError, match="bins must be at least 2"):
        epoch_nodes(np.zeros(5), np.zeros(5), bins=1)


def test_the_implied_correlation_inverts_the_copula(short_climatology):
    """Equation (11) round-trips: insert ``rho``, recover ``rho``.

    The target is generated BY the copula at a known correlation, so the
    inversion must return it.

    Both tolerances are DERIVED from the root finder's declared ``xtol`` of
    1e-10 and not chosen. The tolerance on ``rho`` is that ``xtol``, with a
    hundredfold margin. The tolerance on the residual is ``xtol`` times the
    derivative of equation (11)'s left-hand side, which by equation (9) is the
    epoch mean of ``phi_2`` and is of order 0.05 to 0.2 here, so a residual of
    order 1e-11 is expected and an initial 1e-12 tolerance on it was simply
    wrong: the measured value is 1.99e-12.
    """
    threshold = short_climatology.threshold_series()
    nodes = epoch_nodes(threshold[:, 0], threshold[:, 1])
    for rho in (-0.4, 0.0, 0.25, 0.7, 0.93):
        target = nodes.joint(rho)
        result = implied_correlation(nodes, target)
        assert not result.at_bracket_end
        assert result.rho == pytest.approx(rho, rel=0.0, abs=1e-8)
        assert result.residual == pytest.approx(0.0, rel=0.0, abs=1e-10)


def test_the_implied_correlation_flags_a_target_above_the_bracket(short_climatology):
    """A joint rate the copula cannot reach returns the bracket end, flagged."""
    threshold = short_climatology.threshold_series()
    nodes = epoch_nodes(threshold[:, 0], threshold[:, 1])
    result = implied_correlation(nodes, 0.999)
    assert result.at_bracket_end
    assert result.rho == RHO_MAX


def test_the_implied_correlation_flags_a_target_below_the_bracket(short_climatology):
    """And the other end."""
    threshold = short_climatology.threshold_series()
    nodes = epoch_nodes(threshold[:, 0], threshold[:, 1])
    result = implied_correlation(nodes, 0.0)
    assert result.at_bracket_end
    assert result.rho == RHO_MIN


def test_the_implied_correlation_rejects_a_target_outside_zero_one(short_climatology):
    """Input validation."""
    threshold = short_climatology.threshold_series()
    nodes = epoch_nodes(threshold[:, 0], threshold[:, 1])
    with pytest.raises(ValueError, match="observed_joint must lie in"):
        implied_correlation(nodes, 1.5)


def test_the_implied_correlation_rejects_a_non_positive_tolerance(short_climatology):
    """Input validation."""
    threshold = short_climatology.threshold_series()
    nodes = epoch_nodes(threshold[:, 0], threshold[:, 1])
    with pytest.raises(ValueError, match="tolerance must be strictly positive"):
        implied_correlation(nodes, 0.1, tolerance=0.0)


def test_pair_excess_exact_agrees_with_the_difference_of_the_two_joints(
    short_climatology,
):
    """The epoch-averaged excess is the average of the per-epoch excesses."""
    threshold = short_climatology.threshold_series()
    rho = 0.6
    excess = pair_excess_exact(threshold[:, 0], threshold[:, 1], rho)
    joint = pair_joint_exact(threshold[:, 0], threshold[:, 1], rho)
    independent = pair_joint_exact(threshold[:, 0], threshold[:, 1], 0.0)
    assert excess == pytest.approx(joint - independent, rel=1e-11, abs=1e-18)


def test_pair_joint_exact_rejects_mismatched_series():
    """Input validation."""
    with pytest.raises(ValueError, match="same shape"):
        pair_joint_exact(np.zeros(4), np.zeros(5), 0.5)


def test_the_network_monte_carlo_agrees_with_the_exact_pair_quadrature(
    short_climatology,
):
    """Equation (10) against equation (8): two routes, one number.

    The Monte Carlo of equation (10) does not use the quadrature of equation
    (9) and the quadrature does not use random numbers, so this is an
    independent cross-check. The criterion is the declared four standard errors
    of the Monte Carlo estimator and not an absolute constant.
    """
    threshold = short_climatology.threshold_series()[:, [0, 1]]
    rho = 0.68
    matrix = np.array([[1.0, rho], [rho, 1.0]])
    estimate = network_joint_monte_carlo(threshold, matrix, 200_000, seed=31337)
    exact = pair_joint_exact(threshold[:, 0], threshold[:, 1], rho)
    assert abs(estimate.estimate - exact) < 4.0 * estimate.standard_error


def test_the_network_monte_carlo_is_deterministic_in_its_seed(short_climatology):
    """Two calls with the same seed give the same estimate."""
    threshold = short_climatology.threshold_series()[:, [0, 1, 2]]
    matrix = np.eye(3) * 0.4 + 0.6
    first = network_joint_monte_carlo(threshold, matrix, 20_000, seed=5)
    second = network_joint_monte_carlo(threshold, matrix, 20_000, seed=5)
    assert first.estimate == second.estimate


def test_the_network_monte_carlo_error_falls_as_the_root_of_the_sample_count(
    short_climatology,
):
    """``sqrt(p(1-p)/M)``: four times the samples halves the error."""
    threshold = short_climatology.threshold_series()[:, [0, 1, 2]]
    matrix = np.eye(3) * 0.4 + 0.6
    small = network_joint_monte_carlo(threshold, matrix, 20_000, seed=9)
    large = network_joint_monte_carlo(threshold, matrix, 320_000, seed=9)
    assert large.standard_error == pytest.approx(
        small.standard_error / 4.0, rel=0.1, abs=0.0
    )


def test_the_network_monte_carlo_rejects_an_indefinite_matrix(short_climatology):
    """A matrix no Gaussian field has is a modelling error, not a repair job."""
    threshold = short_climatology.threshold_series()[:, [0, 1, 2]]
    bad = np.array([[1.0, 0.99, -0.99], [0.99, 1.0, 0.99], [-0.99, 0.99, 1.0]])
    with pytest.raises(ValueError, match="not positive definite"):
        network_joint_monte_carlo(threshold, bad, 1000, seed=1)


def test_the_network_monte_carlo_rejects_a_mismatched_matrix(short_climatology):
    """Input validation."""
    threshold = short_climatology.threshold_series()[:, [0, 1]]
    with pytest.raises(ValueError, match="correlation must be"):
        network_joint_monte_carlo(threshold, np.eye(3), 1000, seed=1)


def test_the_network_monte_carlo_rejects_too_few_samples(short_climatology):
    """Input validation."""
    threshold = short_climatology.threshold_series()[:, [0, 1]]
    with pytest.raises(ValueError, match="samples must be at least 2"):
        network_joint_monte_carlo(threshold, np.eye(2), 1, seed=1)


def test_the_monte_carlo_relative_error_is_infinite_at_a_zero_estimate():
    """An estimate of zero has no relative error, and ``inf`` says so."""
    from sitediverse.copula import MonteCarloJoint

    assert np.isinf(MonteCarloJoint(0.0, 0.0, 100).relative_error)
