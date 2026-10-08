"""Tests for the limit states and their reference probabilities."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from rareverify.limitstates import (
    LinearGaussianLimitState,
    LognormalRatioLimitState,
    RippledLimitState,
    limit_state_from_name,
)


def test_linear_gaussian_reference_is_the_normal_tail():
    """g = beta - a.x with |a| = 1 gives P = Phi(-beta) exactly.

    Hand calculation at beta = 3.719: Phi(-3.719) = 1.000065e-4, i.e. the
    reliability index the whole package uses as its 1e-4 instance.
    """
    for beta in (1.0, 2.5, 3.719, 4.753, 6.0):
        state = LinearGaussianLimitState(beta=beta, dimension=3)
        assert state.analytic_probability() == pytest.approx(
            float(stats.norm.cdf(-beta)), rel=1e-14
        )
    assert LinearGaussianLimitState(beta=3.719).analytic_probability() == pytest.approx(
        1.000065259e-4, rel=1e-8
    )


def test_linear_gaussian_probability_is_direction_invariant():
    """Rotating a must not change the probability: an invariance check."""
    base = LinearGaussianLimitState(beta=3.0, dimension=4).analytic_probability()
    for direction in ([1, 1, 1, 1], [0, 0, 1, 0], [-2, 3, 0, 1]):
        rotated = LinearGaussianLimitState(beta=3.0, dimension=4, direction=direction)
        assert rotated.analytic_probability() == pytest.approx(base, rel=1e-14)
        assert float(np.linalg.norm(rotated.a)) == pytest.approx(1.0, rel=1e-14)


def test_linear_gaussian_empirical_probability_matches_reference():
    """A direct count at a mild beta where crude Monte Carlo is adequate."""
    state = LinearGaussianLimitState(beta=2.0, dimension=2)
    rng = np.random.default_rng(7)
    x = rng.standard_normal((400_000, 2))
    empirical = float(np.mean(state.failure(x)))
    reference = state.analytic_probability()
    noise = math.sqrt(reference * (1 - reference) / 400_000)
    assert abs(empirical - reference) < 4.0 * noise


def test_lognormal_ratio_reference_and_nonlinearity():
    """g = R - S is nonlinear in x, but {R <= S} is linear, so P is exact.

    Hand check of the default: beta = (1.190661 - 0) / sqrt(0.2^2 + 0.25^2)
                                    = 1.190661 / 0.3201562119 = 3.719000
    so P = Phi(-3.7190003) = 1.0000647e-4, which differs from the linear
    instance in the seventh significant figure because mu_r is a rounded
    decimal rather than beta * sqrt(sigma_r^2 + sigma_s^2) to full precision.
    """
    state = LognormalRatioLimitState()
    assert state.beta == pytest.approx(3.719000, abs=1e-6)
    assert state.analytic_probability() == pytest.approx(1.0000647e-4, rel=1e-6)
    # Nonlinearity: g is not affine in x.
    x = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    g = state.g(x)
    assert abs((g[2] - g[1]) - (g[1] - g[0])) > 1e-3


def test_lognormal_failure_event_matches_the_linear_form():
    state = LognormalRatioLimitState()
    rng = np.random.default_rng(3)
    x = rng.standard_normal((20_000, 2))
    nonlinear = state.g(x) <= 0.0
    linear = (
        state.sigma_r * x[:, 0] - state.sigma_s * x[:, 1]
    ) <= state.mu_s - state.mu_r
    assert np.array_equal(nonlinear, linear)


def test_rippled_reduces_to_linear_at_zero_amplitude():
    linear = LinearGaussianLimitState(beta=3.2, dimension=2)
    rippled = RippledLimitState(beta=3.2, amplitude=0.0, dimension=2)
    assert rippled.analytic_probability() == pytest.approx(
        linear.analytic_probability(), rel=1e-12
    )
    x = np.random.default_rng(1).standard_normal((100, 2))
    assert np.allclose(rippled.g(x), linear.g(x))


def test_rippled_quadrature_reference_is_converged():
    """The Gauss-Hermite reference must be stable in the node count."""
    state = RippledLimitState()
    values = [state.analytic_probability(nodes=n) for n in (100, 200, 400, 800)]
    for value in values[1:]:
        assert value == pytest.approx(values[0], rel=1e-9)
    assert values[2] == pytest.approx(4.4462124889798e-4, rel=1e-10)


def test_rippled_empirical_probability_matches_the_quadrature_reference():
    state = RippledLimitState(beta=2.0, amplitude=0.8, frequency=1.5)
    rng = np.random.default_rng(11)
    x = rng.standard_normal((400_000, 2))
    empirical = float(np.mean(state.failure(x)))
    reference = state.analytic_probability()
    noise = math.sqrt(reference * (1 - reference) / 400_000)
    assert abs(empirical - reference) < 4.0 * noise


def test_design_points_lie_on_the_limit_state_boundary():
    for state in (
        LinearGaussianLimitState(),
        LognormalRatioLimitState(),
    ):
        point = state.design_point()
        assert float(state.g(point)[0]) == pytest.approx(0.0, abs=1e-9)
        assert float(np.linalg.norm(point)) == pytest.approx(state.beta, rel=1e-12)


def test_rippled_design_point_is_the_smooth_one_and_is_not_optimal():
    """The documented deliberate gap: the smooth design point is off-boundary."""
    state = RippledLimitState(beta=3.719, amplitude=0.8, frequency=1.5)
    point = state.design_point()
    assert np.allclose(point, [3.719, 0.0])
    assert float(state.g(point)[0]) == pytest.approx(0.0, abs=1e-12)
    # A nearer point of the failure region exists, so |design_point| is not
    # the true reliability index.
    nearer = np.array([2.944183, -0.879480])
    assert float(state.g(nearer)[0]) <= 1e-5
    assert float(np.linalg.norm(nearer)) < float(np.linalg.norm(point))


def test_single_row_input_is_accepted():
    state = LinearGaussianLimitState(dimension=3)
    assert state.g(np.zeros(3)).shape == (1,)


def test_wrong_column_count_raises():
    state = LinearGaussianLimitState(dimension=3)
    with pytest.raises(ValueError, match="columns"):
        state.g(np.zeros((4, 2)))


@pytest.mark.parametrize(
    "factory",
    [
        lambda: LinearGaussianLimitState(beta=math.nan),
        lambda: LinearGaussianLimitState(dimension=0),
        lambda: LinearGaussianLimitState(dimension=3, direction=[1, 1]),
        lambda: LinearGaussianLimitState(dimension=2, direction=[0, 0]),
        lambda: LognormalRatioLimitState(sigma_r=0.0),
        lambda: LognormalRatioLimitState(sigma_s=-1.0),
        lambda: LognormalRatioLimitState(mu_r=math.inf),
        lambda: RippledLimitState(dimension=1),
        lambda: RippledLimitState(quadrature_nodes=4),
        lambda: RippledLimitState(amplitude=math.nan),
    ],
)
def test_invalid_constructions_raise(factory):
    with pytest.raises(ValueError):
        factory()


def test_rippled_rejects_too_few_quadrature_nodes_at_call_time():
    with pytest.raises(ValueError):
        RippledLimitState().analytic_probability(nodes=2)


def test_limit_state_from_name():
    assert isinstance(limit_state_from_name("linear"), LinearGaussianLimitState)
    assert isinstance(limit_state_from_name("lognormal"), LognormalRatioLimitState)
    assert isinstance(limit_state_from_name("rippled"), RippledLimitState)
    with pytest.raises(ValueError, match="unknown limit state"):
        limit_state_from_name("quadratic")


def test_reference_kind_is_declared():
    assert LinearGaussianLimitState().reference_kind == "closed-form"
    assert LognormalRatioLimitState().reference_kind == "closed-form"
    assert RippledLimitState().reference_kind == "quadrature"
    assert "closed-form" in LinearGaussianLimitState().describe()
