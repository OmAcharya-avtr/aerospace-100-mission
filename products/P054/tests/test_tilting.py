"""Tests for the mean-shift tilting family and the importance-sampling estimator."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from rareverify.estimate import binomial_interval, counting_noise_floor, interval_for
from rareverify.limitstates import (
    LinearGaussianLimitState,
    LognormalRatioLimitState,
    RippledLimitState,
)
from rareverify.montecarlo import crude_monte_carlo
from rareverify.tilting import (
    MeanShiftTilt,
    analytic_mean_shift,
    find_design_point,
    find_design_point_radial,
    importance_sampling,
    oracle_mean_shift,
    orthogonal_mean_shift,
    scaled_mean_shift,
)


def test_weight_identity_hand_calculation():
    """w(x) = exp(-theta.x + |theta|^2 / 2).

    With theta = (2, 0) and x = (3, 1):
        theta.x      = 6
        |theta|^2/2  = 2
        log w        = -4
        w            = exp(-4) = 0.01831563888873418
    """
    tilt = MeanShiftTilt(np.array([2.0, 0.0]))
    assert float(tilt.log_weight(np.array([[3.0, 1.0]]))[0]) == pytest.approx(-4.0)
    assert float(tilt.weight(np.array([[3.0, 1.0]]))[0]) == pytest.approx(
        math.exp(-4.0), rel=1e-14
    )


@given(
    theta=st.lists(
        st.floats(min_value=-4.0, max_value=4.0, allow_nan=False), min_size=1, max_size=4
    ),
    x=st.lists(
        st.floats(min_value=-5.0, max_value=5.0, allow_nan=False), min_size=1, max_size=4
    ),
)
def test_weight_equals_the_density_ratio(theta, x):
    """w(x) must equal phi(x) / phi(x - theta) for every theta and x."""
    size = min(len(theta), len(x))
    theta_array = np.asarray(theta[:size], dtype=float)
    x_array = np.asarray(x[:size], dtype=float)
    tilt = MeanShiftTilt(theta_array)
    log_phi_x = -0.5 * float(x_array @ x_array)
    shifted = x_array - theta_array
    log_phi_shift = -0.5 * float(shifted @ shifted)
    assert float(tilt.log_weight(x_array[None, :])[0]) == pytest.approx(
        log_phi_x - log_phi_shift, abs=1e-9
    )


def test_zero_tilt_reproduces_plain_monte_carlo_exactly():
    """theta = 0 makes every weight 1, so the estimator is the crude count."""
    state = LinearGaussianLimitState(beta=2.0, dimension=2)
    tilt = MeanShiftTilt(np.zeros(2))
    estimate = importance_sampling(state, tilt, 50_000, rng=np.random.default_rng(4))
    crude = crude_monte_carlo(state, 50_000, rng=np.random.default_rng(4))
    assert estimate.estimate == pytest.approx(crude.estimate, rel=1e-12)
    assert estimate.is_binomial is True
    assert binomial_interval(estimate).upper == pytest.approx(
        binomial_interval(crude).upper, rel=1e-12
    )


def test_analytic_tilt_is_unbiased_and_far_more_precise_than_crude():
    state = LinearGaussianLimitState(beta=3.719, dimension=2)
    reference = state.analytic_probability()
    tilted = importance_sampling(
        state, analytic_mean_shift(state), 100_000, rng=np.random.default_rng(6)
    )
    floor = counting_noise_floor(reference, 100_000)
    assert abs(tilted.estimate - reference) < 4.0 * max(floor, tilted.standard_error)
    assert tilted.standard_error < 0.2 * floor
    assert tilted.is_binomial is False


def test_analytic_tilt_works_on_the_lognormal_limit_state():
    state = LognormalRatioLimitState()
    reference = state.analytic_probability()
    tilted = importance_sampling(
        state, analytic_mean_shift(state), 100_000, rng=np.random.default_rng(8)
    )
    assert abs(tilted.estimate / reference - 1.0) < 0.05


def test_interval_for_a_weighted_estimator_is_not_binomial():
    state = LinearGaussianLimitState()
    tilted = importance_sampling(
        state, analytic_mean_shift(state), 20_000, rng=np.random.default_rng(1)
    )
    with pytest.raises(ValueError, match="not a binomial proportion"):
        binomial_interval(tilted)
    interval = interval_for(tilted)
    assert interval.kind == "normal-weighted"


def test_wrong_direction_tilt_finds_nothing():
    """scale = -1 tilts away from the failure region: a documented failure case."""
    state = LinearGaussianLimitState(beta=3.719, dimension=2)
    tilted = importance_sampling(
        state, scaled_mean_shift(state, -1.0), 20_000, rng=np.random.default_rng(1)
    )
    assert tilted.n_failures == 0
    assert tilted.estimate == 0.0
    assert tilted.effective_sample_size == 0.0


def test_over_tilt_collapses_the_estimate_while_shrinking_its_standard_error():
    """The measured trap: tiny variance, and an answer wrong by many decades."""
    state = LinearGaussianLimitState(beta=3.719, dimension=2)
    reference = state.analytic_probability()
    tilted = importance_sampling(
        state, scaled_mean_shift(state, 3.0), 50_000, rng=np.random.default_rng(1)
    )
    assert tilted.estimate < reference / 1e4
    assert tilted.standard_error < reference / 1e4
    assert tilted.effective_sample_size < 10.0


def test_orthogonal_tilt_is_available_and_orthogonal():
    state = LinearGaussianLimitState(beta=3.0, dimension=3)
    tilt = orthogonal_mean_shift(state, 1.0)
    assert float(tilt.theta @ state.design_point()) == pytest.approx(0.0, abs=1e-12)
    assert tilt.norm == pytest.approx(3.0, rel=1e-12)
    with pytest.raises(ValueError, match="two input dimensions"):
        orthogonal_mean_shift(LinearGaussianLimitState(dimension=1))


def test_design_point_searches_agree_with_the_closed_form():
    """Two independent search paths plus the closed form, on three limit states."""
    for state, expected in (
        (LinearGaussianLimitState(beta=3.719, dimension=2), 3.719),
        (LognormalRatioLimitState(), 3.719),
    ):
        radial, converged = find_design_point_radial(
            state.g, state.dimension, rng=np.random.default_rng(2)
        )
        slsqp, converged_slsqp = find_design_point(
            state.g, state.dimension, n_starts=16, rng=np.random.default_rng(2)
        )
        assert converged and converged_slsqp
        assert float(np.linalg.norm(radial)) == pytest.approx(expected, rel=1e-5)
        assert float(np.linalg.norm(slsqp)) == pytest.approx(expected, rel=1e-6)


def test_radial_and_slsqp_agree_on_the_rippled_design_point():
    """Known answer: beta_true = 3.072735 for the default rippled limit state."""
    state = RippledLimitState()
    radial, _ = find_design_point_radial(
        state.g, 2, n_directions=512, rng=np.random.default_rng(1)
    )
    slsqp, _ = find_design_point(state.g, 2, n_starts=24, rng=np.random.default_rng(1))
    assert float(np.linalg.norm(radial)) == pytest.approx(3.072735, rel=1e-5)
    assert float(np.linalg.norm(slsqp)) == pytest.approx(3.072735, rel=1e-5)
    assert float(np.linalg.norm(radial)) < float(np.linalg.norm(state.design_point()))


def test_oracle_tilt_beats_the_smooth_design_point_on_a_rough_limit_state():
    """Measured: the smooth design point is not optimal once the ripple is there."""
    state = RippledLimitState(beta=5.5, amplitude=2.5, frequency=2.0)
    naive = importance_sampling(
        state, analytic_mean_shift(state), 100_000, rng=np.random.default_rng(3)
    )
    oracle = importance_sampling(
        state,
        oracle_mean_shift(state, rng=np.random.default_rng(1)),
        100_000,
        rng=np.random.default_rng(3),
    )
    assert oracle.standard_error < naive.standard_error
    assert oracle.effective_sample_size > naive.effective_sample_size


def test_design_point_search_without_a_crossing_reports_non_convergence():
    state = LinearGaussianLimitState(beta=12.0, dimension=2)
    point, converged = find_design_point_radial(
        state.g, 2, max_radius=4.0, rng=np.random.default_rng(1)
    )
    assert converged is False
    assert np.allclose(point, np.zeros(2))


@pytest.mark.parametrize(
    "factory",
    [
        lambda: MeanShiftTilt(np.array([])),
        lambda: MeanShiftTilt(np.array([np.nan])),
    ],
)
def test_invalid_tilts_raise(factory):
    with pytest.raises(ValueError):
        factory()


def test_importance_sampling_validation():
    state = LinearGaussianLimitState(dimension=2)
    tilt = analytic_mean_shift(state)
    with pytest.raises(ValueError, match="n_samples"):
        importance_sampling(state, tilt, 0)
    with pytest.raises(ValueError, match="batch_size"):
        importance_sampling(state, tilt, 10, batch_size=0)
    with pytest.raises(ValueError, match="does not match"):
        importance_sampling(state, MeanShiftTilt(np.zeros(3)), 10)
    with pytest.raises(ValueError, match="scale must be finite"):
        scaled_mean_shift(state, math.nan)
    with pytest.raises(ValueError):
        find_design_point_radial(state.g, 0)
    with pytest.raises(ValueError):
        find_design_point_radial(state.g, 2, n_grid=1)
    with pytest.raises(ValueError):
        find_design_point_radial(state.g, 2, n_bisect=0)
    with pytest.raises(ValueError):
        find_design_point_radial(state.g, 2, n_directions=0)
    with pytest.raises(ValueError):
        find_design_point_radial(state.g, 2, max_radius=0.0)
    with pytest.raises(ValueError):
        find_design_point_radial(state.g, 2, n_refine=-1)
    with pytest.raises(ValueError):
        find_design_point_radial(state.g, 2, refine_spread=0.0)
    with pytest.raises(ValueError):
        find_design_point(state.g, 2, n_starts=0)
    with pytest.raises(ValueError):
        tilt.sample(0, np.random.default_rng(0))
    with pytest.raises(ValueError, match="columns"):
        tilt.log_weight(np.zeros((3, 5)))
