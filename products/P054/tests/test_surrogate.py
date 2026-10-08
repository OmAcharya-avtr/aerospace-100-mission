"""Tests for the learned surrogate and its two uses.

Compute note: a Gaussian-process fit is cubic in the design size, so the fits
are built once per module by fixtures and ``n_train`` is kept at or below 120.
The surrogate numbers quoted in the README come from
validation/validate_surrogate.py, which uses larger designs and replications.
"""

from __future__ import annotations

import numpy as np
import pytest

from rareverify.limitstates import LinearGaussianLimitState, RippledLimitState
from rareverify.surrogate import (
    SurrogateLimitState,
    fit_surrogate,
    radial_design,
    surrogate_design_point,
    surrogate_guided_importance_sampling,
    surrogate_probability,
)
from rareverify.tilting import analytic_mean_shift, importance_sampling

SMOOTH = LinearGaussianLimitState(beta=3.719, dimension=2)
MILD_RIPPLE = RippledLimitState()
ROUGH = RippledLimitState(beta=5.5, amplitude=2.5, frequency=2.0)
HIGH_FREQUENCY = RippledLimitState(beta=3.719, amplitude=0.8, frequency=6.0)


@pytest.fixture(scope="module")
def smooth_fit():
    return fit_surrogate(SMOOTH, n_train=100, rng=np.random.default_rng(11))


@pytest.fixture(scope="module")
def ripple_fit():
    return fit_surrogate(MILD_RIPPLE, n_train=120, rng=np.random.default_rng(12))


@pytest.fixture(scope="module")
def rough_fit():
    return fit_surrogate(ROUGH, n_train=100, rng=np.random.default_rng(22))


def test_radial_design_respects_its_declared_radius():
    """r_max = radius_scale * |Phi^-1(p_prior)|; at p=1e-4 this is 1.4*3.71902."""
    design = radial_design(
        3, 200, p_prior=1e-4, radius_scale=1.4, rng=np.random.default_rng(1)
    )
    radii = np.linalg.norm(design, axis=1)
    assert design.shape == (200, 3)
    assert radii.max() <= 1.4 * 3.71902 + 1e-9
    assert radii.min() >= 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dimension": 0, "n_train": 10},
        {"dimension": 2, "n_train": 0},
        {"dimension": 2, "n_train": 5000},
        {"dimension": 2, "n_train": 10, "p_prior": 0.0},
        {"dimension": 2, "n_train": 10, "p_prior": 0.7},
        {"dimension": 2, "n_train": 10, "radius_scale": 0.0},
    ],
)
def test_radial_design_validation(kwargs):
    with pytest.raises(ValueError):
        radial_design(**kwargs)


def test_surrogate_recovers_the_closed_form_design_point_on_a_linear_limit_state(
    smooth_fit,
):
    """The structural reason the surrogate cannot win on a smooth limit state.

    It recovers beta = 3.719 essentially exactly, so it ends up with the same
    tilt as the analytic baseline and no variance advantage, having spent
    n_train true evaluations the baseline did not need.
    """
    design = surrogate_design_point(smooth_fit, rng=np.random.default_rng(1))
    assert design.converged
    assert design.beta == pytest.approx(3.719, rel=1e-3)
    assert smooth_fit.n_train == 100
    assert smooth_fit.diagnostics["train_rmse"] < 1e-2


def test_surrogate_finds_the_rippled_design_point_the_smooth_one_misses(ripple_fit):
    """Known answer: beta_true = 3.072735, smooth design point 3.719."""
    design = surrogate_design_point(ripple_fit, rng=np.random.default_rng(2))
    assert design.beta == pytest.approx(3.072735, rel=5e-3)
    assert design.beta < float(np.linalg.norm(MILD_RIPPLE.design_point()))


def test_surrogate_uncertainty_band_is_a_band_and_contains_the_mean_estimate(ripple_fit):
    design = surrogate_design_point(ripple_fit, k_sigma=2.0, rng=np.random.default_rng(3))
    low = min(design.probability_lower, design.probability_upper)
    high = max(design.probability_lower, design.probability_upper)
    assert low <= design.probability <= high
    assert high > low
    assert design.k_sigma == 2.0
    assert "band" in design.describe()


def test_straddle_fraction_is_bounded_and_validated(smooth_fit):
    rng = np.random.default_rng(4)
    fraction = smooth_fit.straddle_fraction(rng.standard_normal((400, 2)))
    assert 0.0 <= fraction <= 1.0
    with pytest.raises(ValueError):
        smooth_fit.straddle_fraction(rng.standard_normal((10, 2)), k=0.0)


def test_surrogate_guided_importance_sampling_is_unbiased_and_counts_training(
    ripple_fit,
):
    reference = MILD_RIPPLE.analytic_probability()
    estimate, design = surrogate_guided_importance_sampling(
        MILD_RIPPLE, ripple_fit, 40_000, rng=np.random.default_rng(15)
    )
    assert estimate.true_evaluations == 40_120
    assert estimate.n_samples == 40_000
    assert abs(estimate.estimate / reference - 1.0) < 0.05
    assert estimate.diagnostics["n_train"] == 120.0
    assert estimate.diagnostics["surrogate_beta"] == pytest.approx(design.beta)


def test_surrogate_only_estimator_spends_no_further_true_evaluations(ripple_fit):
    estimate = surrogate_probability(ripple_fit, 10_000, rng=np.random.default_rng(16))
    assert estimate.true_evaluations == 120
    assert estimate.method == "surrogate-only"
    assert estimate.is_binomial is False


def test_surrogate_only_bias_is_not_covered_by_its_own_standard_error():
    """The point of implementing it: the bias is on the record.

    On the high-frequency rippled limit state (frequency 6) with only 30
    training evaluations, the surrogate's boundary is wrong, and the
    surrogate-only estimator reports a small standard error around a badly
    wrong answer. Reference probability 4.489382e-4 by quadrature.

    The nuance, measured in validation/validate_surrogate.py and not hidden
    here: at n_train of 60 or more on these two-dimensional limit states the
    surrogate-only bias falls below its own Monte-Carlo noise and is no longer
    detectable at that budget. The failure is a small-design failure, not a
    universal one.
    """
    reference = HIGH_FREQUENCY.analytic_probability()
    rng = np.random.default_rng(203)
    fit = fit_surrogate(HIGH_FREQUENCY, n_train=30, rng=rng)
    estimate = surrogate_probability(fit, 20_000, rng=rng)
    error = abs(estimate.estimate - reference)
    assert abs(estimate.estimate / reference - 1.0) > 0.2
    assert error > 5.0 * estimate.standard_error


def test_surrogate_limit_state_refuses_to_pretend_it_has_a_reference(smooth_fit):
    surrogate = SurrogateLimitState(smooth_fit)
    with pytest.raises(NotImplementedError, match="no reference probability"):
        surrogate.analytic_probability()
    with pytest.raises(NotImplementedError, match="surrogate_design_point"):
        surrogate.design_point()
    assert surrogate.dimension == 2
    assert surrogate.g(np.zeros((3, 2))).shape == (3,)


def test_sigma_offset_shifts_the_surrogate_boundary_in_the_expected_direction(
    smooth_fit,
):
    x = np.zeros((5, 2))
    mean = smooth_fit.surrogate_limit_state(0.0).g(x)
    optimistic = smooth_fit.surrogate_limit_state(2.0).g(x)
    pessimistic = smooth_fit.surrogate_limit_state(-2.0).g(x)
    assert np.all(optimistic >= mean)
    assert np.all(pessimistic <= mean)


def test_predict_chunking_gives_the_same_answer(smooth_fit):
    x = np.random.default_rng(1).standard_normal((2500, 2))
    whole = smooth_fit.predict(x, chunk=5000)
    chunked = smooth_fit.predict(x, chunk=100)
    assert np.allclose(whole, chunked, atol=1e-12)
    with pytest.raises(ValueError, match="columns"):
        smooth_fit.predict(np.zeros((3, 5)))


def test_fit_validation(smooth_fit):
    with pytest.raises(ValueError):
        fit_surrogate(SMOOTH, n_train=0)
    with pytest.raises(ValueError):
        fit_surrogate(SMOOTH, n_train=99_999)
    with pytest.raises(ValueError):
        surrogate_design_point(smooth_fit, k_sigma=0.0)


def test_surrogate_guided_is_beats_the_naive_tilt_on_a_rough_limit_state(rough_fit):
    """Measured outcome on the rough instance: the surrogate wins here.

    Single paired run at one seed; the replicated version with its measured
    variance reduction factor is in validation/validate_surrogate.py.
    """
    guided, _ = surrogate_guided_importance_sampling(
        ROUGH, rough_fit, 40_000, rng=np.random.default_rng(22)
    )
    naive = importance_sampling(
        ROUGH, analytic_mean_shift(ROUGH), 40_100, rng=np.random.default_rng(23)
    )
    assert guided.standard_error < naive.standard_error
