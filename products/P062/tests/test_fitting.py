"""Fitter tests: recovery, covariance, caveats and input validation."""

from __future__ import annotations

import numpy as np
import pytest

from twincalib.fitting import (
    DIVERGENCE_PENALTY,
    CalibrationProblem,
    GuardedResidual,
    chi_square,
    fit,
    residual_vector,
)
from twincalib.scenarios import calibration_model, design, make_dataset


def test_fit_recovers_known_parameters_on_exact_model_data(exact_baseline_fit):
    problem, result, d = exact_baseline_fit
    assert result.success
    rel = result.relative_error(d.theta_true)
    # All six within 10 per cent; the weakest parameter (q_par) carries a
    # 6 per cent Cramer-Rao relative standard deviation on this design.
    assert float(np.max(np.abs(rel))) < 0.10
    z = result.z_score(d.theta_true)
    assert float(np.max(np.abs(z))) < 4.0
    assert problem.n_observations == d.n_observations


def test_reduced_chi_squared_is_about_one_when_the_model_is_right(exact_baseline_fit):
    _problem, result, _d = exact_baseline_fit
    assert 0.7 < result.reduced_chi2 < 1.4
    assert result.caveats() == []


def test_reduced_chi_squared_flags_model_error(truth_baseline):
    d, _data, problem = truth_baseline
    result = fit(problem, np.asarray(d.theta_true, dtype=float))
    assert result.reduced_chi2 > 1.5
    assert any("reduced chi-squared" in c for c in result.caveats())


def test_negative_log_likelihood_is_half_chi_squared(exact_baseline_fit):
    _problem, result, _d = exact_baseline_fit
    assert result.neg_log_likelihood == pytest.approx(0.5 * result.chi2, rel=1e-12)


def test_chi_square_matches_the_residual_vector(exact_baseline_fit):
    problem, result, _d = exact_baseline_fit
    r = residual_vector(problem, result.theta)
    assert chi_square(problem, result.theta) == pytest.approx(float(r @ r), rel=1e-12)
    assert chi_square(problem, result.theta) == pytest.approx(result.chi2, rel=1e-9)


def test_optimum_residual_is_orthogonal_to_the_sensitivity_columns(exact_baseline_fit):
    # A necessary condition at an interior least-squares optimum: J_w^T r = 0.
    problem, result, _d = exact_baseline_fit
    r = residual_vector(problem, result.theta)
    grad = result.sensitivity.weighted.T @ r
    scale = float(np.linalg.norm(result.sensitivity.weighted, axis=0).max())
    assert float(np.max(np.abs(grad))) / (scale * float(np.linalg.norm(r))) < 1e-4


def test_covariance_is_the_inverse_fisher_information(exact_baseline_fit):
    _problem, result, _d = exact_baseline_fit
    product = result.covariance_log @ result.sensitivity.fisher()
    assert np.max(np.abs(product - np.eye(6))) < 1e-6


def test_relative_sd_is_the_covariance_diagonal(exact_baseline_fit):
    _problem, result, _d = exact_baseline_fit
    assert np.allclose(
        result.relative_sd, np.sqrt(np.diag(result.covariance_log)), rtol=1e-12
    )


def test_correlation_is_bounded_and_unit_on_the_diagonal(exact_baseline_fit):
    _problem, result, _d = exact_baseline_fit
    corr = result.correlation
    assert np.allclose(np.diag(corr), 1.0, atol=1e-12)
    assert float(np.max(np.abs(corr))) <= 1.0 + 1e-9


def test_singular_covariance_is_flagged_on_the_degenerate_design(exact_full_sun):
    d, _data, problem = exact_full_sun
    result = fit(problem, np.asarray(d.theta_true, dtype=float))
    assert result.covariance_singular
    assert any("pseudo-inverse" in c for c in result.caveats())


def test_fit_is_reproducible(exact_baseline):
    d, _data, problem = exact_baseline
    start = np.asarray(d.theta_true, dtype=float) * 1.1
    a = fit(problem, start)
    b = fit(problem, start)
    assert np.max(np.abs(a.theta - b.theta)) == pytest.approx(0.0, abs=0.0)


def test_fit_from_two_different_starts_reaches_the_same_optimum(exact_baseline):
    d, _data, problem = exact_baseline
    theta = np.asarray(d.theta_true, dtype=float)
    a = fit(problem, theta * 0.7)
    b = fit(problem, theta * 1.4)
    assert np.max(np.abs(np.log(a.theta / b.theta))) < 1e-3


def test_offset_shifts_the_prediction(exact_baseline):
    d, data, problem = exact_baseline
    offset = np.full(problem.n_observations, 0.5)
    shifted = CalibrationProblem(
        model=problem.model,
        inputs=problem.inputs,
        initial_state=problem.initial_state,
        channels=problem.channels,
        sample_stride=problem.sample_stride,
        y=problem.y,
        sigma=problem.sigma,
        offset=offset,
    )
    base = problem.predict(d.theta_true)
    with_offset = shifted.predict(d.theta_true)
    assert np.allclose(with_offset - base, 0.5, rtol=0.0, atol=1e-12)
    assert data.n_observations == problem.n_observations


def test_problem_validation():
    d = design("baseline-eclipse")
    data = make_dataset(d, seed=1)
    kwargs = dict(
        model=calibration_model(),
        inputs=data.inputs,
        initial_state=d.initial_state,
        channels=d.channels,
        sample_stride=d.sample_stride,
    )
    with pytest.raises(ValueError, match="y must not be empty"):
        CalibrationProblem(y=np.array([]), sigma=0.3, **kwargs)
    with pytest.raises(ValueError, match="y must be finite"):
        CalibrationProblem(y=np.full(4, np.nan), sigma=0.3, **kwargs)
    with pytest.raises(ValueError, match="sigma"):
        CalibrationProblem(y=data.y, sigma=-0.3, **kwargs)
    with pytest.raises(ValueError, match="offset must have shape"):
        CalibrationProblem(
            y=data.y, sigma=0.3, offset=np.zeros(3), **kwargs
        )


def test_fit_rejects_an_underdetermined_problem():
    d = design("baseline-eclipse")
    data = make_dataset(d, seed=1)
    problem = CalibrationProblem(
        model=calibration_model(),
        inputs=data.inputs,
        initial_state=d.initial_state,
        channels=(0,),
        sample_stride=1,
        y=data.y[:4],
        sigma=0.3,
    )
    with pytest.raises(ValueError, match="cannot determine"):
        fit(problem, d.theta_true)


def test_fit_tolerance_validation(exact_baseline):
    d, _data, problem = exact_baseline
    with pytest.raises(ValueError, match="xtol"):
        fit(problem, d.theta_true, xtol=0.0)
    with pytest.raises(ValueError, match="ftol"):
        fit(problem, d.theta_true, ftol=-1.0)
    with pytest.raises(ValueError, match="bounds_factor"):
        fit(problem, d.theta_true, bounds_factor=1.0)


def test_relative_error_and_z_score_validation(exact_baseline_fit):
    _problem, result, _d = exact_baseline_fit
    with pytest.raises(ValueError, match="shape"):
        result.relative_error([1.0, 2.0])
    with pytest.raises(ValueError, match="zeros"):
        result.relative_error(np.zeros(6))
    with pytest.raises(ValueError, match="must have shape"):
        result.z_score([1.0])


def test_guarded_residual_penalises_a_divergent_parameter_vector(exact_baseline):
    d, _data, problem = exact_baseline
    guard = GuardedResidual(problem)
    good = guard(np.log(np.asarray(d.theta_true, dtype=float)))
    assert guard.n_rejected == 0
    assert float(np.max(np.abs(good))) < 100.0
    bad = np.log(np.asarray(d.theta_true, dtype=float))
    bad[2] = np.log(500.0)
    penalty = guard(bad)
    assert guard.n_rejected == 1
    assert np.all(penalty == DIVERGENCE_PENALTY)


def test_bound_hit_is_reported_as_not_converged(exact_baseline):
    _d, _data, problem = exact_baseline
    # A deliberately tiny search box around a bad start cannot reach the
    # optimum, so the fit must end on the bound and say so.
    result = fit(problem, [80.0, 400.0, 0.25, 1.2e-9, 5.5, 1.8], bounds_factor=1.05)
    assert not result.success
    assert "search bound" in result.message


def test_degrees_of_freedom(exact_baseline):
    d, _data, problem = exact_baseline
    assert problem.degrees_of_freedom == d.n_observations - 6
    assert problem.n_parameters == 6
