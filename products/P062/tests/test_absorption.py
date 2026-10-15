"""Absorption tests: the subspace decomposition and the parameter drift."""

from __future__ import annotations

import numpy as np
import pytest

from twincalib.absorption import (
    DEFAULT_STRONG_CUT,
    decompose_residual,
    parameter_drift,
)
from twincalib.fitting import CalibrationProblem, fit
from twincalib.scenarios import calibration_model, design, make_dataset
from twincalib.sensitivity import log_sensitivity


def _sens(name: str = "baseline-eclipse", seed: int = 21):
    d = design(name)
    model = calibration_model()
    data = make_dataset(d, seed=seed, model=model)
    sens = log_sensitivity(
        model,
        d.theta_true,
        data.inputs,
        d.initial_state,
        d.channels,
        data.sigma,
        d.sample_stride,
    )
    return d, data, sens


def test_default_cut_is_the_documented_value():
    assert DEFAULT_STRONG_CUT == pytest.approx(1e-2)


def test_fractions_sum_to_one():
    _d, _data, sens = _sens()
    rng = np.random.default_rng(0)
    field = rng.normal(0.0, 0.4, size=sens.n_observations)
    split = decompose_residual(sens, field)
    total = split.strong_fraction + split.weak_fraction + split.orthogonal_fraction
    assert total == pytest.approx(1.0, abs=1e-10)
    energy = split.strong_energy + split.weak_energy + split.orthogonal_energy
    assert energy == pytest.approx(split.total_energy, rel=1e-10)


def test_subspace_dimensions_add_to_the_parameter_count():
    _d, _data, sens = _sens()
    rng = np.random.default_rng(1)
    split = decompose_residual(sens, rng.normal(0.0, 1.0, sens.n_observations))
    assert split.n_strong + split.n_weak == sens.n_parameters


def test_a_field_built_from_the_sensitivity_columns_is_entirely_in_the_column_space():
    # Known answer: a residual field that is exactly a parameter perturbation
    # has zero energy orthogonal to the parameters, because it is by
    # construction in the column space of the sensitivity matrix.
    _d, _data, sens = _sens()
    direction = np.array([1.0, -0.5, 0.25, 0.1, -0.3, 0.2])
    field = sens.jacobian_log @ direction
    split = decompose_residual(sens, field, strong_cut=1e-6)
    assert split.orthogonal_fraction < 1e-10
    assert split.strong_fraction > 1.0 - 1e-10


def test_a_field_orthogonal_to_the_column_space_is_entirely_orthogonal():
    _d, _data, sens = _sens()
    rng = np.random.default_rng(2)
    field = rng.normal(0.0, 1.0, sens.n_observations) * sens.sigma
    weighted = field / sens.sigma
    u, _s, _vt = np.linalg.svd(sens.weighted, full_matrices=False)
    residual = weighted - u @ (u.T @ weighted)
    split = decompose_residual(sens, residual * sens.sigma, strong_cut=1e-8)
    assert split.orthogonal_fraction > 1.0 - 1e-8


def test_a_field_along_the_weakest_direction_lands_in_the_weak_subspace():
    _d, _data, sens = _sens()
    u, s, _vt = np.linalg.svd(sens.weighted, full_matrices=False)
    # The weakest left singular direction, scaled back into kelvin.
    field = u[:, -1] * sens.sigma
    cut = float(s[-1] / s[0]) * 2.0
    split = decompose_residual(sens, field, strong_cut=cut)
    assert split.weak_fraction > 1.0 - 1e-8
    assert split.n_weak >= 1


def test_decompose_validation():
    _d, _data, sens = _sens()
    with pytest.raises(ValueError, match="strong_cut"):
        decompose_residual(sens, np.ones(sens.n_observations), strong_cut=0.0)
    with pytest.raises(ValueError, match="residual must have shape"):
        decompose_residual(sens, np.ones(5))
    bad = np.ones(sens.n_observations)
    bad[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        decompose_residual(sens, bad)
    with pytest.raises(ValueError, match="identically zero"):
        decompose_residual(sens, np.zeros(sens.n_observations))


def test_split_summary_names_the_three_parts():
    _d, _data, sens = _sens()
    rng = np.random.default_rng(3)
    text = decompose_residual(
        sens, rng.normal(0.0, 1.0, sens.n_observations)
    ).summary()
    assert "ABSORPTION" in text
    assert "model-form error" in text
    assert "strong-sensitivity subspace" in text


def _problem(name: str = "baseline-eclipse", seed: int = 31):
    d = design(name)
    model = calibration_model()
    data = make_dataset(d, seed=seed, model=model)
    problem = CalibrationProblem(
        model=model,
        inputs=data.inputs,
        initial_state=d.initial_state,
        channels=d.channels,
        sample_stride=d.sample_stride,
        y=data.y,
        sigma=data.sigma,
    )
    return d, problem


def test_zero_offset_leaves_the_parameters_where_they_were():
    d, problem = _problem()
    result = fit(problem, np.asarray(d.theta_true, dtype=float))
    report = parameter_drift(
        problem, result, np.zeros(problem.n_observations), theta_true=d.theta_true
    )
    assert float(np.max(np.abs(report.log_shift))) < 1e-4
    assert report.rms_relative_error_after == pytest.approx(
        report.rms_relative_error_before, abs=1e-4
    )
    assert report.chi2_after == pytest.approx(report.chi2_before, rel=1e-6)


def test_a_constant_offset_moves_the_parameters():
    d, problem = _problem()
    result = fit(problem, np.asarray(d.theta_true, dtype=float))
    report = parameter_drift(
        problem,
        result,
        np.full(problem.n_observations, 1.5),
        theta_true=d.theta_true,
    )
    assert float(np.max(np.abs(report.log_shift))) > 1e-3
    assert report.chi2_after != report.chi2_before


def test_drift_without_truth_declines_to_judge_the_direction():
    d, problem = _problem()
    result = fit(problem, np.asarray(d.theta_true, dtype=float))
    report = parameter_drift(
        problem, result, np.full(problem.n_observations, 0.4)
    )
    assert report.parameters_got_worse is None
    assert np.isnan(report.rms_relative_error_before)
    assert "truth unknown" in report.summary()


def test_drift_summary_lists_every_parameter():
    d, problem = _problem()
    result = fit(problem, np.asarray(d.theta_true, dtype=float))
    report = parameter_drift(
        problem,
        result,
        np.full(problem.n_observations, 0.2),
        theta_true=d.theta_true,
    )
    text = report.summary()
    for name in result.parameter_names:
        assert name in text
    assert "RMS log error against truth" in text


def test_drift_validation():
    d, problem = _problem()
    result = fit(problem, np.asarray(d.theta_true, dtype=float))
    with pytest.raises(ValueError, match="learned_offset must have shape"):
        parameter_drift(problem, result, np.zeros(3))
    with pytest.raises(ValueError, match="one entry per parameter"):
        parameter_drift(
            problem,
            result,
            np.zeros(problem.n_observations),
            theta_true=[1.0, 2.0],
        )
    already = CalibrationProblem(
        model=problem.model,
        inputs=problem.inputs,
        initial_state=problem.initial_state,
        channels=problem.channels,
        sample_stride=problem.sample_stride,
        y=problem.y,
        sigma=problem.sigma,
        offset=np.zeros(problem.n_observations),
    )
    with pytest.raises(ValueError, match="already carries an offset"):
        parameter_drift(already, result, np.zeros(problem.n_observations))
