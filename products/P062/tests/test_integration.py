"""End-to-end tests: the whole intended workflow, in order."""

from __future__ import annotations

import numpy as np
import pytest

from twincalib import (
    CalibrationProblem,
    analyse,
    calibration_model,
    decompose_residual,
    design,
    fit,
    log_sensitivity,
    make_dataset,
    profile_all,
    run_benchmark,
    truth_model,
)
from twincalib.residual import ResidualModel, build_features


def test_identifiability_first_then_fit_then_profile():
    # The documented sequence, on the design that supports it, with every stage
    # consuming the previous stage's output.
    d = design("coarse-sampled")
    model = calibration_model()
    data = make_dataset(d, seed=777, model=model)

    sens = log_sensitivity(
        model,
        d.theta_true,
        data.inputs,
        d.initial_state,
        d.channels,
        data.sigma,
        d.sample_stride,
    )
    report = analyse(sens)
    assert report.structurally_identifiable

    problem = CalibrationProblem(
        model=model,
        inputs=data.inputs,
        initial_state=d.initial_state,
        channels=d.channels,
        sample_stride=d.sample_stride,
        y=data.y,
        sigma=data.sigma,
    )
    result = fit(problem, np.asarray(d.theta_true, dtype=float) * 1.25)
    assert result.success
    assert float(np.max(np.abs(result.relative_error(d.theta_true)))) < 0.25

    curves = profile_all(problem, result, indices=[0, 2], n_points=3, refine_steps=2)
    for c in curves:
        assert c.bounded
        assert c.contains(d.theta_true[c.index])


def test_the_degenerate_design_is_refused_at_every_stage():
    d = design("full-sun")
    model = calibration_model()
    data = make_dataset(d, seed=778, model=model)
    sens = log_sensitivity(
        model,
        d.theta_true,
        data.inputs,
        d.initial_state,
        d.channels,
        data.sigma,
        d.sample_stride,
    )
    report = analyse(sens)
    # Stage 1 says no.
    assert not report.structurally_identifiable
    assert report.verdict().startswith("NOT IDENTIFIABLE")
    problem = CalibrationProblem(
        model=model,
        inputs=data.inputs,
        initial_state=d.initial_state,
        channels=d.channels,
        sample_stride=d.sample_stride,
        y=data.y,
        sigma=data.sigma,
    )
    result = fit(problem, np.asarray(d.theta_true, dtype=float))
    # Stage 2 says no.
    assert result.covariance_singular
    assert not np.isfinite(result.relative_sd[4])
    assert not np.isfinite(result.relative_sd[5])
    # Stage 3 says no.
    curve = profile_all(problem, result, indices=[4], n_points=2, refine_steps=2)[0]
    assert not curve.bounded


def test_the_sum_of_the_two_heat_loads_is_identifiable_even_though_neither_is():
    # The positive statement the package should also make: the degenerate
    # design determines q_sun + q_par well, and that is what the fit is good
    # for. Measured over three seeds.
    d = design("full-sun")
    model = calibration_model()
    totals = []
    for seed in (801, 802, 803):
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
        result = fit(problem, np.asarray(d.theta_true, dtype=float) * 1.1)
        totals.append(float(result.theta[4] + result.theta[5]))
    true_total = d.theta_true[4] + d.theta_true[5]
    for value in totals:
        assert value == pytest.approx(true_total, rel=0.03)


def test_residual_model_pipeline_on_held_out_data():
    # Fit the physics on one trajectory, learn the discrepancy, score on a
    # trajectory the pipeline has never seen.
    d = design("baseline-eclipse")
    model = calibration_model()
    train = make_dataset(d, seed=901, phase=0.0, model=truth_model())
    test = make_dataset(d, seed=902, phase=0.6, model=truth_model())
    problem = CalibrationProblem(
        model=model,
        inputs=train.inputs,
        initial_state=d.initial_state,
        channels=d.channels,
        sample_stride=d.sample_stride,
        y=train.y,
        sigma=train.sigma,
    )
    result = fit(problem, np.asarray(d.theta_true, dtype=float))
    features_train, pred_train = build_features(
        model, result.theta, train.inputs, d.initial_state, d.channels, d.sample_stride
    )
    residual = ResidualModel(
        noise_level=d.sigma_k**2, max_training_points=200
    ).fit(features_train, train.y - pred_train)
    features_test, pred_test = build_features(
        model, result.theta, test.inputs, d.initial_state, d.channels, d.sample_stride
    )
    learned = residual.predict(features_test)
    physics_rmse = float(np.sqrt(np.mean((test.y - pred_test) ** 2)))
    combined_rmse = float(
        np.sqrt(np.mean((test.y - pred_test - learned.mean) ** 2))
    )
    assert combined_rmse < physics_rmse
    # And the absorption question, asked on the held-out design.
    sens_test = log_sensitivity(
        model,
        result.theta,
        test.inputs,
        d.initial_state,
        d.channels,
        test.sigma,
        d.sample_stride,
    )
    split = decompose_residual(sens_test, learned.mean)
    assert split.orthogonal_fraction > 0.0
    assert split.weak_fraction >= 0.0
    assert split.strong_fraction + split.weak_fraction + split.orthogonal_fraction == (
        pytest.approx(1.0, abs=1e-10)
    )


def test_absorption_is_larger_on_the_non_identifiable_design():
    # The headline measurement of the package, as an end-to-end check: the
    # learned residual puts a far larger share of its energy into directions
    # the data cannot constrain when the design is degenerate. The measured
    # ratio is reported in validation/validate_residual_output.txt; the test
    # asserts the ordering, which is the claim.
    kwargs = dict(n_train=3, n_test=2, max_training_points=250, strong_cuts=(1e-2,))
    good = run_benchmark(design("baseline-eclipse"), **kwargs)
    bad = run_benchmark(design("full-sun"), **kwargs)
    assert bad.splits[1e-2].weak_fraction > good.splits[1e-2].weak_fraction
    assert bad.splits[1e-2].weak_fraction > 10.0 * good.splits[1e-2].weak_fraction
