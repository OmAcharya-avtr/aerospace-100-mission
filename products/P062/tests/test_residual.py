"""Residual-model tests: features, fitting, uncertainty and validation."""

from __future__ import annotations

import numpy as np
import pytest

from twincalib.residual import (
    FEATURE_NAMES,
    ResidualModel,
    ResidualPrediction,
    build_features,
    residual_targets,
)
from twincalib.scenarios import build_inputs, calibration_model, design, make_dataset


def test_feature_names_are_the_documented_six():
    assert FEATURE_NAMES == (
        "T1_model",
        "T2_model",
        "dT_model",
        "dissipation",
        "sun",
        "is_channel_T2",
    )


def test_build_features_shape_and_prediction_consistency():
    d = design("baseline-eclipse")
    model = calibration_model()
    inputs = build_inputs(d)
    features, prediction = build_features(
        model, d.theta_true, inputs, d.initial_state, d.channels, d.sample_stride
    )
    assert features.shape == (d.n_observations, len(FEATURE_NAMES))
    assert prediction.shape == (d.n_observations,)
    direct = model.observe(
        d.theta_true, inputs, d.initial_state, d.channels, d.sample_stride
    )
    assert np.max(np.abs(prediction - direct)) == pytest.approx(0.0, abs=0.0)


def test_build_features_channel_indicator_and_interface_difference():
    d = design("baseline-eclipse")
    features, _p = build_features(
        calibration_model(),
        d.theta_true,
        build_inputs(d),
        d.initial_state,
        d.channels,
        d.sample_stride,
    )
    n = d.n_samples
    assert np.all(features[:n, 5] == 0.0)
    assert np.all(features[n:, 5] == 1.0)
    assert np.allclose(features[:, 2], features[:, 0] - features[:, 1], atol=1e-12)


def test_build_features_single_channel_design():
    d = design("lumped-nodes")
    features, prediction = build_features(
        calibration_model(),
        d.theta_true,
        build_inputs(d),
        d.initial_state,
        d.channels,
        d.sample_stride,
    )
    assert features.shape == (d.n_samples, len(FEATURE_NAMES))
    assert prediction.shape == (d.n_samples,)
    assert np.all(features[:, 5] == 0.0)


def test_build_features_rejects_a_bad_channel():
    d = design("baseline-eclipse")
    with pytest.raises(ValueError, match="out of range"):
        build_features(
            calibration_model(),
            d.theta_true,
            build_inputs(d),
            d.initial_state,
            (4,),
            1,
        )


def _training_set(name: str = "baseline-eclipse", seed: int = 7):
    d = design(name)
    model = calibration_model()
    data = make_dataset(d, seed=seed)
    features, prediction = build_features(
        model, d.theta_true, data.inputs, d.initial_state, d.channels, d.sample_stride
    )
    return d, features, data.y - prediction


def test_residual_model_learns_a_real_discrepancy():
    # The declared discrepancy has a 0.51 K root-mean-square on this design
    # against 0.3 K of noise, so a fitted model must explain part of it.
    d, features, target = _training_set()
    model = ResidualModel(noise_level=d.sigma_k**2, max_training_points=200).fit(
        features, target
    )
    assert model.fitted
    pred = model.predict(features)
    before = float(np.sqrt(np.mean(target**2)))
    after = float(np.sqrt(np.mean((target - pred.mean) ** 2)))
    assert after < before
    assert "RBF" in model.kernel_description


def test_prediction_shapes_and_positive_standard_deviation():
    d, features, target = _training_set()
    model = ResidualModel(noise_level=d.sigma_k**2, max_training_points=150).fit(
        features, target
    )
    pred = model.predict(features[:50])
    assert pred.mean.shape == (50,)
    assert pred.std.shape == (50,)
    assert np.all(pred.std > 0.0)


def test_interval_is_symmetric_about_the_mean():
    pred = ResidualPrediction(mean=np.array([1.0, -2.0]), std=np.array([0.5, 0.25]))
    lower, upper = pred.interval(z=2.0)
    assert np.allclose(upper - pred.mean, pred.mean - lower, atol=0.0)
    assert np.allclose(upper, np.array([2.0, -1.5]), atol=1e-12)
    with pytest.raises(ValueError, match="z"):
        pred.interval(z=0.0)


def test_training_point_cap_is_honoured_and_recorded():
    d, features, target = _training_set()
    model = ResidualModel(noise_level=d.sigma_k**2, max_training_points=120).fit(
        features, target
    )
    assert model.n_training_offered == d.n_observations
    assert model.n_training_used == 120
    assert model.n_training_used <= model.n_training_offered


def test_fit_is_reproducible_from_its_seed():
    d, features, target = _training_set()
    a = ResidualModel(
        noise_level=d.sigma_k**2, random_state=3, max_training_points=120
    ).fit(features, target)
    b = ResidualModel(
        noise_level=d.sigma_k**2, random_state=3, max_training_points=120
    ).fit(features, target)
    pa = a.predict(features[:40]).mean
    pb = b.predict(features[:40]).mean
    assert np.max(np.abs(pa - pb)) == pytest.approx(0.0, abs=1e-12)


def test_constructor_validation():
    with pytest.raises(ValueError, match="length_scale"):
        ResidualModel(length_scale=0.0)
    with pytest.raises(ValueError, match="noise_level"):
        ResidualModel(noise_level=-1.0)
    with pytest.raises(ValueError, match="n_restarts"):
        ResidualModel(n_restarts=-1)
    with pytest.raises(ValueError, match="max_training_points"):
        ResidualModel(max_training_points=2)


def test_fit_input_validation():
    model = ResidualModel()
    with pytest.raises(ValueError, match="features must have shape"):
        model.fit(np.zeros((20, 3)), np.zeros(20))
    with pytest.raises(ValueError, match="same number of rows"):
        model.fit(np.zeros((20, 6)), np.zeros(19))
    with pytest.raises(ValueError, match="at least 10"):
        model.fit(np.zeros((5, 6)), np.zeros(5))
    bad = np.zeros((20, 6))
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        model.fit(bad, np.zeros(20))


def test_predict_before_fit_is_an_error():
    model = ResidualModel()
    assert not model.fitted
    assert model.kernel_description == "not fitted"
    assert model.training_residual_scale() == pytest.approx(0.0)
    with pytest.raises(RuntimeError, match="fit the model"):
        model.predict(np.zeros((3, 6)))


def test_predict_shape_validation():
    d, features, target = _training_set()
    model = ResidualModel(noise_level=d.sigma_k**2, max_training_points=60).fit(
        features, target
    )
    with pytest.raises(ValueError, match="features must have shape"):
        model.predict(np.zeros((4, 2)))


def test_constant_feature_column_does_not_produce_nan():
    # The single-channel designs have a constant channel indicator, so the
    # standardiser must survive a zero-variance column.
    d, features, target = _training_set("lumped-nodes", seed=9)
    assert np.ptp(features[:, 5]) == pytest.approx(0.0, abs=0.0)
    model = ResidualModel(noise_level=d.sigma_k**2, max_training_points=100).fit(
        features, target
    )
    pred = model.predict(features[:30])
    assert np.all(np.isfinite(pred.mean))
    assert np.all(np.isfinite(pred.std))


def test_residual_targets_matches_the_definition():
    from twincalib.fitting import CalibrationProblem

    d = design("baseline-eclipse")
    data = make_dataset(d, seed=4)
    problem = CalibrationProblem(
        model=calibration_model(),
        inputs=data.inputs,
        initial_state=d.initial_state,
        channels=d.channels,
        sample_stride=d.sample_stride,
        y=data.y,
        sigma=data.sigma,
    )
    target = residual_targets(problem, d.theta_true)
    assert target.shape == (d.n_observations,)
    assert np.max(
        np.abs(target - (data.y - problem.predict(d.theta_true)))
    ) == pytest.approx(0.0, abs=0.0)


def test_training_residual_scale_is_positive_after_fit():
    d, features, target = _training_set()
    model = ResidualModel(noise_level=d.sigma_k**2, max_training_points=80).fit(
        features, target
    )
    assert model.training_residual_scale() > 0.0
