"""Feature construction, the grouped split, and the uncertainty output."""

from __future__ import annotations

import numpy as np
import pytest

from bitflipsim.criticality import sweep_bit_criticality
from bitflipsim.predictor import (
    FEATURE_NAMES,
    CriticalityPredictor,
    build_features,
    split_parameters,
    uncertainty_calibration,
)


def test_feature_matrix_shape_and_names(params, problem, f32):
    features = build_features(params, problem.calibration.x, f32)
    assert features.shape == (params.layout.size * 32, len(FEATURE_NAMES))
    assert np.isfinite(features).all()


def test_features_encode_the_bit_role_exactly_once(params, problem, f32):
    features = build_features(params, problem.calibration.x, f32)
    index = {name: i for i, name in enumerate(FEATURE_NAMES)}
    roles = features[:, [index["is_sign_bit"], index["is_exponent_bit"], index["is_mantissa_bit"]]]
    assert np.array_equal(roles.sum(axis=1), np.ones(features.shape[0]))
    # one sign bit and eight exponent bits per 32-bit parameter
    assert roles[:, 0].sum() == params.layout.size
    assert roles[:, 1].sum() == params.layout.size * 8
    assert roles[:, 2].sum() == params.layout.size * 23


def test_heuristic_feature_equals_the_baseline_score(params, problem, f32):
    features = build_features(params, problem.calibration.x, f32)
    index = FEATURE_NAMES.index("heuristic_log2_relative")
    per_site = features[:, index].reshape(params.layout.size, 32)
    assert list(per_site[0, 23:31]) == [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]
    assert np.allclose(per_site.std(axis=0), 0.0)


def test_grouped_split_shares_no_parameter(params):
    split = split_parameters(params.layout.size, 32, 0.6, seed=7)
    assert set(split.train_parameters.tolist()) & set(split.test_parameters.tolist()) == set()
    assert split.train_parameters.size + split.test_parameters.size == params.layout.size
    train_mask = split.site_mask("train")
    test_mask = split.site_mask("test")
    assert not (train_mask & test_mask).any()
    assert (train_mask | test_mask).all()
    assert train_mask.sum() == split.train_parameters.size * 32


def test_split_validation():
    with pytest.raises(ValueError, match="train_fraction"):
        split_parameters(10, 32, 0.0)
    with pytest.raises(ValueError, match="at least 2 parameters"):
        split_parameters(1, 32)
    with pytest.raises(ValueError, match="'train' or 'test'"):
        split_parameters(10, 32).site_mask("validation")


def test_predictor_requires_fitting_before_predicting(params, problem, f32):
    predictor = CriticalityPredictor(n_estimators=8, seed=1)
    features = build_features(params, problem.calibration.x, f32)
    with pytest.raises(ValueError, match="not fitted"):
        predictor.predict(features)
    with pytest.raises(ValueError, match="not fitted"):
        _ = predictor.feature_importances


def test_predictor_validation(params, problem, f32):
    features = build_features(params, problem.calibration.x, f32)
    predictor = CriticalityPredictor(n_estimators=8, seed=1)
    with pytest.raises(ValueError, match="disagree"):
        predictor.fit(features, np.zeros(5))
    with pytest.raises(ValueError, match="must be >= 0"):
        predictor.fit(features, -np.ones(features.shape[0]))
    with pytest.raises(ValueError, match="n_estimators"):
        CriticalityPredictor(n_estimators=1)


def test_predictor_produces_a_non_degenerate_uncertainty(params, problem, f32):
    sweep = sweep_bit_criticality(params, problem.calibration.x, problem.calibration.y)
    features = build_features(params, problem.calibration.x, f32)
    split = split_parameters(params.layout.size, 32, seed=7)
    predictor = CriticalityPredictor(n_estimators=40, seed=3).fit(
        features[split.site_mask("train")],
        sweep.flat_degradation()[split.site_mask("train")],
    )
    prediction = predictor.predict(features[split.site_mask("test")])
    assert prediction.expected.shape == (split.test_parameters.size * 32,)
    assert prediction.log_sigma.shape == prediction.expected.shape
    assert (prediction.log_sigma >= 0.0).all()
    assert prediction.log_sigma.max() > 0.0, "an ensemble with zero spread is not an uncertainty"
    assert np.isfinite(prediction.relative_sigma).all()
    assert (prediction.expected >= -1e-9).all()


def test_uncertainty_calibration_reports_coverage(params, problem, f32):
    sweep = sweep_bit_criticality(params, problem.calibration.x, problem.calibration.y)
    features = build_features(params, problem.calibration.x, f32)
    split = split_parameters(params.layout.size, 32, seed=7)
    predictor = CriticalityPredictor(n_estimators=40, seed=3).fit(
        features[split.site_mask("train")],
        sweep.flat_degradation()[split.site_mask("train")],
    )
    prediction = predictor.predict(features[split.site_mask("test")])
    report = uncertainty_calibration(prediction, sweep.flat_degradation()[split.site_mask("test")])
    assert 0.0 <= report["coverage"] <= 1.0
    assert report["gaussian_reference"] == pytest.approx(0.9544997361036416)
    assert report["n_sites"] == split.test_parameters.size * 32
    assert report["mean_abs_error_log1p"] >= 0.0
    with pytest.raises(ValueError, match="shapes disagree"):
        uncertainty_calibration(prediction, np.zeros(3))


def test_importance_table_sums_to_one(params, problem, f32):
    sweep = sweep_bit_criticality(params, problem.calibration.x, problem.calibration.y)
    features = build_features(params, problem.calibration.x, f32)
    predictor = CriticalityPredictor(n_estimators=20, seed=5).fit(
        features, sweep.flat_degradation()
    )
    table = predictor.importance_table()
    assert len(table) == len(FEATURE_NAMES)
    assert sum(value for _, value in table) == pytest.approx(1.0)
    assert table == sorted(table, key=lambda item: -item[1])
