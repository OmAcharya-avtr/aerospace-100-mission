"""Unit tests for the learned regressor and the learned weight estimator."""

from __future__ import annotations

import numpy as np
import pytest

from conformalband.data import make_dataset
from conformalband.learned import LearnedRegressor, LearnedWeightEstimator
from conformalband.shift import CovariateShift


@pytest.fixture(scope="module")
def trained():
    data = make_dataset(1200, seed=57601)
    return LearnedRegressor(random_state=0).fit(data.features, data.energy), data


def test_prediction_shape(trained):
    model, data = trained
    assert model.predict(data.features).shape == (len(data),)


def test_prediction_is_deterministic(trained):
    model, data = trained
    assert np.array_equal(model.predict(data.features[:20]), model.predict(data.features[:20]))


def test_refit_with_the_same_seed_reproduces(trained):
    _, data = trained
    a = LearnedRegressor(random_state=3).fit(data.features, data.energy)
    b = LearnedRegressor(random_state=3).fit(data.features, data.energy)
    assert np.allclose(a.predict(data.features[:50]), b.predict(data.features[:50]), rtol=0.0)


def test_training_rmse_is_below_the_noise_level(trained):
    model, data = trained
    rmse = float(np.sqrt(np.mean((data.energy - model.predict(data.features)) ** 2)))
    assert rmse < float(np.std(data.energy - data.truth))


def test_feature_importances_sum_to_one(trained):
    model, _ = trained
    assert float(model.feature_importances.sum()) == pytest.approx(1.0, rel=1e-9)


def test_mass_is_an_important_feature(trained):
    # Induced power goes as m^2 and is the larger term at the design airspeed,
    # so mass must carry non-trivial importance.
    model, _ = trained
    assert model.feature_importances[1] > 0.05


def test_predict_before_fit_is_refused():
    with pytest.raises(RuntimeError, match="fit"):
        LearnedRegressor().predict(np.zeros((2, 5)))


def test_importances_before_fit_are_refused():
    with pytest.raises(RuntimeError, match="fit"):
        _ = LearnedRegressor().feature_importances


@pytest.mark.parametrize(
    "kwargs", [{"n_estimators": 0}, {"max_depth": 0}, {"learning_rate": 0.0}]
)
def test_learned_regressor_rejects_bad_hyperparameters(kwargs):
    with pytest.raises(ValueError):
        LearnedRegressor(**kwargs)


def test_learned_regressor_rejects_wrong_width():
    with pytest.raises(ValueError, match="shape"):
        LearnedRegressor().fit(np.zeros((10, 3)), np.zeros(10))


def test_learned_regressor_rejects_mismatched_target():
    with pytest.raises(ValueError, match="energy"):
        LearnedRegressor().fit(np.zeros((10, 5)), np.zeros(8))


def test_weight_estimator_auc_is_near_half_without_a_shift():
    calibration = make_dataset(1000, seed=57602, severity=0.0)
    test = make_dataset(1000, seed=57603, severity=0.0)
    estimator = LearnedWeightEstimator().fit(calibration.features, test.features)
    assert 0.45 <= estimator.auc <= 0.58


def test_weight_estimator_auc_rises_with_severity():
    calibration = make_dataset(1500, seed=57604, severity=0.0)
    previous = 0.0
    for severity in (0.0, 2.0, 5.0):
        test = make_dataset(1500, seed=57605, severity=severity)
        auc = LearnedWeightEstimator().fit(calibration.features, test.features).auc
        assert auc >= previous - 0.02
        previous = auc
    assert previous > 0.55


def test_weight_estimator_weights_are_near_one_without_a_shift():
    calibration = make_dataset(1500, seed=57606, severity=0.0)
    test = make_dataset(1500, seed=57607, severity=0.0)
    estimator = LearnedWeightEstimator().fit(calibration.features, test.features)
    weights = estimator.weights(calibration.features)
    assert float(np.mean(weights)) == pytest.approx(1.0, abs=0.08)


def test_weight_estimator_correlates_with_the_exact_ratio():
    calibration = make_dataset(3000, seed=57608, severity=0.0)
    test = make_dataset(3000, seed=57609, severity=3.0)
    estimator = LearnedWeightEstimator().fit(calibration.features, test.features)
    learned = estimator.weights(calibration.features)
    exact = CovariateShift(severity=3.0).likelihood_ratio(calibration.mass, calibration.headwind)
    assert float(np.corrcoef(learned, exact)[0, 1]) > 0.8


def test_weight_estimator_weights_are_positive():
    calibration = make_dataset(400, seed=57610, severity=0.0)
    test = make_dataset(400, seed=57611, severity=2.0)
    estimator = LearnedWeightEstimator().fit(calibration.features, test.features)
    assert np.all(estimator.weights(calibration.features) > 0.0)


def test_weight_estimator_respects_the_clip():
    calibration = make_dataset(400, seed=57612, severity=0.0)
    test = make_dataset(400, seed=57613, severity=3.0)
    estimator = LearnedWeightEstimator(clip=1.5).fit(calibration.features, test.features)
    weights = estimator.weights(calibration.features)
    assert np.all(weights <= 1.5 + 1e-12)
    assert np.all(weights >= 1.0 / 1.5 - 1e-12)
    assert estimator.clipped_fraction > 0.0


def test_weight_estimator_all_columns_option():
    calibration = make_dataset(600, seed=57614, severity=0.0)
    test = make_dataset(600, seed=57615, severity=2.0)
    estimator = LearnedWeightEstimator(columns=None).fit(calibration.features, test.features)
    assert estimator.weights(calibration.features).shape == (600,)


def test_weight_estimator_refuses_before_fit():
    with pytest.raises(RuntimeError, match="fit"):
        LearnedWeightEstimator().weights(np.zeros((3, 5)))


def test_weight_estimator_refuses_auc_before_fit():
    with pytest.raises(RuntimeError, match="fit"):
        _ = LearnedWeightEstimator().auc


@pytest.mark.parametrize("kwargs", [{"regularisation": 0.0}, {"clip": 1.0}, {"clip": 0.5}])
def test_weight_estimator_rejects_bad_hyperparameters(kwargs):
    with pytest.raises(ValueError):
        LearnedWeightEstimator(**kwargs)


def test_weight_estimator_rejects_tiny_samples():
    with pytest.raises(ValueError, match="at least 2 rows"):
        LearnedWeightEstimator().fit(np.zeros((1, 5)), np.zeros((5, 5)))


def test_weight_estimator_rejects_one_dimensional_input():
    with pytest.raises(ValueError, match="2-D"):
        LearnedWeightEstimator(columns=None).fit(np.zeros(5), np.zeros((5, 5)))
