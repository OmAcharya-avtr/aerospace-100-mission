"""The learned predictor: determinism, interval ordering, validation."""

from __future__ import annotations

import numpy as np
import pytest

from latencynet.dataset import log_target
from latencynet.learned import BoostingHyperparameters, LearnedTailPredictor


def test_fit_and_predict(tiny_dataset):
    model = LearnedTailPredictor(
        BoostingHyperparameters(n_estimators=40, max_depth=2)
    ).fit(tiny_dataset.train, 0.99)
    assert model.name == "learned_gbt"
    assert model.uses_dependence_features is True
    point = model.predict_log(tiny_dataset.test)
    assert point.shape == (len(tiny_dataset.test),)
    assert np.all(np.isfinite(point))


def test_fit_is_deterministic(tiny_dataset):
    hp = BoostingHyperparameters(n_estimators=40, max_depth=2)
    a = LearnedTailPredictor(hp).fit(tiny_dataset.train, 0.99).predict_log(tiny_dataset.test)
    b = LearnedTailPredictor(hp).fit(tiny_dataset.train, 0.99).predict_log(tiny_dataset.test)
    assert np.array_equal(a, b)


def test_native_interval_is_ordered(tiny_dataset):
    model = LearnedTailPredictor(
        BoostingHyperparameters(n_estimators=40, max_depth=2), native_interval_level=0.8
    ).fit(tiny_dataset.train, 0.99)
    pred = model.predict_log_interval(tiny_dataset.test, 0.8)
    assert np.all(pred.log_upper >= pred.log_lower)
    assert pred.level == 0.8


def test_native_interval_rejects_a_different_level(tiny_dataset):
    model = LearnedTailPredictor(
        BoostingHyperparameters(n_estimators=40, max_depth=2), native_interval_level=0.9
    ).fit(tiny_dataset.train, 0.99)
    with pytest.raises(ValueError, match="quantile heads were fitted"):
        model.predict_log_interval(tiny_dataset.test, 0.5)


def test_fits_training_data_better_than_a_constant(tiny_dataset):
    model = LearnedTailPredictor(
        BoostingHyperparameters(n_estimators=100, max_depth=2)
    ).fit(tiny_dataset.train, 0.99)
    truth = log_target(tiny_dataset.train, 0.99)
    fitted = np.mean(np.abs(model.predict_log(tiny_dataset.train) - truth))
    constant = np.mean(np.abs(np.mean(truth) - truth))
    assert fitted < constant


def test_feature_importances(tiny_dataset):
    model = LearnedTailPredictor(
        BoostingHyperparameters(n_estimators=40, max_depth=2)
    ).fit(tiny_dataset.train, 0.99)
    imp = model.feature_importances()
    assert imp.shape == (13,)
    assert imp.sum() == pytest.approx(1.0, rel=1e-6)
    assert np.all(imp >= 0.0)


def test_requires_enough_training_pipelines(tiny_dataset):
    with pytest.raises(ValueError, match="at least 20 training pipelines"):
        LearnedTailPredictor().fit(tiny_dataset.train[:5], 0.99)


def test_requires_fit_before_predict():
    with pytest.raises(RuntimeError, match="before predicting"):
        LearnedTailPredictor().predict_log(())
    with pytest.raises(RuntimeError, match="before predicting"):
        LearnedTailPredictor().feature_importances()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_estimators": 0},
        {"max_depth": 0},
        {"learning_rate": 0.0},
        {"learning_rate": 1.5},
        {"min_samples_leaf": 0},
    ],
)
def test_hyperparameter_validation(kwargs):
    with pytest.raises(ValueError):
        BoostingHyperparameters(**kwargs)


def test_level_validation():
    with pytest.raises(ValueError, match="native_interval_level"):
        LearnedTailPredictor(native_interval_level=1.0)
    with pytest.raises(ValueError, match="strictly in"):
        LearnedTailPredictor().fit((), 0.0)
