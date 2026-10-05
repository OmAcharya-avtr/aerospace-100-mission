"""The analytic predictor adapter and the shared interval contract."""

from __future__ import annotations

import numpy as np
import pytest

from latencynet.dataset import log_target
from latencynet.learned import BoostingHyperparameters, LearnedTailPredictor
from latencynet.linear import LinearTailPredictor
from latencynet.predictors import AnalyticTailPredictor, TailPredictor


def test_analytic_predictor_names_and_flags():
    indep = AnalyticTailPredictor(assume_independent=True)
    dep = AnalyticTailPredictor(assume_independent=False)
    assert indep.name == "analytic_sum_indep"
    assert dep.name == "analytic_sum_cov"
    assert indep.uses_dependence_features is False
    assert dep.uses_dependence_features is True
    assert isinstance(indep, TailPredictor)


def test_analytic_predictor_has_nothing_to_fit(tiny_dataset):
    model = AnalyticTailPredictor()
    # Fitting on the train split and on the test split must give identical
    # predictions: a baseline with no parameters cannot be influenced by data.
    a = model.fit(tiny_dataset.train, 0.99).predict_log(tiny_dataset.test)
    b = model.fit(tiny_dataset.test, 0.99).predict_log(tiny_dataset.test)
    assert np.array_equal(a, b)


def test_analytic_predictor_interval(tiny_dataset):
    model = AnalyticTailPredictor().fit(tiny_dataset.train, 0.99)
    pred = model.predict_log_interval(tiny_dataset.test, 0.9)
    assert np.all(pred.log_lower < pred.log_point)
    assert np.all(pred.log_point < pred.log_upper)
    assert np.allclose(model.predict_log(tiny_dataset.test), pred.log_point, rtol=1e-14)
    assert np.all(pred.log_width > 0.0)


def test_covariance_variant_predicts_a_larger_tail_when_correlated(tiny_dataset):
    indep = AnalyticTailPredictor(True).fit(tiny_dataset.train, 0.99)
    dep = AnalyticTailPredictor(False).fit(tiny_dataset.train, 0.99)
    # Every pipeline in the correlated regime has positive latent rho, so the
    # covariance-aware variance exceeds the independence variance and the
    # predicted p99 is larger for every pipeline.
    assert np.all(dep.predict_log(tiny_dataset.test) > indep.predict_log(tiny_dataset.test))


def test_requires_fit_first():
    for model in (AnalyticTailPredictor(), AnalyticTailPredictor(False)):
        with pytest.raises(RuntimeError, match="before predicting"):
            model.predict_log(())
    with pytest.raises(ValueError, match="strictly in"):
        AnalyticTailPredictor().fit((), 1.5)


@pytest.mark.parametrize("level", [0.5, 0.8, 0.95])
def test_interval_widens_with_level(tiny_dataset, level):
    model = AnalyticTailPredictor().fit(tiny_dataset.train, 0.99)
    narrow = model.predict_log_interval(tiny_dataset.test, 0.5)
    wider = model.predict_log_interval(tiny_dataset.test, level)
    if level > 0.5:
        assert np.all(wider.log_width > narrow.log_width)
    else:
        assert np.allclose(wider.log_width, narrow.log_width, rtol=1e-12)


def test_all_three_models_share_the_interface(tiny_dataset):
    models = (
        AnalyticTailPredictor(),
        LinearTailPredictor(),
        LearnedTailPredictor(BoostingHyperparameters(n_estimators=30, max_depth=2)),
    )
    truth = log_target(tiny_dataset.test, 0.99)
    for model in models:
        model.fit(tiny_dataset.train, 0.99)
        point = model.predict_log(tiny_dataset.test)
        pred = model.predict_log_interval(tiny_dataset.test, 0.9)
        assert point.shape == truth.shape
        assert pred.log_point.shape == truth.shape
        assert pred.level == 0.9
        assert np.all(np.isfinite(pred.log_lower))
        assert np.all(np.isfinite(pred.log_upper))
