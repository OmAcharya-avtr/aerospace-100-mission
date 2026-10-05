"""The OLS baseline and its exact prediction interval."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from latencynet.dataset import feature_matrix, log_target
from latencynet.linear import (
    LinearTailPredictor,
    ols_fit,
    ols_predict,
    ols_prediction_interval,
)


def test_recovers_an_exact_linear_relation():
    # y = 2 + 3 x1 - 1 x2 exactly: the fit must reproduce it and the residual
    # standard deviation must be zero to rounding.
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 2))
    y = 2.0 + 3.0 * x[:, 0] - 1.0 * x[:, 1]
    fit = ols_fit(x, y)
    assert fit.residual_std < 1e-12
    assert np.allclose(ols_predict(fit, x), y, atol=1e-10)
    assert fit.n_params == 3
    assert fit.dof == 37


def test_standardisation_does_not_change_predictions():
    # Scaling a feature by 1000 is a reparameterisation; predictions and
    # interval widths must be unchanged.
    rng = np.random.default_rng(1)
    x = rng.normal(size=(60, 3))
    y = x @ np.array([1.0, -2.0, 0.5]) + rng.normal(scale=0.1, size=60)
    a = ols_prediction_interval(ols_fit(x, y), x, 0.9)
    scaled = x * np.array([1.0, 1000.0, 0.001])
    b = ols_prediction_interval(ols_fit(scaled, y), scaled, 0.9)
    for u, v in zip(a, b, strict=True):
        assert np.allclose(u, v, rtol=1e-8)


def test_interval_half_width_matches_the_closed_form():
    # Recompute t * s * sqrt(1 + leverage) independently for one row.
    rng = np.random.default_rng(2)
    x = rng.normal(size=(50, 2))
    y = x @ np.array([1.0, 2.0]) + rng.normal(scale=0.2, size=50)
    fit = ols_fit(x, y)
    lo, pt, hi = ols_prediction_interval(fit, x[:1], 0.9)
    design = np.hstack([[[1.0]], (x[:1] - fit.centre) / fit.scale])
    leverage = float((design @ fit.xtx_inv @ design.T)[0, 0])
    t = float(stats.t.ppf(0.95, fit.dof))
    expected = t * fit.residual_std * np.sqrt(1.0 + leverage)
    assert float(hi[0] - pt[0]) == pytest.approx(expected, rel=1e-10)
    assert float(pt[0] - lo[0]) == pytest.approx(expected, rel=1e-10)


def test_interval_widens_with_level():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(50, 2))
    y = x[:, 0] + rng.normal(scale=0.3, size=50)
    fit = ols_fit(x, y)
    w50 = ols_prediction_interval(fit, x, 0.5)
    w99 = ols_prediction_interval(fit, x, 0.99)
    assert np.all((w99[2] - w99[0]) > (w50[2] - w50[0]))


def test_prediction_interval_covers_about_its_nominal_rate():
    # 2000 Gaussian observations, nominal 90 %: the measured rate has a
    # standard error of sqrt(0.9 * 0.1 / 2000) = 0.0067, so a 0.03 band is
    # about 4.5 standard errors and is a loose check of a sound derivation.
    rng = np.random.default_rng(4)
    x = rng.normal(size=(2000, 3))
    y = x @ np.array([1.0, -1.0, 2.0]) + rng.normal(scale=0.5, size=2000)
    fit = ols_fit(x[:1000], y[:1000])
    lo, _, hi = ols_prediction_interval(fit, x[1000:], 0.9)
    covered = float(np.mean((y[1000:] >= lo) & (y[1000:] <= hi)))
    assert covered == pytest.approx(0.9, abs=0.03)


def test_predictor_end_to_end(tiny_dataset):
    model = LinearTailPredictor().fit(tiny_dataset.train, 0.99)
    assert model.name == "linear_ols"
    assert model.uses_dependence_features is True
    point = model.predict_log(tiny_dataset.test)
    assert point.shape == (len(tiny_dataset.test),)
    pred = model.predict_log_interval(tiny_dataset.test, 0.9)
    assert np.all(pred.log_lower < pred.log_point)
    assert np.all(pred.log_point < pred.log_upper)
    assert np.allclose(point, pred.log_point, rtol=1e-14)
    # Predicted latencies must be positive after exponentiation.
    lo, pt, hi = pred.seconds()
    assert np.all(lo > 0.0) and np.all(hi > lo)


def test_predictor_beats_a_constant_predictor(tiny_dataset):
    model = LinearTailPredictor().fit(tiny_dataset.train, 0.99)
    truth = log_target(tiny_dataset.test, 0.99)
    fitted = np.mean(np.abs(model.predict_log(tiny_dataset.test) - truth))
    constant = np.mean(np.abs(np.mean(log_target(tiny_dataset.train, 0.99)) - truth))
    assert fitted < constant


def test_feature_matrix_shape_is_what_the_fit_expects(tiny_dataset):
    assert feature_matrix(tiny_dataset.train).shape[1] == 13


def test_validation():
    with pytest.raises(RuntimeError, match="before predicting"):
        LinearTailPredictor().predict_log(())
    with pytest.raises(ValueError, match="strictly in"):
        LinearTailPredictor().fit((), 1.0)
    with pytest.raises(ValueError, match="must be 2-D"):
        ols_fit(np.zeros(5), np.zeros(5))
    with pytest.raises(ValueError, match="rows"):
        ols_fit(np.zeros((5, 2)), np.zeros(4))
    with pytest.raises(ValueError, match="finite"):
        ols_fit(np.full((10, 2), np.nan), np.zeros(10))
    with pytest.raises(ValueError, match="at least"):
        ols_fit(np.zeros((3, 2)), np.zeros(3))
    with pytest.raises(ValueError, match="level"):
        ols_prediction_interval(ols_fit(np.zeros((10, 1)), np.arange(10.0)), np.zeros((1, 1)), 0.0)


def test_zero_variance_column_is_tolerated():
    # A constant feature has zero sample standard deviation; the fit must not
    # divide by zero and must simply ignore it.
    rng = np.random.default_rng(5)
    x = np.column_stack([rng.normal(size=30), np.ones(30)])
    y = x[:, 0] * 2.0
    fit = ols_fit(x, y)
    assert np.all(np.isfinite(fit.coef))
    assert fit.residual_std < 1e-12
