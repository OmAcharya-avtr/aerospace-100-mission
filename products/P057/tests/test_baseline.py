"""Unit tests for the analytic and parametric baselines."""

from __future__ import annotations

import numpy as np
import pytest

from conformalband.baseline import (
    PARAMETER_NAMES,
    PARAMETER_UNITS,
    GaussianResidualInterval,
    PhysicsRegressor,
)
from conformalband.data import make_dataset
from conformalband.physics import DEFAULT_AIRFRAME


@pytest.fixture(scope="module")
def fitted():
    data = make_dataset(1200, seed=57501)
    return PhysicsRegressor().fit(data.features, data.energy), data


def test_parameter_names_and_units_align():
    assert len(PARAMETER_NAMES) == len(PARAMETER_UNITS) == 4


def test_n_parameters_is_four():
    assert PhysicsRegressor.n_parameters == 4


def test_fitted_parameters_shape(fitted):
    model, _ = fitted
    assert model.parameters.shape == (4,)


def test_parameter_table_keys(fitted):
    model, _ = fitted
    assert set(model.parameter_table()) == set(PARAMETER_NAMES)


def test_parameters_are_inside_the_declared_bounds(fitted):
    model, _ = fitted
    table = model.parameter_table()
    assert 1e-4 <= table["cd0"] <= 0.5
    assert 0.05 <= table["oswald"] <= 1.0
    assert 0.05 <= table["eta_prop"] <= 1.0
    assert 0.0 <= table["avionics_power"] <= 200.0


def test_fitted_cd0_is_near_the_truth(fitted):
    # The hypothesis class is missing the efficiency curvature, so the fit
    # absorbs part of it into the drag terms. Within 30 % is what is measured,
    # not a tight recovery claim.
    model, _ = fitted
    assert model.parameter_table()["cd0"] == pytest.approx(DEFAULT_AIRFRAME.cd0, rel=0.30)


def test_predictions_are_positive(fitted):
    model, data = fitted
    assert np.all(model.predict(data.features) > 0.0)


def test_prediction_correlates_with_truth(fitted):
    model, data = fitted
    correlation = float(np.corrcoef(model.predict(data.features), data.truth)[0, 1])
    assert correlation > 0.95


def test_residual_rmse_is_within_a_factor_of_two_of_the_noise(fitted):
    model, data = fitted
    rmse = float(np.sqrt(np.mean((data.energy - model.predict(data.features)) ** 2)))
    noise = float(np.std(data.energy - data.truth))
    assert noise < rmse < 2.0 * noise


def test_misspecification_bias_is_airspeed_dependent(fitted):
    # The baseline cannot represent eta(V), so its bias must vary with airspeed.
    model, data = fitted
    bias = data.truth - model.predict(data.features)
    slow = bias[data.features[:, 0] < 19.0]
    fast = bias[data.features[:, 0] > 21.0]
    assert abs(float(np.mean(slow)) - float(np.mean(fast))) > 0.01


def test_predict_before_fit_is_refused():
    with pytest.raises(RuntimeError, match="fit"):
        PhysicsRegressor().predict(np.zeros((2, 5)))


def test_parameters_before_fit_is_refused():
    with pytest.raises(RuntimeError, match="fit"):
        _ = PhysicsRegressor().parameters


def test_fit_rejects_wrong_feature_width():
    with pytest.raises(ValueError, match="shape"):
        PhysicsRegressor().fit(np.zeros((10, 4)), np.zeros(10))


def test_fit_rejects_mismatched_target():
    with pytest.raises(ValueError, match="energy"):
        PhysicsRegressor().fit(np.zeros((10, 5)), np.zeros(9))


def test_fit_rejects_too_few_samples():
    with pytest.raises(ValueError, match="more than 4 samples"):
        PhysicsRegressor().fit(np.ones((4, 5)), np.ones(4))


def test_predict_rejects_wrong_feature_width(fitted):
    model, _ = fitted
    with pytest.raises(ValueError, match="shape"):
        model.predict(np.zeros((3, 2)))


def test_gaussian_sigma_recovers_a_known_standard_deviation():
    rng = np.random.default_rng(57502)
    truth = rng.normal(0.0, 0.25, 50_000)
    interval = GaussianResidualInterval(0.1).fit(truth, np.zeros(50_000))
    assert interval.sigma == pytest.approx(0.25, rel=0.02)


def test_gaussian_half_width_is_z_times_sigma():
    # z_{0.95} = 1.6448536269514722 for alpha = 0.1.
    interval = GaussianResidualInterval(0.1).fit(np.array([1.0, -1.0]), np.zeros(2))
    assert interval.sigma == pytest.approx(1.0, rel=1e-15)
    assert interval.half_width == pytest.approx(1.6448536269514722, rel=1e-12)


def test_gaussian_t_quantile_is_wider_than_normal():
    rng = np.random.default_rng(57503)
    residual = rng.normal(0.0, 1.0, 12)
    normal = GaussianResidualInterval(0.1, use_t=False).fit(residual, np.zeros(12))
    student = GaussianResidualInterval(0.1, use_t=True).fit(residual, np.zeros(12))
    assert student.half_width > normal.half_width


def test_gaussian_dof_correction_raises_sigma():
    rng = np.random.default_rng(57504)
    residual = rng.normal(0.0, 1.0, 20)
    plain = GaussianResidualInterval(0.1, n_parameters=0).fit(residual, np.zeros(20))
    corrected = GaussianResidualInterval(0.1, n_parameters=4).fit(residual, np.zeros(20))
    assert corrected.sigma > plain.sigma
    assert corrected.sigma == pytest.approx(plain.sigma * np.sqrt(20 / 16), rel=1e-12)


def test_gaussian_coverage_on_gaussian_residuals():
    rng = np.random.default_rng(57505)
    residual = rng.normal(0.0, 0.3, 40_000)
    model = GaussianResidualInterval(0.1).fit(residual, np.zeros(40_000))
    interval = model.interval(np.zeros(40_000))
    assert float(interval.covers(residual).mean()) == pytest.approx(0.9, abs=0.01)


def test_gaussian_interval_is_constant_width():
    model = GaussianResidualInterval(0.1).fit(np.array([1.0, -1.0]), np.zeros(2))
    interval = model.interval(np.array([1.0, 50.0]))
    assert float(interval.width[0]) == pytest.approx(float(interval.width[1]), rel=0.0)


def test_gaussian_refuses_before_fit():
    with pytest.raises(RuntimeError, match="fit"):
        _ = GaussianResidualInterval(0.1).sigma


def test_gaussian_refuses_half_width_before_fit():
    with pytest.raises(RuntimeError, match="fit"):
        _ = GaussianResidualInterval(0.1).half_width


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
def test_gaussian_rejects_bad_alpha(alpha):
    with pytest.raises(ValueError, match="alpha"):
        GaussianResidualInterval(alpha)


def test_gaussian_rejects_negative_parameter_count():
    with pytest.raises(ValueError, match="n_parameters"):
        GaussianResidualInterval(0.1, n_parameters=-1)


def test_gaussian_rejects_zero_degrees_of_freedom():
    with pytest.raises(ValueError, match="degrees of freedom"):
        GaussianResidualInterval(0.1, n_parameters=4).fit(np.zeros(4), np.zeros(4))


def test_gaussian_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="must agree"):
        GaussianResidualInterval(0.1).fit(np.zeros(5), np.zeros(4))
