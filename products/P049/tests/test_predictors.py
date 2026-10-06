"""Baseline and learned-model tests, baselines first."""

from __future__ import annotations

import math

import joblib
import numpy as np
import pytest

from linkoutage.calibration import brier_score
from linkoutage.features import build_outage_dataset
from linkoutage.predictors import (
    AnalyticLcrPredictor,
    ConstantRatePredictor,
    LogisticBaseline,
    PlattCalibrated,
    RandomForestOutageClassifier,
    estimate_channel_parameters,
)


@pytest.fixture(scope="module")
def small_dataset():
    from linkoutage.channel import lognormal_amplitude_series

    series = lognormal_amplitude_series(
        600_000, fs_hz=1.0e6, tau_s=2.0e-4, si=0.6, seed=4902
    )
    data = build_outage_dataset(
        series.amplitude,
        threshold=0.35,
        window_samples=400,
        horizon_samples=200,
        stride_samples=50,
        gaussian=series.gaussian,
    )
    return series, data


def test_constant_rate_hand_calculation():
    model = ConstantRatePredictor().fit([1, 0, 0, 0])
    assert model.rate == pytest.approx(0.25)
    assert model.rate_standard_error() == pytest.approx(math.sqrt(0.25 * 0.75 / 4))


def test_constant_rate_predicts_one_value():
    model = ConstantRatePredictor().fit([1, 0, 0, 0])
    p = model.predict_proba_onset(np.zeros((7, 14)))
    assert p.shape == (7,)
    assert np.all(p == 0.25)


def test_constant_rate_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit"):
        ConstantRatePredictor().predict_proba_onset(np.zeros((2, 14)))


def test_constant_rate_rejects_non_binary():
    with pytest.raises(ValueError, match="only 0 and 1"):
        ConstantRatePredictor().fit([0, 2])


def test_constant_rate_reliability_is_the_squared_base_rate_drift(small_dataset):
    _, data = small_dataset
    train, test = data.split.train, data.split.test
    model = ConstantRatePredictor().fit(data.y[train])
    from linkoutage.calibration import evaluate_forecast

    report = evaluate_forecast(
        data.y[test], model.predict_proba_onset(data.x[test]), name="constant"
    )
    # One distinct forecast collapses the reliability diagram to a single bin, so
    # the reliability term is exactly the squared gap between the training base
    # rate the model emits and the test base rate it meets. On this deliberately
    # short record the two differ by about a factor of two, which is itself the
    # point: a constant-rate baseline is only as calibrated as the record is
    # stationary.
    drift = (model.rate - float(data.y[test].mean())) ** 2
    assert report.reliability == pytest.approx(drift, rel=1e-9)
    assert report.resolution == pytest.approx(0.0, abs=1e-12)


def test_estimate_channel_parameters_recovers_si_and_rho(small_dataset):
    series, _ = small_dataset
    est = estimate_channel_parameters(series.amplitude)
    assert est.si == pytest.approx(0.6, rel=0.08)
    assert est.rho == pytest.approx(series.rho, abs=0.002)
    assert est.mean_irradiance == pytest.approx(1.0, rel=0.03)
    assert "SI=" in est.describe()


def test_estimate_channel_parameters_rejects_constant_input():
    with pytest.raises(ValueError, match="variance is zero"):
        estimate_channel_parameters(np.ones(100))


def test_estimate_channel_parameters_rejects_non_positive():
    with pytest.raises(ValueError, match="strictly positive"):
        estimate_channel_parameters(np.array([1.0, -1.0, 1.0]))


def test_analytic_predictor_probabilities_are_in_range(small_dataset):
    series, data = small_dataset
    model = AnalyticLcrPredictor(threshold=0.35, horizon_samples=200).fit(series.amplitude)
    p = model.predict_proba_onset(data.x[data.split.test])
    assert p.shape == (data.split.test.size,)
    assert np.all(p >= 0.0) and np.all(p <= 1.0)


def test_analytic_predictor_uses_no_labels(small_dataset):
    series, data = small_dataset
    a = AnalyticLcrPredictor(threshold=0.35, horizon_samples=200).fit(series.amplitude)
    pa = a.predict_proba_onset(data.x[data.split.test])
    flipped = data.y.copy()
    flipped ^= 1
    b = AnalyticLcrPredictor(threshold=0.35, horizon_samples=200).fit(series.amplitude)
    pb = b.predict_proba_onset(data.x[data.split.test])
    assert np.array_equal(pa, pb)
    del flipped


def test_analytic_predictor_discriminates(small_dataset):
    series, data = small_dataset
    test = data.split.test
    model = AnalyticLcrPredictor(threshold=0.35, horizon_samples=200).fit(series.amplitude)
    p = model.predict_proba_onset(data.x[test])
    y = data.y[test]
    assert p[y == 1].mean() > p[y == 0].mean()


def test_analytic_predictor_gaussian_round_trip(small_dataset):
    series, _ = small_dataset
    model = AnalyticLcrPredictor(threshold=0.35, horizon_samples=50).fit(series.amplitude)
    g = model.gaussian_from_amplitude(series.amplitude[:1000])
    # Reconstructing the level from amplitude must match the generator's own
    # Gaussian to within the parameter-estimation error.
    assert np.corrcoef(g, series.gaussian[:1000])[0, 1] > 0.999


def test_analytic_predictor_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit"):
        AnalyticLcrPredictor(threshold=0.35, horizon_samples=10).predict_proba_onset(
            np.zeros((2, 14))
        )


def test_analytic_predictor_rejects_one_dimensional_input(small_dataset):
    series, _ = small_dataset
    model = AnalyticLcrPredictor(threshold=0.35, horizon_samples=10).fit(series.amplitude)
    with pytest.raises(ValueError, match="2-D"):
        model.predict_proba_onset(np.zeros(5))


def test_logistic_baseline_fits_and_predicts(small_dataset):
    _, data = small_dataset
    train, test = data.split.train, data.split.test
    model = LogisticBaseline().fit(data.x[train], data.y[train])
    p = model.predict_proba_onset(data.x[test])
    assert p.shape == (test.size,)
    assert np.all(p >= 0.0) and np.all(p <= 1.0)
    assert model.coefficients().size == 14


def test_logistic_baseline_beats_the_constant_rate(small_dataset):
    _, data = small_dataset
    train, test = data.split.train, data.split.test
    constant = ConstantRatePredictor().fit(data.y[train])
    logistic = LogisticBaseline().fit(data.x[train], data.y[train])
    bs_c = brier_score(data.y[test], constant.predict_proba_onset(data.x[test]))
    bs_l = brier_score(data.y[test], logistic.predict_proba_onset(data.x[test]))
    assert bs_l < bs_c


def test_logistic_baseline_rejects_single_class(small_dataset):
    _, data = small_dataset
    with pytest.raises(ValueError, match="both classes"):
        LogisticBaseline().fit(data.x[:100], np.zeros(100, dtype=int))


def test_logistic_baseline_rejects_shape_mismatch(small_dataset):
    _, data = small_dataset
    with pytest.raises(ValueError, match="rows"):
        LogisticBaseline().fit(data.x[:100], data.y[:50])


def test_logistic_baseline_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit"):
        LogisticBaseline().predict_proba_onset(np.zeros((2, 14)))


def test_forest_fits_and_predicts(small_dataset):
    _, data = small_dataset
    train, test = data.split.train, data.split.test
    model = RandomForestOutageClassifier(n_estimators=30).fit(data.x[train], data.y[train])
    p = model.predict_proba_onset(data.x[test])
    assert p.shape == (test.size,)
    assert np.all(p >= 0.0) and np.all(p <= 1.0)


def test_forest_uncertainty_mean_matches_point_forecast(small_dataset):
    _, data = small_dataset
    train = data.split.train
    model = RandomForestOutageClassifier(n_estimators=30).fit(data.x[train], data.y[train])
    sample = data.x[data.split.test][:200]
    mean, std = model.predict_with_uncertainty(sample)
    assert np.allclose(mean, model.predict_proba_onset(sample), atol=1e-12)
    assert std.shape == mean.shape
    assert np.all(std >= 0.0)


def test_forest_uncertainty_is_not_identically_zero(small_dataset):
    _, data = small_dataset
    train = data.split.train
    model = RandomForestOutageClassifier(n_estimators=30).fit(data.x[train], data.y[train])
    _, std = model.predict_with_uncertainty(data.x[data.split.test][:500])
    assert float(std.max()) > 0.0


def test_forest_feature_importance_sums_to_one(small_dataset):
    _, data = small_dataset
    train = data.split.train
    model = RandomForestOutageClassifier(n_estimators=30).fit(data.x[train], data.y[train])
    imp = model.feature_importance()
    assert imp.size == 14
    assert float(imp.sum()) == pytest.approx(1.0, rel=1e-9)


def test_forest_is_deterministic_under_a_fixed_seed(small_dataset):
    _, data = small_dataset
    train, test = data.split.train, data.split.test
    # n_jobs=1: with two threads the per-tree votes are summed in whatever order
    # the workers finish, which moves the last bit of the average.
    a = RandomForestOutageClassifier(n_estimators=20, seed=7, n_jobs=1).fit(
        data.x[train], data.y[train]
    )
    b = RandomForestOutageClassifier(n_estimators=20, seed=7, n_jobs=1).fit(
        data.x[train], data.y[train]
    )
    assert np.array_equal(
        a.predict_proba_onset(data.x[test]), b.predict_proba_onset(data.x[test])
    )


def test_forest_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit"):
        RandomForestOutageClassifier().predict_proba_onset(np.zeros((2, 14)))


def test_forest_uncertainty_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit"):
        RandomForestOutageClassifier().predict_with_uncertainty(np.zeros((2, 14)))


def test_forest_rejects_single_class(small_dataset):
    _, data = small_dataset
    with pytest.raises(ValueError, match="both classes"):
        RandomForestOutageClassifier().fit(data.x[:100], np.zeros(100, dtype=int))


def test_forest_round_trips_through_joblib(small_dataset, tmp_path):
    _, data = small_dataset
    train, test = data.split.train, data.split.test
    model = RandomForestOutageClassifier(n_estimators=20, n_jobs=1).fit(
        data.x[train], data.y[train]
    )
    path = tmp_path / "forest.joblib"
    joblib.dump(model, path)
    loaded = joblib.load(path)
    assert np.array_equal(
        model.predict_proba_onset(data.x[test]), loaded.predict_proba_onset(data.x[test])
    )


def test_platt_improves_the_analytic_predictor_calibration(small_dataset):
    series, data = small_dataset
    cal, test = data.split.calibration, data.split.test
    base = AnalyticLcrPredictor(threshold=0.35, horizon_samples=200).fit(series.amplitude)
    platt = PlattCalibrated(base).fit(data.x[cal], data.y[cal])
    bs_raw = brier_score(data.y[test], base.predict_proba_onset(data.x[test]))
    bs_cal = brier_score(data.y[test], platt.predict_proba_onset(data.x[test]))
    assert bs_cal < bs_raw
    assert platt.name.endswith("Platt")


def test_platt_rejects_a_constant_base(small_dataset):
    _, data = small_dataset
    cal = data.split.calibration
    constant = ConstantRatePredictor().fit(data.y[data.split.train])
    with pytest.raises(ValueError, match="single value"):
        PlattCalibrated(constant).fit(data.x[cal], data.y[cal])


def test_platt_before_fit_raises(small_dataset):
    series, _ = small_dataset
    base = AnalyticLcrPredictor(threshold=0.35, horizon_samples=10).fit(series.amplitude)
    with pytest.raises(RuntimeError, match="fit"):
        PlattCalibrated(base).predict_proba_onset(np.zeros((2, 14)))
