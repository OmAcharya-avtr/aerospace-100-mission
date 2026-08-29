"""Tests for turbscope.model."""

from __future__ import annotations

import numpy as np
import pytest

from turbscope.dataset import FEATURE_NAMES
from turbscope.dimm import invert_cn2_from_variance
from turbscope.model import (
    DimmOnlyBaseline,
    MeanTrainingBaseline,
    ScintillometerWeakBaseline,
    TurbScopeModel,
    interval_coverage,
    train_default_model,
)
from turbscope.scintillometer import invert_cn2_weak
from turbscope.synthetic import (
    APERTURE_DIAM_M,
    DIMM_WAVELENGTH_M,
    SCINT_WAVELENGTH_M,
    SEPARATION_M,
    WAVE_TYPE,
)


def _row(sigma_i2, var_l, var_t, length):
    return np.array([[np.log10(sigma_i2), np.log10(var_l), np.log10(var_t), np.log10(length)]])


# ------------------------------------------------------------- baselines
def test_scintillometer_weak_baseline_matches_closed_form():
    x = _row(0.05, 1e-12, 1e-12, 500.0)
    base = ScintillometerWeakBaseline()
    pred = 10.0 ** base.predict_log10_cn2(x)[0]
    expected = invert_cn2_weak(0.05, 500.0, SCINT_WAVELENGTH_M, WAVE_TYPE)
    assert pred == pytest.approx(expected, rel=1e-6)


def test_dimm_only_baseline_matches_closed_form_average():
    x = _row(0.05, 1e-12, 8e-13, 500.0)
    base = DimmOnlyBaseline()
    pred = 10.0 ** base.predict_log10_cn2(x)[0]
    cn2_l = invert_cn2_from_variance(
        1e-12, 500.0, DIMM_WAVELENGTH_M, APERTURE_DIAM_M, SEPARATION_M, "longitudinal"
    )
    cn2_t = invert_cn2_from_variance(
        8e-13, 500.0, DIMM_WAVELENGTH_M, APERTURE_DIAM_M, SEPARATION_M, "transverse"
    )
    assert pred == pytest.approx(0.5 * (cn2_l + cn2_t), rel=1e-6)


def test_mean_training_baseline_ignores_inputs():
    y = np.array([-15.0, -14.0, -13.0])
    base = MeanTrainingBaseline().fit(None, y)
    x1 = _row(0.01, 1e-13, 1e-13, 200.0)
    x2 = _row(1.0, 1e-10, 1e-10, 2000.0)
    p1 = base.predict_log10_cn2(x1)[0]
    p2 = base.predict_log10_cn2(x2)[0]
    assert p1 == pytest.approx(p2)
    assert p1 == pytest.approx(np.mean(y))


def test_mean_training_baseline_requires_fit_before_predict():
    base = MeanTrainingBaseline()
    with pytest.raises(RuntimeError):
        base.predict_log10_cn2(_row(0.01, 1e-13, 1e-13, 200.0))


def test_mean_training_baseline_rejects_empty_target():
    with pytest.raises(ValueError):
        MeanTrainingBaseline().fit(None, np.array([]))


# ------------------------------------------------------------------ model
def test_turbscope_model_requires_fit_before_predict():
    model = TurbScopeModel()
    with pytest.raises(RuntimeError):
        model.predict_log10_cn2(_row(0.01, 1e-13, 1e-13, 200.0))
    with pytest.raises(RuntimeError):
        model.predict(0.01, 1e-13, 1e-13, 200.0)
    with pytest.raises(RuntimeError):
        model.fit_report()


def test_turbscope_model_rejects_bad_coverage():
    with pytest.raises(ValueError):
        TurbScopeModel(coverage=0.0)
    with pytest.raises(ValueError):
        TurbScopeModel(coverage=1.0)


def test_turbscope_model_fit_rejects_shape_mismatch():
    model = TurbScopeModel()
    x = np.random.default_rng(0).normal(size=(10, 4))
    y = np.zeros(9)
    with pytest.raises(ValueError):
        model.fit(x, y)


def test_turbscope_model_fit_rejects_wrong_column_count():
    model = TurbScopeModel()
    x = np.random.default_rng(0).normal(size=(10, 3))
    y = np.zeros(10)
    with pytest.raises(ValueError):
        model.fit(x, y)


def test_turbscope_model_fit_rejects_non_finite():
    model = TurbScopeModel()
    x = np.full((10, 4), np.nan)
    y = np.zeros(10)
    with pytest.raises(ValueError):
        model.fit(x, y)


def test_turbscope_model_predict_rejects_non_positive_inputs():
    model, _ = train_default_model(n_scenarios=60, n_realisations=2, calibrate=False)
    with pytest.raises(ValueError):
        model.predict(-1.0, 1e-12, 1e-12, 500.0)
    with pytest.raises(ValueError):
        model.predict(0.05, 0.0, 1e-12, 500.0)
    with pytest.raises(ValueError):
        model.predict(0.05, 1e-12, 1e-12, -500.0)


def test_calibrate_requires_fit_first():
    model = TurbScopeModel()
    with pytest.raises(RuntimeError):
        model.calibrate(np.zeros((30, 4)), np.zeros(30))


def test_calibrate_requires_enough_points():
    model, art = train_default_model(n_scenarios=60, n_realisations=2, calibrate=False)
    with pytest.raises(ValueError):
        model.calibrate(art["x_cal"][:5], art["y_cal"][:5])


def test_calibrate_rejects_shape_mismatch():
    model, art = train_default_model(n_scenarios=60, n_realisations=2, calibrate=False)
    with pytest.raises(ValueError):
        model.calibrate(art["x_cal"], art["y_cal"][:-1])


# ---------------------------------------------------------- interval_coverage
def test_interval_coverage_known_case():
    y = np.array([1.0, 2.0, 3.0, 10.0])
    lo = np.array([0.0, 0.0, 0.0, 0.0])
    hi = np.array([5.0, 5.0, 5.0, 5.0])
    cov, width = interval_coverage(y, lo, hi)
    assert cov == pytest.approx(0.75)  # 3 of 4 inside [0,5]
    assert width == pytest.approx(5.0)


def test_interval_coverage_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        interval_coverage(np.array([1.0]), np.array([0.0, 0.0]), np.array([1.0]))


def test_interval_coverage_rejects_empty():
    with pytest.raises(ValueError):
        interval_coverage(np.array([]), np.array([]), np.array([]))


# ------------------------------------------------------------- integration
def test_train_default_model_is_reproducible(small_model):
    model1, art1 = small_model
    model2, art2 = train_default_model(
        n_scenarios=180, n_realisations=2, random_state=11
    )
    p1 = model1.predict_log10_cn2(art1["x_test"])
    p2 = model2.predict_log10_cn2(art2["x_test"])
    np.testing.assert_array_equal(art1["x_test"], art2["x_test"])
    np.testing.assert_allclose(p1, p2, atol=0.0)


def test_train_default_model_fit_report_has_no_crossing_blowup(small_model):
    model, _ = small_model
    report = model.fit_report()
    assert 0.0 <= report["quantile_crossing_fraction"] <= 0.15


def test_prediction_interval_contains_median(small_model):
    model, art = small_model
    lo, mid, hi = model._three(art["x_test"])  # noqa: SLF001 - internal, but stable API for tests
    assert np.all(lo <= mid + 1e-12)
    assert np.all(mid <= hi + 1e-12)


def test_predict_returns_sane_prediction(small_model):
    model, _ = small_model
    pred = model.predict(0.05, 1e-12, 8e-13, 500.0)
    assert pred.cn2_path > 0.0
    assert pred.cn2_lower <= pred.cn2_path <= pred.cn2_upper
    assert pred.coverage == pytest.approx(0.90)
    assert pred.interval_width_dex >= 0.0


def test_feature_names_length_matches_model_input_dim():
    assert len(FEATURE_NAMES) == 4
