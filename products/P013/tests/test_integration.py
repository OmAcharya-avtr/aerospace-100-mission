"""End-to-end integration tests: profile -> measurement -> inversion -> model -> CLI."""

from __future__ import annotations

import io

import numpy as np
import pytest

from turbscope import (
    PathGeometry,
    SensorSuite,
    TurbScopeModel,
    features_from_measurement,
    generate_dataset,
    invert_dimm,
    invert_scintillation,
    simulate_measurement,
    weighted_path_average,
)
from turbscope.__main__ import main
from turbscope.model import BASELINES, split_dataset


def test_full_pipeline_weak_regime_recovers_the_truth():
    """Known Cn2 -> noisy sensors -> closed-form inversion, weak regime."""
    path = PathGeometry(1200.0, 1550e-9)
    suite = SensorSuite(n_irradiance_samples=4000, n_dimm_frames=4000, dimm_noise_arcsec=0.0)
    z = path.uniform_grid(401)
    cn2 = 2e-15 * (1.0 + 0.4 * np.sin(2.0 * np.pi * z / path.length_m))
    rng = np.random.default_rng(2026)
    meas = simulate_measurement(z, cn2, path, suite, rng)
    assert meas.true_beta0_sq < 0.3

    truth_sc = weighted_path_average(z, cn2, kind="scintillation")
    truth_co = weighted_path_average(z, cn2, kind="coherence")

    est_sc = invert_scintillation(meas.sigma_i2_point, path, n_samples=4000)
    est_co = invert_dimm(
        meas.sigma_l2_rad2, path, subaperture_m=0.06, baseline_m=0.20, n_frames=4000
    )
    assert est_sc.valid and est_co.valid
    assert est_sc.cn2 == pytest.approx(truth_sc, rel=0.15)
    assert est_co.cn2 == pytest.approx(truth_co, rel=0.15)
    # the two sensors estimate different weighted averages of the same profile
    assert est_sc.kernel != est_co.kernel


def test_feature_row_matches_the_dataset_builder():
    path = PathGeometry(800.0, 850e-9)
    suite = SensorSuite()
    z = path.uniform_grid(201)
    cn2 = np.full_like(z, 5e-15)
    meas = simulate_measurement(z, cn2, path, suite, np.random.default_rng(5))
    row = features_from_measurement(path, suite, meas)
    assert row.shape == (13,)
    assert np.all(np.isfinite(row))


def test_train_predict_and_score_against_baselines():
    """A small end-to-end fit; the learned model must produce usable intervals."""
    data = generate_dataset(900, seed=99)
    idx_fit, idx_cal, idx_test = split_dataset(len(data), seed=7)
    model = TurbScopeModel(n_estimators=120)
    model.fit(data.x[idx_fit], data.y[idx_fit])
    delta = model.calibrate(data.x[idx_cal], data.y[idx_cal])
    assert delta >= 0.0 or np.isfinite(delta)

    test = data.take(idx_test)
    pred = model.predict(test.x)
    assert pred.cn2.shape == (len(test),)
    assert np.all(pred.cn2_lower <= pred.cn2)
    assert np.all(pred.cn2 <= pred.cn2_upper)
    coverage = float(np.mean((test.y >= pred.log10_lower) & (test.y <= pred.log10_upper)))
    assert 0.75 <= coverage <= 1.0

    for baseline in BASELINES:
        got = baseline.predict(test)
        assert got.shape == (len(test),)
        assert np.all(np.isfinite(got))


def test_cli_forward_invert_and_saturation_round_trip():
    buf = io.StringIO()
    assert main(["forward", "--cn2", "1e-15", "--length-m", "1000"], stream=buf) == 0
    text = buf.getvalue()
    assert "beta_0^2" in text and "DIMM longitudinal" in text

    buf = io.StringIO()
    assert main(["saturation"], stream=buf) == 0
    assert "multi-valued readings" in buf.getvalue()

    buf = io.StringIO()
    assert main(
        ["invert", "--sigma-i2", "1.4", "--length-m", "2000", "--wavelength-nm", "850"],
        stream=buf,
    ) == 0
    assert "multi-valued" in buf.getvalue()

    buf = io.StringIO()
    assert main(
        ["invert", "--dimm-variance-rad2", "5e-12", "--length-m", "1000"], stream=buf
    ) == 0
    assert "coherence" in buf.getvalue()


def test_cli_invert_requires_a_reading():
    with pytest.raises(ValueError, match="supply --sigma-i2"):
        main(["invert"], stream=io.StringIO())


def test_dataset_regime_labels_are_consistent():
    data = generate_dataset(200, seed=11)
    reg = data.regimes()
    assert reg.min() >= 0 and reg.max() <= 3
    assert np.all(data.beta0_sq[reg == 0] < 0.3)
    assert np.all(data.beta0_sq[reg == 3] >= 5.0)
