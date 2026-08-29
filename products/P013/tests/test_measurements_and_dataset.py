"""Tests for the synthetic measurement generator and the dataset builder."""

from __future__ import annotations

import numpy as np
import pytest

from turbscope import (
    FEATURE_NAMES,
    PathGeometry,
    SensorSuite,
    generate_dataset,
    scintillation_index,
    simulate_measurement,
)
from turbscope.dataset import REGIME_NAMES, regime_labels
from turbscope.measurements import ARCSEC_TO_RAD


def test_sampled_scintillation_index_converges_to_the_forward_model():
    """Sampling the gamma-gamma model must reproduce sigma_I^2 in the mean."""
    path = PathGeometry(1000.0, 1550e-9)
    suite = SensorSuite(n_irradiance_samples=20000, n_dimm_frames=100)
    z = path.uniform_grid(201)
    rng = np.random.default_rng(4242)
    for level in (2e-16, 2e-15, 2e-14):
        cn2 = np.full_like(z, level)
        estimates = [
            simulate_measurement(z, cn2, path, suite, rng).sigma_i2_point for _ in range(6)
        ]
        target = float(scintillation_index(
            simulate_measurement(z, cn2, path, suite, rng).true_beta0_sq, 0.0
        ))
        assert float(np.mean(estimates)) == pytest.approx(target, rel=0.15)


def test_aperture_channel_is_always_weaker_than_the_point_channel():
    path = PathGeometry(1500.0, 850e-9)
    suite = SensorSuite(receiver_diameter_m=0.25, n_irradiance_samples=5000)
    z = path.uniform_grid(201)
    cn2 = np.full_like(z, 1e-14)
    meas = simulate_measurement(z, cn2, path, suite, np.random.default_rng(1))
    assert meas.true_sigma_i2_aperture < meas.true_sigma_i2_point


def test_dimm_noise_floor_biases_the_reading_upward():
    path = PathGeometry(1000.0, 1550e-9)
    z = path.uniform_grid(201)
    cn2 = np.full_like(z, 3e-17)  # very weak: the noise floor dominates
    quiet = SensorSuite(n_dimm_frames=20000, dimm_noise_arcsec=0.0)
    noisy = SensorSuite(n_dimm_frames=20000, dimm_noise_arcsec=0.10)
    a = simulate_measurement(z, cn2, path, quiet, np.random.default_rng(3))
    b = simulate_measurement(z, cn2, path, noisy, np.random.default_rng(3))
    assert b.sigma_l2_rad2 > a.sigma_l2_rad2
    assert noisy.dimm_noise_variance_rad2 == pytest.approx((0.10 * ARCSEC_TO_RAD) ** 2, rel=1e-12)


def test_simulated_dimm_variance_is_unbiased():
    path = PathGeometry(1000.0, 1550e-9)
    suite = SensorSuite(n_dimm_frames=1000, dimm_noise_arcsec=0.0)
    z = path.uniform_grid(201)
    cn2 = np.full_like(z, 5e-15)
    rng = np.random.default_rng(77)
    vals = [simulate_measurement(z, cn2, path, suite, rng).sigma_l2_rad2 for _ in range(80)]
    truth = simulate_measurement(z, cn2, path, suite, rng).true_sigma_l2_rad2
    assert float(np.mean(vals)) == pytest.approx(truth, rel=0.02)


def test_dataset_shapes_and_feature_names():
    data = generate_dataset(120, seed=1234)
    assert data.x.shape == (120, len(FEATURE_NAMES))
    assert data.y.shape == (120,)
    assert len(FEATURE_NAMES) == 13
    assert np.all(np.isfinite(data.x))
    assert np.all(np.isfinite(data.y))


def test_dataset_target_is_the_scintillation_weighted_average():
    data = generate_dataset(80, seed=55)
    assert np.allclose(data.y, np.log10(data.cn2_scint))
    # the coherence-weighted average differs -- that is the sensor-bias effect
    ratio = data.cn2_scint / data.cn2_coherence
    assert float(np.std(np.log10(ratio))) > 0.05


def test_dataset_take_preserves_rows():
    data = generate_dataset(60, seed=8)
    idx = np.array([0, 5, 17])
    sub = data.take(idx)
    assert np.array_equal(sub.x, data.x[idx])
    assert np.array_equal(sub.beta0_sq, data.beta0_sq[idx])


def test_regime_labels_boundaries():
    labels = regime_labels(np.array([0.01, 0.29, 0.31, 0.99, 1.01, 4.9, 5.1, 100.0]))
    assert list(labels) == [0, 0, 1, 1, 2, 2, 3, 3]
    assert REGIME_NAMES[labels[0]] == "weak"
    assert REGIME_NAMES[labels[-1]] == "saturated"


def test_all_regimes_are_populated_in_the_default_dataset():
    data = generate_dataset(600, seed=20260829)
    counts = np.bincount(data.regimes(), minlength=4)
    assert np.all(counts > 20), f"regime counts {counts} must all be populated"
