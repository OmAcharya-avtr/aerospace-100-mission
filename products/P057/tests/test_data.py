"""Unit tests for the synthetic dataset generator."""

from __future__ import annotations

import numpy as np
import pytest

from conformalband.data import (
    AIRSPEED_MEAN,
    DISTANCE_MEAN,
    FEATURE_NAMES,
    FEATURE_UNITS,
    NOISE_FRACTION,
    Dataset,
    make_audit_split,
    make_dataset,
    sample_covariates,
)
from conformalband.shift import CALIBRATION_MASS_MEAN, CovariateShift


def test_feature_names_and_units_align():
    assert len(FEATURE_NAMES) == len(FEATURE_UNITS) == 5


def test_make_dataset_shapes():
    data = make_dataset(300, seed=57401)
    assert data.features.shape == (300, 5)
    assert data.energy.shape == (300,)
    assert data.truth.shape == (300,)
    assert len(data) == 300


def test_make_dataset_is_deterministic():
    a = make_dataset(200, seed=57402)
    b = make_dataset(200, seed=57402)
    assert np.array_equal(a.features, b.features)
    assert np.array_equal(a.energy, b.energy)


def test_different_seeds_give_different_data():
    a = make_dataset(200, seed=57403)
    b = make_dataset(200, seed=57404)
    assert not np.allclose(a.energy, b.energy)


def test_energy_is_positive():
    data = make_dataset(2000, seed=57405)
    assert np.all(data.energy > 0.0)
    assert np.all(data.truth > 0.0)


def test_noise_is_multiplicative_in_scale():
    data = make_dataset(20000, seed=57406)
    relative = (data.energy - data.truth) / data.truth
    assert float(np.std(relative)) == pytest.approx(NOISE_FRACTION, rel=0.05)
    assert abs(float(np.mean(relative))) < 0.002


def test_zero_noise_gives_the_truth():
    data = make_dataset(100, seed=57407, noise_fraction=0.0)
    assert np.allclose(data.energy, data.truth, rtol=0.0, atol=0.0)


def test_residual_scale_grows_with_energy():
    data = make_dataset(20000, seed=57408)
    order = np.argsort(data.truth)
    residual = np.abs(data.energy - data.truth)[order]
    low = float(np.mean(residual[:5000]))
    high = float(np.mean(residual[-5000:]))
    assert high > 1.3 * low


def test_covariate_means_match_the_declaration():
    data = make_dataset(40000, seed=57409)
    assert float(np.mean(data.features[:, 0])) == pytest.approx(AIRSPEED_MEAN, abs=0.05)
    assert float(np.mean(data.mass)) == pytest.approx(CALIBRATION_MASS_MEAN, abs=0.02)
    assert float(np.mean(data.features[:, 4])) == pytest.approx(DISTANCE_MEAN, abs=2.0)


@pytest.mark.parametrize("severity", [1.0, 2.0, 3.0])
def test_shifted_mass_mean_matches_the_declaration(severity):
    data = make_dataset(40000, seed=57410, severity=severity)
    expected = CovariateShift(severity=severity).mass_mean
    assert float(np.mean(data.mass)) == pytest.approx(expected, abs=0.02)


@pytest.mark.parametrize("severity", [1.0, 2.0, 3.0])
def test_shifted_headwind_mean_matches_the_declaration(severity):
    data = make_dataset(40000, seed=57411, severity=severity)
    expected = CovariateShift(severity=severity).headwind_mean
    assert float(np.mean(data.headwind)) == pytest.approx(expected, abs=0.05)


def test_shift_raises_mean_energy():
    low = make_dataset(20000, seed=57412, severity=0.0)
    high = make_dataset(20000, seed=57412, severity=3.0)
    assert float(np.mean(high.truth)) > float(np.mean(low.truth))


def test_explicit_shift_overrides_severity():
    data = make_dataset(10, seed=57413, severity=0.0, shift=CovariateShift(severity=2.0))
    assert data.shift.severity == 2.0


def test_sample_covariates_uses_the_given_generator():
    rng = np.random.default_rng(57414)
    a = sample_covariates(50, rng)
    rng = np.random.default_rng(57414)
    b = sample_covariates(50, rng)
    assert np.array_equal(a, b)


def test_audit_split_sizes():
    split = make_audit_split(seed=57415, n_fit=200, n_calibration=100, n_test=150, severity=1.0)
    assert len(split.fit) == 200
    assert len(split.calibration) == 100
    assert len(split.test) == 150
    assert split.shift.severity == 1.0


def test_audit_split_samples_are_disjoint_draws():
    split = make_audit_split(seed=57416, n_fit=100, n_calibration=100, n_test=100)
    assert not np.allclose(split.fit.energy, split.calibration.energy)


def test_audit_split_calibration_is_in_distribution():
    split = make_audit_split(seed=57417, n_fit=10, n_calibration=10, n_test=10, severity=3.0)
    assert split.calibration.shift.is_identity
    assert not split.test.shift.is_identity


def test_make_dataset_rejects_zero_samples():
    with pytest.raises(ValueError, match="n_samples"):
        make_dataset(0, seed=1)


def test_make_dataset_rejects_negative_noise():
    with pytest.raises(ValueError, match="noise_fraction"):
        make_dataset(10, seed=1, noise_fraction=-0.1)


def test_dataset_rejects_wrong_feature_width():
    with pytest.raises(ValueError, match="shape"):
        Dataset(
            features=np.zeros((4, 3)),
            energy=np.zeros(4),
            truth=np.zeros(4),
            shift=CovariateShift(),
        )


def test_dataset_rejects_mismatched_energy():
    with pytest.raises(ValueError, match="energy"):
        Dataset(
            features=np.zeros((4, 5)),
            energy=np.zeros(3),
            truth=np.zeros(3),
            shift=CovariateShift(),
        )


def test_dataset_rejects_mismatched_truth():
    with pytest.raises(ValueError, match="truth"):
        Dataset(
            features=np.zeros((4, 5)),
            energy=np.zeros(4),
            truth=np.zeros(3),
            shift=CovariateShift(),
        )
