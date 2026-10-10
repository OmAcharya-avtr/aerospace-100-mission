"""Unit tests for the declared covariate shift."""

from __future__ import annotations

import math

import numpy as np
import pytest

from conformalband.shift import (
    CALIBRATION_HEADWIND_MEAN,
    CALIBRATION_HEADWIND_SD,
    CALIBRATION_MASS_MEAN,
    CALIBRATION_MASS_SD,
    HEADWIND_SHIFT_PER_SEVERITY,
    MASS_SHIFT_PER_SEVERITY,
    SHIFT_LEVELS,
    CovariateShift,
)


def test_shift_levels_are_the_audit_levels():
    assert SHIFT_LEVELS == (0.0, 1.0, 2.0, 3.0)


def test_identity_shift():
    shift = CovariateShift()
    assert shift.is_identity
    assert shift.mass_mean == CALIBRATION_MASS_MEAN
    assert shift.headwind_mean == CALIBRATION_HEADWIND_MEAN
    assert shift.mahalanobis == 0.0


@pytest.mark.parametrize("severity", [0.5, 1.0, 2.0, 3.0])
def test_non_identity_shift(severity):
    shift = CovariateShift(severity=severity)
    assert not shift.is_identity
    assert shift.mass_delta == pytest.approx(MASS_SHIFT_PER_SEVERITY * severity, rel=1e-15)
    assert shift.headwind_delta == pytest.approx(
        HEADWIND_SHIFT_PER_SEVERITY * severity, rel=1e-15
    )


@pytest.mark.parametrize("severity", [0.0, 1.0, 2.0, 3.0, 4.0])
def test_mahalanobis_is_linear_in_severity(severity):
    unit = CovariateShift(severity=1.0).mahalanobis
    assert CovariateShift(severity=severity).mahalanobis == pytest.approx(
        severity * unit, rel=1e-13
    )


def test_mass_and_headwind_shifts_are_both_three_tenths_of_a_sigma():
    assert MASS_SHIFT_PER_SEVERITY / CALIBRATION_MASS_SD == pytest.approx(0.3, rel=1e-15)
    assert HEADWIND_SHIFT_PER_SEVERITY / CALIBRATION_HEADWIND_SD == pytest.approx(0.3, rel=1e-15)


def test_scaled_halves_the_severity():
    assert CovariateShift(severity=2.0).scaled(0.5).severity == pytest.approx(1.0, rel=1e-15)


def test_scaled_zero_is_the_identity():
    assert CovariateShift(severity=3.0).scaled(0.0).is_identity


def test_scaled_preserves_standard_deviations():
    shift = CovariateShift(severity=2.0, mass_sd=0.7, headwind_sd=1.9).scaled(0.4)
    assert shift.mass_sd == 0.7
    assert shift.headwind_sd == 1.9


@pytest.mark.parametrize("severity", [0.0, 1.0, 2.0, 3.0])
def test_likelihood_ratio_has_unit_mean_under_calibration(severity):
    # E_cal[w(X)] = 1 for any true likelihood ratio. With 200 000 draws the
    # standard error at severity 3 is a few parts in a thousand.
    rng = np.random.default_rng(57301)
    shift = CovariateShift(severity=severity)
    mass = rng.normal(CALIBRATION_MASS_MEAN, CALIBRATION_MASS_SD, 200_000)
    wind = rng.normal(CALIBRATION_HEADWIND_MEAN, CALIBRATION_HEADWIND_SD, 200_000)
    assert float(np.mean(shift.likelihood_ratio(mass, wind))) == pytest.approx(1.0, abs=0.02)


@pytest.mark.parametrize("severity", [1.0, 2.0, 3.0])
def test_likelihood_ratio_second_moment_is_exp_mahalanobis_squared(severity):
    # E_cal[w^2] = exp(mahalanobis^2) for a Gaussian mean shift.
    rng = np.random.default_rng(57302)
    shift = CovariateShift(severity=severity)
    mass = rng.normal(CALIBRATION_MASS_MEAN, CALIBRATION_MASS_SD, 400_000)
    wind = rng.normal(CALIBRATION_HEADWIND_MEAN, CALIBRATION_HEADWIND_SD, 400_000)
    got = float(np.mean(shift.likelihood_ratio(mass, wind) ** 2))
    assert got == pytest.approx(math.exp(shift.mahalanobis**2), rel=0.05)


def test_log_likelihood_ratio_is_affine_in_mass():
    shift = CovariateShift(severity=2.0)
    mass = np.array([5.0, 6.0, 7.0])
    log_w = shift.log_likelihood_ratio(mass, 0.0)
    assert np.allclose(np.diff(log_w, 2), 0.0, atol=1e-12)


def test_log_likelihood_ratio_slope_in_mass():
    # d log w / d mass = delta / sd^2 = 0.36 / 0.36 = 1.0 at severity 2.
    shift = CovariateShift(severity=2.0)
    slope = float(
        shift.log_likelihood_ratio(7.0, 0.0) - shift.log_likelihood_ratio(6.0, 0.0)
    )
    assert slope == pytest.approx(shift.mass_delta / CALIBRATION_MASS_SD**2, rel=1e-13)


def test_likelihood_ratio_is_positive():
    shift = CovariateShift(severity=3.0)
    assert np.all(shift.likelihood_ratio(np.linspace(2.0, 10.0, 40), np.linspace(-9, 9, 40)) > 0.0)


def test_describe_mentions_mahalanobis():
    assert "Mahalanobis" in CovariateShift(severity=1.0).describe()


@pytest.mark.parametrize(
    "kwargs", [{"severity": -0.1}, {"mass_sd": 0.0}, {"headwind_sd": -1.0}]
)
def test_shift_rejects_bad_parameters(kwargs):
    with pytest.raises(ValueError):
        CovariateShift(**kwargs)


def test_scaled_rejects_negative_fraction():
    with pytest.raises(ValueError, match="fraction"):
        CovariateShift(severity=1.0).scaled(-0.5)
