"""Baseline 2: least squares and the clock-model Kalman filter."""

from __future__ import annotations

import numpy as np
import pytest

from timetransfer.errors import ConfigurationError, InsufficientDataError
from timetransfer.skew import (
    KalmanConfig,
    fit_least_squares,
    kalman_skew,
    measurement_variance_from_data,
)


def test_least_squares_recovers_an_exact_polynomial() -> None:
    # theta(t) = 1e-3 + 2e-6 t + 0.5 * 1e-11 t^2, sampled exactly.
    t = np.arange(200, dtype=np.float64)
    y = 1.0e-3 + 2.0e-6 * t + 0.5 * 1.0e-11 * t * t
    fit = fit_least_squares(t, y, order=2, t_ref_s=0.0)
    assert fit.offset0_s == pytest.approx(1.0e-3, rel=1e-9)
    assert fit.skew_s_per_s == pytest.approx(2.0e-6, rel=1e-7)
    assert fit.drift_per_s == pytest.approx(1.0e-11, rel=1e-5)
    assert fit.rss == pytest.approx(0.0, abs=1e-28)
    assert fit.dof == 197


def test_least_squares_order_one_ignores_drift() -> None:
    t = np.arange(50, dtype=np.float64)
    y = 1.0 + 2.0 * t
    fit = fit_least_squares(t, y, order=1, t_ref_s=0.0)
    assert fit.drift_per_s == 0.0
    assert fit.skew_s_per_s == pytest.approx(2.0, rel=1e-12)
    assert "drift" not in fit.as_text()


def test_least_squares_default_epoch_is_the_mean_time() -> None:
    t = np.asarray([0.0, 10.0, 20.0, 30.0, 40.0])
    fit = fit_least_squares(t, t * 2.0, order=1)
    assert fit.t_ref_s == pytest.approx(20.0)
    value, stderr = fit.offset_at(20.0)
    assert value == pytest.approx(40.0, rel=1e-12)
    assert stderr == pytest.approx(0.0, abs=1e-12)


def test_offset_at_uncertainty_grows_away_from_the_epoch() -> None:
    rng = np.random.default_rng(1)
    t = np.arange(100, dtype=np.float64)
    y = 1.0e-3 + 2.0e-6 * t + rng.normal(0.0, 1.0e-6, size=t.size)
    fit = fit_least_squares(t, y, order=1)
    _, at_epoch = fit.offset_at(fit.t_ref_s)
    _, far = fit.offset_at(fit.t_ref_s + 200.0)
    assert far > at_epoch


def test_detrended_residual_has_the_polynomial_removed() -> None:
    t = np.arange(120, dtype=np.float64)
    y = 5.0e-4 + 1.0e-6 * t + 0.5 * 2.0e-12 * t * t
    fit = fit_least_squares(t, y, order=2)
    residual = fit.detrended(t, y)
    assert float(np.max(np.abs(residual))) < 1e-15


def test_translation_of_the_offset_leaves_the_skew_unchanged() -> None:
    t = np.arange(80, dtype=np.float64)
    rng = np.random.default_rng(4)
    y = 1.0e-3 + 3.0e-6 * t + rng.normal(0.0, 1.0e-7, size=t.size)
    a = fit_least_squares(t, y, order=1)
    b = fit_least_squares(t, y + 1.0, order=1)
    assert b.skew_s_per_s == pytest.approx(a.skew_s_per_s, rel=1e-9)
    assert b.offset0_s - a.offset0_s == pytest.approx(1.0, rel=1e-12)


@pytest.mark.parametrize(("order", "n"), [(1, 2), (2, 3)])
def test_least_squares_refuses_too_few_points(order: int, n: int) -> None:
    with pytest.raises(InsufficientDataError):
        fit_least_squares(np.arange(n, dtype=float), np.zeros(n), order=order)


def test_least_squares_validates() -> None:
    with pytest.raises(ConfigurationError, match="order"):
        fit_least_squares(np.arange(10.0), np.zeros(10), order=3)
    with pytest.raises(ConfigurationError, match="match in length"):
        fit_least_squares(np.arange(10.0), np.zeros(9), order=1)
    bad = np.zeros(10)
    bad[0] = np.nan
    with pytest.raises(ConfigurationError, match="finite"):
        fit_least_squares(np.arange(10.0), bad, order=1)


def test_measurement_variance_from_data_recovers_a_gaussian_sigma() -> None:
    rng = np.random.default_rng(7)
    y = rng.normal(0.0, 1.0e-6, size=4000)
    sigma = float(np.sqrt(measurement_variance_from_data(y, window=5)))
    # A running median of width 5 removes part of the variance, so the recovered
    # sigma is biased low. The measured factor is checked, not assumed away.
    assert 0.5e-6 < sigma < 1.2e-6


def test_measurement_variance_is_insensitive_to_a_linear_trend() -> None:
    rng = np.random.default_rng(8)
    noise = rng.normal(0.0, 1.0e-6, size=2000)
    flat = measurement_variance_from_data(noise, window=5)
    sloped = measurement_variance_from_data(noise + 1.0e-6 * np.arange(2000), window=5)
    assert sloped == pytest.approx(flat, rel=0.35)


def test_measurement_variance_has_a_positive_floor() -> None:
    assert measurement_variance_from_data(np.zeros(100)) == pytest.approx(1e-24)


def test_measurement_variance_validates() -> None:
    with pytest.raises(ConfigurationError, match="window"):
        measurement_variance_from_data(np.zeros(100), window=4)
    with pytest.raises(InsufficientDataError):
        measurement_variance_from_data(np.zeros(3), window=5)


def test_kalman_with_zero_process_noise_matches_a_straight_line_fit() -> None:
    # With Q = 0 the two-state filter is recursive least squares on a line, so
    # its final state must agree with the batch fit evaluated at the last epoch.
    rng = np.random.default_rng(11)
    t = np.arange(300, dtype=np.float64)
    y = 1.0e-3 + 2.0e-6 * t + rng.normal(0.0, 1.0e-7, size=t.size)
    fit = fit_least_squares(t, y, order=1)
    result = kalman_skew(t, y, config=KalmanConfig(measurement_variance_s2=1.0e-14))
    expected, _ = fit.offset_at(float(t[-1]))
    assert result.final_offset_s == pytest.approx(expected, rel=1e-4)
    assert result.final_skew_s_per_s == pytest.approx(fit.skew_s_per_s, rel=1e-3)


def test_kalman_three_state_recovers_a_drift() -> None:
    t = np.arange(600, dtype=np.float64)
    y = 1.0e-3 + 2.0e-6 * t + 0.5 * 5.0e-12 * t * t
    result = kalman_skew(
        t,
        y,
        config=KalmanConfig(
            n_states=3, measurement_variance_s2=1.0e-18, initial_drift_variance=1e-18
        ),
    )
    assert result.states.shape == (600, 3)
    assert float(result.states[-1, 2]) == pytest.approx(5.0e-12, rel=0.2)


def test_kalman_covariance_stays_symmetric_and_positive() -> None:
    rng = np.random.default_rng(13)
    t = np.arange(200, dtype=np.float64)
    y = rng.normal(0.0, 1.0e-7, size=t.size)
    result = kalman_skew(t, y, config=KalmanConfig(q1_s2_per_s=1e-20, q2_per_s=1e-26))
    for k in (0, 50, 199):
        cov = result.covariances[k]
        assert cov == pytest.approx(cov.T, abs=1e-30)
        assert float(np.min(np.linalg.eigvalsh(cov))) > -1e-30
    assert result.normalised_innovation_mean > 0.0


def test_kalman_records_the_measurement_variance_it_used() -> None:
    t = np.arange(100, dtype=np.float64)
    rng = np.random.default_rng(17)
    y = rng.normal(0.0, 1.0e-6, size=t.size)
    supplied = kalman_skew(t, y, config=KalmanConfig(measurement_variance_s2=4.0e-12))
    assert supplied.measurement_variance_s2 == pytest.approx(4.0e-12)
    from_data = kalman_skew(t, y)
    assert from_data.measurement_variance_s2 != 4.0e-12
    assert from_data.measurement_variance_s2 > 0.0
    assert "mean normalised innov" in from_data.as_text()


def test_kalman_validates() -> None:
    t = np.arange(10, dtype=np.float64)
    with pytest.raises(ConfigurationError, match="match in length"):
        kalman_skew(t, np.zeros(9))
    with pytest.raises(ConfigurationError, match="non-decreasing"):
        kalman_skew(np.asarray([0.0, 2.0, 1.0, 3.0]), np.zeros(4))
    with pytest.raises(InsufficientDataError):
        kalman_skew(np.arange(2, dtype=float), np.zeros(2))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"q1_s2_per_s": -1.0}, "q1_s2_per_s"),
        ({"n_states": 4}, "n_states"),
        ({"measurement_variance_s2": 0.0}, "measurement_variance_s2"),
        ({"initial_skew_variance": 0.0}, "initial_skew_variance"),
    ],
)
def test_kalman_config_validates(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ConfigurationError, match=match):
        KalmanConfig(**kwargs)


def test_measurement_variance_floor_is_configurable() -> None:
    # The default floor is not physically meaningful; a caller on a quiet link
    # passes the quantisation variance instead. Measured in
    # validation/validate_skew_estimators.py: with no jitter the data-derived R
    # hits the default floor and the filter becomes over-confident by a factor
    # of several hundred.
    quiet = np.zeros(100)
    assert measurement_variance_from_data(quiet) == pytest.approx(1e-24)
    quantisation_variance = (1e-9 / np.sqrt(12.0)) ** 2
    assert measurement_variance_from_data(
        quiet, floor_variance_s2=quantisation_variance
    ) == pytest.approx(quantisation_variance)
    with pytest.raises(ConfigurationError, match="floor_variance_s2"):
        measurement_variance_from_data(quiet, floor_variance_s2=0.0)
