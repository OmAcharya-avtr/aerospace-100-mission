"""Scoring rules, coverage splits and calibration diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from degradepredict.forecast import QuantileForecast
from degradepredict.metrics import (
    calibration_curve,
    coverage_by_censoring,
    crps,
    falsification_rate,
    ks_uniform,
    mean_pinball,
    pinball_loss,
    pit_values,
)

LEVELS = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])


def _uniform_forecast(a: float, b: float, n_levels: int = 401) -> QuantileForecast:
    """Exact quantile function of Uniform(a, b) on a dense level grid."""
    levels = np.linspace(1.0 / (n_levels + 1), n_levels / (n_levels + 1), n_levels)
    return QuantileForecast(levels=levels, values_days=a + levels * (b - a))


def test_pinball_loss_hand_calculated():
    # level 0.9, realisation above the prediction: 0.9 * (10 - 4) = 5.4
    assert pinball_loss(np.array([10.0]), np.array([4.0]), 0.9)[0] == pytest.approx(5.4)
    # level 0.9, realisation below the prediction: (0.9 - 1) * (4 - 10) = 0.6
    assert pinball_loss(np.array([4.0]), np.array([10.0]), 0.9)[0] == pytest.approx(0.6)
    # the median level halves the absolute error
    assert pinball_loss(np.array([10.0]), np.array([4.0]), 0.5)[0] == pytest.approx(3.0)


def test_pinball_loss_rejects_a_level_outside_the_unit_interval():
    with pytest.raises(ValueError, match="level must be in"):
        pinball_loss(np.array([1.0]), np.array([1.0]), 1.0)


def test_pinball_loss_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        pinball_loss(np.array([1.0, 2.0]), np.array([1.0]), 0.5)


def test_crps_of_a_uniform_predictive_matches_the_closed_form():
    """CRPS(U(a,b), y) = (b - a) * (u**2 - u + 1/3) with u = (y - a)/(b - a).

    Derived from CRPS = E|X - y| - 0.5 E|X - X'| for X, X' ~ U(0, 1) and then
    rescaled; see Gneiting and Raftery (2007) section 4.2. Agreement is to the
    midpoint-rule quadrature error of the 199-point grid, not exact.
    """
    a, b, y = 100.0, 300.0, 180.0
    u = (y - a) / (b - a)
    expected = (b - a) * (u**2 - u + 1.0 / 3.0)
    got = float(crps([_uniform_forecast(a, b)], np.array([y]))[0])
    assert got == pytest.approx(expected, rel=1e-5)


def test_crps_of_a_nearly_degenerate_forecast_is_the_absolute_error():
    """A predictive distribution concentrated at m scores CRPS -> |y - m|."""
    m = 500.0
    f = QuantileForecast(levels=LEVELS, values_days=m + np.linspace(-0.01, 0.01, LEVELS.size))
    got = float(crps([f], np.array([620.0]))[0])
    assert got == pytest.approx(120.0, rel=1e-3)


def test_crps_rejects_a_length_mismatch():
    with pytest.raises(ValueError, match="forecasts for"):
        crps([_uniform_forecast(1.0, 2.0)], np.array([1.0, 2.0]))


def test_mean_pinball_is_lower_for_the_better_centred_forecast():
    good = QuantileForecast(levels=LEVELS, values_days=np.array([400, 430, 470, 500, 530, 570,
                                                                 600], dtype=float))
    bad = QuantileForecast(levels=LEVELS, values_days=np.array([900, 930, 970, 1000, 1030, 1070,
                                                                1100], dtype=float))
    y = np.array([500.0])
    assert float(mean_pinball([good], y)[0]) < float(mean_pinball([bad], y)[0])


def test_coverage_is_reported_separately_for_complete_and_censored():
    # Two units inside the interval and two outside, arranged so that the
    # pooled figure (0.5) matches neither half (1.0 and 0.0).
    f = QuantileForecast(levels=LEVELS, values_days=np.array([100, 150, 300, 500, 800, 1200,
                                                              1500], dtype=float))
    forecasts = [f, f, f, f]
    y = np.array([200.0, 400.0, 5000.0, 6000.0])
    event = np.array([True, True, False, False])
    out = coverage_by_censoring(forecasts, y, event, level=0.90)
    assert out["coverage_complete"] == pytest.approx(1.0)
    assert out["coverage_censored"] == pytest.approx(0.0)
    assert out["coverage_pooled"] == pytest.approx(0.5)
    assert out["n_complete"] == 2.0
    assert out["n_censored"] == 2.0


def test_coverage_rejects_mismatched_shapes():
    f = QuantileForecast(levels=LEVELS, values_days=np.linspace(100, 900, LEVELS.size))
    with pytest.raises(ValueError, match="same shape"):
        coverage_by_censoring([f], np.array([1.0, 2.0]), np.array([True]))


def test_falsification_rate_counts_intervals_the_censoring_alone_refutes():
    f = QuantileForecast(levels=LEVELS, values_days=np.array([10, 20, 30, 40, 50, 60, 70],
                                                             dtype=float))
    # Censored at 500 d, so an upper endpoint of 70 d is already refuted.
    out = falsification_rate([f, f], np.array([500.0, 5.0]), np.array([False, False]))
    assert out["falsified_fraction"] == pytest.approx(0.5)


def test_falsification_rate_with_no_censored_units_returns_nan():
    f = QuantileForecast(levels=LEVELS, values_days=np.linspace(100, 900, LEVELS.size))
    out = falsification_rate([f], np.array([10.0]), np.array([True]))
    assert np.isnan(out["falsified_fraction"])


def test_pit_of_a_calibrated_forecaster_is_close_to_uniform():
    rng = np.random.default_rng(0)
    truth = rng.uniform(100.0, 300.0, 4000)
    forecasts = [_uniform_forecast(100.0, 300.0) for _ in truth]
    pit = pit_values(forecasts, truth)
    assert float(pit.mean()) == pytest.approx(0.5, abs=0.02)
    assert ks_uniform(pit)["ks_statistic"] < 0.05


def test_pit_of_a_biased_forecaster_is_detected():
    rng = np.random.default_rng(1)
    truth = rng.uniform(400.0, 600.0, 2000)
    forecasts = [_uniform_forecast(100.0, 300.0) for _ in truth]
    pit = pit_values(forecasts, truth)
    assert float(pit.mean()) > 0.9
    assert ks_uniform(pit)["ks_pvalue"] < 1e-6


def test_calibration_curve_of_a_uniform_sample_tracks_the_diagonal():
    rng = np.random.default_rng(2)
    nominal, empirical = calibration_curve(rng.uniform(0.0, 1.0, 20000), n_bins=10)
    np.testing.assert_allclose(empirical, nominal, atol=0.02)


def test_calibration_curve_validation():
    with pytest.raises(ValueError, match="n_bins"):
        calibration_curve(np.array([0.1, 0.2]), n_bins=1)
    with pytest.raises(ValueError, match="no PIT values"):
        calibration_curve(np.array([]))


def test_ks_uniform_needs_more_than_one_value():
    with pytest.raises(ValueError, match="at least two"):
        ks_uniform(np.array([0.5]))
