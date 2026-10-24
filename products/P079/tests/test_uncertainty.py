"""The uncertainty budget, its serial-correlation inflation, and coverage."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from timetransfer.errors import ConfigurationError, InsufficientDataError
from timetransfer.skew import fit_least_squares
from timetransfer.uncertainty import (
    AsymmetrySource,
    effective_sample_size,
    measure_coverage,
    offset_uncertainty,
)


def _fit(n: int = 200, sigma: float = 1.0e-6, seed: int = 1):
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64)
    y = 1.0e-3 + 2.0e-6 * t + rng.normal(0.0, sigma, size=n)
    return fit_least_squares(t, y, order=1)


def test_effective_sample_size_equals_n_for_white_residuals() -> None:
    rng = np.random.default_rng(3)
    ess = effective_sample_size(rng.normal(size=20000))
    assert ess.n == 20000
    assert ess.n_effective == pytest.approx(20000.0, rel=0.05)
    assert ess.inflation == pytest.approx(1.0, rel=0.05)


def test_effective_sample_size_known_answer() -> None:
    # n_eff = n (1 - r1) / (1 + r1). For a series with r1 measured at exactly
    # 0.5 and n = 300 the formula gives 300 * 0.5 / 1.5 = 100.
    # A deterministic series with r1 = 0.5 is hard to build, so the formula is
    # checked directly against a synthesised r1 by constructing the series
    # x_k = cos(2 pi k / P), whose lag-1 autocorrelation is cos(2 pi / P).
    period = 2.0 * np.pi / np.arccos(0.5)
    x = np.cos(2.0 * np.pi * np.arange(3000) / period)
    ess = effective_sample_size(x)
    assert ess.lag1_autocorrelation == pytest.approx(0.5, abs=0.01)
    assert ess.n_effective == pytest.approx(3000.0 * 0.5 / 1.5, rel=0.03)


def test_effective_sample_size_is_clipped_for_anti_correlated_residuals() -> None:
    x = np.asarray([1.0, -1.0] * 500)
    ess = effective_sample_size(x)
    assert ess.lag1_autocorrelation < -0.9
    assert ess.n_effective == 1000.0
    assert ess.clipped
    assert ess.inflation == pytest.approx(1.0)


def test_effective_sample_size_handles_a_constant_series() -> None:
    ess = effective_sample_size(np.zeros(100))
    assert ess.n_effective == 100.0
    assert ess.inflation == 1.0


def test_effective_sample_size_rejects_short_input() -> None:
    with pytest.raises(InsufficientDataError):
        effective_sample_size(np.zeros(3))


def test_unaccounted_asymmetry_is_flagged() -> None:
    unc = offset_uncertainty(_fit())
    assert unc.asymmetry_source is AsymmetrySource.UNACCOUNTED
    assert unc.u_asymmetry_s == 0.0
    assert not unc.accounted
    assert "WARNING" in unc.as_text()


def test_declared_bound_contributes_bound_over_two_root_three() -> None:
    # B = 6e-4 -> u = (6e-4 / 2) / sqrt(3) = 3e-4 / 1.7320508 = 1.7320508e-4
    unc = offset_uncertainty(
        _fit(),
        asymmetry_source=AsymmetrySource.DECLARED_BOUND,
        asymmetry_bound_s=6.0e-4,
    )
    assert unc.u_asymmetry_s == pytest.approx(1.7320508075688773e-4, rel=1e-12)
    assert unc.accounted
    assert unc.dominant_term == "asymmetry"


def test_estimated_asymmetry_uses_the_measured_residual_spread() -> None:
    unc = offset_uncertainty(
        _fit(),
        asymmetry_source=AsymmetrySource.ESTIMATED,
        asymmetry_stderr_s=5.0e-5,
    )
    assert unc.u_asymmetry_s == pytest.approx(5.0e-5)


def test_combination_is_in_quadrature() -> None:
    unc = offset_uncertainty(
        _fit(),
        asymmetry_source=AsymmetrySource.ESTIMATED,
        asymmetry_stderr_s=5.0e-5,
        resolution_ns=1000,
    )
    expected = float(
        np.sqrt(unc.u_statistical_s**2 + unc.u_quantisation_s**2 + unc.u_asymmetry_s**2)
    )
    assert unc.u_combined_s == pytest.approx(expected, rel=1e-12)
    assert unc.expanded_s == pytest.approx(1.96 * unc.u_combined_s, rel=1e-12)
    lo, hi = unc.interval_s()
    assert hi - lo == pytest.approx(2.0 * unc.expanded_s, rel=1e-12)


def test_quantisation_term_scales_with_resolution() -> None:
    a = offset_uncertainty(_fit(), resolution_ns=1)
    b = offset_uncertainty(_fit(), resolution_ns=1000)
    assert b.u_quantisation_s == pytest.approx(1000.0 * a.u_quantisation_s, rel=1e-9)


def test_statistical_term_is_inflated_by_serial_correlation() -> None:
    # Build residuals with a strong positive lag-1 autocorrelation by filtering
    # white noise, and check the inflation factor is applied.
    rng = np.random.default_rng(21)
    n = 400
    t = np.arange(n, dtype=np.float64)
    white = rng.normal(0.0, 1.0e-6, size=n)
    correlated = np.zeros(n)
    for k in range(1, n):
        correlated[k] = 0.8 * correlated[k - 1] + white[k]
    fit = fit_least_squares(t, 1.0e-3 + 2.0e-6 * t + correlated, order=1)
    unc = offset_uncertainty(fit)
    assert unc.lag1_autocorrelation > 0.5
    assert unc.correlation_inflation > 1.5
    plain, raw_stderr = fit.offset_at(fit.t_ref_s)
    assert unc.u_statistical_s == pytest.approx(raw_stderr * unc.correlation_inflation, rel=1e-12)
    assert plain == pytest.approx(unc.offset_s, rel=1e-12)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"resolution_ns": 0}, "resolution_ns"),
        ({"coverage_factor": 0.0}, "coverage_factor"),
        (
            {"asymmetry_source": AsymmetrySource.DECLARED_BOUND},
            "asymmetry_bound_s",
        ),
        (
            {"asymmetry_source": AsymmetrySource.ESTIMATED},
            "asymmetry_stderr_s",
        ),
    ],
)
def test_offset_uncertainty_validates(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ConfigurationError, match=match):
        offset_uncertainty(_fit(), **kwargs)


def test_measure_coverage_on_exactly_normal_errors_is_calibrated() -> None:
    rng = np.random.default_rng(31)
    n = 4000
    sigma = 2.0e-6
    errors = rng.normal(0.0, sigma, size=n)
    result = measure_coverage(errors, np.full(n, sigma))
    assert result.nominal == pytest.approx(float(2.0 * stats.norm.cdf(1.96) - 1.0), rel=1e-12)
    assert result.verdict == "calibrated"
    assert result.nominal_inside_interval
    assert result.median_normalised_error == pytest.approx(0.6744897501960817, rel=0.05)


def test_measure_coverage_detects_an_optimistic_uncertainty() -> None:
    rng = np.random.default_rng(32)
    n = 2000
    errors = rng.normal(0.0, 1.0e-5, size=n)
    result = measure_coverage(errors, np.full(n, 1.0e-6))
    assert result.verdict == "optimistic"
    assert result.measured < 0.5


def test_measure_coverage_detects_a_conservative_uncertainty() -> None:
    rng = np.random.default_rng(33)
    n = 2000
    errors = rng.normal(0.0, 1.0e-7, size=n)
    result = measure_coverage(errors, np.full(n, 1.0e-5))
    assert result.verdict == "conservative"
    assert result.measured == 1.0
    assert result.upper == 1.0


def test_measure_coverage_is_monotone_in_k() -> None:
    rng = np.random.default_rng(34)
    errors = rng.normal(0.0, 1.0, size=1000)
    unc = np.ones(1000)
    previous = -1.0
    for k in (0.5, 1.0, 1.5, 2.0, 3.0):
        measured = measure_coverage(errors, unc, coverage_factor=k).measured
        assert measured >= previous
        previous = measured


def test_measure_coverage_validates() -> None:
    with pytest.raises(ConfigurationError, match="match in length"):
        measure_coverage(np.zeros(5), np.ones(4))
    with pytest.raises(InsufficientDataError):
        measure_coverage(np.zeros(1), np.ones(1))
    with pytest.raises(ConfigurationError, match="must be > 0 to measure coverage"):
        measure_coverage(np.zeros(5), np.zeros(5))
    with pytest.raises(ConfigurationError, match="interval_confidence"):
        measure_coverage(np.zeros(5), np.ones(5), interval_confidence=1.5)
