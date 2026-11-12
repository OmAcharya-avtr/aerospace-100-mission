"""The equilibrium test, its known answers, and the two statuses that are not a pass."""

from __future__ import annotations

import math

import numpy as np
import pytest

from thermalbudget.equilibrium import (
    DEFAULT_SLOPE_THRESHOLD_C_PER_S,
    THRESHOLD_LADDER,
    equilibrium_sensitivity,
    equilibrium_test,
    fit_linear,
)


def test_fit_linear_known_answer_exact_line():
    # y = 1 + 2x through (0,1), (1,3), (2,5), (3,7). Slope 2, intercept 1,
    # residuals identically zero, so the standard error is zero too.
    fit = fit_linear(np.array([0.0, 1.0, 2.0, 3.0]), np.array([1.0, 3.0, 5.0, 7.0]))
    assert fit.slope == pytest.approx(2.0, rel=1e-12, abs=1e-12)
    assert fit.intercept == pytest.approx(1.0, rel=1e-12, abs=1e-12)
    assert fit.rms_residual == pytest.approx(0.0, rel=0.0, abs=1e-12)
    assert fit.slope_stderr_ols == pytest.approx(0.0, rel=0.0, abs=1e-12)
    assert fit.n == 4


def test_fit_linear_known_answer_with_one_residual():
    # x = 0,1,2; y = 0,0,2.  mean x = 1, mean y = 2/3.
    # Sxx = 1 + 0 + 1 = 2.  Sxy = (-1)(-2/3) + 0(-2/3) + (1)(4/3) = 2.
    # slope = 1, intercept = 2/3 - 1 = -1/3.
    fit = fit_linear(np.array([0.0, 1.0, 2.0]), np.array([0.0, 0.0, 2.0]))
    assert fit.slope == pytest.approx(1.0, rel=1e-12, abs=1e-12)
    assert fit.intercept == pytest.approx(-1.0 / 3.0, rel=1e-12, abs=1e-12)
    # residuals are 1/3, -2/3, 1/3; sum of squares = 2/3; dof = 1.
    assert fit.rms_residual == pytest.approx(math.sqrt((2.0 / 3.0) / 3.0), rel=1e-12, abs=1e-12)
    assert fit.slope_stderr_ols == pytest.approx(math.sqrt((2.0 / 3.0) / 2.0), rel=1e-12,
                                                 abs=1e-12)


def test_fit_linear_drops_non_finite_pairs():
    fit = fit_linear(
        np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        np.array([1.0, 3.0, math.nan, 7.0, 9.0]),
    )
    assert fit.n == 4
    assert fit.slope == pytest.approx(2.0, rel=1e-12, abs=1e-12)


def test_fit_linear_input_validation():
    with pytest.raises(ValueError, match="same shape"):
        fit_linear(np.zeros(3), np.zeros(4))
    with pytest.raises(ValueError, match="at least 3 finite"):
        fit_linear(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="not identifiable"):
        fit_linear(np.ones(5), np.arange(5.0))


def test_ar1_correction_inflates_the_standard_error_for_positive_rho():
    rng = np.random.default_rng(4)
    n = 400
    noise = np.zeros(n)
    for index in range(1, n):
        noise[index] = 0.9 * noise[index - 1] + rng.normal(0.0, 0.05)
    x = np.arange(float(n))
    fit = fit_linear(x, 0.001 * x + noise)
    assert fit.residual_autocorrelation > 0.5
    assert fit.slope_stderr_ar1 > fit.slope_stderr_ols
    assert fit.inflation > 1.5


def test_an_equilibrated_soak_passes_and_reports_a_small_slope(no_throttle_campaign):
    verdict = equilibrium_test(no_throttle_campaign["soak_1.00_r1"])
    assert verdict.status == "EQUILIBRATED"
    assert verdict.equilibrated
    assert abs(verdict.slope_c_per_s) <= DEFAULT_SLOPE_THRESHOLD_C_PER_S
    assert verdict.n_samples >= 100
    assert "within" in verdict.reason


def test_a_short_loaded_run_is_reported_not_equilibrated_not_averaged(short_run):
    verdict = equilibrium_test(short_run)
    assert verdict.status == "NOT_EQUILIBRATED"
    assert not verdict.equilibrated
    assert "EXCEEDS" in verdict.reason
    assert math.isfinite(verdict.slope_c_per_s)


def test_a_burst_has_too_few_samples_and_that_is_not_a_pass(no_throttle_campaign):
    verdict = equilibrium_test(no_throttle_campaign["burst_1.00_a"])
    assert verdict.status == "INSUFFICIENT_SAMPLES"
    assert not verdict.equilibrated
    assert math.isnan(verdict.slope_c_per_s)
    assert "not a pass" in verdict.reason


def test_a_cooldown_fails_the_test_which_is_the_right_answer(no_throttle_campaign):
    verdict = equilibrium_test(no_throttle_campaign["cool_r1"])
    assert verdict.status == "NOT_EQUILIBRATED"
    assert verdict.slope_c_per_s < 0.0


def test_the_interval_check_is_stricter_than_the_point_check(no_throttle_campaign):
    verdict = equilibrium_test(no_throttle_campaign["soak_1.00_r1"])
    assert verdict.equilibrated
    # Point estimate inside the band implies nothing about the interval, so the
    # two properties are reported separately rather than one standing for both.
    assert isinstance(verdict.interval_inside_threshold, bool)
    low, high = verdict.slope_ci_c_per_s
    assert low < verdict.slope_c_per_s < high


def test_the_threshold_ladder_is_monotone_in_the_threshold(no_throttle_campaign):
    ladder = equilibrium_sensitivity(no_throttle_campaign["soak_0.75"])
    assert tuple(ladder) == THRESHOLD_LADDER
    statuses = [ladder[threshold] for threshold in THRESHOLD_LADDER]
    # Once a threshold admits the run, every looser threshold must admit it too.
    first_pass = next(
        (index for index, status in enumerate(statuses) if status == "EQUILIBRATED"), None
    )
    if first_pass is not None:
        assert all(status == "EQUILIBRATED" for status in statuses[first_pass:])


def test_input_validation(no_throttle_campaign):
    trace = no_throttle_campaign["soak_1.00_r1"]
    with pytest.raises(ValueError, match="window_s must be positive"):
        equilibrium_test(trace, window_s=0.0)
    with pytest.raises(ValueError, match="threshold_c_per_s must be positive"):
        equilibrium_test(trace, threshold_c_per_s=-1.0)


def test_missing_samples_are_excluded_and_counted(throttled_campaign):
    # soak_0.50 carries three sentinel-coded rows by construction, but they are
    # early in the run and the equilibrium window is the final 120 s, so the
    # window itself is clean. Both facts are asserted so that a change to the
    # fixture cannot silently invalidate the test's premise.
    trace = throttled_campaign["soak_0.50"]
    assert trace.missing["temp_tj-thermal"] == 3
    verdict = equilibrium_test(trace)
    assert verdict.n_missing == 0
    assert verdict.n_samples >= 100
