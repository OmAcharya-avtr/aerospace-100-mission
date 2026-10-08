"""Tests for threshold setting from a declared false-alarm target."""

from __future__ import annotations

import pytest

from twininvalidate import (
    DetectorSpec,
    arl0_from_rate,
    bracket_threshold,
    calibrate_threshold,
    closed_form_threshold,
    ewma_lambda1_threshold,
    glr_window1_threshold,
    rate_from_arl0,
)


def test_glr_window1_threshold_known_answer():
    # Equation (7): h = Phi^{-1}(1 - 1/(2T))^2 / 2.
    # T = 1000: 1 - 1/2000 = 0.9995, Phi^{-1}(0.9995) = 3.2905267315,
    # h = 3.2905267315^2 / 2 = 10.8275662 / 2 = 5.4137831.
    assert glr_window1_threshold(1000.0) == pytest.approx(5.4137830853, rel=1e-9)


def test_ewma_lambda1_threshold_known_answer():
    # Equation (8): L = Phi^{-1}(1 - 1/(2T)) = 3.2905267315 for T = 1000.
    assert ewma_lambda1_threshold(1000.0) == pytest.approx(3.2905267315, rel=1e-9)


def test_glr_window1_threshold_known_answer_t100():
    # T = 100: 1 - 1/200 = 0.995, Phi^{-1}(0.995) = 2.5758293035,
    # h = 2.5758293035^2 / 2 = 6.6348966 / 2 = 3.3174483.
    assert glr_window1_threshold(100.0) == pytest.approx(3.3174483005, rel=1e-9)


def test_the_two_closed_forms_are_the_same_test():
    # GLR with window 1 alarms iff |z| > sqrt(2h); EWMA with lam 1 alarms iff
    # |z| > L. The two thresholds must therefore satisfy h = L^2 / 2 exactly.
    for target in (10.0, 100.0, 1000.0, 1e5):
        h = glr_window1_threshold(target)
        lam = ewma_lambda1_threshold(target)
        assert h == pytest.approx(lam**2 / 2.0, rel=1e-12)


def test_closed_form_threshold_is_offered_only_where_it_exists():
    assert closed_form_threshold(DetectorSpec("glr", window=1), 1000.0) is not None
    assert closed_form_threshold(DetectorSpec("ewma", lam=1.0), 1000.0) is not None
    assert closed_form_threshold(DetectorSpec("glr", window=100), 1000.0) is None
    assert closed_form_threshold(DetectorSpec("ewma", lam=0.1), 1000.0) is None
    assert closed_form_threshold(DetectorSpec("cusum"), 1000.0) is None
    assert closed_form_threshold(DetectorSpec("varcusum"), 1000.0) is None


def test_closed_form_calibration_reports_its_method(in_control):
    cal = calibrate_threshold(DetectorSpec("glr", window=1), in_control, 1000.0)
    assert cal.method == "closed-form"
    assert cal.threshold == pytest.approx(glr_window1_threshold(1000.0))


def test_closed_form_arl0_matches_monte_carlo(in_control):
    # The exact threshold of equation (7) should reproduce the target ARL0 on
    # simulated in-control data to within the counting noise. The censored MLE
    # has relative standard error 1/sqrt(n_alarms), so the check is at three
    # standard errors.
    cal = calibrate_threshold(DetectorSpec("glr", window=1), in_control, 1000.0)
    assert cal.n_alarms > 50
    assert abs(cal.achieved_arl0 - 1000.0) < 3.0 * cal.achieved_stderr


@pytest.mark.parametrize("name", ["cusum", "ewma", "glr", "varcusum"])
def test_monte_carlo_calibration_hits_the_target(in_control, name):
    cal = calibrate_threshold(DetectorSpec(name), in_control, 500.0)
    assert cal.method == "monte-carlo-bisection"
    assert cal.threshold > 0.0
    # Bisection drives the censored-MLE estimate onto the target, so the
    # agreement here is tight by construction; it is the achieved ARL0 on
    # *fresh* data that carries sampling error, and that is measured in
    # validation/validate_thresholds.py.
    assert cal.achieved_arl0 == pytest.approx(500.0, rel=0.05)
    assert cal.relative_error == pytest.approx(cal.achieved_arl0 / 500.0 - 1.0)


def test_a_higher_target_needs_a_higher_threshold(in_control):
    low = calibrate_threshold(DetectorSpec("cusum"), in_control, 200.0)
    high = calibrate_threshold(DetectorSpec("cusum"), in_control, 2000.0)
    assert high.threshold > low.threshold


def test_bracket_threshold_brackets_the_target(in_control):
    stat = DetectorSpec("cusum").statistic(in_control)
    bracket = bracket_threshold(stat, 500.0)
    assert bracket.conservative_arl0 >= 500.0
    assert bracket.generous is not None
    assert bracket.generous < bracket.conservative
    assert bracket.generous_arl0 is not None
    assert bracket.generous_arl0 < 500.0
    assert bracket.n_levels > 0


def test_bracket_threshold_refuses_an_unreachable_target(in_control):
    stat = DetectorSpec("cusum").statistic(in_control[:5, :200])
    with pytest.raises(ValueError, match="no achievable threshold"):
        bracket_threshold(stat, 1e12)


def test_rate_conversion_known_answer():
    # 20 Hz, 1 false alarm per 1000 h: ARL0 = 1000 * 3600 * 20 = 7.2e7 samples.
    assert arl0_from_rate(1.0, 20.0) == pytest.approx(7.2e7)
    # ARL0 = 1000 samples at 20 Hz is 50 s between false alarms, i.e.
    # 3.6e6 s / 50 s = 72000 per 1000 h.
    assert rate_from_arl0(1000.0, 20.0) == pytest.approx(72000.0)


def test_rate_conversion_round_trips():
    for rate in (0.1, 1.0, 37.0, 1e4):
        assert arl0_from_rate(rate_from_arl0(arl0_from_rate(rate, 20.0), 20.0), 20.0) == (
            pytest.approx(arl0_from_rate(rate, 20.0))
        )


@pytest.mark.parametrize(
    "func, args, message",
    [
        (glr_window1_threshold, (1.0,), "must exceed 1 sample"),
        (glr_window1_threshold, (0.5,), "must exceed 1 sample"),
        (ewma_lambda1_threshold, (1.0,), "must exceed 1 sample"),
        (arl0_from_rate, (0.0, 20.0), "false_alarms_per_1000h must be positive"),
        (arl0_from_rate, (1.0, 0.0), "sample_rate_hz must be positive"),
        (rate_from_arl0, (0.0, 20.0), "arl0_samples must be positive"),
        (rate_from_arl0, (1.0, -1.0), "sample_rate_hz must be positive"),
    ],
)
def test_threshold_helpers_validate_their_inputs(func, args, message):
    with pytest.raises(ValueError, match=message):
        func(*args)


def test_calibrate_rejects_a_bad_target(in_control):
    with pytest.raises(ValueError, match="must exceed 1 sample"):
        calibrate_threshold(DetectorSpec("cusum"), in_control, 1.0)


def test_calibrate_rejects_a_decreasing_bracket(in_control):
    with pytest.raises(ValueError, match="bracket must be increasing"):
        calibrate_threshold(DetectorSpec("cusum"), in_control, 500.0, bracket=(5.0, 1.0))


def test_calibration_records_the_bank_size(in_control):
    cal = calibrate_threshold(DetectorSpec("cusum"), in_control, 500.0)
    assert (cal.n_runs, cal.n_samples) == in_control.shape
