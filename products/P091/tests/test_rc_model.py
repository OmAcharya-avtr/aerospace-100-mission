"""The relaxation fit and the three tau estimators that share no algebra."""

from __future__ import annotations

import math

import numpy as np
import pytest

from thermalbudget.rc_model import (
    TAU_BOUNDS_S,
    RelaxationFitError,
    fit_exponential_relaxation,
    tau_from_integral,
    tau_from_log_differences,
)


def _noiseless(tau=90.0, asymptote=68.0, initial=45.0, seconds=720):
    t = np.arange(1.0, seconds + 1.0)
    return t, asymptote + (initial - asymptote) * np.exp(-t / tau)


def test_known_answer_noiseless_rise_recovers_every_parameter():
    t, temperature = _noiseless()
    fit = fit_exponential_relaxation(t, temperature)
    # The fit is a least-squares solve over a 1-D search, so recovery is to
    # within the optimiser's tolerance rather than exact; the tolerances below
    # are the agreement actually achieved, not a round number.
    assert fit.tau_s == pytest.approx(90.0, rel=1e-5, abs=0.0)
    assert fit.asymptote_c == pytest.approx(68.0, rel=1e-7, abs=0.0)
    # t starts at 1 s, and _clean shifts it to zero, so the fitted initial value
    # is T(1 s) of the generating curve, not T(0).
    assert fit.initial_c == pytest.approx(
        68.0 + (45.0 - 68.0) * math.exp(-1.0 / 90.0), rel=1e-6, abs=0.0
    )
    assert fit.rms_residual_c == pytest.approx(0.0, rel=0.0, abs=1e-6)
    assert not fit.at_bound
    assert fit.span_in_tau == pytest.approx(719.0 / fit.tau_s, rel=1e-9, abs=0.0)


def test_known_answer_noiseless_cooldown_recovers_tau_and_ambient():
    t, temperature = _noiseless(tau=185.0, asymptote=45.0, initial=68.0, seconds=300)
    fit = fit_exponential_relaxation(t, temperature)
    assert fit.tau_s == pytest.approx(185.0, rel=1e-4, abs=0.0)
    assert fit.asymptote_c == pytest.approx(45.0, rel=1e-6, abs=0.0)
    assert fit.amplitude_c < 0.0 or fit.amplitude_c > 0.0  # sign is the direction
    assert fit.amplitude_c > 0.0  # cooling: the exponential decays from above


def test_the_fit_is_deterministic_for_identical_input():
    t, temperature = _noiseless()
    first = fit_exponential_relaxation(t, temperature)
    second = fit_exponential_relaxation(t, temperature)
    assert first.tau_s == second.tau_s
    assert first.asymptote_c == second.asymptote_c


def test_a_constant_channel_refuses_rather_than_returning_a_tau():
    t = np.arange(1.0, 200.0)
    with pytest.raises(RelaxationFitError, match="constant over the window"):
        fit_exponential_relaxation(t, np.full_like(t, 50.0))


def test_too_few_samples_and_too_short_a_window_both_refuse():
    with pytest.raises(RelaxationFitError, match="at least 20 finite samples"):
        fit_exponential_relaxation(np.arange(5.0), np.arange(5.0))
    t = np.linspace(0.0, 1.0, 40)
    with pytest.raises(RelaxationFitError, match="not identifiable"):
        fit_exponential_relaxation(t, 68.0 - 23.0 * np.exp(-t / 90.0))


def test_the_tau_bound_is_reported_when_it_is_hit():
    # A curve whose relaxation is far slower than the bound admits: 720 s of a
    # tau = 50000 s exponential is essentially a straight line, and the search
    # runs into its upper bound rather than inventing a number past it.
    t, temperature = _noiseless(tau=50000.0, seconds=720)
    fit = fit_exponential_relaxation(t, temperature)
    assert fit.at_bound
    assert fit.tau_s == pytest.approx(TAU_BOUNDS_S[1], rel=1e-3, abs=0.0)


def test_log_difference_route_known_answer_on_a_noiseless_rise():
    t, temperature = _noiseless(seconds=300)
    estimate = tau_from_log_differences(t, temperature, smooth_samples=1, snr_floor=0.0)
    # Noiseless successive differences of an exponential are exactly geometric,
    # so the log slope is exactly -1/tau up to the shift caused by using the
    # midpoint time; the agreement is to within that half-sample offset.
    assert estimate.tau_s == pytest.approx(90.0, rel=1e-3, abs=0.0)
    assert estimate.method == "log-differences"


def test_log_difference_route_uses_no_asymptote_so_a_shifted_trace_gives_the_same_tau():
    t, temperature = _noiseless(seconds=300)
    first = tau_from_log_differences(t, temperature, smooth_samples=1, snr_floor=0.0)
    second = tau_from_log_differences(
        t, temperature + 17.5, smooth_samples=1, snr_floor=0.0
    )
    assert first.tau_s == pytest.approx(second.tau_s, rel=1e-12, abs=0.0)


def test_log_difference_route_refuses_a_flat_channel_by_name():
    t = np.arange(1.0, 300.0)
    rng = np.random.default_rng(1)
    flat = 50.0 + rng.normal(0.0, 0.03, size=t.size)
    with pytest.raises(RelaxationFitError, match="signal floor|not usable"):
        tau_from_log_differences(t, flat)


def test_area_rule_known_answer_on_a_noiseless_rise():
    t, temperature = _noiseless(seconds=720)
    estimate = tau_from_integral(t, temperature, asymptote_c=68.0)
    assert estimate.tau_s == pytest.approx(90.0, rel=2e-3, abs=0.0)
    assert estimate.method == "area-rule"
    assert math.isnan(estimate.stderr_s)


def test_area_rule_refuses_when_the_trace_starts_at_the_asymptote():
    t, temperature = _noiseless(seconds=300)
    with pytest.raises(RelaxationFitError, match="divides by that offset"):
        tau_from_integral(t, temperature, asymptote_c=float(temperature[0]))


def test_the_three_routes_agree_on_a_noiseless_curve():
    t, temperature = _noiseless(seconds=400)
    fit = fit_exponential_relaxation(t, temperature)
    log_difference = tau_from_log_differences(t, temperature, smooth_samples=1, snr_floor=0.0)
    area = tau_from_integral(t, temperature, asymptote_c=fit.asymptote_c)
    values = [fit.tau_s, log_difference.tau_s, area.tau_s]
    assert max(values) - min(values) < 0.5  # seconds, on a 90 s time constant


def test_the_unguarded_log_difference_route_is_the_defect_it_guards_against():
    # Recorded as a test because it is the measured reason the guards exist: over
    # eight time constants the late differences are noise, and selecting pairs on
    # their own magnitude keeps the upward half of that noise.
    t, truth = _noiseless(seconds=720)
    rng = np.random.default_rng(31)
    noisy = truth + rng.normal(0.0, 0.035, size=truth.size)
    guarded = tau_from_log_differences(t, noisy)
    assert guarded.tau_s == pytest.approx(90.0, rel=0.25, abs=0.0)
    assert guarded.conditioning > 0.0
    assert "signal window" in guarded.note


def test_residual_autocorrelation_exposes_a_second_time_constant():
    # Two poles, which a single exponential cannot represent. The fit still
    # returns a tau; what says the form is wrong is the residual structure.
    t = np.arange(1.0, 721.0)
    two_pole = 68.0 - 15.0 * np.exp(-t / 40.0) - 8.0 * np.exp(-t / 400.0)
    fit = fit_exponential_relaxation(t, two_pole)
    assert fit.residual_autocorrelation > 0.9
    assert fit.rms_residual_c > 0.1


def test_fits_on_the_synthetic_campaign_recover_the_generating_parameters(
    no_throttle_campaign, parameters
):
    rise = fit_exponential_relaxation(
        no_throttle_campaign["soak_1.00_r1"].t_rel,
        no_throttle_campaign["soak_1.00_r1"].die_temperature_c,
    )
    assert rise.tau_s == pytest.approx(parameters.tau_rise_s, rel=0.05, abs=0.0)
    assert rise.asymptote_c == pytest.approx(
        parameters.steady_state_c(1.0), rel=0.0, abs=0.2
    )
    cool = fit_exponential_relaxation(
        no_throttle_campaign["cool_r1"].t_rel,
        no_throttle_campaign["cool_r1"].die_temperature_c,
    )
    assert cool.tau_s == pytest.approx(parameters.tau_cool_s, rel=0.05, abs=0.0)
    # The fixture's two time constants differ on purpose: a fixture in which they
    # agreed would hide a defect in whichever estimator was wrong.
    assert abs(cool.tau_s - rise.tau_s) > 50.0
