"""End-to-end integration: declare a twin, calibrate, inject, measure, compare.

This test walks the whole pipeline the README describes, in one go, and asserts
the properties that make the result meaningful rather than any particular
number. Locked numbers live in ``test_regression.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from twininvalidate import (
    SCENARIOS,
    AssetChange,
    DetectorSpec,
    InvalidationMonitor,
    StreamSpec,
    arl0_from_rate,
    bracket_threshold,
    calibrate_threshold,
    changed_streams,
    delay_after_onset,
    delay_curve,
    in_control_streams,
    rate_from_arl0,
    reference_twin,
    simulate_residuals,
    threshold_grid,
)

TARGET_ARL0 = 1000.0


def test_end_to_end_pipeline_for_every_scenario(in_control):
    twin = reference_twin()
    filt = twin.steady_state()
    assert filt.spectral_radius() < 1.0

    specs = [DetectorSpec(n) for n in ("cusum", "ewma", "glr")]
    monitors = {
        s.name: InvalidationMonitor.from_target(s, in_control, TARGET_ARL0) for s in specs
    }
    for name, monitor in monitors.items():
        assert monitor.calibration is not None
        assert monitor.calibration.achieved_arl0 == pytest.approx(TARGET_ARL0, rel=0.1), name

    fresh = in_control_streams(n_runs=60, n_samples=2000, seed=53110)
    for name, monitor in monitors.items():
        arl0 = monitor.arl0(fresh)
        # On a fresh in-control bank the achieved ARL0 must agree with the
        # target to within three standard errors of the censored MLE.
        assert abs(arl0.value - TARGET_ARL0) < 3.0 * arl0.stderr + 0.05 * TARGET_ARL0, name

    for scenario in SCENARIOS:
        oc = changed_streams(scenario, n_runs=60, n_samples=2000)
        for name, monitor in monitors.items():
            arl1 = monitor.arl1(oc)
            assert arl1.detection_fraction > 0.9, (scenario, name)
            assert arl1.value < TARGET_ARL0, (scenario, name)


def test_the_curve_trades_delay_against_false_alarms(in_control):
    spec = DetectorSpec("cusum")
    oc = changed_streams("parameter_step", n_runs=60, n_samples=2000)
    grid = threshold_grid(spec.statistic(in_control), n_points=6)
    curve = delay_curve(spec, in_control, oc, grid)
    arl0 = np.array([p.arl0.value for p in curve])
    arl1 = np.array([p.arl1.value for p in curve])
    # Both must rise together: there is no threshold that buys a longer ARL0
    # and a shorter delay, which is the whole content of the trade-off.
    assert np.all(np.diff(arl0) > 0)
    assert np.all(np.diff(arl1) >= 0)
    rates = np.array([rate_from_arl0(a, 20.0) for a in arl0 if np.isfinite(a)])
    assert np.all(np.diff(rates) < 0)


def test_false_alarm_requirement_converts_and_round_trips():
    # A requirement of 10 false alarms per 1000 h at 20 Hz.
    target = arl0_from_rate(10.0, 20.0)
    assert target == pytest.approx(7.2e6)
    assert rate_from_arl0(target, 20.0) == pytest.approx(10.0)


def test_noise_variance_is_the_hardest_scenario_for_the_baselines(in_control):
    # Measured claim, stated as an inequality so it is a test and not a quote:
    # at a matched ARL0 the delay on the noise-variance change is at least
    # twice the delay on the parameter step, for all three baselines, even
    # though both changes are fully present from sample 0.
    delays = {}
    for name in ("cusum", "ewma", "glr"):
        spec = DetectorSpec(name)
        threshold = calibrate_threshold(spec, in_control, TARGET_ARL0).threshold
        for scenario in ("parameter_step", "noise_variance"):
            oc = changed_streams(scenario, n_runs=80, n_samples=2000)
            delays[(name, scenario)] = spec.statistic(oc)
            est = InvalidationMonitor(spec=spec, threshold=threshold).arl1(oc)
            delays[(name, scenario)] = est.value
    for name in ("cusum", "ewma", "glr"):
        assert delays[(name, "noise_variance")] > 2.0 * delays[(name, "parameter_step")], name


def test_learned_model_and_baselines_are_compared_at_a_matched_arl0(
    classifier, in_control
):
    specs = [DetectorSpec(n) for n in ("cusum", "ewma", "glr")]
    thresholds = {
        s.name: calibrate_threshold(s, in_control, TARGET_ARL0).threshold for s in specs
    }
    bracket = bracket_threshold(classifier.statistic(in_control[:60]), TARGET_ARL0)
    assert bracket.conservative_arl0 >= TARGET_ARL0
    onset = classifier.window

    change = SCENARIOS["parameter_step"]
    oc = simulate_residuals(
        StreamSpec(
            change=AssetChange(change.kind, onset, change.magnitude, change.ramp_samples),
            n_runs=60,
            n_samples=1500,
            seed=53200,
        )
    )
    for spec in specs:
        est, _ = delay_after_onset(spec.statistic(oc), thresholds[spec.name], onset)
        assert est.detection_fraction > 0.9
        assert est.value > 0.0
    learned, n_pre = delay_after_onset(classifier.statistic(oc), bracket.conservative, onset)
    assert learned.detection_fraction > 0.5
    assert n_pre < 0.2 * oc.shape[0]
    # The window-fill floor: a windowed classifier cannot alarm before its
    # window is full, so its delay is bounded below by 1 and its *statistic*
    # is identically zero for the first window-1 samples.
    assert np.all(classifier.statistic(oc)[:, : onset - 1] == 0.0)


def test_learned_model_collapses_on_a_sign_flipped_change(classifier, in_control):
    # The decisive out-of-distribution check: the training set contains only
    # negative gain changes, so a positive one of the same size is outside it.
    bracket = bracket_threshold(classifier.statistic(in_control[:60]), TARGET_ARL0)
    spec = DetectorSpec("cusum")
    threshold = calibrate_threshold(spec, in_control, TARGET_ARL0).threshold
    onset = classifier.window
    oc = simulate_residuals(
        StreamSpec(
            change=AssetChange("parameter_step", onset, +0.01),
            n_runs=60,
            n_samples=1500,
            seed=53300,
        )
    )
    baseline, _ = delay_after_onset(spec.statistic(oc), threshold, onset)
    learned, _ = delay_after_onset(classifier.statistic(oc), bracket.conservative, onset)
    assert baseline.detection_fraction > 0.95
    assert learned.detection_fraction < 0.5
