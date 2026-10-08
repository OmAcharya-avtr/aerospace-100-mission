"""Regression and benchmark tests: locked numbers and the compute budget.

Every value locked here was produced by running this repository's own code in
the session that wrote the test, with the seeds the test passes. The locks are
exact where the quantity is deterministic (the Riccati solution, the
steady-state gain, the closed-form thresholds) and have a stated Monte-Carlo
tolerance where the quantity is an estimate from a finite sample. No tolerance
here was widened to make a failing check pass; where a check cannot be made
tight it is stated as an inequality instead.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from twininvalidate import (
    SCENARIOS,
    DetectorSpec,
    arl1_estimate,
    calibrate_threshold,
    changed_streams,
    ewma_lambda1_threshold,
    glr_window1_threshold,
    in_control_streams,
    max_absolute_difference,
    paired_streams,
    reference_twin,
)

# Locked in the build session, seeds as written below.
LOCKED_S = 1.165114435e-04
LOCKED_SQRT_S = 1.079404667e-02
LOCKED_SPECTRAL_RADIUS = 0.917218411
LOCKED_GAIN = (0.15168442, 0.18259584)

LOCKED_THRESHOLDS = {
    "cusum": 9.893276,
    "ewma": 3.046571,
    "glr": 6.721209,
    "varcusum": 33.933860,
}

LOCKED_DELAYS = {
    ("parameter_step", "cusum"): 62.1,
    ("parameter_step", "ewma"): 76.7,
    ("parameter_step", "glr"): 77.1,
    ("slow_ramp", "cusum"): 236.5,
    ("slow_ramp", "ewma"): 258.6,
    ("slow_ramp", "glr"): 262.3,
    ("noise_variance", "cusum"): 208.1,
    ("noise_variance", "ewma"): 221.9,
    ("noise_variance", "glr"): 234.9,
}


@pytest.fixture(scope="module")
def bank():
    """The locked 120 x 2000 in-control calibration bank, seed 53001."""
    return in_control_streams(n_runs=120, n_samples=2000, seed=53001)


def test_steady_state_quantities_are_unchanged():
    filt = reference_twin().steady_state()
    assert filt.S == pytest.approx(LOCKED_S, rel=1e-9)
    assert np.sqrt(filt.S) == pytest.approx(LOCKED_SQRT_S, rel=1e-9)
    assert filt.spectral_radius() == pytest.approx(LOCKED_SPECTRAL_RADIUS, rel=1e-9)
    assert filt.K.ravel().tolist() == pytest.approx(list(LOCKED_GAIN), rel=1e-7)


def test_closed_form_thresholds_are_unchanged():
    assert glr_window1_threshold(1000.0) == pytest.approx(5.4137830853, rel=1e-9)
    assert ewma_lambda1_threshold(1000.0) == pytest.approx(3.2905267315, rel=1e-9)


@pytest.mark.parametrize("name", sorted(LOCKED_THRESHOLDS))
def test_calibrated_thresholds_are_unchanged(bank, name):
    # Bisection on a fixed bank is deterministic, so these are exact to the
    # bisection tolerance of 1e-4 rather than to Monte-Carlo noise.
    cal = calibrate_threshold(DetectorSpec(name), bank, 1000.0)
    assert cal.threshold == pytest.approx(LOCKED_THRESHOLDS[name], abs=2e-4)
    assert cal.achieved_arl0 == pytest.approx(1000.0, rel=0.02)


@pytest.mark.parametrize("key", sorted(LOCKED_DELAYS))
def test_detection_delays_are_unchanged(bank, key):
    scenario, name = key
    spec = DetectorSpec(name)
    threshold = calibrate_threshold(spec, bank, 1000.0).threshold
    oc = changed_streams(scenario, n_runs=120, n_samples=2000)
    est = arl1_estimate(spec.statistic(oc), threshold)
    assert est.detection_fraction == pytest.approx(1.0)
    # The streams are seeded, so this is a deterministic recomputation and the
    # tolerance only absorbs the bisection tolerance on the threshold.
    assert est.value == pytest.approx(LOCKED_DELAYS[key], rel=0.01)


def test_cusum_beats_the_other_two_baselines_on_every_scenario(bank):
    # The measured ordering, locked as an ordering rather than as numbers:
    # the unbounded-accumulation CUSUM is faster than both the EWMA and the
    # windowed GLR on all three declared scenarios at a matched ARL0.
    thresholds = {
        n: calibrate_threshold(DetectorSpec(n), bank, 1000.0).threshold
        for n in ("cusum", "ewma", "glr")
    }
    for scenario in SCENARIOS:
        oc = changed_streams(scenario, n_runs=120, n_samples=2000)
        delays = {
            n: arl1_estimate(DetectorSpec(n).statistic(oc), thresholds[n]).value
            for n in thresholds
        }
        assert delays["cusum"] < delays["ewma"], scenario
        assert delays["cusum"] < delays["glr"], scenario


def test_ambiguity_difference_is_unchanged():
    assert max_absolute_difference(paired_streams()) == pytest.approx(6.173e-14, rel=0.05)


def test_simulation_throughput_is_within_the_compute_budget():
    # Wall-clock figures on this container move 10-20 % between runs, so the
    # budget is set an order of magnitude above the measured 0.06 s. This is a
    # guard against an accidental de-vectorisation, not a hardware
    # characteristic: 2 cores, measured in the build session.
    start = time.perf_counter()
    z = in_control_streams(n_runs=100, n_samples=2000, seed=1)
    elapsed = time.perf_counter() - start
    assert z.shape == (100, 2000)
    assert elapsed < 5.0


def test_glr_statistic_throughput_is_within_the_compute_budget():
    z = in_control_streams(n_runs=100, n_samples=2000, seed=1)
    start = time.perf_counter()
    stat = DetectorSpec("glr").statistic(z)
    elapsed = time.perf_counter() - start
    assert stat.shape == z.shape
    # Measured 0.07 s; the guard is at 5 s for the same reason as above.
    assert elapsed < 5.0


def test_classifier_single_row_latency_is_measured_not_asserted(classifier):
    # Single-row inference is what a streaming monitor does. The forest's
    # n_jobs is forced to 1 after fitting because n_jobs > 1 is several times
    # slower per row on this container. The assertion is loose on purpose: the
    # number belongs in validation/, not in a pass/fail gate.
    x = np.zeros((1, 9))
    classifier.confidence(x)  # warm up
    start = time.perf_counter()
    for _ in range(20):
        classifier.confidence(x)
    per_call_ms = (time.perf_counter() - start) / 20.0 * 1e3
    assert classifier.forest is not None
    assert classifier.forest.n_jobs == 1
    assert per_call_ms < 200.0
