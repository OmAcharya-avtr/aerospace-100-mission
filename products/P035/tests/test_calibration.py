"""Tests for window trigger levels, threshold calibration and binomial intervals."""

from __future__ import annotations

import numpy as np
import pytest

from telemetryool.calibration import (
    binomial_se,
    calibrate_threshold,
    clopper_pearson_interval,
    estimate_rate,
    wilson_interval,
    window_trigger_level,
    windows_for_precision,
)
from telemetryool.runs import any_run


def test_window_trigger_level_persistence_one_is_the_row_max() -> None:
    scores = np.array([[1.0, 5.0, 2.0], [0.0, -1.0, -2.0]])
    assert window_trigger_level(scores, 1).tolist() == [5.0, 0.0]


def test_window_trigger_level_hand_values() -> None:
    """persistence = 2 over [1, 5, 2, 4, 4]:

    rolling pairwise minima: min(1,5)=1, min(5,2)=2, min(2,4)=2, min(4,4)=4
    maximum of those = 4, so the window alarms at any threshold below 4.
    """
    scores = np.array([[1.0, 5.0, 2.0, 4.0, 4.0]])
    assert window_trigger_level(scores, 2).tolist() == [4.0]
    # persistence = 3: triples are min(1,5,2)=1, min(5,2,4)=2, min(2,4,4)=2 -> 2
    assert window_trigger_level(scores, 3).tolist() == [2.0]


def test_trigger_level_is_exactly_the_alarm_boundary() -> None:
    """The defining property: a window alarms at threshold h iff w > h."""
    rng = np.random.default_rng(1)
    scores = rng.standard_normal((300, 30))
    for persistence in (1, 2, 4):
        w = window_trigger_level(scores, persistence)
        for h in (-1.0, 0.0, 0.5, 1.5, 2.5):
            assert np.array_equal(any_run(scores > h, persistence), w > h)


def test_trigger_level_minus_inf_when_window_is_too_short() -> None:
    scores = np.zeros((2, 3))
    assert np.all(np.isneginf(window_trigger_level(scores, 4)))


def test_calibration_hits_the_discrete_target_exactly() -> None:
    """With 20000 windows and alpha = 0.05 the target is representable exactly:
    ceil(0.05 * 20000) = 1000 windows must alarm in sample."""
    rng = np.random.default_rng(2)
    scores = rng.standard_normal((20000, 100))
    calib = calibrate_threshold(scores, 3, 0.05)
    assert calib.n_alarming == 1000
    assert calib.achieved_alpha_w == pytest.approx(0.05, abs=1e-12)
    assert calib.n_calibration_windows == 20000


def test_calibration_generalises_out_of_sample() -> None:
    """The threshold calibrated on one nominal sample must deliver close to the
    same rate on an independent one.  With n_cal = n_meas = 20000 the combined
    standard error is sqrt(0.05 * 0.95 * 2 / 20000) = 0.00218, so a 4-sigma band
    is [0.0413, 0.0587]."""
    rng = np.random.default_rng(3)
    calib = calibrate_threshold(rng.standard_normal((20000, 100)), 3, 0.05)
    measured = any_run(rng.standard_normal((20000, 100)) > calib.threshold, 3).mean()
    assert 0.0413 < measured < 0.0587


def test_calibration_rejects_unrepresentable_target() -> None:
    rng = np.random.default_rng(4)
    with pytest.raises(ValueError, match="needs at least"):
        calibrate_threshold(rng.standard_normal((50, 20)), 1, 0.001)


def test_calibration_target_validation() -> None:
    scores = np.zeros((100, 10))
    with pytest.raises(ValueError, match=r"must lie in \(0, 1\)"):
        calibrate_threshold(scores, 1, 0.0)
    with pytest.raises(ValueError, match=r"must lie in \(0, 1\)"):
        calibrate_threshold(scores, 1, 1.0)


def test_calibration_rejects_persistence_longer_than_the_window() -> None:
    with pytest.raises(ValueError, match="no window can ever alarm"):
        calibrate_threshold(np.zeros((100, 3)), 4, 0.05)


def test_trigger_level_shape_validation() -> None:
    with pytest.raises(ValueError, match="must be 2-D"):
        window_trigger_level(np.zeros(5), 1)
    with pytest.raises(ValueError, match="persistence must be an integer >= 1"):
        window_trigger_level(np.zeros((2, 5)), 0)


def test_binomial_se_hand_values() -> None:
    """sqrt(0.05 * 0.95 / 20000) = 0.0015411035007422441 and
    sqrt(0.5 * 0.5 / 100) = 0.05."""
    assert binomial_se(0.05, 20000) == pytest.approx(
        float(np.sqrt(0.05 * 0.95 / 20000)), abs=1e-18
    )
    assert binomial_se(0.5, 100) == pytest.approx(0.05, abs=1e-15)
    assert binomial_se(0.0, 10) == 0.0


def test_wilson_interval_known_value() -> None:
    """Wilson 95 % interval for 0 successes in 10 trials.

    At p = 0 the centre z^2/(2n) / (1 + z^2/n) and the half-width
    z sqrt(z^2 / (4 n^2)) / (1 + z^2/n) are equal, so the lower bound is exactly
    0 and the upper bound collapses to (z^2/n) / (1 + z^2/n).  With
    z = 1.959963984540054 and n = 10 that is 0.2775327998628892.
    """
    z = 1.959963984540054
    closed_form = (z * z / 10.0) / (1.0 + z * z / 10.0)
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0
    assert hi == pytest.approx(closed_form, rel=1e-12)
    assert hi == pytest.approx(0.27753, abs=1e-5)
    # The Clopper-Pearson upper bound 1 - 0.025**(1/10) = 0.3085 is wider, as it
    # must be: the exact interval is conservative relative to the score interval.
    assert clopper_pearson_interval(0, 10)[1] > hi


def test_clopper_pearson_known_values() -> None:
    """0 successes in 10 trials: lower 0, upper 1 - 0.025**(1/10) = 0.3085.
    10 successes in 10: lower 0.025**(1/10) = 0.6915, upper 1."""
    lo, hi = clopper_pearson_interval(0, 10)
    assert lo == 0.0
    assert hi == pytest.approx(1.0 - 0.025 ** (1 / 10), abs=1e-12)
    lo, hi = clopper_pearson_interval(10, 10)
    assert lo == pytest.approx(0.025 ** (1 / 10), abs=1e-12)
    assert hi == 1.0


def test_clopper_pearson_contains_wilson_centre() -> None:
    for k in (1, 50, 500, 999):
        cp = clopper_pearson_interval(k, 1000)
        assert cp[0] <= k / 1000 <= cp[1]


def test_estimate_rate_fields_and_z() -> None:
    est = estimate_rate(1000, 20000)
    assert est.rate == 0.05
    assert est.standard_error == pytest.approx(0.0015411, abs=1e-6)
    assert est.z_against(0.05) == pytest.approx(0.0, abs=1e-12)
    # 1100 of 20000 is 0.055, which is (0.055 - 0.05) / 0.0015411 = 3.24 null SE.
    assert estimate_rate(1100, 20000).z_against(0.05) == pytest.approx(3.2445, abs=1e-3)
    assert np.isnan(est.z_against(0.0))


def test_interval_validation() -> None:
    with pytest.raises(ValueError, match="trials must be >= 1"):
        wilson_interval(0, 0)
    with pytest.raises(ValueError, match="successes must lie in"):
        wilson_interval(11, 10)
    with pytest.raises(ValueError, match="confidence must lie in"):
        wilson_interval(1, 10, 1.0)
    with pytest.raises(ValueError, match="trials must be >= 1"):
        clopper_pearson_interval(0, 0)
    with pytest.raises(ValueError, match="successes must lie in"):
        clopper_pearson_interval(-1, 10)
    with pytest.raises(ValueError, match="confidence must lie in"):
        clopper_pearson_interval(1, 10, 0.0)
    with pytest.raises(ValueError, match="rate must lie in"):
        binomial_se(1.5, 10)
    with pytest.raises(ValueError, match="trials must be >= 1"):
        binomial_se(0.5, 0)


def test_windows_for_precision_hand_values() -> None:
    """M = (1 - alpha) / (alpha * r^2): alpha = 0.05, r = 0.10 gives
    0.95 / (0.05 * 0.01) = 1900 exactly."""
    assert windows_for_precision(0.05, 0.10) == 1900
    assert windows_for_precision(0.5, 0.01) == 10000
    with pytest.raises(ValueError, match="alpha must lie in"):
        windows_for_precision(0.0, 0.1)
    with pytest.raises(ValueError, match="relative_se must be > 0"):
        windows_for_precision(0.05, 0.0)


def test_windows_for_precision_round_trips_through_binomial_se() -> None:
    alpha = 0.05
    m = windows_for_precision(alpha, 0.031)
    assert binomial_se(alpha, m) / alpha == pytest.approx(0.031, rel=1e-4)
