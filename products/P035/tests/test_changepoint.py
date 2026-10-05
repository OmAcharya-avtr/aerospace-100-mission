"""Tests for the change-point statistic, its Monte-Carlo threshold and segmentation."""

from __future__ import annotations

import numpy as np
import pytest

from telemetryool.changepoint import (
    calibrate_change_point_threshold,
    detect_change_points,
    max_mean_shift_statistic,
)


def test_statistic_hand_value_on_a_clean_step() -> None:
    """x = [0, 0, 2, 2], sigma = 1.  The cumulative sum is [0, 0, 2, 4] and
    S_4 = 4, n = 4.  For tau = 2 (m = 2): |S_2 - (2/4) * 4| = |0 - 2| = 2 and
    the scale is sqrt(2 * 2 / 4) = 1, so D(2) = 2.
    For tau = 1 (m = 1): |0 - 1| = 1, scale sqrt(1 * 3 / 4) = 0.8660254, D = 1.1547.
    For tau = 3 (m = 3): |2 - 3| = 1, same scale, D = 1.1547.
    So the maximum is 2.0 at index 2.
    """
    st = max_mean_shift_statistic(np.array([0.0, 0.0, 2.0, 2.0]))
    assert st.statistic == pytest.approx(2.0, abs=1e-14)
    assert st.index == 2
    assert (st.start, st.stop) == (0, 4)


def test_statistic_scales_inversely_with_sigma() -> None:
    x = np.array([0.0, 0.0, 2.0, 2.0])
    assert max_mean_shift_statistic(x, sigma=2.0).statistic == pytest.approx(1.0, abs=1e-14)


def test_statistic_is_zero_for_a_constant_series() -> None:
    st = max_mean_shift_statistic(np.full(10, 3.0))
    assert st.statistic == pytest.approx(0.0, abs=1e-14)


def test_statistic_on_a_segment_slice() -> None:
    x = np.concatenate([np.zeros(5), np.array([0.0, 0.0, 2.0, 2.0]), np.zeros(5)])
    st = max_mean_shift_statistic(x, start=5, stop=9)
    assert st.statistic == pytest.approx(2.0, abs=1e-14)
    assert st.index == 7


def test_statistic_degenerate_segment() -> None:
    st = max_mean_shift_statistic(np.array([1.0]), start=0, stop=1)
    assert np.isneginf(st.statistic)
    assert st.index == -1


def test_statistic_validation() -> None:
    with pytest.raises(ValueError, match="must be 1-D"):
        max_mean_shift_statistic(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="sigma must be > 0"):
        max_mean_shift_statistic(np.zeros(5), sigma=0.0)
    with pytest.raises(ValueError, match="invalid segment bounds"):
        max_mean_shift_statistic(np.zeros(5), start=3, stop=2)


def test_threshold_achieves_its_target_in_sample() -> None:
    th = calibrate_change_point_threshold(200, 0.05, 20000, np.random.default_rng(0))
    assert th.achieved_alpha == pytest.approx(0.05, abs=1e-12)
    assert th.n_segments == 200 and th.n_simulations == 20000
    assert th.standard_error == pytest.approx(np.sqrt(0.05 * 0.95 / 20000), rel=1e-9)


def test_threshold_is_larger_for_a_tighter_alpha() -> None:
    loose = calibrate_change_point_threshold(200, 0.10, 20000, np.random.default_rng(1))
    tight = calibrate_change_point_threshold(200, 0.01, 20000, np.random.default_rng(1))
    assert tight.threshold > loose.threshold


def test_threshold_generalises_out_of_sample() -> None:
    """Calibrated on one Monte-Carlo sample, applied to an independent one.
    With 20000 replicates each, the combined SE is sqrt(0.05 * 0.95 * 2 / 20000)
    = 0.00218, so a 5-sigma band is [0.039, 0.061]."""
    th = calibrate_change_point_threshold(150, 0.05, 20000, np.random.default_rng(2))
    rng = np.random.default_rng(3)
    draws = rng.standard_normal((20000, 150))
    stats = np.array([max_mean_shift_statistic(row).statistic for row in draws])
    assert 0.039 < float((stats > th.threshold).mean()) < 0.061


def test_threshold_validation() -> None:
    with pytest.raises(ValueError, match="n_segments must be >= 2"):
        calibrate_change_point_threshold(1, 0.05)
    with pytest.raises(ValueError, match=r"alpha must lie in \(0, 1\)"):
        calibrate_change_point_threshold(100, 0.0)
    with pytest.raises(ValueError, match="n_simulations must be >= 1"):
        calibrate_change_point_threshold(100, 0.05, 0)
    with pytest.raises(ValueError, match="needs at least"):
        calibrate_change_point_threshold(100, 0.001, 100)


def test_single_change_point_localisation_over_many_seeds() -> None:
    """A 3-sigma step at index 120 of a 200-sample standard-normal series.

    Over 100 seeds the global maximiser of the statistic lands exactly on 120 in
    the large majority of trials and within one sample in nearly all of them.
    The thresholds below are the measured behaviour recorded as a regression
    bound, not a claim of exactness: localisation error does not go to zero at
    finite shift size, and the measured distribution over 200 seeds is
    exact 0.84, within 1 sample 0.97, within 2 samples 0.99, worst case 4
    (validation/validate_changepoint.py).
    """
    th = calibrate_change_point_threshold(200, 0.05, 4000, np.random.default_rng(4))
    maximiser_error = []
    nearest_found = []
    for seed in range(100):
        rng = np.random.default_rng(1000 + seed)
        x = rng.standard_normal(200)
        x[120:] += 3.0
        maximiser_error.append(abs(max_mean_shift_statistic(x).index - 120))
        found = detect_change_points(x, th.threshold, min_segment=20)
        assert found, f"seed {seed}: a 3-sigma step was missed entirely"
        nearest_found.append(min(abs(f - 120) for f in found))
    err = np.array(maximiser_error)
    near = np.array(nearest_found)
    assert float((err == 0).mean()) >= 0.75
    assert float((err <= 1).mean()) >= 0.92
    assert float((near <= 2).mean()) >= 0.95


def test_two_change_points_are_both_found() -> None:
    rng = np.random.default_rng(5)
    x = rng.standard_normal(300)
    x[100:200] += 4.0
    th = calibrate_change_point_threshold(300, 0.05, 4000, np.random.default_rng(6))
    found = detect_change_points(x, th.threshold, min_segment=25)
    assert len(found) >= 2
    assert min(abs(f - 100) for f in found) <= 2
    assert min(abs(f - 200) for f in found) <= 2


def test_no_change_point_in_pure_noise_at_a_tight_threshold() -> None:
    rng = np.random.default_rng(7)
    assert detect_change_points(rng.standard_normal(200), 10.0, min_segment=20) == []


def test_min_segment_blocks_splitting_short_segments() -> None:
    x = np.concatenate([np.zeros(10), np.full(10, 10.0)])
    assert detect_change_points(x, 1.0, min_segment=15) == []
    assert detect_change_points(x, 1.0, min_segment=5) == [10]


def test_max_depth_limits_recursion() -> None:
    rng = np.random.default_rng(8)
    x = rng.standard_normal(400)
    shallow = detect_change_points(x, 0.5, min_segment=2, max_depth=1)
    deep = detect_change_points(x, 0.5, min_segment=2, max_depth=6)
    assert len(shallow) == 1
    assert len(deep) > len(shallow)


def test_detect_validation() -> None:
    with pytest.raises(ValueError, match="must be 1-D"):
        detect_change_points(np.zeros((2, 2)), 1.0)
    with pytest.raises(ValueError, match="min_segment must be >= 2"):
        detect_change_points(np.zeros(20), 1.0, min_segment=1)
    with pytest.raises(ValueError, match="max_depth must be >= 1"):
        detect_change_points(np.zeros(20), 1.0, max_depth=0)
