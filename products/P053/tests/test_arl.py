"""Tests for the average-run-length estimators and the delay curve."""

from __future__ import annotations

import numpy as np
import pytest

from twininvalidate import (
    DetectorSpec,
    arl0_estimate,
    arl1_estimate,
    delay_after_onset,
    delay_curve,
    threshold_grid,
)


def test_arl0_known_answer_all_runs_alarm_immediately():
    # Four runs, statistic 5 at sample 0: each contributes 1 sample at risk and
    # one alarm, so ARL0 = 4/4 = 1 sample.
    stat = np.full((4, 10), 5.0)
    est = arl0_estimate(stat, 1.0)
    assert est.value == pytest.approx(1.0)
    assert est.n_detected == 4
    assert est.detection_fraction == pytest.approx(1.0)
    assert est.censored_fraction == pytest.approx(0.0)


def test_arl0_known_answer_with_censoring():
    # Two runs of horizon 10. Run 0 alarms at index 4, contributing 5 samples
    # at risk and 1 alarm. Run 1 never alarms, contributing 10 samples and no
    # alarm. ARL0 = (5 + 10) / 1 = 15 samples; the plain censored mean is
    # (5 + 10) / 2 = 7.5, which under-estimates it.
    stat = np.zeros((2, 10))
    stat[0, 4] = 9.0
    est = arl0_estimate(stat, 1.0)
    assert est.value == pytest.approx(15.0)
    assert est.lower_bound == pytest.approx(7.5)
    assert est.n_detected == 1
    assert est.stderr == pytest.approx(15.0)  # 15 / sqrt(1)


def test_arl0_returns_infinity_when_nothing_alarms():
    est = arl0_estimate(np.zeros((3, 50)), 1.0)
    assert np.isinf(est.value)
    assert np.isnan(est.stderr)
    assert est.lower_bound == pytest.approx(50.0)
    assert est.n_detected == 0
    assert est.censored_fraction == pytest.approx(1.0)


def test_arl0_standard_error_scales_as_one_over_root_alarms():
    stat = np.zeros((100, 20))
    stat[:, 9] = 9.0  # every run alarms at index 9
    est = arl0_estimate(stat, 1.0)
    assert est.value == pytest.approx(10.0)
    assert est.stderr == pytest.approx(10.0 / np.sqrt(100))


def test_arl1_known_answer():
    # Three runs: alarms at index 1, 3 and never, horizon 10.
    # Mean delay over detected runs = (2 + 4) / 2 = 3 samples.
    # Lower bound with the censored run at the horizon = (2 + 4 + 10) / 3 = 5.333
    stat = np.zeros((3, 10))
    stat[0, 1] = 9.0
    stat[1, 3] = 9.0
    est = arl1_estimate(stat, 1.0)
    assert est.value == pytest.approx(3.0)
    assert est.lower_bound == pytest.approx(16.0 / 3.0)
    assert est.detection_fraction == pytest.approx(2.0 / 3.0)


def test_arl1_returns_infinity_when_nothing_alarms():
    est = arl1_estimate(np.zeros((3, 40)), 1.0)
    assert np.isinf(est.value)
    assert est.lower_bound == pytest.approx(40.0)


def test_arl1_standard_error_is_nan_for_a_single_detection():
    stat = np.zeros((2, 10))
    stat[0, 2] = 9.0
    est = arl1_estimate(stat, 1.0)
    assert np.isnan(est.stderr)


def test_delay_after_onset_excludes_pre_onset_alarms():
    # Run 0 alarms at index 2, before the onset at 5: it is a false alarm and
    # is excluded. Run 1 alarms at index 7, i.e. 3 samples after the onset.
    stat = np.zeros((2, 12))
    stat[0, 2] = 9.0
    stat[1, 7] = 9.0
    est, n_pre = delay_after_onset(stat, 1.0, onset=5)
    assert n_pre == 1
    assert est.n_runs == 1
    assert est.value == pytest.approx(3.0)


def test_delay_after_onset_with_zero_onset_is_the_zero_state_delay():
    stat = np.zeros((2, 12))
    stat[0, 3] = 9.0
    stat[1, 5] = 9.0
    est, n_pre = delay_after_onset(stat, 1.0, onset=0)
    assert n_pre == 0
    assert est.value == pytest.approx(arl1_estimate(stat, 1.0).value)


def test_delay_after_onset_raises_if_every_run_false_alarms():
    stat = np.full((3, 12), 9.0)
    with pytest.raises(ValueError, match="false alarm before the change onset"):
        delay_after_onset(stat, 1.0, onset=5)


@pytest.mark.parametrize("onset", [-1, 12, 99])
def test_delay_after_onset_validates_the_onset(onset):
    with pytest.raises(ValueError, match="onset must lie"):
        delay_after_onset(np.zeros((2, 12)), 1.0, onset)


def test_delay_after_onset_requires_two_dimensions():
    with pytest.raises(ValueError, match="must be 2-D"):
        delay_after_onset(np.zeros(12), 1.0, 2)


def test_delay_curve_is_monotone_in_the_threshold(in_control, stepped):
    spec = DetectorSpec("cusum")
    grid = np.array([3.0, 6.0, 9.0, 12.0])
    curve = delay_curve(spec, in_control, stepped, grid)
    arl0 = [p.arl0.value for p in curve]
    arl1 = [p.arl1.value for p in curve]
    assert arl0 == sorted(arl0)
    assert arl1 == sorted(arl1)
    assert [p.threshold for p in curve] == pytest.approx(grid.tolist())
    assert curve[0].row[0] == pytest.approx(3.0)


def test_delay_curve_rejects_bad_thresholds(in_control, stepped):
    spec = DetectorSpec("cusum")
    with pytest.raises(ValueError, match="at least one value"):
        delay_curve(spec, in_control, stepped, np.array([]))
    with pytest.raises(ValueError, match="must be non-negative"):
        delay_curve(spec, in_control, stepped, np.array([-1.0]))


def test_threshold_grid_spans_and_is_increasing(in_control):
    stat = DetectorSpec("cusum").statistic(in_control)
    grid = threshold_grid(stat, n_points=8)
    assert grid.size == 8
    assert np.all(np.diff(grid) > 0)
    assert grid[0] > 0.0


def test_threshold_grid_validates_its_inputs(in_control):
    stat = DetectorSpec("cusum").statistic(in_control)
    with pytest.raises(ValueError, match="n_points must be at least 2"):
        threshold_grid(stat, n_points=1)
    with pytest.raises(ValueError, match="no spread"):
        threshold_grid(np.zeros((4, 10)))
