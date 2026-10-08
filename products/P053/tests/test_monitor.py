"""Tests for the streaming monitor wrapper."""

from __future__ import annotations

import numpy as np
import pytest

from twininvalidate import DetectorSpec, InvalidationMonitor


@pytest.fixture(scope="module")
def monitor(in_control):
    return InvalidationMonitor.from_target(DetectorSpec("cusum"), in_control, 1000.0)


def test_from_target_records_its_calibration(monitor):
    assert monitor.calibration is not None
    assert monitor.calibration.target_arl0 == pytest.approx(1000.0)
    assert monitor.threshold == pytest.approx(monitor.calibration.threshold)
    assert monitor.sample_rate_hz == pytest.approx(20.0)


def test_running_on_in_control_data_rarely_alarms(monitor, in_control):
    verdict = monitor.run(in_control[:, :500])
    # ARL0 is 1000 samples, so over 500 samples the chance a run alarms is
    # roughly 1 - exp(-0.5) = 0.39. Anything above 0.8 would mean the
    # threshold is wrong by a wide margin.
    assert verdict.alarm_fraction < 0.8


def test_running_on_a_changed_stream_alarms(monitor, stepped):
    verdict = monitor.run(stepped)
    assert verdict.alarm_fraction > 0.95
    assert "alarmed at threshold" in verdict.describe()


def test_describe_handles_the_no_alarm_case(monitor):
    verdict = monitor.run(np.zeros((3, 50)))
    assert verdict.alarm_fraction == pytest.approx(0.0)
    assert "no alarm in 3 run(s)" in verdict.describe()


def test_verdict_exposes_the_statistic_and_threshold(monitor, stepped):
    verdict = monitor.run(stepped[:5])
    assert verdict.statistic.shape == (5, stepped.shape[1])
    assert verdict.threshold == pytest.approx(monitor.threshold)
    assert verdict.alarmed.dtype == bool


def test_arl0_and_arl1_round_trip(monitor, in_control, stepped):
    arl0 = monitor.arl0(in_control)
    arl1 = monitor.arl1(stepped)
    assert arl0.value > 10.0 * arl1.value
    assert arl1.detection_fraction > 0.95


def test_false_alarm_rate_conversion(monitor):
    # ARL0 1000 samples at 20 Hz is 50 s, so 1000 h / 50 s = 72000 per 1000 h.
    assert monitor.false_alarms_per_1000h(1000.0) == pytest.approx(72000.0)


def test_direct_construction_has_no_calibration_record():
    monitor = InvalidationMonitor(spec=DetectorSpec("ewma"), threshold=3.0)
    assert monitor.calibration is None
    assert monitor.threshold == pytest.approx(3.0)
