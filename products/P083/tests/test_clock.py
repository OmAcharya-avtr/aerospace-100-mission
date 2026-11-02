"""Host-to-device drift estimation, reset detection and the duration correction."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hilverdict.clock import (
    CLOCK_CLEAN_VERDICT,
    CLOCK_DRIFTING,
    CLOCK_RESET,
    MAD_TO_SIGMA,
    detect_resets,
    drift_corrected_log,
    estimate_clock,
    least_squares_rate,
    theil_sen_rate,
)
from hilverdict.synthetic import ClockSpec, generate_clock, generate_run
from hilverdict.timeline import miss_events

PERIOD = 0.010


def _host(samples: int = 201) -> np.ndarray:
    return np.arange(samples, dtype=float) * PERIOD


def _device(drift_ppm: float, samples: int = 201, noise: float = 0.0,
            seed: int = 0) -> np.ndarray:
    host = _host(samples)
    generator = np.random.default_rng(seed)
    return (1.0 + drift_ppm * 1e-6) * host + noise * generator.standard_normal(samples)


def test_mad_scale_constant_is_the_published_one():
    assert MAD_TO_SIGMA == pytest.approx(1.4826, rel=1e-15, abs=0.0)


def test_least_squares_recovers_an_exact_rate():
    rate, error = least_squares_rate(_host(), _device(500.0))
    assert (rate - 1.0) * 1e6 == pytest.approx(500.0, rel=1e-9, abs=1e-9)
    assert error == pytest.approx(0.0, rel=0.0, abs=1e-15)


def test_least_squares_recovers_a_zero_rate():
    rate, _error = least_squares_rate(_host(), _device(0.0))
    assert rate == pytest.approx(1.0, rel=1e-15, abs=1e-15)


def test_least_squares_is_invariant_to_an_offset():
    first, _e1 = least_squares_rate(_host(), _device(300.0))
    second, _e2 = least_squares_rate(_host(), _device(300.0) + 1.234)
    assert first == pytest.approx(second, rel=1e-13, abs=0.0)


def test_least_squares_standard_error_is_nan_with_two_samples():
    _rate, error = least_squares_rate(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    assert math.isnan(error)


def test_least_squares_rejects_mismatched_series():
    with pytest.raises(ValueError, match="must match"):
        least_squares_rate(np.zeros(5), np.zeros(4))


def test_least_squares_rejects_a_single_sample():
    with pytest.raises(ValueError, match="at least 2 samples"):
        least_squares_rate(np.zeros(1), np.zeros(1))


def test_least_squares_rejects_a_constant_host_series():
    with pytest.raises(ValueError, match="not identified"):
        least_squares_rate(np.zeros(5), np.arange(5.0))


def test_theil_sen_recovers_an_exact_rate():
    rate = theil_sen_rate(_host(), _device(-250.0))
    assert (rate - 1.0) * 1e6 == pytest.approx(-250.0, rel=1e-9, abs=1e-9)


def test_theil_sen_matches_least_squares_without_noise():
    first, _error = least_squares_rate(_host(), _device(120.0))
    second = theil_sen_rate(_host(), _device(120.0))
    assert first == pytest.approx(second, rel=1e-12, abs=0.0)


def test_theil_sen_resists_a_single_corrupted_timestamp():
    device = _device(200.0)
    device[73] += 0.5
    robust = theil_sen_rate(_host(), device)
    squares, _error = least_squares_rate(_host(), device)
    assert abs((robust - 1.0) * 1e6 - 200.0) < abs((squares - 1.0) * 1e6 - 200.0)


def test_theil_sen_caps_its_pair_count():
    host = _host(400)
    device = (1.0 + 100e-6) * host
    rate = theil_sen_rate(host, device, max_pairs=500)
    assert (rate - 1.0) * 1e6 == pytest.approx(100.0, rel=1e-6, abs=1e-6)


def test_theil_sen_rejects_mismatched_series():
    with pytest.raises(ValueError, match="must match"):
        theil_sen_rate(np.zeros(5), np.zeros(4))


def test_theil_sen_rejects_a_single_sample():
    with pytest.raises(ValueError, match="at least 2 samples"):
        theil_sen_rate(np.zeros(1), np.zeros(1))


def test_theil_sen_rejects_a_constant_host_series():
    with pytest.raises(ValueError, match="not identified"):
        theil_sen_rate(np.zeros(5), np.arange(5.0))


def test_no_reset_is_detected_in_a_clean_series():
    assert detect_resets(_host(), _device(300.0, noise=2e-7, seed=3)) == ()


def test_a_large_reset_is_detected_at_its_index():
    device = _device(300.0, noise=2e-7, seed=3)
    device[120:] -= 1.2
    assert detect_resets(_host(), device) == (120,)


def test_two_resets_are_both_detected():
    device = _device(0.0, noise=2e-7, seed=5)
    device[60:] -= 0.6
    device[150:] -= 1.5
    assert detect_resets(_host(), device) == (60, 150)


def test_a_reset_below_the_declared_floor_is_not_detected():
    device = _device(0.0, noise=2e-7, seed=5)
    device[100:] -= 1e-5
    assert detect_resets(_host(), device) == ()


def test_the_detection_floor_can_be_lowered_by_declaration():
    device = _device(0.0, noise=1e-9, seed=5)
    device[100:] -= 1e-5
    assert detect_resets(_host(), device, floor_s=1e-7) == (100,)


def test_detect_resets_on_a_two_sample_series_returns_nothing():
    assert detect_resets(np.zeros(2), np.zeros(2)) == ()


def test_detect_resets_rejects_a_non_positive_scale():
    with pytest.raises(ValueError, match="scales must be positive"):
        detect_resets(_host(), _device(0.0), scales=0.0)


def test_detect_resets_rejects_a_non_positive_floor():
    with pytest.raises(ValueError, match="floor_s must be positive"):
        detect_resets(_host(), _device(0.0), floor_s=0.0)


def test_detect_resets_rejects_mismatched_series():
    with pytest.raises(ValueError, match="must match"):
        detect_resets(np.zeros(5), np.zeros(4))


def test_estimate_clock_reports_clean_for_a_small_drift():
    estimate = estimate_clock(_host(), _device(10.0, noise=2e-7, seed=1), 100.0)
    assert estimate.verdict == CLOCK_CLEAN_VERDICT
    assert estimate.usable is True


def test_estimate_clock_reports_drifting_beyond_the_allowance():
    estimate = estimate_clock(_host(), _device(900.0, noise=2e-7, seed=1), 100.0)
    assert estimate.verdict == CLOCK_DRIFTING
    assert estimate.usable is True


def test_estimate_clock_reports_reset_whatever_the_rate():
    device = _device(10.0, noise=2e-7, seed=1)
    device[80:] -= 0.8
    estimate = estimate_clock(_host(), device, 100.0)
    assert estimate.verdict == CLOCK_RESET
    assert estimate.usable is False
    assert estimate.resets == (80,)


def test_estimate_clock_segments_at_every_reset():
    device = _device(0.0, noise=2e-7, seed=2)
    device[70:] -= 0.7
    device[140:] -= 1.4
    estimate = estimate_clock(_host(), device, 100.0)
    assert [(s.start, s.stop) for s in estimate.segments] == [
        (0, 70), (70, 140), (140, 201)
    ]


def test_segment_samples_is_the_half_open_length():
    estimate = estimate_clock(_host(), _device(0.0), 100.0)
    assert estimate.segments[0].samples == 201


def test_pooled_drift_is_the_sample_weighted_mean():
    device = _device(400.0, noise=0.0)
    device[100:] -= 1.0
    estimate = estimate_clock(_host(), device, 1000.0)
    weights = np.array([segment.samples for segment in estimate.segments])
    drifts = np.array([segment.drift_ppm for segment in estimate.segments])
    assert estimate.drift_ppm == pytest.approx(
        float(np.sum(weights * drifts) / np.sum(weights)), rel=1e-12, abs=0.0
    )


def test_estimate_clock_rejects_a_negative_allowance():
    with pytest.raises(ValueError, match="non-negative"):
        estimate_clock(_host(), _device(0.0), -1.0)


def test_estimate_clock_raises_when_no_segment_is_identifiable():
    # A deliberately tiny threshold calls every sample a reset, so no segment
    # has two samples and no rate exists. This is the error path, exercised.
    host = np.array([0.0, 0.01, 0.02])
    device = np.array([0.0, 0.004, 0.011])
    with pytest.raises(ValueError, match="no segment between resets"):
        estimate_clock(host, device, 100.0, scales=0.001, floor_s=1e-12)


def test_estimate_clock_raises_on_a_constant_host_series():
    with pytest.raises(ValueError, match="not identified"):
        estimate_clock(np.zeros(4), np.zeros(4), 100.0)


def test_generate_clock_returns_the_injected_reset_indices():
    spec = ClockSpec(
        drift_ppm=0.0, jitter_s=1e-9, tick_s=1e-9, reset_fractions=(0.5,), name="t"
    )
    _device_series, resets = generate_clock(
        spec, _host(), np.random.default_rng(0)
    )
    assert resets == (100,)


def test_generate_clock_quantises_to_the_tick():
    spec = ClockSpec(
        drift_ppm=0.0, jitter_s=1e-9, tick_s=1e-6, reset_fractions=(), name="t"
    )
    device, _resets = generate_clock(spec, _host(), np.random.default_rng(0))
    residual = device / 1e-6 - np.round(device / 1e-6)
    assert float(np.max(np.abs(residual))) < 1e-9


def test_clock_spec_rejects_a_negative_jitter():
    with pytest.raises(ValueError, match="jitter_s must be non-negative"):
        ClockSpec(drift_ppm=0.0, jitter_s=-1.0, tick_s=1e-7,
                  reset_fractions=(), name="t")


def test_clock_spec_rejects_a_non_positive_tick():
    with pytest.raises(ValueError, match="tick_s must be positive"):
        ClockSpec(drift_ppm=0.0, jitter_s=1e-7, tick_s=0.0,
                  reset_fractions=(), name="t")


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.2, 1.5])
def test_clock_spec_rejects_reset_fractions_outside_the_open_interval(fraction):
    with pytest.raises(ValueError, match="reset fractions must lie"):
        ClockSpec(drift_ppm=0.0, jitter_s=1e-7, tick_s=1e-7,
                  reset_fractions=(fraction,), name="t")


def test_drift_correction_divides_the_durations():
    log = generate_run(seed=5, profile="loaded", amplitude_deg=6.0, steps=50)
    corrected = drift_corrected_log(log, 10000.0)
    assert np.asarray(corrected.durations_s) == pytest.approx(
        np.asarray(log.durations_s) / 1.01, rel=1e-12, abs=0.0
    )


def test_drift_correction_leaves_everything_else_alone():
    log = generate_run(seed=5, profile="loaded", amplitude_deg=6.0, steps=50)
    corrected = drift_corrected_log(log, 500.0)
    assert np.array_equal(corrected.states, log.states)
    assert np.array_equal(corrected.computed_rad, log.computed_rad)
    assert corrected.label.endswith("drift-corrected")


def test_drift_correction_rejects_an_impossible_rate():
    log = generate_run(seed=5, profile="loaded", amplitude_deg=6.0, steps=50)
    with pytest.raises(ValueError, match="non-positive clock rate"):
        drift_corrected_log(log, -2.0e6)


def test_drift_correction_recovers_the_true_miss_set():
    spec = ClockSpec(
        drift_ppm=20000.0, jitter_s=2e-7, tick_s=1e-7, reset_fractions=(), name="t"
    )
    drifted = generate_run(seed=31, profile="loaded", amplitude_deg=6.0, steps=200,
                           clock=spec)
    truth = generate_run(seed=31, profile="loaded", amplitude_deg=6.0, steps=200)
    estimate = estimate_clock(drifted.host_time_s, drifted.device_time_s, 100.0)
    corrected = drift_corrected_log(drifted, estimate.drift_ppm)
    assert {event.index for event in miss_events(corrected)} == {
        event.index for event in miss_events(truth)
    }


def test_uncorrected_drift_changes_the_apparent_miss_set():
    spec = ClockSpec(
        drift_ppm=20000.0, jitter_s=2e-7, tick_s=1e-7, reset_fractions=(), name="t"
    )
    drifted = generate_run(seed=31, profile="loaded", amplitude_deg=6.0, steps=200,
                           clock=spec)
    truth = generate_run(seed=31, profile="loaded", amplitude_deg=6.0, steps=200)
    assert {event.index for event in miss_events(drifted)} != {
        event.index for event in miss_events(truth)
    }
