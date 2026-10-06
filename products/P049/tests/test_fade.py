"""Known-answer, definitional and edge-case tests for the fade core."""

from __future__ import annotations

import math

import numpy as np
import pytest

from linkoutage.fade import (
    DEFAULT_DEFINITIONS,
    FadeDefinitions,
    availability,
    below_threshold,
    down_crossing_indices,
    fade_durations,
    fade_runs,
    fade_statistics,
    level_crossing_rate,
    mean_fade_duration,
    outage_fraction,
    record_duration_s,
    up_crossing_indices,
)


def test_hand_down_crossings(hand_series):
    assert down_crossing_indices(hand_series, 0.6).tolist() == [1, 5]


def test_hand_up_crossings(hand_series):
    assert up_crossing_indices(hand_series, 0.6).tolist() == [3, 6]


def test_hand_mask(hand_series):
    assert below_threshold(hand_series, 0.6).tolist() == [
        False,
        True,
        True,
        False,
        False,
        True,
        False,
    ]


def test_hand_mean_fade_duration(hand_series):
    # (2 samples + 1 sample) / 2 fades = 1.5 samples, fs = 1 Hz -> 1.5 s
    assert mean_fade_duration(hand_series, 0.6, 1.0) == pytest.approx(1.5)


def test_hand_level_crossing_rate(hand_series):
    # 2 down-crossings over (7 - 1) / 1.0 = 6 s
    assert level_crossing_rate(hand_series, 0.6, 1.0) == pytest.approx(2.0 / 6.0)


def test_hand_outage_and_availability(hand_series):
    assert outage_fraction(hand_series, 0.6) == pytest.approx(3.0 / 7.0)
    assert availability(hand_series, 0.6) == pytest.approx(4.0 / 7.0)


def test_hand_statistics_block(hand_series):
    s = fade_statistics(hand_series, 0.6, 1.0)
    assert (s.n_down_crossings, s.n_up_crossings, s.n_runs) == (2, 2, 2)
    assert (s.n_complete_fades, s.n_left_censored, s.n_right_censored) == (2, 0, 0)
    assert s.n_single_sample_fades == 1
    assert s.in_fade_samples == 3
    assert s.max_complete_fade_duration_s == pytest.approx(2.0)
    assert s.median_fade_duration_s == pytest.approx(1.5)


def test_hand_report_contains_definitions(hand_series):
    text = fade_statistics(hand_series, 0.6, 1.0).report()
    assert "a[n] < T (strict)" in text
    assert "duration = L / fs" in text
    assert "record duration = (N - 1) / fs" in text


def test_rice_residual_is_exactly_one_over_n_minus_one(hand_series):
    # With no censoring and every below-threshold sample inside a complete run,
    # mean_fade_duration / (outage_fraction / LCR) = N / (N - 1) exactly.
    s = fade_statistics(hand_series, 0.6, 1.0)
    assert s.rice_relative_residual == pytest.approx(1.0 / (s.n_samples - 1), rel=1e-12)


def test_censored_series_counts(censored_series):
    s = fade_statistics(censored_series, 0.6, 2.0)
    assert s.n_down_crossings == 1
    assert s.n_up_crossings == 1
    assert s.n_runs == 2
    assert s.n_left_censored == 1
    assert s.n_right_censored == 1
    assert s.n_complete_fades == 0
    assert math.isnan(s.mean_fade_duration_s)
    assert s.level_crossing_rate_hz == pytest.approx(0.5)
    assert s.outage_fraction == pytest.approx(3.0 / 5.0)


def test_censored_included_as_complete(censored_series):
    defs = FadeDefinitions(censoring="include_as_complete")
    complete, censored = fade_durations(censored_series, 0.6, 2.0, definitions=defs)
    # runs of 2 and 1 samples at fs = 2 Hz -> 1.0 s and 0.5 s
    assert sorted(complete.tolist()) == pytest.approx([0.5, 1.0])
    assert censored.size == 0


def test_censored_lower_bounds_reported(censored_series):
    complete, censored = fade_durations(censored_series, 0.6, 2.0)
    assert complete.size == 0
    assert sorted(censored.tolist()) == pytest.approx([0.5, 1.0])


def test_interval_count_convention():
    a = np.array([1.0, 0.5, 0.5, 0.5, 1.0])
    defs = FadeDefinitions(duration_convention="interval_count")
    # run of 3 samples -> (3 - 1) / 1000 = 0.002 s
    assert mean_fade_duration(a, 0.6, 1000.0, definitions=defs) == pytest.approx(0.002)
    assert mean_fade_duration(a, 0.6, 1000.0) == pytest.approx(0.003)


def test_interpolated_convention_symmetric_case():
    # a[0] = 1.0, a[1] = 0.2, a[2] = 1.0, T = 0.6, fs = 1 Hz.
    # down-crossing fraction (1.0 - 0.6) / (1.0 - 0.2) = 0.5 -> t = 0.5
    # up-crossing   fraction (0.6 - 0.2) / (1.0 - 0.2) = 0.5 -> t = 1.5
    # duration = 1.0 s, versus 1.0 s by sample count: they coincide here.
    a = np.array([1.0, 0.2, 1.0])
    defs = FadeDefinitions(duration_convention="interpolated")
    assert mean_fade_duration(a, 0.6, 1.0, definitions=defs) == pytest.approx(1.0)


def test_interpolated_convention_asymmetric_case():
    # a = [1.0, 0.4, 1.0]; down fraction (1.0-0.6)/(1.0-0.4) = 2/3 -> t = 2/3
    # up fraction (0.6-0.4)/(1.0-0.4) = 1/3 -> t = 1 + 1/3 = 4/3
    # duration = 4/3 - 2/3 = 2/3 s
    a = np.array([1.0, 0.4, 1.0])
    defs = FadeDefinitions(duration_convention="interpolated")
    assert mean_fade_duration(a, 0.6, 1.0, definitions=defs) == pytest.approx(2.0 / 3.0)


def test_strict_versus_non_strict_at_the_threshold():
    a = np.array([1.0, 0.6, 1.0])
    assert down_crossing_indices(a, 0.6).size == 0
    defs = FadeDefinitions(strict_below=False)
    assert down_crossing_indices(a, 0.6, definitions=defs).tolist() == [1]


def test_dropping_single_sample_fades():
    a = np.array([1.0, 0.4, 1.0, 0.4, 0.4, 1.0])
    defs = FadeDefinitions(count_single_sample_fades=False)
    s = fade_statistics(a, 0.6, 1.0, definitions=defs)
    assert s.n_down_crossings == 1
    assert s.n_complete_fades == 1
    assert s.mean_fade_duration_s == pytest.approx(2.0)
    # The outage fraction still counts every below-threshold sample.
    assert s.outage_fraction == pytest.approx(3.0 / 6.0)


def test_record_duration_conventions():
    assert record_duration_s(7, 2.0, FadeDefinitions()) == pytest.approx(3.0)
    assert record_duration_s(
        7, 2.0, FadeDefinitions(record_duration_convention="samples")
    ) == pytest.approx(3.5)


def test_record_duration_changes_rate(hand_series):
    defs = FadeDefinitions(record_duration_convention="samples")
    r_intervals = level_crossing_rate(hand_series, 0.6, 1.0)
    r_samples = level_crossing_rate(hand_series, 0.6, 1.0, definitions=defs)
    assert r_intervals / r_samples == pytest.approx(7.0 / 6.0)


def test_all_below_threshold_is_one_censored_fade():
    a = np.full(10, 0.1)
    s = fade_statistics(a, 0.6, 1.0)
    assert s.n_runs == 1
    assert s.n_left_censored == 1 and s.n_right_censored == 1
    assert s.n_down_crossings == 0
    assert math.isnan(s.mean_fade_duration_s)
    assert s.level_crossing_rate_hz == 0.0
    assert s.outage_fraction == 1.0
    assert math.isnan(s.rice_relative_residual)


def test_never_below_threshold():
    a = np.full(10, 1.0)
    s = fade_statistics(a, 0.6, 1.0)
    assert s.n_runs == 0
    assert s.outage_fraction == 0.0
    assert s.availability == 1.0
    assert math.isnan(s.mean_fade_duration_s)


def test_runs_object_lengths(hand_series):
    runs = fade_runs(hand_series, 0.6, 1.0)
    assert runs.length_samples.tolist() == [2, 1]
    assert runs.n_runs == 2
    assert runs.censored.tolist() == [False, False]


def test_definitions_with_helper():
    d = DEFAULT_DEFINITIONS.with_(strict_below=False)
    assert d.strict_below is False
    assert DEFAULT_DEFINITIONS.strict_below is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"duration_convention": "nonsense"},
        {"censoring": "nonsense"},
        {"record_duration_convention": "nonsense"},
        {"duration_convention": "interpolated", "censoring": "include_as_complete"},
    ],
)
def test_bad_definitions_raise(kwargs):
    with pytest.raises(ValueError):
        FadeDefinitions(**kwargs)


def test_two_dimensional_amplitude_raises():
    with pytest.raises(ValueError, match="one-dimensional"):
        fade_statistics(np.zeros((3, 3)), 0.5, 1.0)


def test_too_short_amplitude_raises():
    with pytest.raises(ValueError, match="at least 2 samples"):
        fade_statistics(np.array([1.0]), 0.5, 1.0)


def test_non_finite_amplitude_raises():
    with pytest.raises(ValueError, match="non-finite"):
        fade_statistics(np.array([1.0, np.nan, 1.0]), 0.5, 1.0)


def test_non_finite_threshold_raises(hand_series):
    with pytest.raises(ValueError, match="finite"):
        fade_statistics(hand_series, np.nan, 1.0)


@pytest.mark.parametrize("fs", [0.0, -1.0, float("inf"), float("nan")])
def test_bad_sample_rate_raises(hand_series, fs):
    with pytest.raises(ValueError, match="sample rate"):
        fade_statistics(hand_series, 0.6, fs)


def test_record_duration_rejects_short_record():
    with pytest.raises(ValueError, match="at least 2"):
        record_duration_s(1, 1.0, DEFAULT_DEFINITIONS)
