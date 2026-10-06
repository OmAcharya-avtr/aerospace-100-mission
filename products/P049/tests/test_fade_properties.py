"""Property-based tests of the algebraic identities the fade core must obey."""

from __future__ import annotations

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from linkoutage.fade import (
    FadeDefinitions,
    availability,
    below_threshold,
    down_crossing_indices,
    fade_runs,
    fade_statistics,
    outage_fraction,
    up_crossing_indices,
)

AMP = st.lists(
    st.floats(min_value=0.01, max_value=2.0, allow_nan=False, allow_infinity=False),
    min_size=2,
    max_size=200,
)

SETTINGS = settings(
    max_examples=150,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


@SETTINGS
@given(AMP)
def test_outage_plus_availability_is_one(values):
    a = np.asarray(values)
    assert abs(outage_fraction(a, 0.6) + availability(a, 0.6) - 1.0) < 1e-12


@SETTINGS
@given(AMP)
def test_crossing_counts_differ_by_at_most_one(values):
    a = np.asarray(values)
    down = down_crossing_indices(a, 0.6).size
    up = up_crossing_indices(a, 0.6).size
    assert abs(down - up) <= 1


@SETTINGS
@given(AMP)
def test_run_lengths_sum_to_in_fade_samples(values):
    a = np.asarray(values)
    runs = fade_runs(a, 0.6, 1.0)
    mask = below_threshold(a, 0.6)
    assert int(runs.length_samples.sum()) == int(np.count_nonzero(mask))


@SETTINGS
@given(AMP)
def test_runs_are_disjoint_and_ordered(values):
    a = np.asarray(values)
    runs = fade_runs(a, 0.6, 1.0)
    starts = runs.start
    stops = runs.stop
    assert np.all(stops > starts)
    if starts.size > 1:
        assert np.all(starts[1:] > stops[:-1])


@SETTINGS
@given(AMP)
def test_down_crossings_equal_non_left_censored_runs(values):
    a = np.asarray(values)
    runs = fade_runs(a, 0.6, 1.0)
    expected = int(np.count_nonzero(~runs.left_censored))
    assert down_crossing_indices(a, 0.6).size == expected


@SETTINGS
@given(AMP)
def test_rice_identity_when_nothing_is_censored(values):
    a = np.asarray(values)
    s = fade_statistics(a, 0.6, 1.0)
    if s.n_complete_fades == 0 or s.n_left_censored or s.n_right_censored:
        return
    # Every below-threshold sample lies in a complete run, so
    # mean_fade_duration / (outage_fraction / LCR) = N / (N - 1) exactly.
    assert abs(s.rice_relative_residual - 1.0 / (s.n_samples - 1)) < 1e-9


@SETTINGS
@given(AMP)
def test_interval_count_never_exceeds_sample_count(values):
    a = np.asarray(values)
    s_sample = fade_statistics(a, 0.6, 1.0)
    s_interval = fade_statistics(
        a, 0.6, 1.0, definitions=FadeDefinitions(duration_convention="interval_count")
    )
    if s_sample.n_complete_fades == 0:
        return
    assert s_interval.mean_fade_duration_s <= s_sample.mean_fade_duration_s + 1e-12


@SETTINGS
@given(AMP)
def test_interpolated_duration_within_one_sample_of_sample_count(values):
    a = np.asarray(values)
    defs = FadeDefinitions(duration_convention="interpolated")
    runs = fade_runs(a, 0.6, 1.0, definitions=defs)
    keep = ~runs.censored
    if not np.any(keep):
        return
    interp = runs.durations_s()[keep]
    counted = runs.length_samples[keep].astype(float)
    # The interpolated crossings lie inside the two sample intervals that
    # bracket the run, so the duration cannot differ from L by more than 1.
    assert np.all(np.abs(interp - counted) <= 1.0 + 1e-9)


@SETTINGS
@given(AMP, st.floats(min_value=0.02, max_value=1.9))
def test_outage_fraction_is_monotone_in_threshold(values, t):
    a = np.asarray(values)
    assert outage_fraction(a, t) <= outage_fraction(a, t + 0.05) + 1e-12


@SETTINGS
@given(AMP)
def test_strict_and_non_strict_bracket_each_other(values):
    a = np.asarray(values)
    strict = outage_fraction(a, 0.6)
    loose = outage_fraction(a, 0.6, definitions=FadeDefinitions(strict_below=False))
    assert strict <= loose + 1e-12


@SETTINGS
@given(AMP, st.floats(min_value=1.0, max_value=1e6))
def test_rate_scales_linearly_with_sample_rate(values, fs):
    from linkoutage.fade import level_crossing_rate

    a = np.asarray(values)
    r1 = level_crossing_rate(a, 0.6, 1.0)
    r2 = level_crossing_rate(a, 0.6, fs)
    assert abs(r2 - r1 * fs) <= 1e-6 * max(1.0, abs(r2))


@SETTINGS
@given(AMP)
def test_dropping_single_sample_fades_cannot_increase_crossings(values):
    a = np.asarray(values)
    full = down_crossing_indices(a, 0.6).size
    dropped = down_crossing_indices(
        a, 0.6, definitions=FadeDefinitions(count_single_sample_fades=False)
    ).size
    assert dropped <= full
