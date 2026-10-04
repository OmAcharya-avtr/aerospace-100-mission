"""Property-based tests for the algebraic identities in the timing layer."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from hilforge.timebase import VirtualTimebase, duration_resolution_uncertainty
from hilforge.timing import (
    LatencyHistogram,
    PeriodSpec,
    overrun_report,
    quantile_standard_error,
    timing_uncertainty,
)

_DURATION = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
_POSITIVE = st.floats(min_value=1e-6, max_value=1.0, allow_nan=False, allow_infinity=False)


@given(st.lists(_DURATION, min_size=1, max_size=300), _POSITIVE)
@settings(max_examples=120, deadline=None)
def test_cascade_overruns_are_a_superset_of_direct_overruns(durations, period):
    """A direct overrun is always a cascade overrun.

    ``d[i] > D`` implies ``c[i] = max(r[i], c[i-1]) + d[i] >= r[i] + d[i] >
    r[i] + D = D[i]``, so the direct set is contained in the cascade set for
    any trace and any period.
    """
    account = overrun_report(durations, period)
    assert set(account.direct_indices) <= set(account.cascade_indices)


@given(st.lists(_DURATION, min_size=1, max_size=300), _POSITIVE)
@settings(max_examples=120, deadline=None)
def test_completion_times_are_non_decreasing(durations, period):
    account = overrun_report(durations, period)
    completions = np.asarray(account.completion_s)
    assert np.all(np.diff(completions) >= -1e-12)


@given(st.lists(_DURATION, min_size=1, max_size=200), _POSITIVE)
@settings(max_examples=120, deadline=None)
def test_lateness_equals_completion_minus_deadline(durations, period):
    account = overrun_report(durations, period)
    spec = PeriodSpec(period_s=period)
    for i, (completion, lateness) in enumerate(
        zip(account.completion_s, account.lateness_s, strict=True)
    ):
        assert lateness == completion - spec.absolute_deadline_s(i)


@given(st.lists(_DURATION, min_size=1, max_size=200), _POSITIVE)
@settings(max_examples=100, deadline=None)
def test_overrun_counts_never_exceed_the_trace_length(durations, period):
    account = overrun_report(durations, period)
    assert 0 <= account.direct_count <= len(durations)
    assert 0 <= account.cascade_count <= len(durations)
    assert account.max_consecutive_cascade <= account.cascade_count


@given(
    st.lists(_DURATION, min_size=2, max_size=400),
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
)
@settings(max_examples=150, deadline=None)
def test_percentiles_are_monotone_and_bracketed(samples, p):
    hist = LatencyHistogram()
    hist.extend(samples)
    for method in ("nearest_rank", "linear"):
        value = hist.percentile(p, method=method)
        assert hist.minimum - 1e-15 <= value <= hist.maximum + 1e-15
    lower = hist.percentile(min(p, 0.5), method="linear")
    upper = hist.percentile(max(p, 0.5), method="linear")
    assert lower <= upper + 1e-15


@given(st.lists(_DURATION, min_size=2, max_size=400))
@settings(max_examples=120, deadline=None)
def test_nearest_rank_percentile_is_always_an_observed_sample(samples):
    hist = LatencyHistogram()
    hist.extend(samples)
    observed = set(hist.samples().tolist())
    for p in (0.0, 0.1, 0.5, 0.9, 0.99, 1.0):
        assert hist.percentile(p, method="nearest_rank") in observed


@given(st.lists(_DURATION, min_size=2, max_size=300))
@settings(max_examples=120, deadline=None)
def test_histogram_mean_matches_numpy(samples):
    hist = LatencyHistogram()
    hist.extend(samples)
    assert hist.mean == np.mean(samples) or abs(hist.mean - np.mean(samples)) < 1e-9
    assert hist.count == len(samples)


@given(
    st.lists(_DURATION, min_size=2, max_size=200),
    st.floats(min_value=1e-12, max_value=1e-3, allow_nan=False),
)
@settings(max_examples=120, deadline=None)
def test_combined_uncertainty_is_at_least_each_component(samples, resolution):
    unc = timing_uncertainty(samples, resolution_s=resolution)
    assert unc.combined_s >= unc.u_statistical_s - 1e-18
    assert unc.combined_s >= unc.u_resolution_s - 1e-18
    assert unc.combined_s <= unc.u_statistical_s + unc.u_resolution_s + 1e-18
    assert unc.expanded_k2_s == 2.0 * unc.combined_s


@given(st.floats(min_value=1e-12, max_value=1.0, allow_nan=False))
@settings(max_examples=100, deadline=None)
def test_resolution_uncertainty_is_linear_in_the_resolution(delta):
    assert duration_resolution_uncertainty(2.0 * delta) == 2.0 * duration_resolution_uncertainty(
        delta
    )
    assert duration_resolution_uncertainty(delta) == delta / math.sqrt(6.0)


@given(
    st.floats(min_value=1e-6, max_value=1.0 - 1e-6, allow_nan=False),
    st.integers(min_value=1, max_value=10**6),
    _POSITIVE,
)
@settings(max_examples=120, deadline=None)
def test_quantile_standard_error_shrinks_as_one_over_sqrt_n(p, n, density):
    se_n = quantile_standard_error(p, n, density)
    se_4n = quantile_standard_error(p, 4 * n, density)
    assert se_4n == se_n / 2.0 or abs(se_4n - se_n / 2.0) < 1e-18


@given(
    st.lists(_POSITIVE, min_size=1, max_size=200),
    st.floats(min_value=1e-9, max_value=1e-6, allow_nan=False),
)
@settings(max_examples=100, deadline=None)
def test_virtual_timebase_total_equals_the_sum_of_advances(steps, resolution):
    assume(resolution > 0.0)
    tb = VirtualTimebase(resolution_s=resolution)
    for step in steps:
        tb.advance(step)
    expected_ticks = sum(round(s / resolution) for s in steps)
    assert tb.now() == expected_ticks * resolution


@given(_POSITIVE, st.integers(min_value=0, max_value=10000))
@settings(max_examples=150, deadline=None)
def test_absolute_deadline_minus_release_is_the_relative_deadline(period, index):
    spec = PeriodSpec(period_s=period)
    gap = spec.absolute_deadline_s(index) - spec.release_time_s(index)
    # Exact in the reals; float64 subtraction of two large multiples of the
    # period loses the last bits, so the identity holds to a relative 1e-12.
    assert gap == pytest.approx(spec.effective_deadline_s, rel=1e-12)


@given(
    st.lists(_DURATION, min_size=1, max_size=200),
    _POSITIVE,
    st.floats(min_value=0.01, max_value=1.0, allow_nan=False),
)
@settings(max_examples=120, deadline=None)
def test_a_tighter_deadline_never_reduces_the_overrun_count(durations, period, fraction):
    loose = overrun_report(durations, period)
    tight = overrun_report(durations, period, deadline_s=fraction * period)
    assert tight.direct_count >= loose.direct_count
    assert tight.cascade_count >= loose.cascade_count
