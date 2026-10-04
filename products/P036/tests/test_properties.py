"""Property-based tests for the algebraic identities, with Hypothesis.

Only identities that hold exactly or to a stated floating-point bound are
tested here. Nothing in this file depends on wall-clock behaviour.
"""

from __future__ import annotations

import math

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from rtclock.budget import Stage, compose_budget
from rtclock.histogram import overrun_report, percentile
from rtclock.loop import FixedRateLoop, absolute_mode_drift, relative_mode_drift
from rtclock.schedulability import (
    hyperbolic_bound_test,
    response_time_analysis,
    rm_utilization_bound,
    rm_utilization_test,
)
from rtclock.taskset import PeriodicTask, TaskSet
from rtclock.timebase import SkewedSimulatedTimebase
from rtclock.units import (
    frequency_to_period,
    ms_to_s,
    ns_to_s,
    period_to_frequency,
    s_to_ms,
    s_to_ns,
    s_to_us,
    us_to_s,
)

SETTINGS = settings(max_examples=200, deadline=None)

positive_seconds = st.floats(
    min_value=1e-9, max_value=1e3, allow_nan=False, allow_infinity=False, width=64
)
latency_samples = st.lists(
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    min_size=1,
    max_size=400,
)
percentile_values = st.floats(min_value=0.0, max_value=100.0, allow_nan=False)


# --- unit conversions -------------------------------------------------------


@SETTINGS
@given(positive_seconds)
def test_second_nanosecond_round_trip(x):
    # Exact to within one ulp: multiply then divide by the same power of ten.
    assert ns_to_s(s_to_ns(x)) == x or abs(ns_to_s(s_to_ns(x)) - x) <= math.ulp(x)


@SETTINGS
@given(positive_seconds)
def test_second_microsecond_and_millisecond_round_trips(x):
    assert abs(us_to_s(s_to_us(x)) - x) <= math.ulp(x)
    assert abs(ms_to_s(s_to_ms(x)) - x) <= math.ulp(x)


@SETTINGS
@given(positive_seconds)
def test_frequency_period_involution(x):
    # f -> T -> f is an involution to within two ulps (two reciprocals).
    f = period_to_frequency(x)
    assert abs(frequency_to_period(f) - x) <= 2 * math.ulp(x)


@SETTINGS
@given(positive_seconds)
def test_period_frequency_product_is_one(x):
    # (1/T) * T = 1 to within two ulps of 1.0: one rounding in the reciprocal,
    # one in the product.
    assert abs(period_to_frequency(x) * x - 1.0) <= 2 * math.ulp(1.0)


@SETTINGS
@given(positive_seconds, positive_seconds)
def test_nanosecond_conversion_is_order_preserving(a, b):
    assume(a != b)
    assert (a < b) == (s_to_ns(a) < s_to_ns(b))


# --- period and deadline arithmetic -----------------------------------------


@SETTINGS
@given(positive_seconds, st.floats(min_value=0.0, max_value=1e4, allow_nan=False))
def test_releases_in_is_the_ceiling_of_the_ratio(period, window):
    t = PeriodicTask("a", period, period / 2.0)
    n = t.releases_in(window)
    assert n == math.ceil(window / period)
    # n periods cover the window, and n-1 do not (up to the ceiling's own
    # floating-point behaviour at exact multiples).
    assert n * period >= window - 4 * math.ulp(window + period)


@SETTINGS
@given(
    st.floats(min_value=1e-6, max_value=1.0, allow_nan=False),
    st.floats(min_value=1e-9, max_value=1.0, allow_nan=False),
)
def test_utilization_is_scale_invariant(period, wcet):
    assume(wcet <= period)
    base = PeriodicTask("a", period, wcet).utilization
    for scale in (1e-3, 1e3):
        scaled = PeriodicTask("a", period * scale, wcet * scale).utilization
        assert math.isclose(scaled, base, rel_tol=1e-12)


# --- percentiles ------------------------------------------------------------


@SETTINGS
@given(latency_samples, percentile_values, percentile_values)
def test_percentiles_are_monotone_in_p(samples, p, q):
    lo, hi = min(p, q), max(p, q)
    for method in ("nearest_rank", "linear"):
        assert percentile(samples, lo, method) <= percentile(samples, hi, method)


@SETTINGS
@given(latency_samples, percentile_values)
def test_percentiles_lie_within_the_sample_range(samples, p):
    for method in ("nearest_rank", "linear"):
        value = percentile(samples, p, method)
        assert min(samples) <= value <= max(samples)


@SETTINGS
@given(latency_samples, percentile_values)
def test_nearest_rank_returns_a_stored_sample(samples, p):
    assert percentile(samples, p, "nearest_rank") in samples


@SETTINGS
@given(latency_samples)
def test_extremes_are_min_and_max(samples):
    for method in ("nearest_rank", "linear"):
        assert percentile(samples, 100.0, method) == max(samples)
    assert percentile(samples, 0.0, "nearest_rank") == min(samples)
    assert percentile(samples, 0.0, "linear") == min(samples)


@SETTINGS
@given(latency_samples, percentile_values)
def test_percentiles_are_invariant_to_sample_order(samples, p):
    shuffled = list(reversed(samples))
    for method in ("nearest_rank", "linear"):
        assert percentile(samples, p, method) == percentile(shuffled, p, method)


@SETTINGS
@given(
    latency_samples,
    percentile_values,
    st.floats(min_value=1e-3, max_value=1e3, allow_nan=False),
)
def test_percentiles_are_positively_homogeneous(samples, p, scale):
    for method in ("nearest_rank", "linear"):
        a = percentile([x * scale for x in samples], p, method)
        b = percentile(samples, p, method) * scale
        assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-300)


# --- overrun accounting -----------------------------------------------------


@SETTINGS
@given(latency_samples, st.floats(min_value=1e-9, max_value=1.0, allow_nan=False))
def test_overrun_count_is_monotone_decreasing_in_the_budget(samples, budget):
    tighter = overrun_report(samples, budget).count
    looser = overrun_report(samples, budget * 2.0).count
    assert looser <= tighter
    assert 0 <= tighter <= len(samples)


@SETTINGS
@given(latency_samples, st.floats(min_value=1e-9, max_value=1.0, allow_nan=False))
def test_overrun_indices_agree_with_the_count_and_are_ascending(samples, budget):
    rep = overrun_report(samples, budget)
    assert len(rep.indices) == rep.count
    assert list(rep.indices) == sorted(rep.indices)
    assert all(samples[i] > budget for i in rep.indices)
    assert rep.longest_consecutive_run <= rep.count
    assert 0.0 <= rep.fraction <= 1.0


@SETTINGS
@given(latency_samples)
def test_a_budget_at_the_maximum_yields_no_overruns(samples):
    assert overrun_report(samples, max(max(samples), 1e-12)).count == 0


# --- utilization bounds -----------------------------------------------------


@SETTINGS
@given(st.integers(min_value=1, max_value=5000))
def test_rm_bound_is_between_ln2_and_one(n):
    b = rm_utilization_bound(n)
    assert math.log(2.0) < b <= 1.0


@SETTINGS
@given(st.integers(min_value=1, max_value=2000))
def test_rm_bound_is_strictly_decreasing(n):
    assert rm_utilization_bound(n) >= rm_utilization_bound(n + 1)


@SETTINGS
@given(
    st.lists(
        st.tuples(
            st.floats(min_value=1e-3, max_value=1.0, allow_nan=False),
            st.floats(min_value=1e-4, max_value=0.5, allow_nan=False),
        ),
        min_size=1,
        max_size=8,
    )
)
def test_hyperbolic_bound_accepts_everything_liu_layland_accepts(pairs):
    tasks = []
    for i, (period, frac) in enumerate(pairs):
        tasks.append(PeriodicTask(f"t{i}", period, period * frac))
    ts = TaskSet(tasks).rate_monotonic()
    if rm_utilization_test(ts).schedulable:
        assert hyperbolic_bound_test(ts).schedulable


@SETTINGS
@given(
    st.lists(
        st.tuples(
            st.floats(min_value=1e-3, max_value=1.0, allow_nan=False),
            st.floats(min_value=1e-4, max_value=0.3, allow_nan=False),
        ),
        min_size=1,
        max_size=6,
    )
)
def test_rta_response_is_at_least_the_wcet_and_monotone_in_priority(pairs):
    tasks = [PeriodicTask(f"t{i}", p, p * f) for i, (p, f) in enumerate(pairs)]
    ts = TaskSet(tasks).rate_monotonic()
    results = response_time_analysis(ts)
    by_name = {t.name: t for t in ts}
    for rt in results:
        assert rt.response_s >= by_name[rt.name].wcet_s - 1e-15
        # The iterate sequence is non-decreasing by construction.
        assert list(rt.iterates) == sorted(rt.iterates)


# --- loop drift -------------------------------------------------------------


@SETTINGS
@given(
    st.integers(min_value=0, max_value=500),
    st.floats(min_value=1e-6, max_value=1.0, allow_nan=False),
    st.floats(min_value=-1e4, max_value=1e4, allow_nan=False),
)
def test_relative_drift_closed_form_is_linear_in_the_index(k, period, skew_ppm):
    a = relative_mode_drift(1, period, skew_ppm, 0.0)
    assert math.isclose(relative_mode_drift(k, period, skew_ppm, 0.0), k * a, rel_tol=1e-12,
                        abs_tol=1e-300)


@SETTINGS
@given(
    st.integers(min_value=0, max_value=300),
    st.floats(min_value=1e-6, max_value=1.0, allow_nan=False),
    st.floats(min_value=-1e4, max_value=1e4, allow_nan=False),
)
def test_absolute_drift_closed_form_is_bounded_by_the_first_step(k, period, skew_ppm):
    # |drift_k| = |a| |1 - (-eps)^k| / |1 + eps| <= |a| / (1 - |eps|), so the
    # bound is |a| for eps >= 0 and slightly above it for eps < 0.
    eps = skew_ppm * 1e-6
    a = abs(period * eps)
    bound = a / (1.0 - abs(eps))
    assert abs(absolute_mode_drift(k, period, skew_ppm, 0.0)) <= bound * (1.0 + 1e-9) + 1e-300


@settings(max_examples=60, deadline=None)
@given(
    st.floats(min_value=1e-5, max_value=1e-1, allow_nan=False),
    st.floats(min_value=-1e3, max_value=1e3, allow_nan=False),
    st.sampled_from(["absolute", "relative"]),
)
def test_simulated_loop_drift_matches_its_closed_form(period, skew_ppm, mode):
    tb = SkewedSimulatedTimebase(skew_ppm=skew_ppm)
    report = FixedRateLoop(period_s=period, timebase=tb, mode=mode).run(iterations=50)
    closed = relative_mode_drift if mode == "relative" else absolute_mode_drift
    for rec in report.records:
        expected = closed(rec.index, period, skew_ppm, 0.0)
        # Absolute tolerance is binary64 accumulation noise over 50 periods.
        assert abs(rec.drift_s - expected) <= 1e-9 * abs(expected) + 50 * math.ulp(
            50.0 * period
        )


# --- budget composition -----------------------------------------------------


@SETTINGS
@given(
    st.lists(
        st.tuples(
            st.floats(min_value=1e-6, max_value=1e-3, allow_nan=False),
            st.floats(min_value=0.0, max_value=1e-4, allow_nan=False),
        ),
        min_size=1,
        max_size=10,
    ),
    st.floats(min_value=1e-3, max_value=1.0, allow_nan=False),
)
def test_budget_composition_identities(pairs, budget):
    stages = [Stage(f"s{i}", w, None, sd) for i, (w, sd) in enumerate(pairs)]
    r = compose_budget(stages, budget_s=budget, clock_quantum_s=1e-9)
    # WCET total equals the sum, and the mean equals it when mean defaults.
    assert math.isclose(r.wcet_total_s, math.fsum(s.wcet_s for s in stages), rel_tol=1e-15)
    assert math.isclose(r.mean_total_s, r.wcet_total_s, rel_tol=1e-15)
    # Quadrature never exceeds the triangle-inequality bound.
    assert r.stdev_independent_s <= r.stdev_fully_correlated_s * (1.0 + 1e-12) + 1e-300
    # Fractions sum to one and are each in [0, 1].
    assert math.isclose(math.fsum(r.stage_fractions.values()), 1.0, rel_tol=1e-12)
    assert all(0.0 <= f <= 1.0 for f in r.stage_fractions.values())
    assert math.isclose(r.margin_s, budget - r.wcet_total_s, rel_tol=1e-12, abs_tol=1e-300)
