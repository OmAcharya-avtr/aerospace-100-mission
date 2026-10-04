"""Percentile and overrun tests. Both percentile definitions are pinned exactly.

The reference for ``"linear"`` is NumPy's ``method="linear"`` (Hyndman & Fan
type 7); the reference for ``"nearest_rank"`` is an explicit order-statistic
index computed in the test, not in the library.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from rtclock.histogram import LatencyHistogram, overrun_report, percentile

# Ten samples, deliberately unsorted, with no ties, so every order statistic
# is distinguishable.
TEN = [5.0, 1.0, 9.0, 3.0, 7.0, 2.0, 8.0, 4.0, 10.0, 6.0]
TEN_SORTED = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]


# Hand computation, nearest-rank with N = 10, k = max(1, ceil(p/100 * 10)):
#   p = 0   -> ceil(0)    = 0  -> clamped to 1 -> x[1]  = 1
#   p = 1   -> ceil(0.1)  = 1  -> x[1]  = 1
#   p = 10  -> ceil(1)    = 1  -> x[1]  = 1
#   p = 11  -> ceil(1.1)  = 2  -> x[2]  = 2
#   p = 50  -> ceil(5)    = 5  -> x[5]  = 5
#   p = 55  -> ceil(5.5)  = 6  -> x[6]  = 6
#   p = 90  -> ceil(9)    = 9  -> x[9]  = 9
#   p = 95  -> ceil(9.5)  = 10 -> x[10] = 10
#   p = 99  -> ceil(9.9)  = 10 -> x[10] = 10
#   p = 100 -> ceil(10)   = 10 -> x[10] = 10
NEAREST_RANK_HAND = {
    0.0: 1.0,
    1.0: 1.0,
    10.0: 1.0,
    11.0: 2.0,
    50.0: 5.0,
    55.0: 6.0,
    90.0: 9.0,
    95.0: 10.0,
    99.0: 10.0,
    100.0: 10.0,
}


@pytest.mark.parametrize(("p", "expected"), sorted(NEAREST_RANK_HAND.items()))
def test_nearest_rank_hand_values(p, expected):
    assert percentile(TEN, p, "nearest_rank") == expected


# Hand computation, linear (type 7) with N = 10, h = 9 * p/100:
#   p = 0   -> h = 0.0  -> x[1] = 1
#   p = 25  -> h = 2.25 -> x[3] + 0.25*(x[4]-x[3]) = 3 + 0.25 = 3.25
#   p = 50  -> h = 4.5  -> x[5] + 0.5*(x[6]-x[5])  = 5 + 0.5  = 5.5
#   p = 90  -> h = 8.1  -> x[9] + 0.1*(x[10]-x[9]) = 9 + 0.1  = 9.1
#   p = 99  -> h = 8.91 -> x[9] + 0.91*(x[10]-x[9]) = 9.91
#   p = 100 -> h = 9.0  -> x[10] = 10
LINEAR_HAND = {0.0: 1.0, 25.0: 3.25, 50.0: 5.5, 90.0: 9.1, 99.0: 9.91, 100.0: 10.0}


@pytest.mark.parametrize(("p", "expected"), sorted(LINEAR_HAND.items()))
def test_linear_hand_values(p, expected):
    assert percentile(TEN, p, "linear") == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("p", [0.0, 0.5, 1.0, 7.3, 25.0, 50.0, 63.1, 90.0, 99.0, 99.9, 100.0])
def test_linear_matches_numpy_type_7_exactly(p):
    rng = np.random.default_rng(20261004)
    for n in (1, 2, 3, 7, 101, 1000):
        xs = rng.random(n) * 1e-3
        assert percentile(xs.tolist(), p, "linear") == pytest.approx(
            float(np.percentile(xs, p, method="linear")), rel=0.0, abs=1e-15
        )


@pytest.mark.parametrize("p", [0.0, 1.0, 13.7, 50.0, 90.0, 99.0, 100.0])
def test_nearest_rank_is_always_a_stored_sample(p):
    rng = np.random.default_rng(7)
    xs = (rng.random(257) * 1e-3).tolist()
    assert percentile(xs, p, "nearest_rank") in xs


def test_nearest_rank_matches_an_explicit_order_statistic_over_many_sizes():
    rng = np.random.default_rng(99)
    for n in (1, 2, 5, 13, 64, 500):
        xs = (rng.random(n) * 1e-3).tolist()
        srt = sorted(xs)
        for p in (0.0, 0.1, 25.0, 33.333, 50.0, 66.667, 90.0, 99.0, 99.99, 100.0):
            k = max(1, math.ceil(p / 100.0 * n))
            assert percentile(xs, p, "nearest_rank") == srt[k - 1]


def test_both_definitions_agree_at_the_extremes():
    for xs in ([1.0], TEN, [0.0, 0.0, 1.0]):
        assert percentile(xs, 100.0, "nearest_rank") == percentile(xs, 100.0, "linear")
        assert percentile(xs, 0.0, "nearest_rank") == percentile(xs, 0.0, "linear")
        assert percentile(xs, 100.0, "linear") == max(xs)
        assert percentile(xs, 0.0, "linear") == min(xs)


def test_definitions_differ_where_they_should():
    # p = 50 on 10 samples: order statistic x[5] = 5 versus interpolated 5.5.
    assert percentile(TEN, 50.0, "nearest_rank") == 5.0
    assert percentile(TEN, 50.0, "linear") == pytest.approx(5.5)


def test_percentile_validates_input():
    with pytest.raises(ValueError, match="at least one value"):
        percentile([], 50.0)
    with pytest.raises(ValueError, match=r"p must be in \[0, 100\]"):
        percentile(TEN, 100.1)
    with pytest.raises(ValueError, match=r"p must be in \[0, 100\]"):
        percentile(TEN, -1e-9)
    with pytest.raises(ValueError, match="nearest_rank"):
        percentile(TEN, 50.0, "p99")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="all be finite"):
        percentile([1.0, float("nan")], 50.0)
    with pytest.raises(ValueError, match="all be finite"):
        percentile([1.0, float("inf")], 50.0)


# Hand computation on the trace [1, 2, 3, 4, 5] ms with a 3 ms budget:
#   strict > 3 ms: indices 3 and 4 -> count 2, consecutive run 2,
#   worst overshoot 5 - 3 = 2 ms. The sample exactly equal to 3 ms is NOT an
#   overrun.
def test_overrun_report_hand_values():
    trace = [1e-3, 2e-3, 3e-3, 4e-3, 5e-3]
    rep = overrun_report(trace, 3e-3)
    assert rep.count == 2
    assert rep.indices == (3, 4)
    assert rep.longest_consecutive_run == 2
    assert rep.worst_overshoot_s == pytest.approx(2e-3, abs=1e-15)
    assert rep.total_samples == 5
    assert rep.fraction == pytest.approx(0.4)


# Hand computation: scattered overruns at indices 0, 2, 4 give count 3 but a
# longest consecutive run of 1 -- the same count as a 3-long cascade, which is
# a different failure mode.
def test_longest_consecutive_run_separates_cascades_from_scatter():
    scattered = overrun_report([2.0, 0.5, 2.0, 0.5, 2.0], 1.0)
    cascade = overrun_report([0.5, 2.0, 2.0, 2.0, 0.5], 1.0)
    assert scattered.count == cascade.count == 3
    assert scattered.longest_consecutive_run == 1
    assert cascade.longest_consecutive_run == 3


def test_overrun_report_on_an_empty_trace():
    rep = overrun_report([], 1.0)
    assert rep.count == 0
    assert rep.indices == ()
    assert rep.fraction == 0.0
    assert rep.worst_overshoot_s == 0.0
    assert rep.longest_consecutive_run == 0


def test_overrun_report_validates_input():
    with pytest.raises(ValueError, match="budget_s must be > 0"):
        overrun_report([1.0], 0.0)
    with pytest.raises(ValueError, match="must be finite"):
        overrun_report([float("nan")], 1.0)
    with pytest.raises(ValueError, match="must be >= 0 s"):
        overrun_report([-1e-9], 1.0)


def test_histogram_statistics_hand_values():
    h = LatencyHistogram(label="t")
    h.extend([1e-3, 2e-3, 3e-3, 4e-3])
    assert h.count == len(h) == 4
    assert h.mean() == pytest.approx(2.5e-3, abs=1e-18)
    assert h.minimum() == 1e-3
    assert h.maximum() == 4e-3
    # Sample stdev of 1,2,3,4 (ms) with n-1 denominator:
    #   mean 2.5; deviations -1.5,-0.5,0.5,1.5; squares 2.25,0.25,0.25,2.25
    #   sum 5.0; /3 = 1.6666667; sqrt = 1.2909944487 ms
    assert h.stdev() == pytest.approx(1.2909944487358e-3, rel=1e-12)


def test_histogram_summary_keys_name_the_method():
    h = LatencyHistogram()
    h.extend(TEN_SORTED)
    s = h.summary(percentiles=(50.0, 99.9), method="linear")
    assert "p50_linear_s" in s
    assert "p99_9_linear_s" in s
    assert s["count"] == 10.0
    assert s["max_s"] == 10.0
    s2 = h.summary(percentiles=(50.0,), method="nearest_rank")
    assert s2["p50_nearest_rank_s"] == 5.0


def test_histogram_bucket_counts_hand_values():
    h = LatencyHistogram()
    h.extend([0.5, 1.0, 1.5, 2.0, 2.5, 10.0, -0.0])
    # edges [1, 2, 3]: below 1 -> {0.5, -0.0} = 2; [1,2) -> {1.0, 1.5} = 2;
    # [2,3) -> {2.0, 2.5} = 2; >= 3 -> {10.0} = 1
    assert h.bucket_counts([1.0, 2.0, 3.0]) == [2, 2, 2, 1]
    assert sum(h.bucket_counts([1.0, 2.0, 3.0])) == h.count


def test_histogram_bucket_counts_validate_edges():
    h = LatencyHistogram()
    h.add(1.0)
    with pytest.raises(ValueError, match="at least 2 entries"):
        h.bucket_counts([1.0])
    with pytest.raises(ValueError, match="strictly increasing"):
        h.bucket_counts([2.0, 1.0])
    with pytest.raises(ValueError, match="strictly increasing"):
        h.bucket_counts([1.0, 1.0])


def test_histogram_rejects_bad_samples_and_empty_queries():
    h = LatencyHistogram(label="empty")
    with pytest.raises(ValueError, match="is empty"):
        h.mean()
    with pytest.raises(ValueError, match="is empty"):
        h.minimum()
    with pytest.raises(ValueError, match="is empty"):
        h.maximum()
    with pytest.raises(ValueError, match="is empty"):
        h.summary()
    with pytest.raises(ValueError, match="at least 2 samples"):
        LatencyHistogram(samples=[1.0]).stdev()
    with pytest.raises(ValueError, match="must be finite"):
        h.add(float("inf"))
    with pytest.raises(ValueError, match="must be >= 0 s"):
        h.add(-1.0)


def test_histogram_overruns_delegates_consistently():
    h = LatencyHistogram()
    h.extend([1e-3, 5e-3])
    assert h.overruns(2e-3).count == 1


# The nearest-rank sharp edge, pinned deliberately. The binary64 value of 99.9
# is 99.900000000000005684..., so 99.9/100 * 20000 = 19980.000000000004 and the
# ceiling is 19981, one rank above the "intended" 19980. No tolerance is
# applied to hide this; see the rtclock.histogram module docstring.
def test_nearest_rank_float_edge_is_pinned_not_smoothed():
    xs = [float(i) for i in range(20_000)]
    assert math.ceil(99.9 / 100.0 * 20_000) == 19_981
    assert percentile(xs, 99.9, "nearest_rank") == xs[19_980]  # the 19981st value
    # The error is bounded by exactly one order statistic.
    assert percentile(xs, 99.9, "nearest_rank") - xs[19_979] == 1.0
    # p values whose decimal form is representable are unaffected.
    assert percentile(xs, 99.0, "nearest_rank") == xs[19_799]
    assert percentile([float(i) for i in range(10)], 50.0, "nearest_rank") == 4.0
