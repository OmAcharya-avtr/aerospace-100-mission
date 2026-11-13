"""Deadline accounting, including the hand-calculated known-answer cases.

The arithmetic under test is the measurement model:
``releases = sum(1 + dropped_before)`` and
``misses = sum(missed + dropped_before)``.
"""

from __future__ import annotations

import numpy as np
import pytest

from deadlinerisk.accounting import account, autocorrelation, miss_indicator
from deadlinerisk.provenance import Origin
from deadlinerisk.trace import JobTrace


def _trace(missed: list[int], dropped: list[int], response: list[float] | None = None) -> JobTrace:
    n = len(missed)
    if n == 0:
        index = np.zeros(0, dtype=np.int64)
    else:
        index = np.cumsum([0] + [1 + value for value in dropped[:-1]], dtype=np.int64)
    return JobTrace(
        run="hand",
        period_s=0.010,
        deadline_s=0.010,
        job_index=index,
        release_s=index * 0.010,
        start_delay_s=np.zeros(n),
        end_s=index * 0.010 + 0.004,
        exec_s=np.full(n, 0.004),
        response_s=np.asarray(response if response is not None else [0.004] * n),
        missed=np.asarray(missed, dtype=np.int64),
        dropped_before=np.asarray(dropped, dtype=np.int64),
        origin=Origin.SYNTHETIC,
    )


def test_known_answer_no_drops_two_misses_in_five() -> None:
    # HAND CALCULATION. Five executed rows, none dropped, two missed:
    #   releases = (1+0)*5                     = 5
    #   misses   = (1+1+0+0+0) + 0             = 2
    #   miss_rate = 2/5                        = 0.4
    #   executed_miss_rate = 2/5               = 0.4
    totals = account(_trace([1, 1, 0, 0, 0], [0, 0, 0, 0, 0]))
    assert totals.executed == 5
    assert totals.releases == 5
    assert totals.dropped == 0
    assert totals.misses == 2
    assert totals.miss_rate == pytest.approx(0.4, rel=1e-12, abs=0.0)
    assert totals.executed_miss_rate == pytest.approx(0.4, rel=1e-12, abs=0.0)


def test_known_answer_with_drops() -> None:
    # HAND CALCULATION. Four executed rows with dropped_before = [0, 2, 0, 1]
    # and missed = [0, 1, 0, 1]:
    #   releases = (1+0) + (1+2) + (1+0) + (1+1) = 1 + 3 + 1 + 2 = 7
    #   misses   = (0 + 1 + 0 + 1) + (0 + 2 + 0 + 1) = 2 + 3     = 5
    #   miss_rate = 5/7 = 0.7142857142857143
    #   executed_miss_rate = 2/4 = 0.5
    # A tool that ignored the dropped releases would report 0.5 where the
    # deadline-correct figure is 5/7, which is the difference this product is
    # about.
    totals = account(_trace([0, 1, 0, 1], [0, 2, 0, 1]))
    assert totals.releases == 7
    assert totals.dropped == 3
    assert totals.misses == 5
    assert totals.miss_rate == pytest.approx(5.0 / 7.0, rel=1e-12, abs=0.0)
    assert totals.executed_miss_rate == pytest.approx(0.5, rel=1e-12, abs=0.0)


def test_miss_indicator_expands_in_release_order() -> None:
    # HAND CALCULATION. missed = [0, 1, 0], dropped_before = [0, 2, 0].
    # Row 0 contributes [0]; row 1 contributes [1] then two dropped ones
    # [1, 1]; row 2 contributes [0]. So the indicator is
    #   [0, 1, 1, 1, 0]
    # of length 5 = releases, summing to 3 = misses, where
    #   releases = (1+0) + (1+2) + (1+0) = 5
    #   misses   = (0 + 1 + 0) + (0 + 2 + 0) = 3
    indicator = miss_indicator(_trace([0, 1, 0], [0, 2, 0]))
    assert indicator.tolist() == [0, 1, 1, 1, 0]
    assert indicator.size == 5
    assert int(indicator.sum()) == 3


def test_indicator_length_and_sum_match_the_totals_on_the_fixture(ground_truth) -> None:
    totals = account(ground_truth)
    indicator = miss_indicator(ground_truth)
    assert indicator.size == totals.releases
    assert int(indicator.sum()) == totals.misses


def test_empty_trace_is_refused() -> None:
    with pytest.raises(ValueError, match="no executed job rows"):
        account(_trace([], []))


def test_autocorrelation_of_white_noise_is_one_at_lag_zero(rng) -> None:
    series = rng.normal(size=4096)
    rho = autocorrelation(series, 8)
    assert rho[0] == pytest.approx(1.0, rel=1e-12, abs=0.0)
    # Sampling error of rho(k) for white noise is about 1/sqrt(n) = 0.0156, so a
    # 5-sigma band is 0.078. This is a tolerance on a statistic, not an identity.
    assert np.all(np.abs(rho[1:]) < 5.0 / np.sqrt(series.size))


def test_autocorrelation_of_a_constant_series_is_all_zero() -> None:
    rho = autocorrelation(np.full(100, 3.5), 5)
    assert rho.tolist() == [0.0] * 6


def test_autocorrelation_of_an_ar1_series_decays_geometrically(rng) -> None:
    # An AR(1) with phi = 0.6 has rho(k) = 0.6**k in the population.
    phi = 0.6
    n = 200_000
    noise = rng.normal(size=n)
    series = np.empty(n)
    series[0] = noise[0]
    for step in range(1, n):
        series[step] = phi * series[step - 1] + noise[step]
    rho = autocorrelation(series, 3)
    for lag in (1, 2, 3):
        assert rho[lag] == pytest.approx(phi**lag, rel=0.08, abs=0.01)


@pytest.mark.parametrize("max_lag", [-1, 100])
def test_autocorrelation_rejects_an_out_of_range_lag(max_lag: int) -> None:
    with pytest.raises(ValueError):
        autocorrelation(np.arange(10.0), max_lag)
