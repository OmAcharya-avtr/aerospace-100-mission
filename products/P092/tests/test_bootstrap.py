"""Moving-block bootstrap: block selection, resampling and intervals."""

from __future__ import annotations

import numpy as np
import pytest

from deadlinerisk.accounting import miss_indicator
from deadlinerisk.bootstrap import (
    block_length_from_autocorrelation,
    counted_rate_interval,
    moving_block_bootstrap,
    moving_block_index,
    percentile_interval,
    understatement_factor,
)


def test_block_length_of_an_independent_indicator_is_small(rng) -> None:
    # An i.i.d. Bernoulli indicator has rho(1) below the 2/sqrt(n) band with
    # high probability, so the cut-off lag is 1, tau is 1 and the block length
    # is 1. That is the rule reducing to the i.i.d. bootstrap when the data are
    # independent, which is the behaviour that makes it a selection and not a
    # guess.
    indicator = (rng.random(40_000) < 0.05).astype(np.int8)
    choice = block_length_from_autocorrelation(indicator)
    assert choice.lag_cut == 1
    assert choice.block_length == 1
    assert choice.tau_integrated == pytest.approx(1.0, rel=1e-12, abs=0.0)


def test_block_length_grows_with_injected_dependence() -> None:
    # A deterministic run-structured indicator: 30 blocks of 20 misses inside
    # 30000 releases. Consecutive releases then share state out to lag 20, so
    # the selected block length must exceed the independent case's 1.
    indicator = np.zeros(30_000, dtype=np.int8)
    for block in range(30):
        start = block * 1000
        indicator[start : start + 20] = 1
    choice = block_length_from_autocorrelation(indicator)
    assert choice.lag_cut > 1
    assert choice.block_length > 1


def test_block_length_is_capped_so_twenty_blocks_span_the_series() -> None:
    # A single long run of misses gives an autocorrelation that stays above the
    # band for hundreds of lags, so tau is large and the n//20 cap binds.
    indicator = np.zeros(2000, dtype=np.int8)
    indicator[:1000] = 1
    choice = block_length_from_autocorrelation(indicator)
    assert choice.block_length <= 2000 // 20
    assert choice.capped is True


def test_block_rule_text_is_carried_with_the_choice(rng) -> None:
    choice = block_length_from_autocorrelation((rng.random(1000) < 0.1).astype(np.int8))
    assert "integrated autocorrelation" in choice.rule


@pytest.mark.parametrize("series", [np.zeros(1), np.zeros((2, 2))])
def test_block_length_input_validation(series: np.ndarray) -> None:
    with pytest.raises(ValueError):
        block_length_from_autocorrelation(series)


def test_moving_block_index_has_the_right_length_and_range(rng) -> None:
    for block in (1, 3, 17, 100):
        index = moving_block_index(100, block, rng)
        assert index.size == 100
        assert index.min() >= 0
        assert index.max() <= 99


def test_moving_block_index_blocks_are_consecutive(rng) -> None:
    index = moving_block_index(100, 5, rng)
    # Within each complete block the indices must step by exactly one.
    for start in range(0, 100 - 5, 5):
        block = index[start : start + 5]
        assert np.array_equal(np.diff(block), np.ones(4, dtype=block.dtype))


@pytest.mark.parametrize(("n", "block"), [(0, 1), (10, 0), (10, 11), (10, -1)])
def test_moving_block_index_input_validation(n: int, block: int) -> None:
    with pytest.raises(ValueError):
        moving_block_index(n, block, np.random.default_rng(0))


def test_bootstrap_is_reproducible_at_a_fixed_seed() -> None:
    values = np.arange(500.0)

    def statistic(index: np.ndarray) -> float:
        return float(values[index].mean())

    first = moving_block_bootstrap(statistic, 500, 10, 64, seed=7)
    second = moving_block_bootstrap(statistic, 500, 10, 64, seed=7)
    assert np.array_equal(first, second)
    third = moving_block_bootstrap(statistic, 500, 10, 64, seed=8)
    assert not np.array_equal(first, third)


def test_bootstrap_rejects_zero_resamples() -> None:
    with pytest.raises(ValueError, match="resamples must be"):
        moving_block_bootstrap(lambda index: 0.0, 10, 1, 0, seed=0)


def test_percentile_interval_brackets_the_median_of_its_input() -> None:
    # HAND CALCULATION. For the 101 values 0, 1, ..., 100 the 2.5th and 97.5th
    # percentiles under numpy's default linear interpolation are at positions
    # 0.025*100 = 2.5 and 0.975*100 = 97.5, so the interval is [2.5, 97.5].
    low, high = percentile_interval(np.arange(101.0))
    assert low == pytest.approx(2.5, rel=1e-12, abs=0.0)
    assert high == pytest.approx(97.5, rel=1e-12, abs=0.0)


def test_percentile_interval_excludes_non_finite_replicates() -> None:
    replicates = np.asarray([1.0, 2.0, 3.0, np.inf, np.nan])
    low, high = percentile_interval(replicates)
    assert np.isfinite(low) and np.isfinite(high)
    assert low >= 1.0 and high <= 3.0


def test_percentile_interval_of_no_finite_replicate_is_nan() -> None:
    low, high = percentile_interval(np.asarray([np.inf, np.nan]))
    assert np.isnan(low) and np.isnan(high)


@pytest.mark.parametrize("confidence", [0.0, 1.0, -0.1, 1.5])
def test_percentile_interval_rejects_a_bad_confidence(confidence: float) -> None:
    with pytest.raises(ValueError, match="confidence must be"):
        percentile_interval(np.arange(10.0), confidence)


def test_counted_rate_interval_contains_the_point_estimate(ground_truth) -> None:
    interval = counted_rate_interval(ground_truth, resamples=400, seed=11)
    assert interval.low <= interval.rate <= interval.high
    indicator = miss_indicator(ground_truth)
    assert interval.rate == pytest.approx(
        float(indicator.mean()), rel=1e-12, abs=0.0
    )
    assert interval.block.n == indicator.size


def test_understatement_factor_is_the_ratio_it_reports(ground_truth) -> None:
    result = understatement_factor(ground_truth, resamples=200, seed=13)
    assert result.factor == pytest.approx(
        result.counted_rate / result.margin_probability, rel=1e-12, abs=0.0
    )
    assert result.block_rows >= 1
    assert result.interval_contains_one == bool(result.low <= 1.0 <= result.high)


def test_understatement_interval_widens_as_resamples_stay_consistent(
    ground_truth,
) -> None:
    # Two seeds at the same resample count must give intervals that overlap:
    # they are two estimates of the same interval, not two different quantities.
    first = understatement_factor(ground_truth, resamples=300, seed=21)
    second = understatement_factor(ground_truth, resamples=300, seed=22)
    assert first.low <= second.high
    assert second.low <= first.high
