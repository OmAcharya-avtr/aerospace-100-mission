"""The bootstrap, and above all its RESAMPLING UNIT.

SPEC §4 declares the pass as the unit and the declaration is not negotiable.
These tests assert it on the data structure itself: every interval the package
produces records ``unit == "pass"``, and a frame-unit interval is measured
against it in ``validation/validate_bootstrap_unit.py`` to show what the wrong
unit costs.
"""

from __future__ import annotations

import numpy as np
import pytest

from hiltimingdiff import bootstrap
from hiltimingdiff.bootstrap import (
    BOOTSTRAP_SEED,
    CONFIDENCE,
    RESAMPLES,
    RESAMPLING_UNIT,
    Interval,
    bootstrap_ratio_of_medians,
    bootstrap_statistic,
    percentile_interval,
    unit_resamples,
)


def test_declared_resampling_unit_is_the_pass():
    assert RESAMPLING_UNIT == "pass"


def test_declared_confidence_is_ninety_five_per_cent():
    assert CONFIDENCE == 0.95


def test_default_resamples_is_four_thousand():
    assert RESAMPLES == 4000


def test_interval_default_unit_is_the_pass():
    interval = Interval(point=1.0, low=0.5, high=1.5, confidence=0.95, resamples=10, units=5)
    assert interval.unit == "pass"


def test_interval_width():
    interval = Interval(point=2.0, low=1.5, high=2.5, confidence=0.95, resamples=10, units=5)
    assert interval.width == pytest.approx(1.0, rel=1e-12, abs=0.0)


def test_interval_relative_width():
    interval = Interval(point=2.0, low=1.5, high=2.5, confidence=0.95, resamples=10, units=5)
    assert interval.relative_width == pytest.approx(0.5, rel=1e-12, abs=0.0)


def test_interval_relative_width_is_nan_at_a_zero_point():
    interval = Interval(point=0.0, low=-1.0, high=1.0, confidence=0.95, resamples=10, units=5)
    assert np.isnan(interval.relative_width)


@pytest.mark.parametrize(
    ("value", "inside"),
    [(0.4, False), (0.5, True), (1.0, True), (1.6, False)],
)
def test_interval_contains_is_closed(value, inside):
    interval = Interval(point=1.0, low=0.5, high=1.5, confidence=0.95, resamples=10, units=5)
    assert interval.contains(value) is inside


def test_interval_format_shows_point_and_both_bounds():
    interval = Interval(point=1.5, low=1.25, high=1.75, confidence=0.95, resamples=10, units=5)
    assert interval.format(2) == "1.50 [1.25, 1.75]"


def test_unit_resamples_shape():
    indices = unit_resamples(40, resamples=17)
    assert indices.shape == (17, 40)


def test_unit_resamples_indices_are_in_range():
    indices = unit_resamples(40, resamples=50)
    assert int(indices.min()) >= 0
    assert int(indices.max()) <= 39


def test_unit_resamples_is_reproducible_at_a_fixed_seed():
    first = unit_resamples(40, resamples=20, seed=7)
    second = unit_resamples(40, resamples=20, seed=7)
    assert np.array_equal(first, second)


def test_unit_resamples_differs_at_a_different_seed():
    first = unit_resamples(40, resamples=20, seed=7)
    second = unit_resamples(40, resamples=20, seed=8)
    assert not np.array_equal(first, second)


def test_unit_resamples_draws_with_replacement():
    # With 4000 draws from 5 units, a sample with no repeated index would be a
    # 4.6e-3 event per row; over 4000 rows the probability that none repeats is
    # astronomically small, so at least one row must repeat an index.
    indices = unit_resamples(5, resamples=4000, seed=3)
    repeated = [len(set(row.tolist())) < 5 for row in indices]
    assert any(repeated)


@pytest.mark.parametrize("n_units", [0, 1])
def test_unit_resamples_refuses_fewer_than_two_units(n_units):
    with pytest.raises(ValueError, match="at least 2 units"):
        unit_resamples(n_units)


def test_unit_resamples_refuses_zero_resamples():
    with pytest.raises(ValueError, match="resamples must be positive"):
        unit_resamples(10, resamples=0)


def test_percentile_interval_brackets_the_replicates():
    replicates = np.linspace(0.0, 10.0, 1001)
    interval = percentile_interval(5.0, replicates, units=50)
    assert interval.low == pytest.approx(0.25, rel=1e-6, abs=0.0)
    assert interval.high == pytest.approx(9.75, rel=1e-6, abs=0.0)


def test_percentile_interval_records_the_point_not_the_replicate_mean():
    replicates = np.linspace(0.0, 10.0, 101)
    interval = percentile_interval(42.0, replicates, units=50)
    assert interval.point == 42.0


def test_percentile_interval_drops_non_finite_replicates():
    replicates = np.array([1.0, 2.0, 3.0, np.nan, np.inf])
    interval = percentile_interval(2.0, replicates, units=10)
    assert interval.resamples == 3


def test_percentile_interval_refuses_all_non_finite_replicates():
    with pytest.raises(ValueError, match="non-finite"):
        percentile_interval(1.0, np.array([np.nan, np.nan]), units=10)


@pytest.mark.parametrize("confidence", [0.0, 1.0, -0.1, 1.5])
def test_percentile_interval_refuses_a_confidence_outside_the_unit_interval(confidence):
    with pytest.raises(ValueError, match="confidence must lie"):
        percentile_interval(1.0, np.array([1.0, 2.0]), confidence=confidence, units=10)


@pytest.mark.parametrize("confidence", [0.80, 0.90, 0.95, 0.99])
def test_a_wider_confidence_gives_a_wider_interval(confidence):
    replicates = np.random.default_rng(1).normal(0.0, 1.0, size=20000)
    narrow = percentile_interval(0.0, replicates, confidence=0.50, units=100)
    wide = percentile_interval(0.0, replicates, confidence=confidence, units=100)
    assert wide.width >= narrow.width


def test_bootstrap_statistic_point_is_the_statistic_on_the_observed_data():
    values = np.arange(1.0, 51.0)
    interval = bootstrap_statistic(values, lambda sample: float(np.median(sample)))
    assert interval.point == pytest.approx(float(np.median(values)), rel=1e-12, abs=0.0)


def test_bootstrap_statistic_records_the_unit_count():
    values = np.arange(1.0, 51.0)
    interval = bootstrap_statistic(values, lambda sample: float(np.mean(sample)))
    assert interval.units == 50


def test_bootstrap_statistic_records_the_pass_unit():
    values = np.arange(1.0, 51.0)
    interval = bootstrap_statistic(values, lambda sample: float(np.mean(sample)))
    assert interval.unit == "pass"


def test_bootstrap_statistic_interval_brackets_the_point_for_a_mean():
    values = np.random.default_rng(5).normal(10.0, 2.0, size=200)
    interval = bootstrap_statistic(values, lambda sample: float(np.mean(sample)))
    assert interval.low <= interval.point <= interval.high


def test_bootstrap_statistic_interval_is_reproducible():
    values = np.random.default_rng(5).normal(10.0, 2.0, size=200)
    first = bootstrap_statistic(values, lambda sample: float(np.mean(sample)), seed=11)
    second = bootstrap_statistic(values, lambda sample: float(np.mean(sample)), seed=11)
    assert (first.low, first.high) == (second.low, second.high)


def test_bootstrap_statistic_interval_narrows_with_more_units():
    generator = np.random.default_rng(13)
    small = bootstrap_statistic(
        generator.normal(0.0, 1.0, size=40), lambda sample: float(np.mean(sample))
    )
    large = bootstrap_statistic(
        generator.normal(0.0, 1.0, size=4000), lambda sample: float(np.mean(sample))
    )
    assert large.width < small.width


def test_bootstrap_statistic_on_a_constant_series_has_zero_width():
    interval = bootstrap_statistic(
        np.full(100, 7.0), lambda sample: float(np.median(sample))
    )
    assert interval.width == pytest.approx(0.0, rel=0.0, abs=1e-12)


def test_bootstrap_ratio_of_medians_point():
    numerator = np.full(100, 8.0)
    denominator = np.full(100, 2.0)
    interval = bootstrap_ratio_of_medians(numerator, denominator)
    assert interval.point == pytest.approx(4.0, rel=1e-12, abs=0.0)


def test_bootstrap_ratio_of_medians_resamples_both_sides_together():
    # Paired series: unit i contributes (2i, i). The ratio of medians is 2 for
    # every resample, because the pairing is preserved. If the two sides were
    # resampled independently the ratio would wander.
    units = np.arange(1.0, 101.0)
    interval = bootstrap_ratio_of_medians(2.0 * units, units)
    assert interval.low == pytest.approx(2.0, rel=1e-12, abs=0.0)
    assert interval.high == pytest.approx(2.0, rel=1e-12, abs=0.0)


def test_bootstrap_ratio_of_medians_refuses_mismatched_lengths():
    with pytest.raises(ValueError, match="same units"):
        bootstrap_ratio_of_medians(np.ones(5), np.ones(6))


def test_bootstrap_ratio_of_medians_returns_nan_replicates_for_a_zero_denominator():
    numerator = np.ones(50)
    denominator = np.zeros(50)
    with pytest.raises(ValueError, match="non-finite"):
        bootstrap_ratio_of_medians(numerator, denominator)


def test_module_seed_is_pinned():
    assert BOOTSTRAP_SEED == 94094


def test_bootstrap_module_exposes_only_the_percentile_method():
    # The interval is a percentile interval and is labelled as one. There is no
    # BCa or normal-theory path to pick by accident.
    assert not hasattr(bootstrap, "bca_interval")
    assert not hasattr(bootstrap, "normal_interval")
