"""Unit and input-validation tests for the paired record and its validation."""

from __future__ import annotations

import numpy as np
import pytest

from transfergap.discrepancy import (
    GapSample,
    clip_fraction,
    gaps_from_pairs,
    validate_confidence,
    validate_gaps,
)


def test_validate_gaps_returns_float64_one_dimensional(simple_gaps):
    out = validate_gaps(simple_gaps, 10.0)
    assert out.dtype == np.float64
    assert out.shape == (4,)


@pytest.mark.parametrize(
    ("bad_range", "fragment"),
    [(0.0, "finite positive"), (-1.0, "finite positive"), (np.inf, "finite positive")],
)
def test_validate_gaps_rejects_bad_range(simple_gaps, bad_range, fragment):
    with pytest.raises(ValueError, match=fragment):
        validate_gaps(simple_gaps, bad_range)


def test_validate_gaps_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        validate_gaps(np.array([]), 10.0)


def test_validate_gaps_rejects_two_dimensional():
    with pytest.raises(ValueError, match="one-dimensional"):
        validate_gaps(np.zeros((2, 2)), 10.0)


def test_validate_gaps_rejects_nan():
    with pytest.raises(ValueError, match="non-finite"):
        validate_gaps(np.array([1.0, np.nan]), 10.0)


def test_validate_gaps_rejects_negative():
    with pytest.raises(ValueError, match="negative"):
        validate_gaps(np.array([1.0, -0.5]), 10.0)


def test_validate_gaps_rejects_above_range():
    """A range smaller than the data is not conservative, it is wrong."""
    with pytest.raises(ValueError, match="above the declared range"):
        validate_gaps(np.array([1.0, 11.0]), 10.0)


@pytest.mark.parametrize("value", [0.0, 1.0, -0.1, 1.5, np.nan])
def test_validate_confidence_rejects_outside_open_unit_interval(value):
    with pytest.raises(ValueError, match="strictly inside"):
        validate_confidence(value)


def test_gaps_from_pairs_is_the_clipped_absolute_difference():
    # Hand calculation. field = [10, 20, 30], simulated = [12, 5, 30], R = 10.
    # |10-12| = 2, |20-5| = 15 -> clipped to 10, |30-30| = 0.
    out = gaps_from_pairs(np.array([10.0, 20.0, 30.0]), np.array([12.0, 5.0, 30.0]), 10.0)
    assert out.tolist() == [2.0, 10.0, 0.0]


def test_gaps_from_pairs_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="PAIRED"):
        gaps_from_pairs(np.array([1.0, 2.0]), np.array([1.0]), 10.0)


def test_clip_fraction_counts_the_censored_pairs():
    # One of three raw discrepancies (15) is above R = 10, so 1/3.
    fraction = clip_fraction(
        np.array([10.0, 20.0, 30.0]), np.array([12.0, 5.0, 30.0]), 10.0
    )
    assert fraction == pytest.approx(1.0 / 3.0, rel=1e-12, abs=0.0)


def test_gap_sample_rejects_feature_row_mismatch():
    with pytest.raises(ValueError, match="operating condition"):
        GapSample(
            gaps=np.array([1.0, 2.0]),
            features=np.zeros((3, 2)),
            feature_names=("a", "b"),
            unit="kN.m",
            bound_range=10.0,
        )


def test_gap_sample_rejects_name_count_mismatch():
    with pytest.raises(ValueError, match="columns"):
        GapSample(
            gaps=np.array([1.0, 2.0]),
            features=np.zeros((2, 3)),
            feature_names=("a", "b"),
            unit="kN.m",
            bound_range=10.0,
        )


def test_gap_sample_subset_keeps_metadata(field_sample):
    subset = field_sample.subset(np.arange(5))
    assert subset.size == 5
    assert subset.unit == field_sample.unit
    assert subset.feature_names == field_sample.feature_names
    assert subset.bound_range == field_sample.bound_range


def test_gap_sample_size_matches_the_gap_count(field_sample):
    assert field_sample.size == field_sample.gaps.size == 60
