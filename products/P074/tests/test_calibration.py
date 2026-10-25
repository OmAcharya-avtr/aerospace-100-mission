"""Coverage measurement, the exact binomial interval, and the stratified table."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from orbitdrift import calibration


def test_clopper_pearson_known_answers() -> None:
    """Published values for the exact binomial interval.

    For 8 successes in 10 trials at 95 per cent confidence the Clopper-Pearson
    interval is (0.4439, 0.9748) to four decimal places; this is a standard
    textbook example (Clopper and Pearson 1934, and reproduced in every
    treatment of the exact interval since). For 0 in 10 the lower limit is
    exactly 0 and the upper is 0.3085; for 10 in 10 the interval is
    (0.6915, 1).
    """
    low, high = calibration.clopper_pearson(8, 10)
    assert low == pytest.approx(0.44390, rel=0.0, abs=5e-5)
    assert high == pytest.approx(0.97475, rel=0.0, abs=5e-5)

    low, high = calibration.clopper_pearson(0, 10)
    assert low == 0.0
    assert high == pytest.approx(0.30850, rel=0.0, abs=5e-5)

    low, high = calibration.clopper_pearson(10, 10)
    assert low == pytest.approx(0.69150, rel=0.0, abs=5e-5)
    assert high == 1.0


def test_clopper_pearson_contains_the_point_estimate_and_widens_with_confidence() -> None:
    for successes, trials in ((3, 20), (17, 20), (50, 100), (1, 3)):
        fraction = successes / trials
        low, high = calibration.clopper_pearson(successes, trials, 0.95)
        assert low <= fraction <= high
        wide_low, wide_high = calibration.clopper_pearson(successes, trials, 0.99)
        assert wide_low <= low
        assert wide_high >= high


def test_clopper_pearson_rejects_impossible_counts() -> None:
    with pytest.raises(ValueError, match="trials must be positive"):
        calibration.clopper_pearson(0, 0)
    with pytest.raises(ValueError, match=r"must be in \[0, 5\]"):
        calibration.clopper_pearson(6, 5)
    with pytest.raises(ValueError, match=r"must be in \[0, 5\]"):
        calibration.clopper_pearson(-1, 5)


def test_interval_metrics_on_a_hand_constructed_case() -> None:
    """Four rows, two covered, so coverage is exactly 0.5.

    Widths are 2, 2, 2, 2 so the median and mean width are 2. The interval
    score for a covered row is its width; for the row at y = 10 with interval
    [1, 3] the penalty is (2/alpha)(10 - 3) = 20 * 7 = 140 at alpha = 0.1, so
    the score for that row is 142.
    """
    lower = np.array([1.0, 1.0, 1.0, 1.0])
    upper = np.array([3.0, 3.0, 3.0, 3.0])
    truth = np.array([2.0, 2.5, 10.0, 0.5])
    metrics = calibration.interval_metrics(lower, upper, truth, nominal=0.90)
    assert metrics.n == 4
    assert metrics.coverage == 0.5
    assert metrics.median_width_km == 2.0
    assert metrics.mean_width_km == 2.0
    assert metrics.median_log_width == pytest.approx(np.log(3.0), rel=1e-14, abs=0.0)
    expected = (2.0 + 2.0 + (2.0 + 20.0 * 7.0) + (2.0 + 20.0 * 0.5)) / 4.0
    assert metrics.interval_score == pytest.approx(expected, rel=1e-13, abs=0.0)
    assert metrics.crossing_fraction == 0.0


def test_interval_metrics_handles_a_crossed_interval_without_sorting_it_away() -> None:
    """A crossing is counted and the coverage test still uses the real range.

    Unconstrained quantile regression can put the lower prediction above the
    upper one. Sorting would hide the fact; this package reports it.
    """
    metrics = calibration.interval_metrics(
        np.array([5.0, 1.0]), np.array([2.0, 3.0]), np.array([3.0, 2.0]), nominal=0.90
    )
    assert metrics.crossing_fraction == 0.5
    assert metrics.coverage == 1.0


def test_interval_metrics_rejects_mismatched_shapes_and_empty_input() -> None:
    with pytest.raises(ValueError, match="shapes must match"):
        calibration.interval_metrics(np.array([1.0]), np.array([2.0, 3.0]), np.array([1.5]))
    with pytest.raises(ValueError, match="no finite rows"):
        calibration.interval_metrics(
            np.array([np.nan]), np.array([np.nan]), np.array([np.nan])
        )


def test_coverage_by_group_is_one_trial_per_object() -> None:
    """Two objects of three rows each; one covered on 2 of 3, one on 0 of 3."""
    lower = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    upper = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
    truth = np.array([1.0, 1.0, 5.0, 5.0, 5.0, 5.0])
    groups = np.array([0, 0, 0, 1, 1, 1])
    fraction, interval, n_groups = calibration.coverage_by_group(lower, upper, truth, groups)
    assert n_groups == 2
    assert fraction == 0.5
    assert interval[0] <= 0.5 <= interval[1]


def test_coverage_table_flags_a_cell_whose_interval_excludes_the_nominal_level() -> None:
    """Two strata of 100 rows: one covering 0.90, one covering 0.50."""
    rng = np.random.default_rng(29)
    truth = rng.uniform(0.0, 1.0, 200)
    lower = np.zeros(200)
    upper = np.where(np.arange(200) < 100, 0.90, 0.50)
    strata = np.where(np.arange(200) < 100, "a good", "b bad")
    cells = calibration.coverage_table(lower, upper, truth, strata, nominal=0.90)
    assert [cell.stratum for cell in cells] == ["a good", "b bad"]
    good, bad = cells
    assert good.n == 100
    assert bad.miscalibrated is True
    assert bad.verdict == "under"
    assert bad.coverage < 0.65


def test_coverage_table_does_not_flag_a_cell_too_small_to_be_inconsistent() -> None:
    """A five-row cell has an interval so wide it cannot contradict anything.

    Flagging it would be noise. It is returned with its count and its
    interval and the verdict ``thin``, which is an absence of evidence rather
    than a pass.
    """
    lower = np.zeros(5)
    upper = np.full(5, 0.3)
    truth = np.array([0.1, 0.2, 0.9, 0.95, 0.99])
    cells = calibration.coverage_table(
        lower, upper, truth, np.array(["tiny"] * 5), nominal=0.90, minimum_cell=20
    )
    assert cells[0].n == 5
    assert cells[0].miscalibrated is False
    assert cells[0].verdict == "thin"


def test_coverage_table_reports_over_coverage_as_over() -> None:
    lower = np.zeros(200)
    upper = np.full(200, 10.0)
    truth = np.full(200, 1.0)
    cells = calibration.coverage_table(
        lower, upper, truth, np.array(["all"] * 200), nominal=0.90
    )
    assert cells[0].coverage == 1.0
    assert cells[0].miscalibrated is True
    assert cells[0].verdict == "over"


def test_altitude_and_age_strata_labels_are_sortable_and_cover_the_range() -> None:
    altitude = np.array([100.0, 399.0, 400.0, 599.0, 700.0, 900.0, 1500.0])
    labels = calibration.altitude_stratum(altitude)
    assert labels[0] == "alt 1 <400 km"
    assert labels[2] == "alt 2 400-600 km"
    assert labels[-1] == "alt 5 >1100 km"
    assert list(labels) == sorted(labels)

    age = np.array([0.1, 1.0, 2.0, 4.0, 9.0])
    age_labels = calibration.age_stratum(age)
    assert age_labels[0] == "age 1 <1 d"
    assert age_labels[-1] == "age 4 >4 d"
    assert list(age_labels) == sorted(age_labels)


@settings(max_examples=200, deadline=None)
@given(
    successes=st.integers(0, 500),
    trials=st.integers(1, 500),
    confidence=st.floats(0.5, 0.999, allow_nan=False),
)
def test_clopper_pearson_always_brackets_the_proportion(
    successes: int, trials: int, confidence: float
) -> None:
    """The exact interval must contain the point estimate, for every count.

    An independently derived property of the construction rather than a
    comparison: the interval is defined by inverting the binomial tails at the
    observed count, so the count's own proportion is inside it by definition.
    Hypothesis covers the boundaries at 0 and at ``n`` and every confidence
    level, where a naive implementation returns a NaN or an inverted interval.
    """
    if successes > trials:
        return
    low, high = calibration.clopper_pearson(successes, trials, confidence)
    assert 0.0 <= low <= successes / trials <= high <= 1.0
    assert np.isfinite(low) and np.isfinite(high)


@settings(max_examples=120, deadline=None)
@given(
    values=st.lists(st.floats(0.01, 100.0, allow_nan=False), min_size=1, max_size=40),
    half_width=st.floats(0.01, 5.0, allow_nan=False),
)
def test_a_wider_interval_never_covers_less(values: list[float], half_width: float) -> None:
    """Monotonicity of coverage in width: a derived property, not a measurement."""
    truth = np.array(values)
    centre = np.full(truth.size, float(np.median(truth)))
    narrow = calibration.interval_metrics(
        centre / (1.0 + half_width), centre * (1.0 + half_width), truth
    )
    wide = calibration.interval_metrics(
        centre / (1.0 + 2.0 * half_width), centre * (1.0 + 2.0 * half_width), truth
    )
    assert wide.coverage >= narrow.coverage
    assert wide.median_width_km >= narrow.median_width_km
