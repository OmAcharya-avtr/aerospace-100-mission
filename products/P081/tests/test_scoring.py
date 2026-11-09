"""Scores: the exact interval, the confusion matrix, the Brier score, calibration."""

from __future__ import annotations

import numpy as np
import pytest

from twindiverge.scoring import (
    CONFIDENCE_BINS,
    CONFIDENT_THRESHOLD,
    brier_score,
    calibration_table,
    confident_error_rate,
    confusion_matrix,
    proportion,
)


def test_clopper_pearson_at_the_textbook_case():
    """0 successes of 10 at 95 per cent: the lower limit is exactly 0 and the
    upper is 1 - 0.025**(1/10) = 0.30849969... , which is the closed form of
    the Clopper-Pearson upper limit when the count is zero."""
    result = proportion(0, 10)
    assert result.lower == pytest.approx(0.0, rel=0.0, abs=0.0)
    assert result.upper == pytest.approx(1.0 - 0.025 ** (1.0 / 10.0), rel=1e-12, abs=0.0)
    assert not np.isnan(result.value)
    assert result.value == pytest.approx(0.0, rel=0.0, abs=0.0)


def test_clopper_pearson_at_full_success():
    """10 of 10: the upper limit is exactly 1 and the lower is 0.025**(1/10) =
    0.69150030..., the mirror of the case above. A normal approximation would
    give a standard error of exactly zero here, which is why this interval is
    used."""
    result = proportion(10, 10)
    assert result.upper == pytest.approx(1.0, rel=0.0, abs=0.0)
    assert result.lower == pytest.approx(0.025 ** (1.0 / 10.0), rel=1e-12, abs=0.0)


def test_clopper_pearson_brackets_the_point_estimate():
    for count in range(0, 21):
        result = proportion(count, 20)
        assert result.lower <= result.value <= result.upper


def test_an_empty_proportion_is_reported_as_not_available():
    result = proportion(0, 0)
    assert np.isnan(result.value)
    assert "n/a" in str(result)


@pytest.mark.parametrize(("count", "total"), [(-1, 10), (11, 10), (1, -1)])
def test_proportion_validates_its_counts(count: int, total: int):
    with pytest.raises(ValueError, match="count <= total"):
        proportion(count, total)


def test_proportion_validates_its_level():
    with pytest.raises(ValueError, match="level"):
        proportion(1, 10, level=0.0)


def test_confusion_matrix_counts_every_pair():
    matrix = confusion_matrix(
        np.array([0, 0, 1, 1, 1]), np.array([0, 1, 1, 1, 0]), (0, 1)
    )
    assert matrix.tolist() == [[1, 1], [1, 2]]
    assert int(matrix.sum()) == 5


def test_confusion_matrix_refuses_a_label_outside_the_declared_set():
    with pytest.raises(ValueError, match="outside the declared label set"):
        confusion_matrix(np.array([0, 2]), np.array([0, 0]), (0, 1))


def test_confusion_matrix_refuses_mismatched_shapes():
    with pytest.raises(ValueError, match="shape"):
        confusion_matrix(np.array([0, 1]), np.array([0]), (0, 1))


def test_brier_score_of_a_perfect_forecast_is_zero():
    probabilities = np.eye(3)
    assert brier_score(probabilities, np.array([0, 1, 2])) == pytest.approx(
        0.0, rel=0.0, abs=0.0
    )


def test_brier_score_of_a_confidently_wrong_forecast_is_two():
    """A one-hot forecast on the wrong class costs (1-0)^2 + (0-1)^2 = 2."""
    assert brier_score(np.eye(2), np.array([1, 0])) == pytest.approx(2.0, rel=0.0, abs=0.0)


def test_brier_score_charges_the_whole_mass_when_the_truth_is_outside_the_columns():
    """The unidentifiable-subspace case for a classifier with no refusal class:
    a one-hot forecast against an absent truth costs 1."""
    assert brier_score(np.eye(2)[:1], np.array([-1])) == pytest.approx(
        1.0, rel=0.0, abs=0.0
    )


def test_brier_score_validates_its_rows():
    with pytest.raises(ValueError, match="truth entries"):
        brier_score(np.eye(2), np.array([0]))
    with pytest.raises(ValueError, match="sum to 1"):
        brier_score(np.array([[0.5, 0.2]]), np.array([0]))


def test_calibration_table_has_one_row_per_declared_bin():
    rows = calibration_table(np.array([0.1, 0.5, 0.95]), np.array([False, True, True]))
    assert len(rows) == len(CONFIDENCE_BINS)
    assert sum(row[2] for row in rows) == 3


def test_calibration_table_marks_empty_bins():
    rows = calibration_table(np.array([0.95]), np.array([True]))
    empty = [row for row in rows if row[2] == 0]
    assert empty
    for row in empty:
        assert np.isnan(row[3]) and np.isnan(row[4])


def test_calibration_table_validates_its_shapes():
    with pytest.raises(ValueError, match="confidence shape"):
        calibration_table(np.array([0.5]), np.array([True, False]))


def test_confident_error_rate_counts_over_the_whole_population():
    """Two of four cases are wrong, and only one of those two is confident, so
    the rate is 1/4 and not 1/2."""
    result = confident_error_rate(
        np.array([0.9, 0.2, 0.9, 0.9]), np.array([False, False, True, True])
    )
    assert result.count == 1
    assert result.total == 4
    assert result.value == pytest.approx(0.25, rel=0.0, abs=0.0)


def test_confident_error_rate_uses_the_declared_threshold():
    assert CONFIDENT_THRESHOLD == pytest.approx(0.5, rel=0.0, abs=0.0)
    just_below = confident_error_rate(np.array([0.499999]), np.array([False]))
    just_above = confident_error_rate(np.array([0.500001]), np.array([False]))
    assert just_below.count == 0
    assert just_above.count == 1


def test_confident_error_rate_validates_its_arguments():
    with pytest.raises(ValueError, match="confidence shape"):
        confident_error_rate(np.array([0.5]), np.array([True, True]))
    with pytest.raises(ValueError, match="threshold"):
        confident_error_rate(np.array([0.5]), np.array([True]), threshold=1.5)
