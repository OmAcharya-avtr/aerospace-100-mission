"""Unit, known-answer, input-validation and edge-case tests for the permutation test."""

from __future__ import annotations

import numpy as np
import pytest

from transfergap.exchangeability import (
    STATISTICS,
    half_split_statistic,
    permutation_test,
    trend_statistic,
)


def test_trend_statistic_is_one_on_a_monotone_increase():
    """Hand calculation: ranks 0..4 against index 0..4 is a perfect correlation.

    Spearman's rho between the ranks of [1,2,3,4,5] (which are 0,1,2,3,4) and
    the index (0,1,2,3,4) is the Pearson correlation of a vector with itself,
    which is 1 in exact arithmetic. It is also 1 to the last bit here because
    the two centred vectors are the identical array.
    """
    assert trend_statistic(np.array([1.0, 2.0, 3.0, 4.0, 5.0])) == pytest.approx(
        1.0, rel=0.0, abs=1e-14
    )


def test_trend_statistic_is_minus_one_on_a_monotone_decrease():
    assert trend_statistic(np.array([5.0, 4.0, 3.0, 2.0, 1.0])) == pytest.approx(
        -1.0, rel=0.0, abs=1e-14
    )


def test_trend_statistic_is_zero_on_a_constant_sample():
    """Edge case: the rank vector is constant and the correlation is undefined.

    A constant sample carries no evidence of drift, so zero is the honest
    value. Returning nan would propagate into a p-value; raising would make a
    legitimate if degenerate sample unusable.
    """
    assert trend_statistic(np.full(10, 3.0)) == 0.0


def test_trend_statistic_known_answer_on_a_small_permutation():
    """Hand calculation on [2, 1, 4, 3].

    ranks       = [1, 0, 3, 2]
    index       = [0, 1, 2, 3]
    centred r   = [-0.5, -1.5, 1.5, 0.5]
    centred i   = [-1.5, -0.5, 0.5, 1.5]
    numerator   = 0.75 + 0.75 + 0.75 + 0.75 = 3.0
    sum r^2     = 0.25 + 2.25 + 2.25 + 0.25 = 5.0
    sum i^2     = 5.0
    rho         = 3.0 / sqrt(25) = 0.6
    """
    assert trend_statistic(np.array([2.0, 1.0, 4.0, 3.0])) == pytest.approx(
        0.6, rel=1e-14, abs=0.0
    )


def test_half_split_statistic_known_answer():
    """Hand calculation: mean([3,4]) - mean([1,2]) = 3.5 - 1.5 = 2.0."""
    assert half_split_statistic(np.array([1.0, 2.0, 3.0, 4.0])) == 2.0


def test_half_split_statistic_drops_the_middle_of_an_odd_sample():
    """[1,2,3,4,5]: halves of 2 each, mean([4,5]) - mean([1,2]) = 4.5 - 1.5 = 3.0."""
    assert half_split_statistic(np.array([1.0, 2.0, 3.0, 4.0, 5.0])) == 3.0


def test_statistics_registry_holds_both_named_alternatives():
    assert set(STATISTICS) == {"trend", "half_split"}


def test_permutation_test_rejects_a_hard_monotone_drift():
    """A sample that is strictly increasing is as non-exchangeable as it gets.

    With 40 strictly increasing values the trend statistic is 1.0, the largest
    value it can take, so no permutation can match it except the identity-like
    ones, and the p-value is the smallest attainable, 1/(B+1).
    """
    result = permutation_test(np.arange(40.0), "trend", permutations=499, seed=1)
    assert result.rejected is True
    assert result.p_value == pytest.approx(1.0 / 500.0, rel=0.0, abs=1e-15)


def test_permutation_test_does_not_reject_a_shuffled_sample():
    """An i.i.d. sample is exchangeable; a single draw should usually pass."""
    values = np.random.default_rng(5).normal(size=60)
    result = permutation_test(values, "trend", permutations=499, alpha=0.05, seed=2)
    assert result.rejected is False


def test_permutation_p_value_is_never_zero():
    """Equation (18): the +1 in both places makes the floor 1/(B+1), not 0."""
    result = permutation_test(np.arange(50.0), "half_split", permutations=199, seed=3)
    assert result.p_value >= 1.0 / 200.0


def test_permutation_test_is_reproducible_from_its_seed():
    values = np.random.default_rng(9).normal(size=40)
    first = permutation_test(values, "trend", permutations=299, seed=17)
    second = permutation_test(values, "trend", permutations=299, seed=17)
    assert first.p_value == second.p_value
    assert first.null_mean == second.null_mean


def test_permutation_test_verdict_does_not_claim_exchangeability():
    """A non-rejection must not be reported as evidence FOR the assumption."""
    values = np.random.default_rng(5).normal(size=60)
    result = permutation_test(values, "trend", permutations=199, seed=2)
    assert "not evidence that" in result.verdict
    assert "exchangeable" in result.verdict


def test_permutation_test_verdict_names_the_consequence_on_rejection():
    result = permutation_test(np.arange(40.0), "trend", permutations=199, seed=1)
    assert "no guarantee" in result.verdict


def test_permutation_test_rejects_an_unknown_statistic():
    with pytest.raises(ValueError, match="statistic must be one of"):
        permutation_test(np.arange(20.0), "autocorrelation", permutations=199)


def test_permutation_test_rejects_too_few_permutations():
    with pytest.raises(ValueError, match="at least 99"):
        permutation_test(np.arange(20.0), "trend", permutations=50)


def test_permutation_test_rejects_a_nan():
    with pytest.raises(ValueError, match="non-finite"):
        permutation_test(np.array([1.0, np.nan, 3.0, 4.0, 5.0]), "trend")


def test_trend_statistic_rejects_a_sample_that_is_too_short():
    with pytest.raises(ValueError, match="at least 3"):
        trend_statistic(np.array([1.0, 2.0]))


def test_half_split_statistic_rejects_a_sample_that_is_too_short():
    with pytest.raises(ValueError, match="at least 4"):
        half_split_statistic(np.array([1.0, 2.0, 3.0]))


def test_permutation_test_detects_the_swept_campaign(swept_sample):
    """The deliberately non-exchangeable synthetic campaign, at n = 120.

    The POWER of this test at smaller n is measured over many draws in
    validation/validate_exchangeability_output.txt; a single draw at n = 40 is
    not reliable and the measured power there is 0.90, not 1.0. At n = 120 both
    statistics reject on every draw measured.
    """
    for name in ("trend", "half_split"):
        result = permutation_test(swept_sample.gaps, name, permutations=499, seed=4)
        assert result.rejected is True, f"{name} p = {result.p_value}"
