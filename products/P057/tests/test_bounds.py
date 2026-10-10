"""Unit and validation tests for the bounds module."""

from __future__ import annotations

import numpy as np
import pytest

from conformalband.bounds import (
    clopper_pearson,
    effective_sample_size,
    split_conformal_coverage_bound,
)


@pytest.mark.parametrize("n", [9, 19, 50, 100, 500, 2000])
def test_bound_is_between_lower_and_upper(n):
    bound = split_conformal_coverage_bound(n, 0.1)
    assert bound.lower <= bound.exact <= bound.upper + 1e-15


@pytest.mark.parametrize("n", [9, 19, 50, 100, 500, 2000])
def test_conservatism_shrinks_as_one_over_n(n):
    bound = split_conformal_coverage_bound(n, 0.1)
    assert 0.0 <= bound.conservatism <= 1.0 / (n + 1) + 1e-15


def test_conservatism_decreases_with_calibration_size():
    small = split_conformal_coverage_bound(20, 0.1).conservatism
    large = split_conformal_coverage_bound(2000, 0.1).conservatism
    assert large <= small


@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.1, 0.2, 0.5])
def test_rank_over_n_plus_one_is_the_exact_value(alpha):
    bound = split_conformal_coverage_bound(997, alpha)
    assert bound.exact == pytest.approx(bound.rank / 998, rel=1e-15)


def test_minimum_calibration_size_for_alpha_one_tenth():
    # n + 1 >= 1 / alpha means n >= 9 for alpha = 0.1.
    assert split_conformal_coverage_bound(9, 0.1).rank == 9
    with pytest.raises(ValueError, match="too small"):
        split_conformal_coverage_bound(8, 0.1)


def test_minimum_calibration_size_for_alpha_one_hundredth():
    assert split_conformal_coverage_bound(99, 0.01).rank == 99
    with pytest.raises(ValueError, match="too small"):
        split_conformal_coverage_bound(98, 0.01)


def test_error_message_names_the_needed_size():
    with pytest.raises(ValueError, match="n_calibration >= 19"):
        split_conformal_coverage_bound(10, 0.05)


@pytest.mark.parametrize("n", [0, -1, -100])
def test_bound_rejects_non_positive_n(n):
    with pytest.raises(ValueError, match="n_calibration"):
        split_conformal_coverage_bound(n, 0.1)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.2, 1.4])
def test_bound_rejects_bad_alpha(alpha):
    with pytest.raises(ValueError, match="alpha"):
        split_conformal_coverage_bound(100, alpha)


@pytest.mark.parametrize("successes", [0, 1, 500, 999, 1000])
def test_clopper_pearson_is_ordered(successes):
    low, high = clopper_pearson(successes, 1000)
    assert 0.0 <= low <= high <= 1.0


def test_clopper_pearson_narrows_with_more_trials():
    narrow = clopper_pearson(9000, 10_000)
    wide = clopper_pearson(90, 100)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


@pytest.mark.parametrize("confidence", [0.80, 0.90, 0.95, 0.99])
def test_clopper_pearson_widens_with_confidence(confidence):
    low, high = clopper_pearson(900, 1000, confidence)
    reference = clopper_pearson(900, 1000, 0.80)
    assert (high - low) >= (reference[1] - reference[0]) - 1e-12


def test_clopper_pearson_rejects_bad_counts():
    with pytest.raises(ValueError, match="trials"):
        clopper_pearson(1, 0)
    with pytest.raises(ValueError, match="successes"):
        clopper_pearson(11, 10)
    with pytest.raises(ValueError, match="successes"):
        clopper_pearson(-1, 10)


def test_clopper_pearson_rejects_bad_confidence():
    with pytest.raises(ValueError, match="confidence"):
        clopper_pearson(5, 10, 1.0)


@pytest.mark.parametrize("n", [1, 2, 10, 500])
def test_effective_sample_size_of_equal_weights(n):
    assert effective_sample_size(np.ones(n)) == pytest.approx(float(n), rel=1e-13)


def test_effective_sample_size_is_bounded_by_n():
    rng = np.random.default_rng(57701)
    weights = rng.lognormal(0.0, 1.5, 400)
    ess = effective_sample_size(weights)
    assert 1.0 <= ess <= 400.0


def test_effective_sample_size_falls_as_weights_spread():
    rng = np.random.default_rng(57702)
    tight = effective_sample_size(rng.lognormal(0.0, 0.1, 500))
    loose = effective_sample_size(rng.lognormal(0.0, 2.0, 500))
    assert loose < tight


def test_effective_sample_size_is_scale_invariant():
    rng = np.random.default_rng(57703)
    weights = rng.lognormal(0.0, 0.8, 200)
    assert effective_sample_size(weights) == pytest.approx(
        effective_sample_size(1234.5 * weights), rel=1e-12
    )


def test_effective_sample_size_rejects_empty():
    with pytest.raises(ValueError, match="non-empty"):
        effective_sample_size(np.array([]))


def test_effective_sample_size_rejects_negative():
    with pytest.raises(ValueError, match="non-negative"):
        effective_sample_size(np.array([1.0, -1.0]))


def test_effective_sample_size_rejects_all_zero():
    with pytest.raises(ValueError, match="all be zero"):
        effective_sample_size(np.zeros(5))
