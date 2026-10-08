"""Tests for the sample-size planner, including hand-calculated known answers."""

from __future__ import annotations

import math

import pytest

from rareverify.intervals import zero_failure_upper
from rareverify.planner import (
    detection_probability,
    expected_violations,
    plan_campaign,
    probability_of_zero_violations,
    samples_for_relative_width,
    samples_for_zero_failure_bound,
)


def test_zero_failure_sample_size_hand_calculation():
    """p_target = 1e-3 at 95 % confidence.

    Hand calculation:
        ln(0.05)       = -2.99573227355399
        ln(1 - 1e-3)   = -0.00100050033358
        ratio          =  2994.2335...
        ceil           =  2995
    """
    assert samples_for_zero_failure_bound(1e-3, 0.95) == 2995


def test_zero_failure_sample_size_hand_calculation_1e4():
    """p_target = 1e-4 at 95 % confidence.

    ln(0.05) / ln(1 - 1e-4) = -2.99573227355399 / -1.00005000333e-4
                            = 29955.8...  -> 29956
    """
    assert samples_for_zero_failure_bound(1e-4, 0.95) == 29956


def test_zero_failure_sample_size_hand_calculation_99_percent():
    """p_target = 1e-3 at 99 % confidence.

    ln(0.01) / ln(1 - 1e-3) = -4.60517018599 / -0.00100050033358
                            = 4602.87...  -> 4603
    """
    assert samples_for_zero_failure_bound(1e-3, 0.99) == 4603


def test_returned_sample_size_actually_meets_the_bound_and_is_minimal():
    for p_target in (1e-2, 1e-3, 1e-4, 3e-5):
        for confidence in (0.90, 0.95, 0.99):
            n = samples_for_zero_failure_bound(p_target, confidence)
            assert zero_failure_upper(n, confidence, side="upper") <= p_target
            assert zero_failure_upper(n - 1, confidence, side="upper") > p_target


def test_wilson_zero_failure_sample_size_solves_its_own_formula():
    for p_target in (1e-2, 1e-3, 1e-4):
        n = samples_for_zero_failure_bound(p_target, 0.95, method="wilson")
        assert zero_failure_upper(n, 0.95, method="wilson", side="upper") <= p_target
        assert (
            zero_failure_upper(n - 1, 0.95, method="wilson", side="upper") > p_target
        )


def test_probability_of_zero_violations_equals_alpha_at_the_planned_size():
    """A consistency identity: by construction P(k=0) at p_target is about alpha."""
    for p_target in (1e-2, 1e-3, 1e-4):
        n = samples_for_zero_failure_bound(p_target, 0.95)
        assert probability_of_zero_violations(n, p_target) == pytest.approx(
            0.05, abs=2e-3
        )


def test_detection_probability_and_expected_violations():
    assert expected_violations(1000, 1e-3) == pytest.approx(1.0)
    assert detection_probability(1000, 1e-3) == pytest.approx(
        1.0 - math.exp(1000 * math.log1p(-1e-3))
    )
    assert detection_probability(1000, 1e-3) == pytest.approx(0.632305, abs=1e-6)
    assert probability_of_zero_violations(1, 0.25) == pytest.approx(0.75)


def test_relative_width_plan_satisfies_its_predicate():
    for method in ("clopper-pearson", "wilson"):
        n = samples_for_relative_width(1e-3, 0.5, 0.95, method=method)
        k = round(n * 1e-3)
        from rareverify.intervals import proportion_interval

        interval = proportion_interval(k, n, confidence=0.95, method=method)
        assert interval.width <= 0.5 * 1e-3


def test_relative_width_is_monotone_in_the_requested_width():
    tight = samples_for_relative_width(1e-3, 0.25)
    loose = samples_for_relative_width(1e-3, 1.0)
    assert tight > loose


def test_relative_width_refuses_an_impossible_budget():
    with pytest.raises(ValueError, match="max_samples"):
        samples_for_relative_width(1e-8, 0.01, max_samples=1000)


def test_plan_campaign_fields_are_consistent():
    plan = plan_campaign(1e-4, confidence=0.95, relative_width=0.5)
    assert plan.n_demonstration == 29956
    assert plan.zero_failure_bound <= 1e-4
    assert plan.expected_violations_demonstration == pytest.approx(
        plan.n_demonstration * 1e-4
    )
    assert plan.n_estimation > plan.n_demonstration
    assert "runs to demonstrate" in plan.describe()


@pytest.mark.parametrize("bad", [0.0, 1.0, -1e-3, 1.5, math.nan])
def test_invalid_probabilities_raise(bad):
    with pytest.raises(ValueError):
        samples_for_zero_failure_bound(bad, 0.95)
    with pytest.raises(ValueError):
        samples_for_zero_failure_bound(1e-3, bad)


def test_invalid_arguments_raise():
    with pytest.raises(ValueError):
        samples_for_zero_failure_bound(1e-3, method="jeffreys")
    with pytest.raises(ValueError):
        samples_for_relative_width(1e-3, relative_width=0.0)
    with pytest.raises(ValueError):
        samples_for_relative_width(1e-3, back_scan=-1)
    with pytest.raises(ValueError):
        samples_for_relative_width(1e-3, max_samples=0)
    with pytest.raises(ValueError):
        expected_violations(0, 1e-3)
    with pytest.raises(ValueError):
        probability_of_zero_violations(0, 1e-3)
