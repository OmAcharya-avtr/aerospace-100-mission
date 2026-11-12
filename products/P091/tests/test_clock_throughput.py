"""The two measured relations, and the degenerate case that is a result."""

from __future__ import annotations

import math

import pytest

from thermalbudget.clock_throughput import (
    IDENTIFIABILITY_FRACTION,
    clock_temperature_relation,
    proportionality_exponent,
)
from thermalbudget.provenance import Origin


def test_a_pinned_clock_gives_a_degenerate_relation_not_a_fitted_slope(no_throttle_campaign):
    relation = clock_temperature_relation(
        no_throttle_campaign.soaks() + no_throttle_campaign.bursts()
    )
    assert not relation.identifiable
    assert not relation.throttling_observed
    assert math.isnan(relation.slope_khz_per_c)
    assert relation.relative_clock_range <= IDENTIFIABILITY_FRACTION
    assert "does not throttle measurably" in relation.reason
    assert relation.origin is Origin.SYNTHETIC


def test_a_degenerate_relation_predicts_the_modal_observed_clock(no_throttle_campaign):
    relation = clock_temperature_relation(no_throttle_campaign.soaks())
    clock, degenerate, extrapolating = relation.clock_at(68.0)
    assert degenerate
    assert clock == pytest.approx(relation.modal_clock_khz, rel=0.0, abs=1e-9)
    assert not extrapolating
    # And it says so for a temperature it never saw, rather than silently
    # extrapolating a relation it does not have.
    _, degenerate_hot, extrapolating_hot = relation.clock_at(150.0)
    assert degenerate_hot and extrapolating_hot


def test_a_rolling_off_clock_gives_an_identifiable_relation(throttled_campaign):
    relation = clock_temperature_relation(
        throttled_campaign.soaks() + throttled_campaign.bursts()
    )
    assert relation.identifiable
    assert relation.throttling_observed
    assert relation.slope_khz_per_c < 0.0
    low, high = relation.slope_ci_khz_per_c
    assert low < relation.slope_khz_per_c < high
    assert not (low <= 0.0 <= high)


def test_the_predicted_clock_is_clamped_at_the_highest_observed(throttled_campaign):
    relation = clock_temperature_relation(
        throttled_campaign.soaks() + throttled_campaign.bursts()
    )
    cold, _, _ = relation.clock_at(0.0)
    assert cold <= relation.clock_range_khz[1]


def test_the_single_line_fit_understates_a_knee_and_that_is_documented(
    throttled_campaign, parameters
):
    # The fixture's true post-knee slope is 42000 kHz/degC. Pooling samples from
    # below the knee, where the slope is zero, gives a smaller magnitude. The
    # relation is reported as the average gradient over the measured range, which
    # is what it is.
    relation = clock_temperature_relation(
        throttled_campaign.soaks() + throttled_campaign.bursts()
    )
    assert abs(relation.slope_khz_per_c) < parameters.throttle_slope_khz_per_c
    assert abs(relation.slope_khz_per_c) > 0.2 * parameters.throttle_slope_khz_per_c


def test_relation_refuses_an_empty_input_set(no_throttle_campaign):
    with pytest.raises(ValueError, match="empty input set"):
        clock_temperature_relation([])


def test_relation_refuses_too_few_samples(no_throttle_campaign):
    with pytest.raises(ValueError, match="fewer than the 60 required"):
        clock_temperature_relation([no_throttle_campaign["burst_1.00_a"]])


def test_the_exponent_is_not_identifiable_without_clock_variation(no_throttle_campaign):
    exponent = proportionality_exponent(
        no_throttle_campaign.soaks(1.0) + no_throttle_campaign.bursts(), duty=1.0
    )
    assert not exponent.identifiable
    assert math.isnan(exponent.exponent)  # NOT 1.0: an assumption is not a measurement
    assert "ASSUMED" in exponent.reason


def test_the_exponent_is_near_one_when_the_clock_does_vary(throttled_campaign):
    exponent = proportionality_exponent(
        throttled_campaign.soaks(1.0) + throttled_campaign.bursts(), duty=1.0
    )
    assert exponent.identifiable
    # The fixture generates throughput exactly proportional to clock at fixed
    # duty, so this measures that the estimator recovers the exponent it was
    # given. It is not evidence about the real payload.
    assert exponent.exponent == pytest.approx(1.0, rel=0.0, abs=0.05)


def test_the_exponent_refuses_a_duty_with_no_run(no_throttle_campaign):
    with pytest.raises(ValueError, match="no trace at commanded duty"):
        proportionality_exponent(no_throttle_campaign.soaks(), duty=0.9)
