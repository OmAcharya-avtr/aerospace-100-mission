"""Poisson-sized campaigns and their standard errors."""

from __future__ import annotations

import numpy as np
import pytest

from bitflipsim.campaign import (
    activation_campaign,
    flux_campaign,
    parameter_campaign,
    single_upset_sites,
    trials_for_standard_error,
)
from bitflipsim.flux import upset_rate


def test_trials_for_standard_error_is_the_closed_form():
    # n = (s/e)**2 = (0.2/0.01)**2 = 400
    assert trials_for_standard_error(0.2, 0.01) == 400
    assert trials_for_standard_error(0.0, 0.01) == 0
    with pytest.raises(ValueError, match="sample_std"):
        trials_for_standard_error(-1.0, 0.01)
    with pytest.raises(ValueError, match="target_standard_error"):
        trials_for_standard_error(0.2, 0.0)


def test_zero_expected_upsets_leaves_the_model_intact(params, problem):
    rng = np.random.default_rng(0)
    result = parameter_campaign(
        params, problem.evaluation.x, problem.evaluation.y, 0.0, 10, rng
    )
    assert result.mean_degradation == 0.0
    # mean of ten identical accuracies differs from the single value by
    # float64 summation round-off (2.2e-16 here), not by any injected effect.
    assert result.mean_accuracy == pytest.approx(result.golden_accuracy, abs=1e-15)
    assert result.mean_accuracy_drop == pytest.approx(0.0, abs=1e-15)
    assert result.degradation_standard_error == 0.0


def test_degradation_rises_with_the_upset_rate(params, problem):
    rng = np.random.default_rng(17)
    low = parameter_campaign(params, problem.evaluation.x, problem.evaluation.y, 1.0, 400, rng)
    high = parameter_campaign(params, problem.evaluation.x, problem.evaluation.y, 20.0, 400, rng)
    # The separation must exceed the combined standard error, or the comparison
    # is not supported by the sample.
    separation = high.mean_degradation - low.mean_degradation
    combined = np.hypot(high.degradation_standard_error, low.degradation_standard_error)
    assert separation > 3.0 * combined, (separation, combined)


def test_drawn_counts_have_the_requested_poisson_mean(params, problem):
    rng = np.random.default_rng(23)
    result = parameter_campaign(params, problem.evaluation.x, problem.evaluation.y, 5.0, 2000, rng)
    # se(mean) = sqrt(5/2000) = 0.05; a 5-sigma window is 5 +- 0.25
    assert abs(result.upset_counts.mean() - 5.0) < 0.25


def test_clamping_reduces_the_mean_degradation(params, problem):
    limit = float(np.abs(params.values).max())
    unmitigated = parameter_campaign(
        params, problem.evaluation.x, problem.evaluation.y, 8.0, 300,
        np.random.default_rng(5),
    )
    mitigated = parameter_campaign(
        params, problem.evaluation.x, problem.evaluation.y, 8.0, 300,
        np.random.default_rng(5), clamp_limit=limit,
    )
    assert mitigated.mean_degradation < unmitigated.mean_degradation


def test_activation_campaign_runs_and_is_bounded(params, problem):
    rng = np.random.default_rng(11)
    result = activation_campaign(
        params, problem.evaluation.x, problem.evaluation.y, 10.0, 60, rng
    )
    assert 0.0 <= result.mean_degradation <= 1.0
    assert result.trials == 60
    assert result.golden_accuracy > 0.5


def test_flux_campaign_matches_the_equivalent_expected_count(params, problem):
    rate = upset_rate(1.0e6, 1.0e-9, params.layout.size * 32)
    exposure = rate.exposure_for_expected_upsets(6.0)
    by_flux = flux_campaign(
        params, problem.evaluation.x, problem.evaluation.y, rate, exposure, 150,
        np.random.default_rng(8),
    )
    by_count = parameter_campaign(
        params, problem.evaluation.x, problem.evaluation.y, 6.0, 150,
        np.random.default_rng(8),
    )
    assert by_flux.expected_upsets == pytest.approx(6.0, rel=1e-9)
    assert by_flux.mean_degradation == pytest.approx(by_count.mean_degradation)


def test_campaign_validation(params, problem):
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="expected_upsets"):
        parameter_campaign(params, problem.evaluation.x, problem.evaluation.y, -1.0, 5, rng)
    with pytest.raises(ValueError, match="trials"):
        parameter_campaign(params, problem.evaluation.x, problem.evaluation.y, 1.0, 0, rng)
    with pytest.raises(ValueError, match="expected_upsets"):
        activation_campaign(params, problem.evaluation.x, problem.evaluation.y, -1.0, 5, rng)
    with pytest.raises(ValueError, match="trials"):
        activation_campaign(params, problem.evaluation.x, problem.evaluation.y, 1.0, 0, rng)


def test_single_upset_sites_enumeration(params):
    everything = single_upset_sites(params)
    assert len(everything) == params.layout.size * 32
    exponent_only = single_upset_sites(params, list(range(23, 31)))
    assert len(exponent_only) == params.layout.size * 8
    with pytest.raises(ValueError, match="outside"):
        single_upset_sites(params, [32])
