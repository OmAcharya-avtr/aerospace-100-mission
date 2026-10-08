"""Tests for the crude Monte-Carlo estimator."""

from __future__ import annotations

import math

import numpy as np
import pytest

from rareverify.estimate import binomial_interval, counting_noise_floor, interval_for
from rareverify.limitstates import LinearGaussianLimitState
from rareverify.montecarlo import crude_monte_carlo, required_samples_for_cov


def test_required_samples_hand_calculation():
    """n = (1 - p) / (p c^2); at p = 1e-6, c = 0.1 this is 9.99999e7 -> 99999900."""
    assert required_samples_for_cov(1e-6, 0.1) == 99_999_900
    assert required_samples_for_cov(1e-4, 0.1) == 999_900
    assert required_samples_for_cov(0.5, 0.1) == 100


def test_crude_estimate_matches_reference_within_four_noise_floors():
    state = LinearGaussianLimitState(beta=2.5, dimension=2)
    reference = state.analytic_probability()
    estimate = crude_monte_carlo(state, 300_000, rng=np.random.default_rng(5))
    floor = counting_noise_floor(reference, 300_000)
    assert abs(estimate.estimate - reference) < 4.0 * floor
    assert estimate.is_binomial is True
    assert estimate.true_evaluations == 300_000
    assert estimate.effective_sample_size == estimate.n_failures


def test_crude_standard_error_is_the_binomial_plug_in():
    state = LinearGaussianLimitState(beta=2.0, dimension=2)
    estimate = crude_monte_carlo(state, 50_000, rng=np.random.default_rng(2))
    p_hat = estimate.estimate
    assert estimate.standard_error == pytest.approx(
        math.sqrt(p_hat * (1 - p_hat) / 50_000), rel=1e-12
    )
    assert estimate.standard_error_kind == "binomial-plugin"


def test_batching_does_not_change_the_result_for_a_given_seed_count():
    """Batch size changes the draw order, not the statistical properties."""
    state = LinearGaussianLimitState(beta=1.5, dimension=2)
    small = crude_monte_carlo(state, 60_000, rng=np.random.default_rng(1), batch_size=1000)
    large = crude_monte_carlo(
        state, 60_000, rng=np.random.default_rng(1), batch_size=60_000
    )
    reference = state.analytic_probability()
    floor = counting_noise_floor(reference, 60_000)
    assert abs(small.estimate - reference) < 4.0 * floor
    assert abs(large.estimate - reference) < 4.0 * floor
    assert small.n_samples == large.n_samples == 60_000


def test_zero_failure_run_gives_a_zero_estimate_and_a_valid_interval():
    state = LinearGaussianLimitState(beta=6.0, dimension=2)
    estimate = crude_monte_carlo(state, 20_000, rng=np.random.default_rng(1))
    assert estimate.n_failures == 0
    assert estimate.estimate == 0.0
    assert estimate.standard_error == 0.0
    assert math.isinf(estimate.coefficient_of_variation)
    interval = binomial_interval(estimate, side="upper")
    assert interval.lower == 0.0
    assert interval.upper == pytest.approx(1.0 - 0.05 ** (1 / 20_000), rel=1e-12)


def test_interval_for_a_crude_run_is_binomial():
    state = LinearGaussianLimitState(beta=2.0, dimension=2)
    estimate = crude_monte_carlo(state, 20_000, rng=np.random.default_rng(1))
    interval = interval_for(estimate)
    assert interval.kind == "clopper-pearson"
    assert "binomial count" in interval.rationale
    assert interval.lower <= estimate.estimate <= interval.upper


@pytest.mark.parametrize(("n", "batch"), [(0, 100), (-1, 100), (10, 0)])
def test_invalid_sample_counts_raise(n, batch):
    with pytest.raises(ValueError):
        crude_monte_carlo(
            LinearGaussianLimitState(), n, rng=np.random.default_rng(0), batch_size=batch
        )


@pytest.mark.parametrize(("p", "cov"), [(0.0, 0.1), (1.0, 0.1), (0.1, 0.0), (0.1, -1.0)])
def test_required_samples_invalid_arguments_raise(p, cov):
    with pytest.raises(ValueError):
        required_samples_for_cov(p, cov)


def test_counting_noise_floor_validation():
    assert counting_noise_floor(1e-4, 10_000) == pytest.approx(
        math.sqrt(1e-4 * (1 - 1e-4) / 10_000), rel=1e-15
    )
    with pytest.raises(ValueError):
        counting_noise_floor(1e-4, 0)
    with pytest.raises(ValueError):
        counting_noise_floor(1.5, 10)
