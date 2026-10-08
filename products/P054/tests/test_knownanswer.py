"""Tests for the known-answer check harness."""

from __future__ import annotations

import math

import numpy as np
import pytest

from rareverify.knownanswer import check_against_reference
from rareverify.limitstates import (
    LinearGaussianLimitState,
    LognormalRatioLimitState,
    RippledLimitState,
)
from rareverify.montecarlo import crude_monte_carlo
from rareverify.tilting import analytic_mean_shift, importance_sampling


def test_tolerance_basis_is_the_counting_noise_floor_for_a_crude_run():
    """floor = sqrt(p (1-p) / n); at p = 1e-4, n = 1e5 this is 3.16226e-5."""
    state = LinearGaussianLimitState()
    estimate = crude_monte_carlo(state, 100_000, rng=np.random.default_rng(1))
    check = check_against_reference(
        "linear crude", estimate, state.analytic_probability()
    )
    assert check.noise_floor == pytest.approx(3.1622e-5, rel=1e-3)
    assert check.tolerance_basis == pytest.approx(
        max(check.noise_floor, estimate.standard_error), rel=1e-12
    )


def test_a_low_variance_estimator_keeps_the_noise_floor_as_its_basis():
    """The tolerance is never tightened below the counting-noise floor."""
    state = LinearGaussianLimitState()
    estimate = importance_sampling(
        state, analytic_mean_shift(state), 100_000, rng=np.random.default_rng(1)
    )
    check = check_against_reference(
        "linear IS", estimate, state.analytic_probability()
    )
    assert estimate.standard_error < check.noise_floor
    assert check.tolerance_basis == check.noise_floor


def test_all_three_limit_states_pass_at_four_noise_floors():
    for state in (
        LinearGaussianLimitState(),
        LognormalRatioLimitState(),
        RippledLimitState(),
    ):
        reference = state.analytic_probability()
        rng = np.random.default_rng(13)
        for estimate in (
            crude_monte_carlo(state, 200_000, rng=rng),
            importance_sampling(
                state, analytic_mean_shift(state), 200_000, rng=rng
            ),
        ):
            check = check_against_reference(
                f"{state.name} {estimate.method}",
                estimate,
                reference,
                state.reference_kind,
                tolerance_multiples=4.0,
            )
            assert check.passed, check.describe()


def test_a_deliberately_wrong_estimate_fails_the_check():
    state = LinearGaussianLimitState()
    estimate = importance_sampling(
        state, analytic_mean_shift(state), 50_000, rng=np.random.default_rng(1)
    )
    check = check_against_reference(
        "deliberately wrong reference", estimate, 1e-2, "closed-form", 4.0
    )
    assert check.passed is False
    assert "FAIL" in check.describe()


def test_reference_kind_is_carried_through():
    state = RippledLimitState()
    estimate = crude_monte_carlo(state, 20_000, rng=np.random.default_rng(1))
    check = check_against_reference(
        "rippled", estimate, state.analytic_probability(), state.reference_kind
    )
    assert check.reference_kind == "quadrature"
    assert "quadrature" in check.describe()


def test_nan_standard_error_is_treated_as_zero_for_the_basis():
    from rareverify.estimate import RareEventEstimate

    estimate = RareEventEstimate(
        method="subset-simulation",
        estimate=1e-4,
        standard_error=math.nan,
        n_samples=8000,
        true_evaluations=7400,
        n_failures=100,
        is_binomial=False,
        effective_sample_size=math.nan,
        wall_seconds=0.0,
    )
    check = check_against_reference("nan se", estimate, 1e-4)
    assert check.reported_standard_error == 0.0
    assert check.tolerance_basis == check.noise_floor


@pytest.mark.parametrize(("reference", "tolerance"), [(0.0, 4.0), (1.0, 4.0), (1e-4, 0.0)])
def test_invalid_arguments_raise(reference, tolerance):
    state = LinearGaussianLimitState()
    estimate = crude_monte_carlo(state, 1000, rng=np.random.default_rng(1))
    with pytest.raises(ValueError):
        check_against_reference("bad", estimate, reference, "closed-form", tolerance)
