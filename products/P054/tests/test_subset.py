"""Tests for subset simulation."""

from __future__ import annotations

import math

import numpy as np
import pytest

from rareverify.estimate import interval_for
from rareverify.limitstates import (
    LinearGaussianLimitState,
    LognormalRatioLimitState,
    RippledLimitState,
)
from rareverify.subset import subset_simulation


def test_subset_simulation_recovers_a_known_probability_on_average():
    """Mean of 20 independent runs against Phi(-3.719) = 1.000065e-4.

    Subset simulation is biased at finite n_per_level, so the check is on the
    mean of replications against the reference, with a tolerance set by the
    replication standard error rather than by a single run.
    """
    state = LinearGaussianLimitState(beta=3.719, dimension=2)
    reference = state.analytic_probability()
    estimates = np.array(
        [
            subset_simulation(
                state, n_per_level=2000, rng=np.random.default_rng([99, i])
            ).estimate
            for i in range(20)
        ]
    )
    standard_error = estimates.std(ddof=1) / math.sqrt(len(estimates))
    assert abs(estimates.mean() - reference) < 4.0 * standard_error


def test_subset_simulation_reaches_a_very_small_probability():
    """beta = 5.5 gives P = 1.8990e-8, far out of reach of a crude run."""
    state = LinearGaussianLimitState(beta=5.5, dimension=2)
    reference = state.analytic_probability()
    estimate = subset_simulation(
        state, n_per_level=4000, rng=np.random.default_rng(5)
    )
    assert reference == pytest.approx(1.898956e-8, rel=1e-5)
    assert 0.2 * reference < estimate.estimate < 5.0 * reference
    assert estimate.diagnostics["levels"] >= 6


def test_subset_simulation_runs_on_every_limit_state():
    for state in (
        LinearGaussianLimitState(),
        LognormalRatioLimitState(),
        RippledLimitState(),
    ):
        estimate = subset_simulation(
            state, n_per_level=1000, rng=np.random.default_rng(3)
        )
        assert estimate.estimate > 0.0
        assert estimate.is_binomial is False
        assert estimate.standard_error_kind == "subset-independence-lower-bound"
        assert 0.0 < estimate.diagnostics["mean_acceptance_rate"] < 1.0


def test_level_bookkeeping_is_consistent():
    state = LinearGaussianLimitState(beta=4.5, dimension=2)
    estimate = subset_simulation(state, n_per_level=2000, rng=np.random.default_rng(2))
    levels = int(estimate.diagnostics["levels"])
    expected = 0.1 ** (levels - 1) * estimate.diagnostics["final_conditional"]
    assert estimate.estimate == pytest.approx(expected, rel=1e-12)
    assert estimate.diagnostics["chain_length"] == 10.0
    assert estimate.diagnostics["n_seeds"] == 200.0
    assert estimate.n_samples == 2000 * levels


def test_evaluation_count_matches_the_chain_arithmetic():
    """n_per_level for level 0, then n_seeds * (chain_length - 1) per later level."""
    state = LinearGaussianLimitState(beta=4.0, dimension=2)
    estimate = subset_simulation(state, n_per_level=1000, rng=np.random.default_rng(1))
    levels = int(estimate.diagnostics["levels"])
    expected = 1000 + (levels - 1) * 100 * 9
    assert estimate.true_evaluations == expected


def test_reported_standard_error_is_declared_as_a_lower_bound():
    state = LinearGaussianLimitState()
    estimate = subset_simulation(state, n_per_level=1000, rng=np.random.default_rng(1))
    interval = interval_for(estimate)
    assert "LOWER BOUND" in interval.rationale
    assert estimate.standard_error == pytest.approx(
        estimate.estimate * estimate.diagnostics["cov_independence_lower_bound"],
        rel=1e-12,
    )


def test_max_levels_exhaustion_raises():
    state = LinearGaussianLimitState(beta=8.0, dimension=2)
    with pytest.raises(RuntimeError, match="did not reach"):
        subset_simulation(
            state, n_per_level=1000, max_levels=2, rng=np.random.default_rng(1)
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_per_level": 5},
        {"p0": 0.0},
        {"p0": 1.0},
        {"max_levels": 0},
        {"proposal_std": 0.0},
        {"n_per_level": 1000, "p0": 0.33},
        {"n_per_level": 1000, "p0": 0.0003},
    ],
)
def test_invalid_arguments_raise(kwargs):
    with pytest.raises(ValueError):
        subset_simulation(LinearGaussianLimitState(), **kwargs)
