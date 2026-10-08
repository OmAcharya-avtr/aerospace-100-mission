"""Regression and benchmark tests: pinned numbers that must not drift.

Every value here was produced by this code in this environment and is pinned so
that a change in behaviour shows up as a test failure rather than as a quietly
different README. Values that depend on a random stream are pinned with the
seed that produced them; values that are deterministic mathematics are pinned to
full precision.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from rareverify.benchmark import replicate, summarise, variance_reduction
from rareverify.intervals import clopper_pearson, exact_coverage, wilson, zero_failure_upper
from rareverify.limitstates import (
    LinearGaussianLimitState,
    LognormalRatioLimitState,
    RippledLimitState,
)
from rareverify.montecarlo import crude_monte_carlo
from rareverify.planner import samples_for_zero_failure_bound
from rareverify.subset import subset_simulation
from rareverify.tilting import (
    analytic_mean_shift,
    find_design_point_radial,
    importance_sampling,
    orthogonal_mean_shift,
)


def test_deterministic_interval_values_are_pinned():
    assert clopper_pearson(0, 100, side="upper").upper == pytest.approx(
        0.029513049607039932, rel=1e-13
    )
    assert clopper_pearson(0, 100).upper == pytest.approx(
        0.03621669264517646, rel=1e-13
    )
    assert wilson(0, 100).upper == pytest.approx(0.03699349820698568, rel=1e-13)
    assert zero_failure_upper(29956, side="upper") == pytest.approx(
        9.999941531979584e-05, rel=1e-13
    )
    assert exact_coverage(50, 0.1, method="clopper-pearson") == pytest.approx(
        0.9703082890880887, rel=1e-12
    )


def test_deterministic_reference_probabilities_are_pinned():
    assert LinearGaussianLimitState(beta=3.719).analytic_probability() == pytest.approx(
        1.0000652593416135e-4, rel=1e-13
    )
    assert LognormalRatioLimitState().analytic_probability() == pytest.approx(
        1.0000646652128005e-4, rel=1e-12
    )
    assert RippledLimitState().analytic_probability() == pytest.approx(
        4.4462124889798390e-4, rel=1e-12
    )
    assert RippledLimitState(
        beta=5.5, amplitude=2.5, frequency=2.0
    ).analytic_probability() == pytest.approx(1.8530815e-4, rel=1e-6)


def test_deterministic_design_points_are_pinned():
    assert samples_for_zero_failure_bound(1e-4) == 29956
    point, converged = find_design_point_radial(
        RippledLimitState().g, 2, n_directions=512, rng=np.random.default_rng(1)
    )
    assert converged
    assert float(np.linalg.norm(point)) == pytest.approx(3.072735, rel=1e-5)
    rough = RippledLimitState(beta=5.5, amplitude=2.5, frequency=2.0)
    point, _ = find_design_point_radial(
        rough.g, 2, n_directions=512, rng=np.random.default_rng(1)
    )
    assert float(np.linalg.norm(point)) == pytest.approx(3.0979, rel=1e-3)


def test_seeded_estimator_values_are_pinned():
    state = LinearGaussianLimitState(beta=3.719, dimension=2)
    crude = crude_monte_carlo(state, 100_000, rng=np.random.default_rng(12345))
    assert crude.n_failures == 10
    assert crude.estimate == pytest.approx(1.0e-4, rel=1e-12)
    tilted = importance_sampling(
        state, analytic_mean_shift(state), 100_000, rng=np.random.default_rng(12345)
    )
    assert tilted.estimate == pytest.approx(1.0060357e-4, rel=1e-6)
    assert tilted.effective_sample_size == pytest.approx(19445.19, rel=1e-5)
    subset = subset_simulation(state, n_per_level=2000, rng=np.random.default_rng(12345))
    assert subset.estimate == pytest.approx(1.345e-4, rel=1e-9)
    assert subset.diagnostics["levels"] == 4.0
    assert subset.true_evaluations == 7400


def test_measured_variance_reduction_is_pinned_within_replication_noise():
    """Benchmark regression. 30 replications at 20000 samples, seeds fixed.

    The empirical standard deviation of 30 replications has a relative
    uncertainty of about 1 / sqrt(2 * 29) = 13 %, so the variance reduction
    factor is pinned to a factor of 1.5 rather than to a digit count.
    """
    state = LinearGaussianLimitState(beta=3.719, dimension=2)
    reference = state.analytic_probability()
    crude = summarise(
        "crude",
        replicate(
            lambda rng: crude_monte_carlo(state, 20_000, rng=rng), 30, seed=2026
        ),
        reference,
    )
    tilted = summarise(
        "analytic-IS",
        replicate(
            lambda rng: importance_sampling(
                state, analytic_mean_shift(state), 20_000, rng=rng
            ),
            30,
            seed=2027,
        ),
        reference,
    )
    bad = summarise(
        "orthogonal=0.5",
        replicate(
            lambda rng: importance_sampling(
                state, orthogonal_mean_shift(state, 0.5), 20_000, rng=rng
            ),
            30,
            seed=2028,
        ),
        reference,
    )
    good = variance_reduction(tilted, crude, reference)
    worse = variance_reduction(bad, crude, reference)
    assert 1500.0 < good.vrf_measured < 5000.0
    assert 0.005 < worse.vrf_measured < 0.5
    assert worse.worse_than_reference is True
    assert worse.worse_in_mse is True


def test_compute_budget_of_a_single_estimator_run():
    """Budget guard: one 200000-sample run on this container.

    Wall-clock timings move 10 to 20 % between runs on this machine, so the
    bound is deliberately loose. It exists to catch an accidental factor of
    ten, not to characterise hardware. The container has 2 cores.
    """
    state = LinearGaussianLimitState(beta=3.719, dimension=2)
    start = time.perf_counter()
    crude_monte_carlo(state, 200_000, rng=np.random.default_rng(1))
    importance_sampling(
        state, analytic_mean_shift(state), 200_000, rng=np.random.default_rng(1)
    )
    subset_simulation(state, n_per_level=2000, rng=np.random.default_rng(1))
    elapsed = time.perf_counter() - start
    assert elapsed < 30.0, f"estimator budget regression: {elapsed:.2f} s"
