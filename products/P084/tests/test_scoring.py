"""Unit and edge-case tests for the coverage-scoring harness."""

from __future__ import annotations

import numpy as np
import pytest

from transfergap import synthetic
from transfergap.bounds import hoeffding_mean_upper, wilks_tolerance_upper
from transfergap.scoring import score_mean_bounds, score_prediction_bounds


def _draw(n: int, seed: int):
    return synthetic.generate(n, "random", seed=seed)


def test_score_mean_bounds_measures_a_rate_and_a_width(truth):
    scores = score_mean_bounds(
        {"hoeffding": lambda s: hoeffding_mean_upper(s.gaps, s.bound_range, 0.95, s.unit)},
        _draw,
        truth.mean,
        n=40,
        repetitions=40,
        seed=1,
    )
    assert len(scores) == 1
    score = scores[0]
    assert score.target == "mean"
    assert 0.0 <= score.coverage <= 1.0
    assert score.declared == pytest.approx(0.95, rel=0.0, abs=1e-15)
    assert score.mean_width > 0.0
    assert score.failures == 0
    assert score.unit == "kN.m"


def test_score_mean_bounds_counts_a_method_that_cannot_run(truth):
    """A method that raises is COUNTED, never silently dropped from the rate.

    An all-zero sample has no variance, so this method's deliberate ValueError
    fires on every repetition; the harness must report 40 failures and a nan
    rate rather than an empty success.
    """

    def always_raises(_sample):
        raise ValueError("deliberate")

    scores = score_mean_bounds(
        {"broken": always_raises}, _draw, truth.mean, n=20, repetitions=40, seed=2
    )
    assert scores[0].failures == 40
    assert np.isnan(scores[0].coverage)


def test_score_mean_bounds_rejects_zero_repetitions(truth):
    with pytest.raises(ValueError, match="at least 1"):
        score_mean_bounds({}, _draw, truth.mean, n=20, repetitions=0, seed=1)


def test_score_prediction_bounds_reports_marginal_and_tolerance_rates(pool):
    def wilks(sample, _conditions):
        return np.full(
            pool.size, wilks_tolerance_upper(sample.gaps, sample.bound_range, 0.95, 0.95).upper
        )

    scores = score_prediction_bounds(
        {"wilks": wilks},
        _draw,
        pool.features,
        pool.gaps,
        n=60,
        repetitions=40,
        seed=3,
        proportion=0.95,
        distribution_free={"wilks": True},
    )
    score = scores[0]
    assert score.target == "quantile"
    assert 0.0 <= score.coverage <= 1.0
    assert score.tolerance_rate is not None
    assert 0.0 <= score.tolerance_rate <= 1.0
    assert score.distribution_free is True


def test_score_prediction_bounds_counts_a_method_that_cannot_run_at_this_n(pool):
    """Wilks at n = 15 raises on every repetition, by equation (7)."""

    def wilks(sample, _conditions):
        return np.full(
            pool.size, wilks_tolerance_upper(sample.gaps, sample.bound_range, 0.95, 0.95).upper
        )

    scores = score_prediction_bounds(
        {"wilks": wilks}, _draw, pool.features, pool.gaps, n=15, repetitions=10, seed=4,
        proportion=0.95,
    )
    assert scores[0].failures == 10
    assert np.isnan(scores[0].coverage)


def test_score_prediction_bounds_accepts_a_scalar_band(pool):
    """A marginal method may return one number; it is broadcast, not rejected."""

    def constant(_sample, _conditions):
        return np.array(1e6)

    scores = score_prediction_bounds(
        {"huge": constant}, _draw, pool.features, pool.gaps, n=20, repetitions=5, seed=5,
        proportion=0.95,
    )
    assert scores[0].coverage == 1.0


def test_score_prediction_bounds_rejects_a_mismatched_pool(pool):
    with pytest.raises(ValueError, match="same rows"):
        score_prediction_bounds(
            {}, _draw, pool.features, pool.gaps[:-1], n=20, repetitions=5, seed=5,
            proportion=0.95,
        )
