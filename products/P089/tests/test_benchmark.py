"""Scoring: the subspace split, equation (24), coverage and the score table."""

from __future__ import annotations

import numpy as np
import pytest

from aligncal import batch, benchmark, dataset, observability


def test_nominal_coverage_values():
    """Standard-normal two-sided coverage, computed from scipy.special.erf at import."""
    assert benchmark.NOMINAL_COVERAGE[1] == pytest.approx(0.6826894921, rel=1e-9, abs=0.0)
    assert benchmark.NOMINAL_COVERAGE[2] == pytest.approx(0.9544997361, rel=1e-9, abs=0.0)
    assert benchmark.NOMINAL_COVERAGE[3] == pytest.approx(0.9973002039, rel=1e-9, abs=0.0)


def test_subspace_split_sums_to_the_error(single_verdict):
    sensor = single_verdict["secondary_1"]
    rng = np.random.default_rng(41)
    for _ in range(20):
        error = rng.normal(size=3) * 1.0e-3
        observable, unobservable = benchmark.subspace_split(error, sensor)
        assert np.max(np.abs(observable + unobservable - error)) < 1.0e-18


def test_subspace_split_is_orthogonal(single_verdict):
    sensor = single_verdict["secondary_1"]
    rng = np.random.default_rng(42)
    for _ in range(20):
        vector = rng.normal(size=3)
        observable, unobservable = benchmark.subspace_split(vector, sensor)
        # The rounding of the projection is governed by the magnitude of the
        # vector being projected, not by the magnitude of the small part of it,
        # so the bound is scaled by |vector|^2. An absolute 1e-15 was the first
        # draft and failed on a vector of norm 2.
        scale = float(np.dot(vector, vector))
        assert abs(float(np.dot(observable, unobservable))) < 1.0e-14 * scale


def test_subspace_split_leaves_nothing_unobservable_on_a_rich_record(rich_verdict):
    sensor = rich_verdict["secondary_1"]
    _, unobservable = benchmark.subspace_split(np.array([1.0, 2.0, 3.0]), sensor)
    assert float(np.linalg.norm(unobservable)) < 1.0e-13


def test_crb_z_scores_are_nan_on_an_unobservable_direction(single_verdict):
    sensor = single_verdict["secondary_1"]
    scores = benchmark.crb_z_scores(np.array([1.0e-4, 2.0e-4, -1.0e-4]), sensor)
    assert np.isnan(scores[2])
    assert np.all(np.isfinite(scores[:2]))


def test_crb_z_scores_are_finite_on_a_rich_record(rich_verdict):
    scores = benchmark.crb_z_scores(np.array([1.0e-5, 0.0, 0.0]), rich_verdict["secondary_1"])
    assert np.all(np.isfinite(scores))


def test_crb_z_score_is_the_component_over_the_bound(rich_verdict):
    sensor = rich_verdict["secondary_1"]
    error = np.array([3.0e-5, -1.0e-5, 2.0e-5])
    scores = benchmark.crb_z_scores(error, sensor)
    for dof in sensor.degrees_of_freedom:
        expected = float(np.dot(error, dof.direction)) / dof.crb_sigma_rad
        assert scores[dof.index] == pytest.approx(expected, rel=1.0e-12, abs=0.0)


def test_batch_z_scores_are_of_order_one_over_many_seeds():
    """Equation (24) on the unbiased batch solution: the distribution must be standard.

    This is a check on the estimator and on the Cramer-Rao bound together, and
    it is the tightest link between the two in the suite. The sample standard
    deviation of 240 draws from a standard normal has a one-sigma spread of
    1/sqrt(480) = 0.046, so the window below is wide enough not to be a seed
    lottery and narrow enough to catch a bound that is wrong by 20 per cent.
    """
    scores: list[float] = []
    for seed in range(80):
        generated = dataset.generate(
            profile="three_axis", samples=900, noise_sigma=2.0e-4, seed=120000 + seed
        )
        verdict = observability.assess(generated.record)
        sensor = verdict["secondary_1"]
        solution = batch.solve(generated.record, verdict, "secondary_1")
        error = solution.theta_rad - generated.truth["secondary_1"]
        scores.extend(float(z) for z in benchmark.crb_z_scores(error, sensor) if np.isfinite(z))
    array = np.array(scores)
    assert array.size == 240
    assert abs(float(np.mean(array))) < 0.25
    assert 0.8 < float(np.std(array)) < 1.25


def test_coverage_hand_computed():
    errors = np.array([0.5, 1.5, 2.5, 3.5])
    sigmas = np.ones(4)
    assert benchmark.coverage(errors, sigmas, k=1.0) == 0.25
    assert benchmark.coverage(errors, sigmas, k=2.0) == 0.5
    assert benchmark.coverage(errors, sigmas, k=4.0) == 1.0


def test_coverage_counts_a_zero_width_interval_as_a_miss():
    assert benchmark.coverage(np.array([1.0]), np.array([0.0])) == 0.0
    assert benchmark.coverage(np.array([0.0]), np.array([0.0])) == 1.0


def test_coverage_counts_an_infinite_sigma_as_a_miss_unless_the_error_is_zero():
    """An infinite interval is the analytic refusal, not a successful prediction."""
    assert benchmark.coverage(np.array([1.0]), np.array([np.inf])) == 0.0


def test_coverage_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="errors and"):
        benchmark.coverage(np.zeros(3), np.zeros(4))


def test_coverage_rejects_empty():
    with pytest.raises(ValueError, match="at least one record"):
        benchmark.coverage(np.zeros(0), np.zeros(0))


def test_score_table_fields():
    sensors = []
    errors = []
    for seed in range(12):
        generated = dataset.generate(samples=400, seed=200 + seed)
        verdict = observability.assess(generated.record)
        sensors.append(verdict["secondary_1"])
        errors.append(np.array([1.0e-4, -2.0e-4, 3.0e-4]))
    table = benchmark.score("probe", np.array(errors), sensors)
    assert table.label == "probe"
    assert table.records == 12
    # Per-component RMSE of a constant error vector, hand-computed in mrad:
    # sqrt((0.1^2 + 0.2^2 + 0.3^2) / 3) = sqrt(0.14/3) = 0.21602469
    assert table.rmse_total_mrad == pytest.approx(0.21602469, rel=1.0e-6, abs=0.0)
    # Median magnitude: sqrt(0.14) = 0.37416574 mrad
    assert table.median_total_mrad == pytest.approx(0.37416574, rel=1.0e-6, abs=0.0)
    assert table.worst_total_mrad == pytest.approx(0.37416574, rel=1.0e-6, abs=0.0)
    assert np.max(np.abs(table.bias_mrad - np.array([0.1, -0.2, 0.3]))) < 1.0e-9


def test_score_observable_and_unobservable_add_in_quadrature():
    sensors = []
    errors = []
    rng = np.random.default_rng(44)
    for seed in range(20):
        generated = dataset.generate(profile="single_axis", samples=400, seed=300 + seed)
        verdict = observability.assess(generated.record)
        sensors.append(verdict["secondary_1"])
        errors.append(rng.normal(size=3) * 1.0e-3)
    table = benchmark.score("probe", np.array(errors), sensors)
    total = np.hypot(table.rmse_observable_mrad, table.rmse_unobservable_mrad)
    assert total == pytest.approx(table.rmse_total_mrad, rel=1.0e-9, abs=0.0)


def test_score_rejects_bad_shapes():
    generated = dataset.generate(samples=200, seed=5)
    sensor = observability.assess(generated.record)["secondary_1"]
    with pytest.raises(ValueError, match=r"shape \(m, 3\)"):
        benchmark.score("x", np.zeros((2, 4)), [sensor, sensor])
    with pytest.raises(ValueError, match="observability verdicts"):
        benchmark.score("x", np.zeros((3, 3)), [sensor])


def test_score_row_and_header_align():
    generated = dataset.generate(samples=200, seed=6)
    sensor = observability.assess(generated.record)["secondary_1"]
    table = benchmark.score("probe", np.zeros((1, 3)), [sensor])
    assert len(table.row()) == len(benchmark.ScoreTable.header())
    assert table.row().startswith("probe")
