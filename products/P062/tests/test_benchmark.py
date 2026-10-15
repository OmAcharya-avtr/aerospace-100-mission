"""Held-out benchmark tests."""

from __future__ import annotations

import numpy as np
import pytest

from twincalib.benchmark import coverage_fraction, run_benchmark
from twincalib.scenarios import design


def test_coverage_fraction_known_answers():
    # Hand answer: three errors of 0.1, 0.5, 2.0 against std 1.0 at z = 1.0
    # puts two inside, so the fraction is 2/3.
    err = np.array([0.1, 0.5, 2.0])
    std = np.ones(3)
    assert coverage_fraction(err, std, z=1.0) == pytest.approx(2.0 / 3.0)
    assert coverage_fraction(err, std, z=3.0) == pytest.approx(1.0)
    assert coverage_fraction(err, std, z=0.05) == pytest.approx(0.0)


def test_coverage_fraction_is_monotone_in_z():
    rng = np.random.default_rng(0)
    err = rng.normal(0.0, 1.0, 500)
    std = np.ones(500)
    values = [coverage_fraction(err, std, z=z) for z in (0.5, 1.0, 2.0, 3.0)]
    assert all(a <= b for a, b in zip(values[:-1], values[1:], strict=True))


def test_coverage_fraction_on_gaussian_errors_is_near_nominal():
    rng = np.random.default_rng(1)
    err = rng.normal(0.0, 1.0, 20000)
    assert coverage_fraction(err, np.ones(20000)) == pytest.approx(0.95, abs=0.01)
    assert coverage_fraction(err, np.ones(20000), z=1.0) == pytest.approx(
        0.6827, abs=0.015
    )


def test_coverage_fraction_validation():
    with pytest.raises(ValueError, match="same shape"):
        coverage_fraction(np.ones(3), np.ones(4))
    with pytest.raises(ValueError, match="at least one sample"):
        coverage_fraction(np.array([]), np.array([]))
    with pytest.raises(ValueError, match="std"):
        coverage_fraction(np.ones(3), np.zeros(3))
    with pytest.raises(ValueError, match="z"):
        coverage_fraction(np.ones(3), np.ones(3), z=-1.0)


@pytest.fixture(scope="module")
def small_benchmark():
    """A deliberately small run: 2 training and 2 held-out trajectories."""
    return run_benchmark(
        design("baseline-eclipse"),
        n_train=2,
        n_test=2,
        max_training_points=150,
        strong_cuts=(1e-2,),
    )


def test_benchmark_reports_every_trajectory(small_benchmark):
    assert len(small_benchmark.train_scores) == 2
    assert len(small_benchmark.test_scores) == 2
    phases = [s.phase for s in small_benchmark.test_scores]
    assert all(p >= 0.5 for p in phases)
    assert all(s.phase < 0.5 for s in small_benchmark.train_scores)
    seeds = {s.seed for s in small_benchmark.train_scores} | {
        s.seed for s in small_benchmark.test_scores
    }
    assert len(seeds) == 4


def test_improvement_is_consistent_with_the_two_rmse_values(small_benchmark):
    expected = 1.0 - (
        small_benchmark.rmse_combined_pooled / small_benchmark.rmse_physics_pooled
    )
    assert small_benchmark.improvement_pooled == pytest.approx(expected, rel=1e-12)
    for s in small_benchmark.test_scores:
        assert s.improvement == pytest.approx(
            1.0 - s.rmse_combined / s.rmse_physics, rel=1e-12
        )


def test_winner_string_is_one_of_three(small_benchmark):
    assert small_benchmark.winner in ("physics-only", "physics+residual", "tie")


def test_winner_is_consistent_with_the_improvement_and_the_margin(small_benchmark):
    # The verdict is a threshold comparison, so it is checked against the
    # measured improvement rather than asserted as a fixed string: a run whose
    # improvement sits within rounding of the margin may legitimately report a
    # tie.
    imp = small_benchmark.improvement_pooled
    margin = small_benchmark.decision_margin
    if small_benchmark.winner == "physics+residual":
        assert imp > margin - 1e-12
    elif small_benchmark.winner == "physics-only":
        assert imp < -margin + 1e-12
    else:
        assert abs(imp) <= margin + 1e-12


def test_the_discrepancy_signal_is_above_the_noise_on_this_design(small_benchmark):
    assert small_benchmark.discrepancy_rms > small_benchmark.noise_sigma
    assert small_benchmark.noise_sigma == pytest.approx(0.3)


def test_residual_model_helps_where_there_is_real_structure(small_benchmark):
    # A measurement, not a hope: the declared discrepancy is 1.8 times the
    # thermometer noise on this design, so a working residual model must reduce
    # the held-out error. The size of the reduction is reported in
    # validation/validate_residual_output.txt and is not asserted here.
    assert small_benchmark.improvement_pooled > 0.05


def test_coverage_is_reported_and_plausible(small_benchmark):
    assert 0.5 < small_benchmark.coverage_95_pooled <= 1.0
    assert 0.3 < small_benchmark.coverage_68_pooled <= 1.0
    assert small_benchmark.coverage_95_pooled >= small_benchmark.coverage_68_pooled


def test_split_and_drift_are_attached(small_benchmark):
    assert set(small_benchmark.splits) == {1e-2}
    split = small_benchmark.splits[1e-2]
    assert split.strong_fraction + split.weak_fraction + split.orthogonal_fraction == (
        pytest.approx(1.0, abs=1e-10)
    )
    assert small_benchmark.drift.theta_true is not None
    assert small_benchmark.drift.log_shift.shape == (6,)


def test_summary_text_contains_the_verdict_and_the_numbers(small_benchmark):
    text = small_benchmark.summary()
    assert "VERDICT" in text
    assert small_benchmark.winner in text
    assert "physics-only pooled test RMSE" in text
    assert "coverage" in text


def test_training_counts_are_recorded(small_benchmark):
    assert small_benchmark.n_training_used <= small_benchmark.n_training_samples
    assert small_benchmark.n_training_used == 150


def test_benchmark_validation():
    d = design("baseline-eclipse")
    with pytest.raises(ValueError, match="at least 1"):
        run_benchmark(d, n_train=0)
    with pytest.raises(ValueError, match="decision_margin"):
        run_benchmark(d, decision_margin=-0.1)


def test_benchmark_is_reproducible():
    kwargs = dict(
        n_train=2, n_test=1, max_training_points=80, strong_cuts=(1e-2,)
    )
    a = run_benchmark(design("lumped-nodes"), **kwargs)
    b = run_benchmark(design("lumped-nodes"), **kwargs)
    assert a.rmse_combined_pooled == pytest.approx(b.rmse_combined_pooled, rel=1e-12)
    assert a.winner == b.winner
