"""Monte Carlo reference estimators and their intervals.

Sample counts here are small on purpose; the convergence evidence lives in
``validation/validate_montecarlo.py``. These tests check determinism, interval
behaviour, the ordering relations between the three estimators and input
validation.
"""

from __future__ import annotations

import numpy as np
import pytest

from conjprob.analytic import pc_foster
from conjprob.geometry import encounter_plane_basis, principal_frame, project_covariance
from conjprob.montecarlo import (
    clopper_pearson,
    pc_monte_carlo,
    pc_monte_carlo_curved,
    pc_monte_carlo_window,
)

COVARIANCE = np.diag([54.0**2, 300.0**2, 80.0**2])
MISS = np.array([120.0, 30.0, 0.0])
VELOCITY = np.array([0.0, 0.0, 5740.0])
RADIUS = 12.0


def _foster_reference() -> float:
    basis = encounter_plane_basis(VELOCITY)
    frame = principal_frame(project_covariance(COVARIANCE, basis), basis.T @ MISS)
    return pc_foster(frame.sigma_x_m, frame.sigma_y_m, frame.x_bar_m, frame.y_bar_m, RADIUS)


def test_clopper_pearson_hand_checkable_edges():
    low, high = clopper_pearson(0, 100, 0.95)
    assert low == 0.0
    # With no successes the upper limit is 1 - (alpha/2)^(1/n) = 1 - 0.025^0.01.
    assert high == pytest.approx(1.0 - 0.025 ** (1.0 / 100.0), rel=1e-9)
    low, high = clopper_pearson(100, 100, 0.95)
    assert high == 1.0
    assert low == pytest.approx(0.025 ** (1.0 / 100.0), rel=1e-9)


def test_clopper_pearson_brackets_the_point_estimate():
    low, high = clopper_pearson(37, 1000, 0.95)
    assert low < 0.037 < high


def test_clopper_pearson_narrows_with_more_samples():
    narrow = clopper_pearson(3700, 100_000)
    wide = clopper_pearson(37, 1000)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


@pytest.mark.parametrize(
    "hits, n, confidence, message",
    [
        (1, 0, 0.95, "n_samples"),
        (-1, 10, 0.95, "hits"),
        (11, 10, 0.95, "hits"),
        (1, 10, 1.5, "confidence"),
    ],
)
def test_clopper_pearson_rejects_bad_input(hits, n, confidence, message):
    with pytest.raises(ValueError, match=message):
        clopper_pearson(hits, n, confidence)


def test_monte_carlo_is_deterministic_for_a_seed():
    first = pc_monte_carlo(COVARIANCE, MISS, VELOCITY, RADIUS, n_samples=200_000, seed=11)
    second = pc_monte_carlo(COVARIANCE, MISS, VELOCITY, RADIUS, n_samples=200_000, seed=11)
    assert first.hits == second.hits
    assert first.pc == second.pc


def test_chunk_size_changes_the_draw_but_not_the_conclusion():
    # The chunk size changes how the generator is consumed, so the hit count is
    # not reproducible across chunk sizes and this is documented rather than
    # asserted away. What must hold is that the two estimates are statistically
    # consistent: each interval contains the other point estimate.
    one = pc_monte_carlo(
        COVARIANCE, MISS, VELOCITY, RADIUS, n_samples=1_000_000, seed=5, chunk=1_000_000
    )
    many = pc_monte_carlo(
        COVARIANCE, MISS, VELOCITY, RADIUS, n_samples=1_000_000, seed=5, chunk=50_000
    )
    assert one.ci_low <= many.pc <= one.ci_high
    assert many.ci_low <= one.pc <= many.ci_high


def test_monte_carlo_interval_contains_the_analytic_value():
    reference = _foster_reference()
    result = pc_monte_carlo(COVARIANCE, MISS, VELOCITY, RADIUS, n_samples=2_000_000, seed=3)
    assert result.ci_low <= reference <= result.ci_high
    assert result.relative_ci_width < 0.25


def test_monte_carlo_result_fields_are_consistent():
    result = pc_monte_carlo(COVARIANCE, MISS, VELOCITY, RADIUS, n_samples=200_000, seed=2)
    assert result.pc == pytest.approx(result.hits / result.n_samples, rel=1e-15)
    assert result.ci_width == pytest.approx(result.ci_high - result.ci_low, rel=1e-15)
    assert result.seed == 2


def test_windowed_estimate_never_exceeds_the_unbounded_one():
    unbounded = pc_monte_carlo(
        COVARIANCE, MISS, VELOCITY, RADIUS, n_samples=400_000, seed=8
    )
    windowed = pc_monte_carlo_window(
        COVARIANCE, MISS, VELOCITY, RADIUS, 0.002, n_samples=400_000, seed=8
    )
    assert windowed.hits <= unbounded.hits


def test_windowed_estimate_converges_to_the_unbounded_one_for_a_long_window():
    unbounded = pc_monte_carlo(
        COVARIANCE, MISS, VELOCITY, RADIUS, n_samples=400_000, seed=9
    )
    windowed = pc_monte_carlo_window(
        COVARIANCE, MISS, VELOCITY, RADIUS, 100.0, n_samples=400_000, seed=9
    )
    assert windowed.hits == unbounded.hits


def test_curved_estimator_with_zero_acceleration_matches_the_windowed_one():
    # Same seed, same offsets. The curved estimator discretises time and the
    # windowed one does not, so the curved hit count can only be lower or equal.
    windowed = pc_monte_carlo_window(
        COVARIANCE, MISS, VELOCITY, RADIUS, 0.5, n_samples=40_000, seed=4
    )
    curved = pc_monte_carlo_curved(
        COVARIANCE,
        MISS,
        VELOCITY,
        RADIUS,
        0.5,
        [0.0, 0.0, 0.0],
        n_samples=40_000,
        seed=4,
        time_nodes=257,
        refinements=8,
    )
    assert curved.hits <= windowed.hits
    assert curved.pc == pytest.approx(windowed.pc, rel=0.15, abs=1e-6)


def test_curved_estimator_grid_refinement_can_only_find_more_hits():
    coarse = pc_monte_carlo_curved(
        COVARIANCE, MISS, VELOCITY, RADIUS, 0.5, [0.0, 0.0, 1e-3],
        n_samples=20_000, seed=6, time_nodes=9, refinements=0,
    )
    fine = pc_monte_carlo_curved(
        COVARIANCE, MISS, VELOCITY, RADIUS, 0.5, [0.0, 0.0, 1e-3],
        n_samples=20_000, seed=6, time_nodes=257, refinements=6,
    )
    assert fine.hits >= coarse.hits


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"n_samples": 0}, "n_samples"),
        ({"chunk": 0}, "chunk"),
    ],
)
def test_monte_carlo_rejects_bad_counts(kwargs, message):
    with pytest.raises(ValueError, match=message):
        pc_monte_carlo(COVARIANCE, MISS, VELOCITY, RADIUS, **kwargs)


def test_monte_carlo_rejects_bad_geometry():
    with pytest.raises(ValueError, match=r"shape \(3, 3\)"):
        pc_monte_carlo(np.eye(2), MISS, VELOCITY, RADIUS, n_samples=10)
    with pytest.raises(ValueError, match="3 components"):
        pc_monte_carlo(COVARIANCE, [1.0, 2.0], VELOCITY, RADIUS, n_samples=10)
    with pytest.raises(ValueError, match="relative velocity is zero"):
        pc_monte_carlo(COVARIANCE, MISS, [0.0, 0.0, 0.0], RADIUS, n_samples=10)
    with pytest.raises(ValueError, match="positive semi-definite"):
        pc_monte_carlo(np.diag([1.0, 1.0, -1.0]), MISS, VELOCITY, RADIUS, n_samples=10)
    with pytest.raises(ValueError, match="non-negative"):
        pc_monte_carlo(COVARIANCE, MISS, VELOCITY, -1.0, n_samples=10)


def test_window_and_curved_reject_bad_windows():
    with pytest.raises(ValueError, match="window_s"):
        pc_monte_carlo_window(COVARIANCE, MISS, VELOCITY, RADIUS, 0.0, n_samples=10)
    with pytest.raises(ValueError, match="window_s"):
        pc_monte_carlo_curved(
            COVARIANCE, MISS, VELOCITY, RADIUS, -1.0, [0.0, 0.0, 0.0], n_samples=10
        )


def test_curved_rejects_bad_acceleration_and_grid():
    with pytest.raises(ValueError, match="3 components"):
        pc_monte_carlo_curved(COVARIANCE, MISS, VELOCITY, RADIUS, 1.0, [0.0], n_samples=10)
    with pytest.raises(ValueError, match="time_nodes"):
        pc_monte_carlo_curved(
            COVARIANCE, MISS, VELOCITY, RADIUS, 1.0, [0.0, 0.0, 0.0],
            n_samples=10, time_nodes=2,
        )
    with pytest.raises(ValueError, match="refinements"):
        pc_monte_carlo_curved(
            COVARIANCE, MISS, VELOCITY, RADIUS, 1.0, [0.0, 0.0, 0.0],
            n_samples=10, refinements=-1,
        )
