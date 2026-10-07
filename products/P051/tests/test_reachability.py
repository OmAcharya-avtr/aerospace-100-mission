"""The exact multi-step predictor: horizon-0 equivalence, monotonicity, soundness."""

from __future__ import annotations

import numpy as np
import pytest

from simplexguard import ExactLeadPredictor


def test_horizon_zero_reproduces_the_one_step_guard_condition(guard, controllers, rng):
    # Inequality (4) at j = 0 must be identical to inequality (2) evaluated on
    # the unsaturated performance input. Disagreement here would mean the
    # unrolling is wrong.
    predictor = ExactLeadPredictor(guard, controllers[1], 0)
    performance = controllers[1]
    disagreements = 0
    for _ in range(3000):
        x = rng.uniform(-1.0, 1.0, 2) * np.array([0.30, 0.50])
        r = np.array([rng.uniform(-0.25, 0.25), 0.0])
        u = performance.unsaturated(x, r)
        exact = guard.condition_margin(x, u)[0] < 0.0
        if exact != predictor.predict(x, r):
            disagreements += 1
    assert disagreements == 0


def test_worst_case_dominates_nominal_on_identical_inputs(guard, controllers, rng):
    # The worst-case offsets are smaller by the non-negative disturbance sum, so
    # anything the nominal variant flags the worst-case variant must flag too.
    xs = rng.uniform(-1.0, 1.0, (1500, 2)) * np.array([0.30, 0.50])
    rs = np.column_stack([rng.uniform(-0.25, 0.25, 1500), np.zeros(1500)])
    for horizon in (1, 5, 10):
        wc = ExactLeadPredictor(guard, controllers[1], horizon, "worst_case")
        nom = ExactLeadPredictor(guard, controllers[1], horizon, "nominal")
        assert np.all(wc.predict_many(xs, rs) >= nom.predict_many(xs, rs))


def test_prediction_is_monotone_in_the_horizon(guard, controllers, rng):
    xs = rng.uniform(-1.0, 1.0, (1000, 2)) * np.array([0.30, 0.50])
    rs = np.column_stack([rng.uniform(-0.25, 0.25, 1000), np.zeros(1000)])
    previous = ExactLeadPredictor(guard, controllers[1], 0).predict_many(xs, rs)
    for horizon in (1, 3, 8):
        current = ExactLeadPredictor(guard, controllers[1], horizon).predict_many(xs, rs)
        assert np.all(current >= previous)
        previous = current


def test_first_step_agrees_with_fires_at(guard, controllers, rng):
    predictor = ExactLeadPredictor(guard, controllers[1], 6)
    for _ in range(300):
        x = rng.uniform(-1.0, 1.0, 2) * np.array([0.30, 0.50])
        r = np.array([rng.uniform(-0.25, 0.25), 0.0])
        first = predictor.first_step(x, r)
        if first < 0:
            assert not any(predictor.fires_at(x, r, j) for j in range(7))
        else:
            assert predictor.fires_at(x, r, first)
            assert not any(predictor.fires_at(x, r, j) for j in range(first))


def test_vectorised_matches_scalar(guard, controllers, rng):
    predictor = ExactLeadPredictor(guard, controllers[1], 4)
    xs = rng.uniform(-1.0, 1.0, (200, 2)) * np.array([0.30, 0.50])
    rs = np.column_stack([rng.uniform(-0.25, 0.25, 200), np.zeros(200)])
    batch = predictor.predict_many(xs, rs)
    scalar = np.array([predictor.predict(x, r) for x, r in zip(xs, rs, strict=True)])
    assert np.array_equal(batch, scalar)
    firsts = predictor.first_step_many(xs, rs)
    scalar_first = np.array(
        [predictor.first_step(x, r) for x, r in zip(xs, rs, strict=True)]
    )
    assert np.array_equal(firsts, scalar_first)


def test_score_is_monotone_decreasing_in_the_first_firing_step(guard, controllers, rng):
    predictor = ExactLeadPredictor(guard, controllers[1], 8)
    xs = rng.uniform(-1.0, 1.0, (400, 2)) * np.array([0.30, 0.50])
    rs = np.column_stack([rng.uniform(-0.25, 0.25, 400), np.zeros(400)])
    firsts = predictor.first_step_many(xs, rs)
    scores = predictor.score_many(xs, rs)
    assert np.all(scores >= 0.0)
    assert np.all(scores <= 1.0)
    assert np.all(scores[firsts < 0] == 0.0)
    fired = firsts >= 0
    # Earlier predicted firing must score at least as high.
    order = np.argsort(firsts[fired])
    assert np.all(np.diff(scores[fired][order]) <= 1e-12)


def test_a_state_deep_inside_the_eroded_set_never_fires(guard, controllers):
    predictor = ExactLeadPredictor(guard, controllers[1], 10)
    assert not predictor.predict(np.zeros(2), np.zeros(2))
    assert predictor.first_step(np.zeros(2), np.zeros(2)) == -1


def test_predictor_validation(guard, controllers):
    with pytest.raises(ValueError, match="horizon must be non-negative"):
        ExactLeadPredictor(guard, controllers[1], -1)
    with pytest.raises(ValueError, match="unknown mode"):
        ExactLeadPredictor(guard, controllers[1], 2, "optimistic")
    predictor = ExactLeadPredictor(guard, controllers[1], 2)
    with pytest.raises(ValueError, match="step must be in"):
        predictor.fires_at(np.zeros(2), np.zeros(2), 3)
    with pytest.raises(ValueError, match="must have length 2"):
        predictor.fires_at(np.zeros(3), np.zeros(2), 0)
    with pytest.raises(ValueError, match="references"):
        predictor.predict_many(np.zeros((3, 2)), np.zeros((2, 2)))
    with pytest.raises(ValueError, match="width 2"):
        predictor.predict_many(np.zeros((3, 3)), np.zeros((3, 3)))
