"""The exact switching condition: known answers, exactness, and the dwell logic."""

from __future__ import annotations

import numpy as np
import pytest

from simplexguard import (
    BaselineController,
    Mode,
    Plant,
    SimplexGuard,
    box,
    robust_invariant_set,
)
from simplexguard.polytope import Polytope


def _one_d_guard():
    """A one-dimensional guard whose condition is hand-computable.

    Plant x' = 1.2 x + u + w, |x| <= 1, |u| <= 5, |w| <= 0.05, baseline
    u = -0.7 x. From ``tests/test_invariant.py`` the robust invariant set is
    S = {|x| <= 1}, so the eroded set is S (-) W = {|x| <= 0.95}.
    """
    plant = Plant(
        A=np.array([[1.2]]),
        B=np.array([[1.0]]),
        disturbance=box([0.05]),
        state_constraints=box([1.0]),
        input_constraints=box([5.0]),
    )
    baseline = BaselineController(gain=np.array([[0.7]]), input_set=box([5.0]))
    invariant = robust_invariant_set(plant, baseline).polytope
    return plant, baseline, SimplexGuard(plant, baseline, invariant), invariant


def test_one_dimensional_switching_condition_known_answer():
    # Hand calculation. S = {|x| <= 1}, W = {|w| <= 0.05}, so S (-) W is
    # {|x| <= 0.95}. At x = 0.5 the next nominal state is 1.2*0.5 + u = 0.6 + u,
    # so the condition 0.6 + u in [-0.95, 0.95] is
    #     -1.55 <= u <= 0.35.
    # The margin is min(0.95 - (0.6 + u), 0.95 + (0.6 + u)).
    #   u = 0.30 -> next = 0.90, margin = min(0.05, 1.85) = 0.05  -> admitted
    #   u = 0.35 -> next = 0.95, margin = min(0.00, 1.90) = 0.00  -> admitted
    #   u = 0.40 -> next = 1.00, margin = min(-0.05, 1.95) = -0.05 -> refused
    _, _, guard, _ = _one_d_guard()
    x = np.array([0.5])
    for u, expected_margin, admitted in (
        (0.30, 0.05, True),
        (0.35, 0.00, True),
        (0.40, -0.05, False),
    ):
        margin, _ = guard.condition_margin(x, np.array([u]))
        assert margin == pytest.approx(expected_margin, abs=1e-12)
        assert guard.allows(x, np.array([u])) is admitted


def test_one_dimensional_eroded_set_known_answer():
    _, _, guard, _ = _one_d_guard()
    assert guard.eroded_set.support(np.array([1.0])) == pytest.approx(0.95, abs=1e-12)
    assert guard.disturbance_offsets == pytest.approx(np.array([0.05, 0.05]))


def test_support_function_form_agrees_with_vertex_enumeration(guard, rng, plant):
    # The exactness claim: inequality (2) must agree with brute-force
    # enumeration of the vertices of the box W at every tested state and input.
    disagreements = 0
    for _ in range(2000):
        x = rng.uniform(-1.0, 1.0, 2) * np.array([0.30, 0.50])
        u = rng.uniform(-3.0, 3.0, 1)
        if guard.allows(x, u) != guard.brute_force_allows(x, u):
            disagreements += 1
    assert disagreements == 0


def test_brute_force_needs_a_box_disturbance_set(plant, controllers, invariant_set):
    generic = plant.with_disturbance(Polytope(plant.disturbance.A, plant.disturbance.b))
    g = SimplexGuard(generic, controllers[0], invariant_set)
    with pytest.raises(TypeError, match="box disturbance set"):
        g.brute_force_allows(np.zeros(2), np.zeros(1))


def test_an_inadmissible_input_is_refused(guard):
    # u outside U must be refused regardless of the invariance margin.
    x = np.zeros(2)
    assert not guard.allows(x, np.array([1e6]))
    decision = guard.decide(x, np.array([1e6]))
    assert decision.mode is Mode.BASELINE
    assert not decision.input_admissible


def test_a_state_outside_the_invariant_set_is_reported_as_certificate_lost(
    plant, controllers, invariant_set
):
    guard = SimplexGuard(plant, controllers[0], invariant_set)
    outside = np.array([0.15, 0.45])
    assert not invariant_set.contains(outside)
    decision = guard.decide(outside, np.zeros(1))
    assert decision.certificate_lost
    assert decision.state_margin < 0.0
    assert decision.mode is Mode.BASELINE


def test_decision_records_both_candidate_inputs(guard, controllers):
    performance = controllers[1]
    x = np.array([0.05, 0.02])
    u_perf = performance(x, np.array([0.18, 0.0]))
    decision = guard.decide(x, u_perf)
    assert decision.proposed_input == pytest.approx(u_perf)
    assert decision.baseline_input == pytest.approx(controllers[0](x))
    if decision.mode is Mode.PERFORMANCE:
        assert decision.applied_input == pytest.approx(u_perf)
    else:
        assert decision.applied_input == pytest.approx(decision.baseline_input)


def test_margin_is_reported_even_when_the_performance_input_is_admitted(guard):
    decision = guard.decide(np.zeros(2), np.zeros(1))
    assert decision.mode is Mode.PERFORMANCE
    assert decision.margin > 0.0
    assert decision.condition_holds


def test_minimum_baseline_dwell_holds_the_baseline(plant, controllers, invariant_set):
    guard = SimplexGuard(plant, controllers[0], invariant_set, min_baseline_dwell=4)
    # Force one switch with an enormous proposed input, then propose zero input
    # (which the condition admits at the origin) and check the baseline is held.
    guard.decide(np.zeros(2), np.array([1e6]))
    held = [guard.decide(np.zeros(2), np.zeros(1)) for _ in range(3)]
    assert all(d.mode is Mode.BASELINE for d in held)
    assert all(d.forced_by_dwell for d in held)
    assert all(d.condition_holds for d in held)
    released = guard.decide(np.zeros(2), np.zeros(1))
    assert released.mode is Mode.PERFORMANCE
    assert not released.forced_by_dwell


def test_reset_clears_the_dwell_counter(plant, controllers, invariant_set):
    guard = SimplexGuard(plant, controllers[0], invariant_set, min_baseline_dwell=10)
    guard.decide(np.zeros(2), np.array([1e6]))
    guard.reset()
    assert guard.decide(np.zeros(2), np.zeros(1)).mode is Mode.PERFORMANCE


def test_advance_dwell_false_leaves_the_counter_alone(plant, controllers, invariant_set):
    guard = SimplexGuard(plant, controllers[0], invariant_set, min_baseline_dwell=5)
    probe = guard.decide(np.zeros(2), np.array([1e6]), advance_dwell=False)
    assert probe.mode is Mode.BASELINE
    assert guard.decide(np.zeros(2), np.zeros(1)).mode is Mode.PERFORMANCE


def test_verify_true_rejects_a_set_that_is_not_a_certificate(plant, controllers):
    with pytest.raises(ValueError, match="not a valid certificate"):
        SimplexGuard(plant, controllers[0], plant.state_constraints, verify=True)


def test_verify_true_accepts_the_computed_set(plant, controllers, invariant_set):
    SimplexGuard(plant, controllers[0], invariant_set, verify=True)


def test_guard_constructor_validation(plant, controllers, invariant_set):
    with pytest.raises(ValueError, match="invariant set has dim"):
        SimplexGuard(plant, controllers[0], box([1.0]))
    with pytest.raises(ValueError, match="min_baseline_dwell"):
        SimplexGuard(plant, controllers[0], invariant_set, min_baseline_dwell=0)


def test_decide_validates_shapes(guard):
    with pytest.raises(ValueError, match="x has length"):
        guard.decide(np.zeros(3), np.zeros(1))
    with pytest.raises(ValueError, match="proposed_input has length"):
        guard.decide(np.zeros(2), np.zeros(2))
    with pytest.raises(ValueError, match="non-finite"):
        guard.decide(np.zeros(2), np.array([np.nan]))


def test_mode_str_is_lowercase():
    assert str(Mode.BASELINE) == "baseline"
    assert str(Mode.PERFORMANCE) == "performance"


def test_describe_reports_the_eroded_radius(guard):
    text = guard.describe()
    assert "Chebyshev radius of S-W" in text
    assert "minimum baseline dwell    1 steps" in text


def test_eroded_set_is_strictly_inside_the_invariant_set(guard, invariant_set):
    assert invariant_set.contains_polytope(guard.eroded_set, tol=1e-9)
    assert guard.eroded_set.chebyshev_radius() < invariant_set.chebyshev_radius()
    assert np.all(guard.disturbance_offsets > 0.0)
