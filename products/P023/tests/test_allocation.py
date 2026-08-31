"""Unit, input-validation, known-answer and edge-case tests for the allocators."""

import numpy as np
import pytest

from alloclab.allocation import (
    METHODS,
    DEFAULT_TORQUE_TOL,
    allocate,
    is_attainable,
    lp_allocate,
    pseudo_inverse_allocate,
    qp_allocate,
    redistributed_pseudo_inverse_allocate,
    weighted_pseudo_inverse_allocate,
)
from alloclab.dataset import reference_thruster_cluster
from alloclab.effectors import (
    general_effector_set,
    orthogonal_effectors,
    pyramid_reaction_wheels,
)


@pytest.fixture
def triad():
    return orthogonal_effectors(1.0)


@pytest.fixture
def pyramid():
    return pyramid_reaction_wheels(0.1)


@pytest.fixture
def cluster():
    return reference_thruster_cluster(1.0, 0.5)


# --------------------------------------------------------------------------
# Known-answer tests (hand-calculated)
# --------------------------------------------------------------------------


def test_pinv_on_identity_returns_the_torque_itself(triad):
    # B = I so B^+ = I and u = tau exactly.
    res = pseudo_inverse_allocate(triad, [0.3, -0.4, 0.5])
    assert np.allclose(res.commands, [0.3, -0.4, 0.5])
    assert res.status == "exact"
    assert res.feasible


def test_pinv_minimum_norm_on_two_parallel_effectors():
    # Two effectors with identical columns b = (1,0,0). The minimum-2-norm
    # solution of u1 + u2 = 1 is u1 = u2 = 0.5.
    e = general_effector_set([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]], [-2.0, -2.0], [2.0, 2.0])
    res = pseudo_inverse_allocate(e, [1.0, 0.0, 0.0])
    assert np.allclose(res.commands, [0.5, 0.5])


def test_weighted_pinv_shifts_effort_to_the_cheap_effector():
    # min w1 u1^2 + w2 u2^2 s.t. u1 + u2 = 1 gives u_i proportional to 1/w_i:
    # with w = (1, 3), u = (3/4, 1/4).
    e = general_effector_set(
        [[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]], [-2.0, -2.0], [2.0, 2.0]
    )
    res = weighted_pseudo_inverse_allocate(
        e, [1.0, 0.0, 0.0], weights=[1.0, 3.0], u_pref=[0.0, 0.0]
    )
    assert np.allclose(res.commands, [0.75, 0.25])


def test_qp_on_identity_saturates_at_the_bound(triad):
    # Commanding 2.0 N*m about x when the limit is 1.0 leaves 1.0 unmet.
    res = qp_allocate(triad, [2.0, 0.0, 0.0])
    assert np.allclose(res.commands, [1.0, 0.0, 0.0], atol=1e-9)
    assert res.residual_norm == pytest.approx(1.0, abs=1e-6)
    assert res.status == "infeasible"
    assert res.bound_violation == pytest.approx(0.0, abs=1e-12)


def test_pinv_reports_the_bound_violation_it_causes(triad):
    res = pseudo_inverse_allocate(triad, [2.0, 0.0, 0.0])
    assert np.allclose(res.commands, [2.0, 0.0, 0.0])
    assert res.bound_violation == pytest.approx(1.0)
    assert res.status == "infeasible"
    assert "violates command bounds" in res.message


def test_lp_min_control_recovers_the_minimum_thrust_solution(cluster):
    # +x torque of 0.2 N*m on the reference cluster: t1 has column (0.5,0,0)
    # and is the only single thruster giving pure +x, so minimum total thrust
    # is u1 = 0.4 N with everything else off.
    res = lp_allocate(cluster, [0.2, 0.0, 0.0], objective="min_control", u_pref=np.zeros(8))
    assert res.feasible
    assert res.commands[0] == pytest.approx(0.4, abs=1e-9)
    assert np.allclose(res.commands[1:], 0.0, atol=1e-9)


def test_qp_matches_hand_solution_for_a_two_effector_split():
    # Two orthogonal effectors, box [0, 1], u_pref = 0, gamma large:
    # tau = (0.3, 0.4, 0) forces u = (0.3, 0.4) uniquely.
    e = general_effector_set(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], [0.0, 0.0], [1.0, 1.0]
    )
    res = qp_allocate(e, [0.3, 0.4, 0.0], u_pref=[0.0, 0.0])
    assert np.allclose(res.commands, [0.3, 0.4], atol=1e-9)


# --------------------------------------------------------------------------
# Every bounded method meets an attainable command exactly
# --------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["lp", "qp", "rpi"])
def test_bounded_methods_meet_an_interior_command(pyramid, method):
    tau = np.array([0.02, -0.015, 0.03])
    res = allocate(pyramid, tau, method=method)
    assert res.feasible, res.message
    assert res.residual_norm <= DEFAULT_TORQUE_TOL
    assert res.bound_violation == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("method", ["lp", "qp"])
def test_bounded_methods_respect_bounds_outside_the_ams(cluster, method):
    huge = np.array([50.0, -30.0, 20.0])
    res = allocate(cluster, huge, method=method)
    assert res.bound_violation == pytest.approx(0.0, abs=1e-9)
    assert res.status == "infeasible"
    assert not res.feasible


def test_lp_solvers_agree(pyramid):
    tau = np.array([0.03, 0.01, -0.02])
    a = lp_allocate(pyramid, tau, solver="highs")
    b = lp_allocate(pyramid, tau, solver="pulp")
    # The 1-norm optimum is a face, so the vertices can differ; the achieved
    # torque must not.
    assert np.allclose(a.achieved_torque, b.achieved_torque, atol=1e-7)


def test_qp_gamma_controls_the_residual(pyramid):
    tau = np.array([0.03, 0.01, -0.02])
    loose = qp_allocate(pyramid, tau, gamma=1e4)
    tight = qp_allocate(pyramid, tau, gamma=1e12)
    assert tight.residual_norm < loose.residual_norm
    assert loose.residual_norm > DEFAULT_TORQUE_TOL


def test_qp_control_weight_biases_away_from_an_expensive_effector(pyramid):
    tau = np.array([0.0, 0.0, -0.05])
    cheap = qp_allocate(pyramid, tau, u_pref=np.zeros(4))
    biased = qp_allocate(
        pyramid, tau, control_weights=[100.0, 1.0, 1.0, 1.0], u_pref=np.zeros(4)
    )
    assert abs(biased.commands[0]) < abs(cheap.commands[0])
    assert biased.feasible


def test_rpi_reports_its_iteration_count(cluster):
    res = redistributed_pseudo_inverse_allocate(cluster, [0.3, 0.0, 0.0])
    assert res.extras["n_iterations"] >= 1
    assert res.extras["n_saturated"] >= 0


def test_zero_torque_is_met_by_every_method(cluster):
    for method in METHODS:
        res = allocate(cluster, np.zeros(3), method=method)
        assert res.residual_norm <= DEFAULT_TORQUE_TOL, method


# --------------------------------------------------------------------------
# Feasibility test
# --------------------------------------------------------------------------


def test_is_attainable_agrees_with_the_box_for_the_triad(triad):
    assert is_attainable(triad, [0.9, -0.9, 0.9])
    assert not is_attainable(triad, [1.1, 0.0, 0.0])


def test_is_attainable_on_the_boundary(triad):
    assert is_attainable(triad, [1.0, 1.0, 1.0])


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("method", METHODS)
def test_wrong_torque_shape_is_rejected(pyramid, method):
    with pytest.raises(ValueError, match=r"torque must have shape \(3,\)"):
        allocate(pyramid, [1.0, 2.0], method=method)


@pytest.mark.parametrize("method", METHODS)
def test_non_finite_torque_is_rejected(pyramid, method):
    with pytest.raises(ValueError, match="finite"):
        allocate(pyramid, [np.nan, 0.0, 0.0], method=method)


def test_unknown_method_is_rejected(pyramid):
    with pytest.raises(ValueError, match="unknown method"):
        allocate(pyramid, np.zeros(3), method="magic")


def test_non_positive_weights_are_rejected(pyramid):
    with pytest.raises(ValueError, match="weights must be strictly positive"):
        weighted_pseudo_inverse_allocate(pyramid, np.zeros(3), weights=[1.0, 0.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="control_weights must be strictly positive"):
        qp_allocate(pyramid, np.zeros(3), control_weights=[-1.0, 1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="torque_weights must be strictly positive"):
        qp_allocate(pyramid, np.zeros(3), torque_weights=[0.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="torque_weights must be strictly positive"):
        lp_allocate(pyramid, np.zeros(3), torque_weights=[0.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="cost must be strictly positive"):
        lp_allocate(pyramid, np.zeros(3), objective="min_control", cost=np.zeros(4))
    with pytest.raises(ValueError, match="weights must be strictly positive"):
        redistributed_pseudo_inverse_allocate(pyramid, np.zeros(3), weights=np.zeros(4))


def test_non_finite_weights_are_rejected(pyramid):
    with pytest.raises(ValueError, match="finite"):
        weighted_pseudo_inverse_allocate(
            pyramid, np.zeros(3), weights=[np.inf, 1.0, 1.0, 1.0]
        )


def test_bad_gamma_is_rejected(pyramid):
    with pytest.raises(ValueError, match="gamma must be finite and > 0"):
        qp_allocate(pyramid, np.zeros(3), gamma=0.0)
    with pytest.raises(ValueError, match="gamma must be finite and > 0"):
        qp_allocate(pyramid, np.zeros(3), gamma=np.inf)


def test_bad_lp_objective_and_solver_are_rejected(pyramid):
    with pytest.raises(ValueError, match="unknown objective"):
        lp_allocate(pyramid, np.zeros(3), objective="cheapest")
    with pytest.raises(ValueError, match="unknown solver"):
        lp_allocate(pyramid, np.zeros(3), solver="glpk")
    with pytest.raises(ValueError, match="solver='highs' only"):
        lp_allocate(pyramid, np.zeros(3), objective="min_control", solver="pulp")


def test_bad_max_iter_is_rejected(pyramid):
    with pytest.raises(ValueError, match="max_iter must be >= 1"):
        redistributed_pseudo_inverse_allocate(pyramid, np.zeros(3), max_iter=0)


# --------------------------------------------------------------------------
# Edge cases
# --------------------------------------------------------------------------


def test_all_effectors_failed_gives_infeasible_but_no_crash(triad):
    dead = triad.with_failures([0, 1, 2])
    for method in ("qp", "lp", "rpi"):
        res = allocate(dead, [0.5, 0.0, 0.0], method=method)
        assert np.allclose(res.commands, 0.0, atol=1e-9), method
        assert res.status == "infeasible", method
        assert res.residual_norm == pytest.approx(0.5, abs=1e-6), method


def test_rank_deficient_set_cannot_meet_an_out_of_span_command():
    # Two effectors spanning only the xy-plane: a +z command is unreachable.
    e = general_effector_set(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], [-1.0, -1.0], [1.0, 1.0]
    )
    res = qp_allocate(e, [0.0, 0.0, 0.5])
    assert res.status == "infeasible"
    assert res.residual_norm == pytest.approx(0.5, abs=1e-8)
    assert not is_attainable(e, [0.0, 0.0, 0.5])


def test_saturated_status_when_on_the_boundary(triad):
    res = qp_allocate(triad, [1.0, 0.0, 0.0], u_pref=np.zeros(3))
    assert res.feasible
    assert res.status == "saturated"


def test_weak_effector_degrades_the_qp_residual():
    """A near-useless effector costs the weighted-least-squares QP accuracy.

    Effector 1 has column norm 6.1e-5: it barely moves the vehicle. The QP's
    control term (u_pref = box centre by default) therefore pulls its command
    a long way for almost no torque penalty, and the achieved torque drifts
    off the commanded zero by 1.6e-8 N*m -- above DEFAULT_TORQUE_TOL, so the
    result is honestly reported infeasible. Found by Hypothesis; kept as a
    pinned demonstration of the limitation documented in README "Limitations".
    """
    e = general_effector_set(
        np.array(
            [
                [0.0, 0.0, 1.0, 0.0],
                [-1.0, -6.10351562e-05, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
        np.zeros(4),
        np.array([1.0, 2.0, 1.0, 1.0]),
    )
    res = qp_allocate(e, np.zeros(3))
    assert res.residual_norm == pytest.approx(1.6379603149e-08, rel=1e-6)
    assert not res.feasible
    # Raising gamma or setting u_pref to the true optimum removes it.
    assert qp_allocate(e, np.zeros(3), u_pref=np.zeros(4)).feasible
    assert qp_allocate(e, np.zeros(3), gamma=1e16).residual_norm < 1e-11


def test_weak_feasible_direction_degrades_the_qp_residual():
    """Same mechanism without any weak column, also found by Hypothesis.

    Every column here has norm >= 1, but the exact null direction of B is
    (2, -1e-4, 1, 0), which the bound u1 >= 0 blocks. The best direction the
    active set leaves open is (2, 0, 1, 0), whose torque effectiveness is only
    1e-4, so the QP moves 1.5e-4 along it to reduce effort and accepts a
    1.5e-8 N*m residual at the default gamma = 1e12.
    """
    e = general_effector_set(
        np.array([[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, -2.0, 0.0], [0.0, 1.0, 1e-4, 0.0]]),
        np.zeros(4),
        np.ones(4),
    )
    res = qp_allocate(e, np.zeros(3))
    assert res.residual_norm == pytest.approx(1.4992503765e-08, rel=1e-6)
    assert not res.feasible
    assert np.min(np.linalg.norm(e.matrix, axis=0)) >= 1.0
    # gamma = 1e16 pushes it four orders down, as 1/gamma predicts.
    assert qp_allocate(e, np.zeros(3), gamma=1e16).residual_norm < 1e-11
    assert qp_allocate(e, np.zeros(3), u_pref=np.zeros(4)).feasible


def test_solve_time_is_recorded(pyramid):
    res = qp_allocate(pyramid, [0.01, 0.0, 0.0])
    assert res.solve_time_s >= 0.0


def test_result_repr_contains_status(pyramid):
    res = qp_allocate(pyramid, [0.01, 0.0, 0.0])
    assert res.status in str(res)
