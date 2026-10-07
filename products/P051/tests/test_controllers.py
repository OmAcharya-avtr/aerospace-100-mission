"""LQR gains, saturation, and the controller callables."""

from __future__ import annotations

import numpy as np
import pytest

from simplexguard import (
    BaselineController,
    PerformanceController,
    box,
    closed_loop_matrix,
    dlqr_gain,
    reference_controllers,
    reference_plant,
    saturate,
)


def test_scalar_lqr_known_answer():
    # Scalar plant x' = a x + u with a = 2, Q = 1, R = 1. The discrete Riccati
    # equation for the scalar case with b = 1 is
    #     p = q + a^2 p - a^2 p^2 / (r + p),
    # whose stabilising root at a = 2, q = r = 1 satisfies
    #     p (1 + p) = (1 + p) + 4 p (1 + p) - 4 p^2
    #  => p + p^2 = 1 + p + 4p + 4p^2 - 4p^2  =>  p^2 - 4 p - 1 = 0
    #  => p = 2 + sqrt(5) = 4.2360679774997896 (the positive root).
    # Then K = b p a / (r + b^2 p) = 2 p / (1 + p) = 8.472135955 / 5.236067977
    #        = 1.6180339887... = the golden ratio.
    from simplexguard import Plant

    plant = Plant(
        A=np.array([[2.0]]),
        B=np.array([[1.0]]),
        disturbance=box([0.0]),
        state_constraints=box([1.0]),
        input_constraints=box([10.0]),
    )
    gain = dlqr_gain(plant, np.array([1.0]), np.array([1.0]))
    assert gain.shape == (1, 1)
    assert float(gain[0, 0]) == pytest.approx((1.0 + np.sqrt(5.0)) / 2.0, abs=1e-10)


def test_lqr_gain_stabilises_the_closed_loop():
    plant = reference_plant()
    gain = dlqr_gain(plant, np.array([1.0, 1.0]), np.array([1.0]))
    rho = float(np.max(np.abs(np.linalg.eigvals(closed_loop_matrix(plant, gain)))))
    assert rho < 1.0


def test_larger_input_weight_gives_a_lower_authority_gain():
    plant = reference_plant()
    soft = dlqr_gain(plant, np.ones(2), np.array([100.0]))
    hard = dlqr_gain(plant, np.ones(2), np.array([0.01]))
    assert np.all(np.abs(soft) < np.abs(hard))


def test_reference_controllers_baseline_is_slower_than_performance():
    plant = reference_plant()
    baseline, performance = reference_controllers(plant)
    rho_b = float(np.max(np.abs(np.linalg.eigvals(closed_loop_matrix(plant, baseline.gain)))))
    rho_p = float(
        np.max(np.abs(np.linalg.eigvals(closed_loop_matrix(plant, performance.gain))))
    )
    assert rho_b > rho_p
    assert np.all(np.abs(performance.gain) > np.abs(baseline.gain))


def test_saturate_known_answer():
    u_set = box([3.0])
    assert saturate(np.array([5.0]), u_set) == pytest.approx(np.array([3.0]))
    assert saturate(np.array([-5.0]), u_set) == pytest.approx(np.array([-3.0]))
    assert saturate(np.array([1.25]), u_set) == pytest.approx(np.array([1.25]))


def test_baseline_controller_is_minus_gain_times_state_then_saturated():
    u_set = box([0.5])
    ctrl = BaselineController(gain=np.array([[2.0, 0.0]]), input_set=u_set)
    assert ctrl.unsaturated(np.array([1.0, 0.0])) == pytest.approx(np.array([-2.0]))
    assert ctrl(np.array([1.0, 0.0])) == pytest.approx(np.array([-0.5]))


def test_performance_controller_tracks_the_reference():
    u_set = box([10.0])
    ctrl = PerformanceController(gain=np.array([[2.0, 1.0]]), input_set=u_set)
    x = np.array([1.0, 0.5])
    r = np.array([1.0, 0.5])
    assert ctrl(x, r) == pytest.approx(np.array([0.0]))
    # With no reference the controller regulates to the origin:
    #   -(2*1.0 + 1*0.5) = -2.5
    assert ctrl(x) == pytest.approx(np.array([-2.5]))


def test_saturates_at_flag():
    ctrl = PerformanceController(gain=np.array([[10.0, 0.0]]), input_set=box([1.0]))
    assert ctrl.saturates_at(np.array([1.0, 0.0]))
    assert not ctrl.saturates_at(np.array([0.05, 0.0]))


def test_dlqr_rejects_bad_weights():
    plant = reference_plant()
    with pytest.raises(ValueError, match="q_diag has length"):
        dlqr_gain(plant, np.ones(3), np.ones(1))
    with pytest.raises(ValueError, match="r_diag has length"):
        dlqr_gain(plant, np.ones(2), np.ones(2))
    with pytest.raises(ValueError, match="strictly positive"):
        dlqr_gain(plant, np.array([0.0, 1.0]), np.ones(1))
    with pytest.raises(ValueError, match="strictly positive"):
        dlqr_gain(plant, np.ones(2), np.array([-1.0]))


def test_closed_loop_matrix_rejects_a_misshaped_gain():
    plant = reference_plant()
    with pytest.raises(ValueError, match="gain has shape"):
        closed_loop_matrix(plant, np.ones((2, 2)))


def test_controller_constructors_validate():
    with pytest.raises(ValueError, match="non-finite"):
        BaselineController(gain=np.array([[np.nan, 0.0]]), input_set=box([1.0]))
    with pytest.raises(ValueError, match="rows, input set has dim"):
        BaselineController(gain=np.ones((2, 2)), input_set=box([1.0]))
    with pytest.raises(ValueError, match="non-finite"):
        PerformanceController(gain=np.array([[np.inf, 0.0]]), input_set=box([1.0]))


def test_controller_calls_validate_shapes():
    ctrl = BaselineController(gain=np.array([[1.0, 1.0]]), input_set=box([1.0]))
    with pytest.raises(ValueError, match="x has length"):
        ctrl(np.zeros(3))
    perf = PerformanceController(gain=np.array([[1.0, 1.0]]), input_set=box([1.0]))
    with pytest.raises(ValueError, match="reference has length"):
        perf(np.zeros(2), np.zeros(3))


def test_saturate_rejects_a_non_box_input_set():
    from simplexguard.polytope import Polytope

    with pytest.raises(TypeError, match="Box"):
        saturate(np.zeros(1), Polytope(np.array([[1.0]]), np.array([1.0])))
