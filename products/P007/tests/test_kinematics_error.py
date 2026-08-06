"""Known-answer and validation tests for kinematics and attitude-error functions."""

import numpy as np
import pytest

from quatkit import (
    Quaternion,
    angle_between,
    attitude_error_vector,
    closed_form_constant_omega,
    error_quaternion,
    propagate,
    quat_derivative,
    quat_identity,
    rk4_step,
)


class TestQuatDerivative:
    def test_known_answer_identity_spin_z(self):
        # q = identity, ω = [0, 0, 2] rad/s:
        # q̇ = ½ [1,0,0,0] ⊗ [0,0,0,2] = ½ [0,0,0,2] = [0,0,0,1].
        qdot = quat_derivative(quat_identity(), [0.0, 0.0, 2.0])
        np.testing.assert_allclose(qdot, [0.0, 0.0, 0.0, 1.0], atol=1e-15)

    def test_derivative_orthogonal_to_q(self):
        # d|q|²/dt = 2 q·q̇ = 0 for pure-imaginary ω quaternion.
        q = Quaternion.from_axis_angle([1.0, 2.0, 3.0], 0.9).as_array()
        qdot = quat_derivative(q, [0.4, -0.2, 0.1])
        assert abs(np.dot(q, qdot)) < 1e-15

    def test_bad_shapes_raise(self):
        with pytest.raises(ValueError):
            quat_derivative([1.0, 0.0, 0.0], [0.0, 0.0, 1.0])
        with pytest.raises(ValueError):
            quat_derivative(quat_identity(), [0.0, 0.0])


class TestRK4Propagation:
    def test_constant_omega_vs_closed_form(self):
        # Constant ω: RK4 at dt=0.05 s over 60 s must match the exact
        # q(t) = q0 ⊗ exp(ω t) solution to well under 1e-9 rad.
        q0 = Quaternion.from_euler_zyx(0.3, -0.2, 0.1).as_array()
        omega = np.array([0.1, 0.2, -0.15])  # rad/s
        times = np.arange(0.0, 60.0 + 1e-9, 0.05)
        qs = propagate(q0, lambda t: omega, times)
        q_exact = closed_form_constant_omega(q0, omega, times)
        errs = angle_between(qs, q_exact)
        assert float(np.max(errs)) < 1e-9

    def test_norm_preserved(self):
        q0 = quat_identity()
        times = np.linspace(0.0, 20.0, 401)

        def omega(t):
            return np.array([0.5 * np.sin(0.3 * t), 0.4, 0.2 * np.cos(0.7 * t)])

        qs = propagate(q0, omega, times)
        np.testing.assert_allclose(np.linalg.norm(qs, axis=1), 1.0, atol=1e-13)

    def test_single_step_matches_exact_to_dt5(self):
        # One RK4 step, |ω| dt = 0.1 rad: local error must scale like dt⁵
        # (coefficient ~ |ω|⁵/2880 for this equation; just bound it loosely).
        omega = np.array([0.0, 0.0, 1.0])
        dt = 0.1
        q1 = rk4_step(quat_identity(), 0.0, dt, lambda t: omega)
        q_exact = closed_form_constant_omega(quat_identity(), omega, dt)
        err = float(angle_between(q1 / np.linalg.norm(q1), q_exact))
        assert err < 1e-8

    def test_non_unit_q0_raises(self):
        # Documented policy: propagation requires a unit initial quaternion.
        with pytest.raises(ValueError, match="unit quaternion"):
            propagate([2.0, 0.0, 0.0, 0.0], lambda t: np.zeros(3), [0.0, 1.0])

    def test_non_increasing_times_raise(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            propagate(quat_identity(), lambda t: np.zeros(3), [0.0, 1.0, 1.0])

    def test_zero_omega_is_constant(self):
        qs = propagate(quat_identity(), lambda t: np.zeros(3), np.linspace(0, 5, 11))
        np.testing.assert_allclose(qs, np.tile(quat_identity(), (11, 1)), atol=1e-15)


class TestAttitudeError:
    def test_error_of_identical_attitudes_is_identity(self):
        q = Quaternion.from_euler_zyx(1.0, 0.5, -0.3).as_array()
        dq = error_quaternion(q, q)
        np.testing.assert_allclose(dq, quat_identity(), atol=1e-15)
        assert float(angle_between(q, q)) == pytest.approx(0.0, abs=1e-12)

    def test_error_vector_small_angle(self):
        # q = q_ref ⊗ δq with δq a 1 mrad rotation about x̂:
        # δθ = 2 vec(δq) ≈ [1e-3, 0, 0] rad to O(θ³) ~ 4e-11.
        q_ref = Quaternion.from_euler_zyx(0.7, -0.1, 0.4)
        dq = Quaternion.from_axis_angle([1.0, 0.0, 0.0], 1e-3)
        q = q_ref * dq
        dtheta = attitude_error_vector(q.as_array(), q_ref.as_array())
        np.testing.assert_allclose(dtheta, [1e-3, 0.0, 0.0], atol=1e-10)

    def test_angle_between_known(self):
        q1 = Quaternion.identity().as_array()
        q2 = Quaternion.from_axis_angle([0.0, 1.0, 0.0], 0.75).as_array()
        assert float(angle_between(q2, q1)) == pytest.approx(0.75, abs=1e-12)
        # Symmetric and double-cover safe:
        assert float(angle_between(q1, q2)) == pytest.approx(0.75, abs=1e-12)
        assert float(angle_between(-q2, q1)) == pytest.approx(0.75, abs=1e-12)

    def test_error_composition_consistency(self):
        # δq = q_ref⁻¹ ⊗ q  =>  q_ref ⊗ δq == q exactly.
        q_ref = Quaternion.from_euler_zyx(-0.4, 0.9, 1.7)
        q = Quaternion.from_euler_zyx(0.2, -0.6, 0.05)
        dq = error_quaternion(q.as_array(), q_ref.as_array())
        recomposed = q_ref * Quaternion.from_array(dq, normalize=True)
        assert recomposed.isclose(q, atol=1e-12)
