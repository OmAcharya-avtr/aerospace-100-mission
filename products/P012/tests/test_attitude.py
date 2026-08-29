"""Unit tests for navbench.attitude — quaternion algebra and rigid-body dynamics."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from conftest import random_unit_quat
from navbench import (
    axis_angle_from_quat,
    dcm_from_quat,
    euler_moment_derivative,
    euler_zyx_from_quat,
    quat_angle_between,
    quat_canonical,
    quat_conjugate,
    quat_derivative,
    quat_from_axis_angle,
    quat_from_dcm,
    quat_from_euler_zyx,
    quat_from_small_angle,
    quat_identity,
    quat_multiply,
    quat_norm,
    quat_normalize,
    quat_propagate,
    quat_rotate,
    skew,
    small_angle_from_quat,
)


class TestSkew:
    def test_known_matrix(self):
        # Hand-checked: skew([1,2,3]) = [[0,-3,2],[3,0,-1],[-2,1,0]]
        assert np.allclose(skew([1, 2, 3]), [[0, -3, 2], [3, 0, -1], [-2, 1, 0]])

    def test_antisymmetric(self, rng):
        v = rng.standard_normal(3)
        assert np.allclose(skew(v), -skew(v).T)

    def test_acts_as_cross_product(self, rng):
        a, b = rng.standard_normal(3), rng.standard_normal(3)
        assert np.allclose(skew(a) @ b, np.cross(a, b))

    def test_zero_vector_gives_zero_matrix(self):
        assert np.allclose(skew([0, 0, 0]), np.zeros((3, 3)))

    def test_own_vector_in_null_space(self, rng):
        v = rng.standard_normal(3)
        assert np.allclose(skew(v) @ v, np.zeros(3), atol=1e-15)

    @pytest.mark.parametrize("bad", [[1, 2], [1, 2, 3, 4], np.zeros((3, 3))])
    def test_wrong_shape_raises(self, bad):
        with pytest.raises(ValueError, match="shape"):
            skew(bad)

    def test_nonfinite_raises(self):
        with pytest.raises(ValueError, match="finite"):
            skew([1.0, np.nan, 3.0])


class TestQuaternionBasics:
    def test_identity(self):
        assert np.allclose(quat_identity(), [1, 0, 0, 0])

    def test_identity_is_identity_of_multiplication(self, rng):
        q = random_unit_quat(rng)
        assert np.allclose(quat_multiply(quat_identity(), q), q)
        assert np.allclose(quat_multiply(q, quat_identity()), q)

    def test_multiply_hand_computed(self):
        # i * j = k: i = [0,1,0,0], j = [0,0,1,0], k = [0,0,0,1]
        i = np.array([0.0, 1.0, 0.0, 0.0])
        j = np.array([0.0, 0.0, 1.0, 0.0])
        assert np.allclose(quat_multiply(i, j), [0.0, 0.0, 0.0, 1.0])
        # j * i = -k
        assert np.allclose(quat_multiply(j, i), [0.0, 0.0, 0.0, -1.0])

    def test_multiply_i_squared_is_minus_one(self):
        i = np.array([0.0, 1.0, 0.0, 0.0])
        assert np.allclose(quat_multiply(i, i), [-1.0, 0.0, 0.0, 0.0])

    def test_conjugate(self):
        assert np.allclose(quat_conjugate([1, 2, 3, 4]), [1, -2, -3, -4])

    def test_conjugate_is_involution(self, rng):
        q = random_unit_quat(rng)
        assert np.allclose(quat_conjugate(quat_conjugate(q)), q)

    def test_unit_quaternion_times_conjugate_is_identity(self, rng):
        q = random_unit_quat(rng)
        assert np.allclose(quat_multiply(q, quat_conjugate(q)), quat_identity(), atol=1e-15)

    def test_conjugate_of_product_reverses_order(self, rng):
        a, b = random_unit_quat(rng), random_unit_quat(rng)
        left = quat_conjugate(quat_multiply(a, b))
        right = quat_multiply(quat_conjugate(b), quat_conjugate(a))
        assert np.allclose(left, right)

    def test_norm(self):
        assert quat_norm([1, 1, 1, 1]) == pytest.approx(2.0)

    def test_normalize(self):
        assert np.allclose(quat_normalize([2, 0, 0, 0]), [1, 0, 0, 0])

    def test_normalize_rejects_tiny(self):
        with pytest.raises(ValueError, match="norm"):
            quat_normalize([1e-15, 0, 0, 0])

    def test_canonical_flips_negative_scalar(self):
        assert np.allclose(quat_canonical([-0.5, 0.5, 0.5, 0.5]), [0.5, -0.5, -0.5, -0.5])

    def test_canonical_leaves_positive_scalar(self, rng):
        q = quat_canonical(random_unit_quat(rng))
        assert q[0] >= 0.0

    @pytest.mark.parametrize("bad", [[1, 2, 3], [1, 2, 3, 4, 5], np.zeros((2, 2))])
    def test_bad_shape_raises(self, bad):
        with pytest.raises(ValueError, match="shape"):
            quat_conjugate(bad)

    def test_nonfinite_raises(self):
        with pytest.raises(ValueError, match="finite"):
            quat_norm([1.0, np.inf, 0.0, 0.0])


class TestDcm:
    def test_identity_quat_gives_identity_dcm(self):
        assert np.allclose(dcm_from_quat(quat_identity()), np.eye(3))

    def test_90_deg_about_z_hand_computed(self):
        # q = [cos45, 0, 0, sin45]; R rotates +x to +y.
        q = quat_from_axis_angle([0, 0, 1], np.pi / 2)
        assert np.allclose(dcm_from_quat(q) @ [1, 0, 0], [0, 1, 0], atol=1e-15)

    def test_180_deg_about_x(self):
        q = quat_from_axis_angle([1, 0, 0], np.pi)
        assert np.allclose(dcm_from_quat(q), np.diag([1.0, -1.0, -1.0]), atol=1e-15)

    def test_matches_scipy(self, rng):
        for _ in range(30):
            q = random_unit_quat(rng)
            ref = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
            assert np.allclose(dcm_from_quat(q), ref, atol=1e-14)

    def test_orthogonal_with_unit_determinant(self, rng):
        r = dcm_from_quat(random_unit_quat(rng))
        assert np.allclose(r @ r.T, np.eye(3), atol=1e-14)
        assert np.linalg.det(r) == pytest.approx(1.0)

    def test_composition_homomorphism(self, rng):
        a, b = random_unit_quat(rng), random_unit_quat(rng)
        assert np.allclose(
            dcm_from_quat(quat_multiply(a, b)), dcm_from_quat(a) @ dcm_from_quat(b), atol=1e-14
        )

    def test_round_trip_quat_dcm_quat(self, rng):
        for _ in range(30):
            q = quat_canonical(random_unit_quat(rng))
            assert np.allclose(quat_from_dcm(dcm_from_quat(q)), q, atol=1e-13)

    @pytest.mark.parametrize(
        "axis,angle",
        [([1, 0, 0], np.pi), ([0, 1, 0], np.pi), ([0, 0, 1], np.pi), ([1, 1, 1], np.pi)],
    )
    def test_shepperd_branches_at_180_degrees(self, axis, angle):
        """All four Shepperd pivots are exercised by 180 deg rotations."""
        q = quat_canonical(quat_from_axis_angle(axis, angle))
        assert np.allclose(quat_from_dcm(dcm_from_quat(q)), q, atol=1e-13)

    def test_non_orthogonal_matrix_raises(self):
        with pytest.raises(ValueError, match="orthogonal"):
            quat_from_dcm(np.array([[1.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))

    def test_reflection_raises(self):
        with pytest.raises(ValueError, match="determinant"):
            quat_from_dcm(np.diag([1.0, 1.0, -1.0]))

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="shape"):
            quat_from_dcm(np.eye(4))

    def test_nonfinite_raises(self):
        m = np.eye(3)
        m[0, 0] = np.nan
        with pytest.raises(ValueError, match="finite"):
            quat_from_dcm(m)


class TestAxisAngle:
    def test_known_value(self):
        q = quat_from_axis_angle([0, 0, 1], np.pi / 3)
        assert q[0] == pytest.approx(np.cos(np.pi / 6))
        assert q[3] == pytest.approx(np.sin(np.pi / 6))

    def test_axis_need_not_be_normalised(self):
        a = quat_from_axis_angle([0, 0, 5], 0.7)
        b = quat_from_axis_angle([0, 0, 1], 0.7)
        assert np.allclose(a, b)

    def test_round_trip(self, rng):
        for _ in range(30):
            axis = rng.standard_normal(3)
            axis /= np.linalg.norm(axis)
            ang = float(rng.uniform(0.01, np.pi - 0.01))
            ax2, an2 = axis_angle_from_quat(quat_from_axis_angle(axis, ang))
            assert np.allclose(ax2 * an2, axis * ang, atol=1e-14)

    def test_identity_gives_zero_angle(self):
        axis, ang = axis_angle_from_quat(quat_identity())
        assert ang == pytest.approx(0.0)
        assert np.allclose(axis, [1, 0, 0])

    def test_tiny_axis_raises(self):
        with pytest.raises(ValueError, match="axis norm"):
            quat_from_axis_angle([0, 0, 0], 1.0)

    def test_nonfinite_angle_raises(self):
        with pytest.raises(ValueError, match="finite"):
            quat_from_axis_angle([0, 0, 1], np.nan)

    def test_angle_is_in_zero_pi(self, rng):
        for _ in range(30):
            _, ang = axis_angle_from_quat(random_unit_quat(rng))
            assert 0.0 <= ang <= np.pi + 1e-15


class TestSmallAngleMap:
    def test_zero_gives_identity(self):
        assert np.allclose(quat_from_small_angle([0, 0, 0]), quat_identity())

    def test_first_order_for_small_input(self):
        v = np.array([1e-6, -2e-6, 3e-6])
        q = quat_from_small_angle(v)
        assert np.allclose(q[1:], v / 2.0, rtol=1e-9)
        assert q[0] == pytest.approx(1.0)

    def test_result_is_unit_norm(self, rng):
        for mag in (1e-12, 1e-6, 0.1, 1.0, 3.0):
            v = rng.standard_normal(3)
            v = mag * v / np.linalg.norm(v)
            assert np.linalg.norm(quat_from_small_angle(v)) == pytest.approx(1.0, abs=1e-15)

    @pytest.mark.parametrize("mag", [1e-14, 1e-12, 1e-9, 1e-6, 1e-3, 0.5, 2.0, 3.0])
    def test_round_trip_relative(self, mag, rng):
        """Regression test for the axis_angle_from_quat small-rotation defect."""
        v = rng.standard_normal(3)
        v = mag * v / np.linalg.norm(v)
        back = small_angle_from_quat(quat_from_small_angle(v))
        assert np.max(np.abs(back - v)) / mag < 1e-14

    def test_matches_axis_angle_construction(self):
        v = np.array([0.3, -0.4, 0.5])
        n = np.linalg.norm(v)
        assert np.allclose(quat_from_small_angle(v), quat_from_axis_angle(v / n, n))

    def test_direction_preserved_at_1e14_rad(self):
        v = np.array([1e-14, 2e-14, -3e-14]) / np.sqrt(14.0)
        back = small_angle_from_quat(quat_from_small_angle(v))
        assert np.allclose(back / np.linalg.norm(back), v / np.linalg.norm(v), atol=1e-6)


class TestRotationAndKinematics:
    def test_rotate_identity(self, rng):
        v = rng.standard_normal(3)
        assert np.allclose(quat_rotate(quat_identity(), v), v)

    def test_rotate_preserves_length(self, rng):
        q, v = random_unit_quat(rng), rng.standard_normal(3)
        assert np.linalg.norm(quat_rotate(q, v)) == pytest.approx(np.linalg.norm(v))

    def test_rotate_matches_dcm(self, rng):
        q, v = random_unit_quat(rng), rng.standard_normal(3)
        assert np.allclose(quat_rotate(q, v), dcm_from_quat(q) @ v)

    def test_derivative_zero_rate_gives_zero(self, rng):
        assert np.allclose(quat_derivative(random_unit_quat(rng), [0, 0, 0]), np.zeros(4))

    def test_derivative_is_orthogonal_to_q(self, rng):
        """d|q|^2/dt = 2 q.qdot = 0, so the norm is conserved analytically."""
        q = random_unit_quat(rng)
        w = rng.standard_normal(3)
        assert float(q @ quat_derivative(q, w)) == pytest.approx(0.0, abs=1e-15)

    def test_propagate_matches_axis_angle_for_constant_rate(self):
        w = np.array([0.0, 0.0, 0.1])
        q = quat_propagate(quat_identity(), w, 3.0)
        assert np.allclose(q, quat_from_axis_angle([0, 0, 1], 0.3))

    def test_propagate_preserves_norm_over_many_steps(self):
        q = quat_normalize([0.3, 0.2, -0.5, 0.7])
        for _ in range(5000):
            q = quat_propagate(q, [0.03, -0.07, 0.11], 0.05)
        assert np.linalg.norm(q) == pytest.approx(1.0, abs=1e-14)

    def test_propagate_composes(self):
        q0 = quat_normalize([0.3, 0.2, -0.5, 0.7])
        w = np.array([0.02, 0.05, -0.01])
        one = quat_propagate(q0, w, 2.0)
        two = quat_propagate(quat_propagate(q0, w, 1.0), w, 1.0)
        assert np.allclose(quat_canonical(one), quat_canonical(two), atol=1e-14)

    def test_propagate_zero_dt_is_identity(self, rng):
        q = random_unit_quat(rng)
        assert np.allclose(quat_propagate(q, [1, 2, 3], 0.0), q)

    def test_propagate_nonfinite_dt_raises(self):
        with pytest.raises(ValueError, match="finite"):
            quat_propagate(quat_identity(), [0, 0, 1], np.nan)

    def test_angle_between_identical(self, rng):
        q = random_unit_quat(rng)
        assert quat_angle_between(q, q) == pytest.approx(0.0, abs=1e-15)

    def test_angle_between_sign_invariant(self, rng):
        q = random_unit_quat(rng)
        assert quat_angle_between(q, -q) == pytest.approx(0.0, abs=1e-7)

    def test_angle_between_known(self):
        a = quat_identity()
        b = quat_from_axis_angle([0, 1, 0], 0.4)
        assert quat_angle_between(a, b) == pytest.approx(0.4)


class TestEuler:
    def test_zero_angles_give_identity(self):
        assert np.allclose(quat_from_euler_zyx(0, 0, 0), quat_identity())

    def test_round_trip(self, rng):
        for _ in range(30):
            yaw = float(rng.uniform(-np.pi, np.pi))
            pitch = float(rng.uniform(-1.2, 1.2))
            roll = float(rng.uniform(-np.pi, np.pi))
            y2, p2, r2 = euler_zyx_from_quat(quat_from_euler_zyx(yaw, pitch, roll))
            assert (y2, p2, r2) == pytest.approx((yaw, pitch, roll), abs=1e-12)

    def test_pure_yaw(self):
        q = quat_from_euler_zyx(0.5, 0.0, 0.0)
        assert np.allclose(q, quat_from_axis_angle([0, 0, 1], 0.5))

    def test_pure_roll(self):
        q = quat_from_euler_zyx(0.0, 0.0, -0.3)
        assert np.allclose(q, quat_from_axis_angle([1, 0, 0], -0.3))

    def test_dcm_is_rz_ry_rx(self):
        yaw, pitch, roll = 0.3, -0.2, 0.5
        cz, sz = np.cos(yaw), np.sin(yaw)
        cy, sy = np.cos(pitch), np.sin(pitch)
        cx, sx = np.cos(roll), np.sin(roll)
        rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        rxm = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        assert np.allclose(dcm_from_quat(quat_from_euler_zyx(yaw, pitch, roll)), rz @ ry @ rxm)

    def test_nonfinite_raises(self):
        with pytest.raises(ValueError, match="finite"):
            quat_from_euler_zyx(np.nan, 0.0, 0.0)


class TestEulerMoment:
    def test_zero_torque_zero_rate(self, inertia):
        assert np.allclose(euler_moment_derivative([0, 0, 0], inertia, [0, 0, 0]), np.zeros(3))

    def test_principal_axis_spin_has_no_gyroscopic_term(self, inertia):
        assert np.allclose(euler_moment_derivative([0, 0, 0.5], inertia, [0, 0, 0]), np.zeros(3))

    def test_hand_computed(self):
        """J = diag(2,3,4), w = [1,1,1], tau = 0.

        Jw = [2,3,4]; w x Jw = [1*4-1*3, 1*2-1*4, 1*3-1*2] = [1,-2,1].
        wdot = -J^-1 [1,-2,1] = [-0.5, 2/3, -0.25].
        """
        j = np.diag([2.0, 3.0, 4.0])
        got = euler_moment_derivative([1, 1, 1], j, [0, 0, 0])
        assert np.allclose(got, [-0.5, 2.0 / 3.0, -0.25])

    def test_torque_only(self):
        j = np.diag([2.0, 4.0, 8.0])
        assert np.allclose(euler_moment_derivative([0, 0, 0], j, [2, 4, 8]), [1, 1, 1])

    def test_asymmetric_inertia_raises(self):
        bad = np.array([[1.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        with pytest.raises(ValueError, match="symmetric"):
            euler_moment_derivative([1, 0, 0], bad, [0, 0, 0])

    def test_singular_inertia_raises(self):
        with pytest.raises(ValueError, match="positive definite"):
            euler_moment_derivative([1, 0, 0], np.diag([1.0, 1.0, 0.0]), [0, 0, 0])

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="shape"):
            euler_moment_derivative([1, 0, 0], np.eye(2), [0, 0, 0])

    def test_nonfinite_inertia_raises(self):
        j = np.eye(3)
        j[1, 1] = np.inf
        with pytest.raises(ValueError, match="finite"):
            euler_moment_derivative([1, 0, 0], j, [0, 0, 0])
