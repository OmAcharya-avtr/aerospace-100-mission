"""Attitude kinematics and rigid-body dynamics."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from detumblesim.attitude import (
    angular_momentum,
    dcm_to_quat,
    kinetic_energy,
    quat_kinematics,
    quat_multiply,
    quat_normalize,
    quat_to_dcm,
    rigid_body_derivative,
    skew,
)

finite = st.floats(-5.0, 5.0, allow_nan=False, allow_infinity=False)
vec3 = st.lists(finite, min_size=3, max_size=3)
vec4 = st.lists(finite, min_size=4, max_size=4)


class TestSkew:
    def test_known_answer(self):
        # [v x] w must equal v x w for v = (1, 2, 3), w = (4, 5, 6):
        # v x w = (2*6 - 3*5, 3*4 - 1*6, 1*5 - 2*4) = (-3, 6, -3)
        v = np.array([1.0, 2.0, 3.0])
        w = np.array([4.0, 5.0, 6.0])
        assert np.allclose(skew(v) @ w, [-3.0, 6.0, -3.0])

    def test_antisymmetric(self):
        s = skew([0.3, -1.2, 4.0])
        assert np.allclose(s, -s.T)

    @pytest.mark.parametrize("bad", [[1.0, 2.0], [[1.0, 2.0, 3.0]], []])
    def test_rejects_wrong_shape(self, bad):
        with pytest.raises(ValueError, match="3-vector"):
            skew(bad)

    @given(a=vec3, b=vec3)
    def test_matches_cross(self, a, b):
        assert np.allclose(skew(a) @ np.array(b), np.cross(a, b))


class TestQuaternion:
    def test_identity_quaternion_gives_identity_dcm(self):
        assert np.allclose(quat_to_dcm([1.0, 0.0, 0.0, 0.0]), np.eye(3))

    def test_known_answer_z_rotation(self):
        # A rotation of the frame by theta about +z has
        # A = [[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]].  For theta = 90 deg
        # the quaternion is (cos45, 0, 0, sin45).
        theta = np.pi / 2
        q = [np.cos(theta / 2), 0.0, 0.0, np.sin(theta / 2)]
        assert np.allclose(
            quat_to_dcm(q), [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            atol=1e-12,
        )

    def test_normalize_rejects_zero(self):
        with pytest.raises(ValueError, match="zero-norm"):
            quat_normalize([0.0, 0.0, 0.0, 0.0])

    def test_normalize_rejects_wrong_shape(self):
        with pytest.raises(ValueError, match="shape"):
            quat_normalize([1.0, 0.0, 0.0])

    def test_dcm_to_quat_rejects_wrong_shape(self):
        with pytest.raises(ValueError, match="shape"):
            dcm_to_quat(np.eye(4))

    def test_kinematics_rejects_wrong_shapes(self):
        with pytest.raises(ValueError, match="quaternion"):
            quat_kinematics([1.0, 0.0, 0.0], [0.0, 0.0, 1.0])
        with pytest.raises(ValueError, match="omega_body"):
            quat_kinematics([1.0, 0.0, 0.0, 0.0], [0.0, 0.0])

    def test_multiply_rejects_wrong_shape(self):
        with pytest.raises(ValueError, match="shape-\\(4,\\)"):
            quat_multiply([1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])

    def test_multiply_identity(self):
        q = quat_normalize([0.3, -0.2, 0.5, 0.1])
        assert np.allclose(quat_multiply([1.0, 0.0, 0.0, 0.0], q), q)

    @given(q=vec4)
    @settings(max_examples=60, deadline=None)
    def test_dcm_is_orthonormal(self, q):
        if np.linalg.norm(q) < 1e-6:
            return
        a = quat_to_dcm(q)
        assert np.allclose(a @ a.T, np.eye(3), atol=1e-10)
        assert np.isclose(np.linalg.det(a), 1.0, atol=1e-10)

    @given(q=vec4)
    @settings(max_examples=60, deadline=None)
    def test_quat_dcm_roundtrip(self, q):
        # q and -q are the same rotation, so the round trip is only defined up
        # to sign; at q0 == 0 the "q0 >= 0" convention cannot pick a branch.
        if np.linalg.norm(q) < 1e-3:
            return
        qn = quat_normalize(q)
        back = dcm_to_quat(quat_to_dcm(qn))
        assert min(
            float(np.max(np.abs(back - qn))), float(np.max(np.abs(back + qn)))
        ) < 1e-8

    @given(q=vec4, w=vec3)
    @settings(max_examples=60, deadline=None)
    def test_kinematics_preserves_norm_to_first_order(self, q, w):
        if np.linalg.norm(q) < 1e-3:
            return
        qn = quat_normalize(q)
        assert abs(float(qn @ quat_kinematics(qn, w))) < 1e-12


class TestRigidBody:
    def test_known_answer_torque_free_symmetric(self):
        # For J = diag(2, 2, 2) and omega = (1, 0, 0), the gyroscopic term
        # omega x (J omega) = (1,0,0) x (2,0,0) = 0, so omega_dot = L / 2.
        j = np.diag([2.0, 2.0, 2.0])
        wd = rigid_body_derivative([1.0, 0.0, 0.0], j, [4.0, 0.0, 0.0])
        assert np.allclose(wd, [2.0, 0.0, 0.0])

    def test_known_answer_gyroscopic(self):
        # J = diag(1, 2, 3), omega = (1, 1, 0):
        #   J omega = (1, 2, 0); omega x J omega = (1*0-0*2, 0*1-1*0, 1*2-1*1)
        #           = (0, 0, 1)
        # so omega_dot = J^-1 (0 - (0,0,1)) = (0, 0, -1/3).
        j = np.diag([1.0, 2.0, 3.0])
        wd = rigid_body_derivative([1.0, 1.0, 0.0], j, [0.0, 0.0, 0.0])
        assert np.allclose(wd, [0.0, 0.0, -1.0 / 3.0])

    def test_precomputed_inverse_matches(self):
        j = np.diag([0.3, 0.4, 0.5])
        a = rigid_body_derivative([1.0, -2.0, 0.5], j, [0.1, 0.0, -0.2])
        b = rigid_body_derivative(
            [1.0, -2.0, 0.5], j, [0.1, 0.0, -0.2], inertia_inv=np.linalg.inv(j)
        )
        assert np.allclose(a, b)

    @pytest.mark.parametrize(
        "w,j,tq",
        [
            ([1.0, 2.0], np.eye(3), [0.0, 0.0, 0.0]),
            ([1.0, 2.0, 3.0], np.eye(2), [0.0, 0.0, 0.0]),
            ([1.0, 2.0, 3.0], np.eye(3), [0.0, 0.0]),
        ],
    )
    def test_rejects_bad_shapes(self, w, j, tq):
        with pytest.raises(ValueError):
            rigid_body_derivative(w, j, tq)

    def test_energy_and_momentum(self):
        j = np.diag([1.0, 2.0, 4.0])
        w = np.array([1.0, 1.0, 1.0])
        assert np.isclose(kinetic_energy(w, j), 0.5 * (1.0 + 2.0 + 4.0))
        assert np.allclose(angular_momentum(w, j), [1.0, 2.0, 4.0])

    @given(w=vec3)
    @settings(max_examples=50, deadline=None)
    def test_torque_free_conserves_energy_rate(self, w):
        # d(T)/dt = omega . (J omega_dot) = -omega . (omega x J omega) = 0
        j = np.diag([0.7, 1.3, 2.1])
        wd = rigid_body_derivative(w, j, [0.0, 0.0, 0.0])
        assert abs(float(np.asarray(w) @ (j @ wd))) < 1e-10 * (1.0 + np.linalg.norm(w) ** 3)
