"""Quaternion algebra: conventions, identities, edge cases, known answers."""

from __future__ import annotations

import math

import numpy as np
import pytest

from slewforge.attitude import (
    axis_angle_from_quat,
    cross3,
    quat_angle,
    quat_conjugate,
    quat_from_axis_angle,
    quat_from_rotvec,
    quat_identity,
    quat_multiply,
    quat_normalize,
    quat_relative,
    quat_rotate,
    quat_slerp,
    quat_to_dcm,
    rotate_about_axis,
    unit_vector,
)


class TestConventions:
    def test_identity_is_scalar_first(self):
        assert quat_identity().tolist() == [1.0, 0.0, 0.0, 0.0]

    def test_hamilton_ijk(self):
        # i j = k with scalar-first storage: [0,1,0,0] * [0,0,1,0] = [0,0,0,1]
        i = np.array([0.0, 1.0, 0.0, 0.0])
        j = np.array([0.0, 0.0, 1.0, 0.0])
        k = np.array([0.0, 0.0, 0.0, 1.0])
        assert np.allclose(quat_multiply(i, j), k)
        assert np.allclose(quat_multiply(j, k), i)
        assert np.allclose(quat_multiply(k, i), j)
        # and i^2 = j^2 = k^2 = -1
        for q in (i, j, k):
            assert np.allclose(quat_multiply(q, q), [-1.0, 0.0, 0.0, 0.0])

    def test_quat_multiply_applies_right_argument_first(self):
        # Rotate x by 90 deg about z (-> y), then by 90 deg about x (-> z).
        rz = quat_from_axis_angle([0, 0, 1], math.pi / 2)
        rx = quat_from_axis_angle([1, 0, 0], math.pi / 2)
        composed = quat_multiply(rx, rz)
        assert np.allclose(quat_rotate(composed, [1.0, 0.0, 0.0]), [0.0, 0.0, 1.0], atol=1e-15)

    def test_active_rotation_of_x_about_z(self):
        # Hand-calculated: rotating +x by 90 deg about +z gives +y (right-handed).
        q = quat_from_axis_angle([0, 0, 1], math.pi / 2)
        assert np.allclose(quat_rotate(q, [1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-15)

    def test_dcm_matches_quat_rotate(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            q = quat_normalize(rng.normal(size=4))
            v = rng.normal(size=3)
            assert np.allclose(quat_to_dcm(q) @ v, quat_rotate(q, v), atol=1e-14)

    def test_dcm_is_a_rotation(self):
        rng = np.random.default_rng(1)
        for _ in range(50):
            r = quat_to_dcm(quat_normalize(rng.normal(size=4)))
            assert np.allclose(r @ r.T, np.eye(3), atol=1e-14)
            assert abs(float(np.linalg.det(r)) - 1.0) < 1e-14


class TestKnownAnswers:
    def test_ninety_degree_quaternion(self):
        # q = [cos 45, sin 45, 0, 0] = [0.7071067811865476, 0.7071067811865475, 0, 0]
        q = quat_from_axis_angle([1, 0, 0], math.pi / 2)
        assert q[0] == pytest.approx(math.sqrt(0.5), abs=1e-16)
        assert q[1] == pytest.approx(math.sqrt(0.5), abs=1e-16)

    def test_axis_angle_round_trip(self):
        rng = np.random.default_rng(2)
        for _ in range(200):
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis)
            angle = float(rng.uniform(1e-9, math.pi - 1e-9))
            a, t = axis_angle_from_quat(quat_from_axis_angle(axis, angle))
            assert t == pytest.approx(angle, abs=1e-12)
            assert np.allclose(a, axis, atol=1e-8)

    def test_axis_angle_takes_the_short_way(self):
        q = quat_from_axis_angle([0, 0, 1], math.radians(350.0))
        axis, angle = axis_angle_from_quat(q)
        assert math.degrees(angle) == pytest.approx(10.0, abs=1e-12)
        assert np.allclose(axis, [0.0, 0.0, -1.0], atol=1e-12)

    def test_sign_ambiguity_gives_the_same_rotation(self):
        rng = np.random.default_rng(3)
        for _ in range(50):
            q = quat_normalize(rng.normal(size=4))
            v = rng.normal(size=3)
            assert np.allclose(quat_rotate(q, v), quat_rotate(-q, v), atol=1e-14)
            assert axis_angle_from_quat(q)[1] == pytest.approx(axis_angle_from_quat(-q)[1])

    def test_rotvec_small_angle_series(self):
        # sin(t/2)/t -> 1/2 as t -> 0; check against the exact value at 1e-9 rad.
        p = np.array([1e-9, 0.0, 0.0])
        q = quat_from_rotvec(p)
        assert q[1] == pytest.approx(math.sin(0.5e-9), rel=1e-14)

    def test_rotvec_zero(self):
        assert np.allclose(quat_from_rotvec([0.0, 0.0, 0.0]), quat_identity())

    def test_quat_angle_of_180_degrees(self):
        q = quat_from_axis_angle([0, 1, 0], math.pi)
        assert quat_angle(quat_identity(), q) == pytest.approx(math.pi, abs=1e-12)


class TestRelative:
    def test_relative_composes_back(self):
        rng = np.random.default_rng(4)
        for _ in range(100):
            a = quat_normalize(rng.normal(size=4))
            b = quat_normalize(rng.normal(size=4))
            r = quat_relative(a, b)
            got = quat_multiply(r, a)
            assert min(
                float(np.linalg.norm(got - b)), float(np.linalg.norm(got + b))
            ) < 1e-13

    def test_relative_of_identical_attitudes_is_identity(self):
        q = quat_normalize([0.3, -0.2, 0.5, 0.1])
        assert quat_angle(q, q) == pytest.approx(0.0, abs=1e-15)


class TestSlerp:
    def test_endpoints(self):
        a = quat_from_axis_angle([0, 0, 1], 0.3)
        b = quat_from_axis_angle([0, 0, 1], 1.7)
        assert np.allclose(quat_slerp(a, b, 0.0), a, atol=1e-15)
        assert np.allclose(quat_slerp(a, b, 1.0), b, atol=1e-15)

    def test_uniform_angular_rate(self):
        a = quat_identity()
        b = quat_from_axis_angle([0, 0, 1], 2.0)
        s = np.linspace(0.0, 1.0, 21)
        qs = quat_slerp(a, b, s)
        angles = np.array([quat_angle(a, q) for q in qs])
        assert np.allclose(angles, 2.0 * s, atol=1e-12)

    def test_takes_the_short_arc(self):
        a = quat_identity()
        b = -quat_from_axis_angle([0, 0, 1], 0.4)
        mid = quat_slerp(a, b, 0.5)
        assert quat_angle(a, mid) == pytest.approx(0.2, abs=1e-12)

    def test_near_identical_quaternions_use_linear_fallback(self):
        a = quat_identity()
        b = quat_from_axis_angle([0, 0, 1], 1e-12)
        mid = quat_slerp(a, b, 0.5)
        assert abs(float(np.linalg.norm(mid)) - 1.0) < 1e-15

    def test_scalar_and_array_shapes(self):
        a, b = quat_identity(), quat_from_axis_angle([1, 0, 0], 1.0)
        assert quat_slerp(a, b, 0.5).shape == (4,)
        assert quat_slerp(a, b, [0.0, 0.5, 1.0]).shape == (3, 4)


class TestRotateAboutAxis:
    def test_matches_quaternion_rotation(self):
        rng = np.random.default_rng(5)
        for _ in range(100):
            axis = rng.normal(size=3)
            v = rng.normal(size=3)
            angle = float(rng.uniform(-6.0, 6.0))
            expect = quat_rotate(quat_from_axis_angle(axis, angle), v)
            assert np.allclose(rotate_about_axis(v, axis, angle), expect, atol=1e-13)

    def test_broadcasts_over_angle(self):
        out = rotate_about_axis([1.0, 0.0, 0.0], [0, 0, 1], np.linspace(0, math.pi, 5))
        assert out.shape == (5, 3)
        assert np.allclose(out[-1], [-1.0, 0.0, 0.0], atol=1e-15)

    def test_preserves_length(self):
        rng = np.random.default_rng(6)
        v = rng.normal(size=3) * 7.0
        out = rotate_about_axis(v, [1, 2, 3], np.linspace(0, 5, 17))
        assert np.allclose(np.linalg.norm(out, axis=1), np.linalg.norm(v), atol=1e-13)


class TestCross3:
    def test_matches_numpy(self):
        rng = np.random.default_rng(7)
        for _ in range(200):
            a, b = rng.normal(size=3), rng.normal(size=3)
            assert np.array_equal(cross3(a, b), np.cross(a, b))


class TestValidation:
    @pytest.mark.parametrize("bad", [[1.0, 0.0, 0.0], np.zeros((2, 5))])
    def test_bad_quaternion_shape(self, bad):
        with pytest.raises(ValueError, match=r"shape"):
            quat_normalize(bad)

    def test_non_finite_quaternion(self):
        with pytest.raises(ValueError, match="non-finite"):
            quat_normalize([np.nan, 0.0, 0.0, 1.0])

    def test_zero_norm_quaternion(self):
        with pytest.raises(ValueError, match="norm"):
            quat_normalize([0.0, 0.0, 0.0, 0.0])

    def test_zero_vector_direction(self):
        with pytest.raises(ValueError, match="undefined"):
            unit_vector([0.0, 0.0, 0.0])

    def test_bad_vector_shape(self):
        with pytest.raises(ValueError, match=r"shape"):
            unit_vector([1.0, 2.0])

    def test_axis_angle_of_identity_returns_zero_angle(self):
        axis, angle = axis_angle_from_quat(quat_identity())
        assert angle == 0.0
        assert np.allclose(axis, [1.0, 0.0, 0.0])

    def test_conjugate_does_not_mutate_input(self):
        q = np.array([1.0, 2.0, 3.0, 4.0])
        quat_conjugate(q)
        assert np.array_equal(q, [1.0, 2.0, 3.0, 4.0])
