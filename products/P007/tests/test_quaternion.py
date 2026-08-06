"""Unit, known-answer, and input-validation tests for the Quaternion class and core ops.

Hand calculations are shown in comments next to each known-answer assertion.
Convention under test: scalar-first [w, x, y, z], Hamilton product, active rotation.
"""

import numpy as np
import pytest

from quatkit import (
    Quaternion,
    quat_exp,
    quat_identity,
    quat_inverse,
    quat_log,
    quat_multiply,
    quat_normalize,
    quat_rotate,
    quat_slerp,
)

S2 = np.sqrt(0.5)  # sin(45 deg) = cos(45 deg)


class TestConstruction:
    def test_identity(self):
        q = Quaternion.identity()
        assert q.as_array() == pytest.approx([1.0, 0.0, 0.0, 0.0])

    def test_non_unit_raises_by_default(self):
        # Documented normalize-or-raise policy: |q| = 2 is rejected.
        with pytest.raises(ValueError, match="normalize=True"):
            Quaternion(2.0, 0.0, 0.0, 0.0)

    def test_non_unit_accepted_with_normalize_flag(self):
        q = Quaternion(2.0, 0.0, 0.0, 0.0, normalize=True)
        assert q.norm == pytest.approx(1.0, abs=1e-15)
        assert q.w == pytest.approx(1.0)

    def test_small_deviation_silently_renormalized(self):
        # Within NORM_TOL=1e-6 of unit: accepted and cleaned up.
        q = Quaternion(1.0 + 5e-7, 0.0, 0.0, 0.0)
        assert q.norm == pytest.approx(1.0, abs=1e-15)

    def test_zero_quaternion_raises(self):
        with pytest.raises(ValueError, match="zero quaternion"):
            Quaternion(0.0, 0.0, 0.0, 0.0)
        with pytest.raises(ValueError):
            Quaternion(0.0, 0.0, 0.0, 0.0, normalize=True)

    def test_nonfinite_raises(self):
        with pytest.raises(ValueError, match="finite"):
            Quaternion(np.nan, 0.0, 0.0, 0.0)
        with pytest.raises(ValueError, match="finite"):
            Quaternion(np.inf, 0.0, 0.0, 0.0, normalize=True)

    def test_from_array_shape_validation(self):
        with pytest.raises(ValueError, match=r"\(4,\)"):
            Quaternion.from_array([1.0, 0.0, 0.0])

    def test_rotation_cannot_receive_non_unit(self):
        # The only paths into rotate() are unit by construction; the array API
        # documents the caller contract instead.
        q = Quaternion(0.5, 0.5, 0.5, 0.5)  # exactly unit: 4 * 0.25 = 1
        v = q.rotate([1.0, 0.0, 0.0])
        assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-12)


class TestHamiltonProduct:
    def test_ij_equals_k(self):
        # Hand check: i ⊗ j = k in Hamilton algebra.
        # [0,1,0,0] ⊗ [0,0,1,0]: w = 0*0 - (1,0,0)·(0,1,0) = 0
        # v = 0*(0,1,0) + 0*(1,0,0) + (1,0,0)×(0,1,0) = (0,0,1)  ->  [0,0,0,1] = k
        i = np.array([0.0, 1.0, 0.0, 0.0])
        j = np.array([0.0, 0.0, 1.0, 0.0])
        k = np.array([0.0, 0.0, 0.0, 1.0])
        np.testing.assert_allclose(quat_multiply(i, j), k, atol=1e-15)
        # Anti-commutativity: j ⊗ i = -k.
        np.testing.assert_allclose(quat_multiply(j, i), -k, atol=1e-15)

    def test_i_squared_is_minus_one(self):
        i = np.array([0.0, 1.0, 0.0, 0.0])
        np.testing.assert_allclose(
            quat_multiply(i, i), [-1.0, 0.0, 0.0, 0.0], atol=1e-15
        )

    def test_composition_order(self):
        # 90° about z then 90° about x (world axes, active) maps x̂ -> ŷ -> ẑ.
        qz = Quaternion.from_axis_angle([0, 0, 1], np.pi / 2)
        qx = Quaternion.from_axis_angle([1, 0, 0], np.pi / 2)
        combined = qx * qz  # apply qz first, then qx
        np.testing.assert_allclose(
            combined.rotate([1.0, 0.0, 0.0]), [0.0, 0.0, 1.0], atol=1e-15
        )

    def test_identity_neutral(self):
        q = Quaternion.from_axis_angle([1, 2, 3], 0.7)
        assert (Quaternion.identity() * q).isclose(q)
        assert (q * Quaternion.identity()).isclose(q)


class TestConjugateInverse:
    def test_q_times_inverse_is_identity(self):
        q = Quaternion.from_axis_angle([1.0, -2.0, 0.5], 1.234)
        prod = q * q.inverse()
        np.testing.assert_allclose(prod.as_array(), quat_identity(), atol=1e-15)

    def test_array_inverse_non_unit(self):
        # q⁻¹ = q*/|q|²: for q = [0,2,0,0], |q|² = 4 -> q⁻¹ = [0,-0.5,0,0].
        np.testing.assert_allclose(
            quat_inverse([0.0, 2.0, 0.0, 0.0]), [0.0, -0.5, 0.0, 0.0], atol=1e-15
        )

    def test_inverse_of_zero_raises(self):
        with pytest.raises(ValueError, match="zero"):
            quat_inverse([0.0, 0.0, 0.0, 0.0])


class TestRotation:
    def test_z90_rotates_x_to_y(self):
        # Hand check: q = [cos45°, 0, 0, sin45°] rotates x̂ to ŷ (right-hand rule
        # about +z, active rotation).
        q = Quaternion(S2, 0.0, 0.0, S2)
        np.testing.assert_allclose(q.rotate([1, 0, 0]), [0, 1, 0], atol=1e-15)

    def test_x90_rotates_y_to_z(self):
        q = Quaternion(S2, S2, 0.0, 0.0)
        np.testing.assert_allclose(q.rotate([0, 1, 0]), [0, 0, 1], atol=1e-15)

    def test_y90_rotates_z_to_x(self):
        q = Quaternion(S2, 0.0, S2, 0.0)
        np.testing.assert_allclose(q.rotate([0, 0, 1]), [1, 0, 0], atol=1e-15)

    def test_vectorized_rotation(self):
        q = Quaternion(S2, 0.0, 0.0, S2)
        vs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]])
        expected = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
        np.testing.assert_allclose(q.rotate(vs), expected, atol=1e-15)

    def test_bad_vector_shape_raises(self):
        with pytest.raises(ValueError, match=r"\(\.\.\., 3\)"):
            quat_rotate(quat_identity(), [1.0, 0.0])


class TestExpLog:
    def test_exp_zero_is_identity(self):
        np.testing.assert_allclose(quat_exp([0, 0, 0]), quat_identity(), atol=1e-15)

    def test_exp_known(self):
        # Rotation vector (π/2) ẑ -> q = [cos45°, 0, 0, sin45°].
        np.testing.assert_allclose(
            quat_exp([0.0, 0.0, np.pi / 2]), [S2, 0.0, 0.0, S2], atol=1e-15
        )

    def test_log_of_identity_is_zero(self):
        np.testing.assert_allclose(quat_log(quat_identity()), [0, 0, 0], atol=1e-15)

    def test_exp_log_roundtrip(self):
        rv = np.array([0.3, -1.1, 0.7])
        np.testing.assert_allclose(quat_log(quat_exp(rv)), rv, atol=1e-12)

    def test_log_resolves_double_cover(self):
        q = Quaternion.from_axis_angle([0, 0, 1], np.pi / 2)
        np.testing.assert_allclose(
            quat_log(-q.as_array()), q.log(), atol=1e-12
        )


class TestSlerp:
    def test_endpoints(self):
        q0 = Quaternion.identity()
        q1 = Quaternion.from_axis_angle([0, 0, 1], np.pi / 2)
        assert q0.slerp(q1, 0.0).isclose(q0, atol=1e-12)
        assert q0.slerp(q1, 1.0).isclose(q1, atol=1e-12)

    def test_midpoint_is_half_angle(self):
        # Halfway from identity to 90° about z must be exactly 45° about z:
        # q = [cos22.5°, 0, 0, sin22.5°].
        q0 = Quaternion.identity()
        q1 = Quaternion.from_axis_angle([0, 0, 1], np.pi / 2)
        mid = q0.slerp(q1, 0.5)
        expected = Quaternion.from_axis_angle([0, 0, 1], np.pi / 4)
        assert mid.isclose(expected, atol=1e-12)

    def test_shortest_path_sign_flip(self):
        # -q1 represents the same attitude; slerp must not take the long way.
        q0 = Quaternion.identity()
        q1 = Quaternion.from_axis_angle([0, 0, 1], np.pi / 2)
        mid = quat_slerp(q0.as_array(), -q1.as_array(), 0.5)
        expected = Quaternion.from_axis_angle([0, 0, 1], np.pi / 4)
        assert Quaternion.from_array(mid, normalize=True).isclose(expected, atol=1e-12)

    def test_near_parallel_fallback(self):
        q0 = Quaternion.identity().as_array()
        q1 = quat_exp([0.0, 0.0, 1e-13])
        out = quat_slerp(q0, q1, 0.5)
        assert np.linalg.norm(out) == pytest.approx(1.0, abs=1e-12)

    def test_array_t(self):
        q0 = Quaternion.identity()
        q1 = Quaternion.from_axis_angle([1, 0, 0], 1.0)
        out = q0.slerp(q1, np.linspace(0, 1, 5))
        assert out.shape == (5, 4)
        np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-12)


class TestNormalizeValidation:
    def test_normalize_zero_raises(self):
        with pytest.raises(ValueError, match="zero or non-finite"):
            quat_normalize([0.0, 0.0, 0.0, 0.0])

    def test_bad_shape_raises(self):
        with pytest.raises(ValueError, match=r"\(\.\.\., 4\)"):
            quat_normalize([1.0, 0.0, 0.0])
