"""Known-answer, round-trip, and gimbal-lock tests for attitude conversions.

Hand-checked reference values are documented in comments.
"""

import numpy as np
import pytest

from quatkit import (
    GimbalLockWarning,
    Quaternion,
    angle_between,
    axis_angle_to_quat,
    dcm_to_quat,
    euler_zyx_to_quat,
    mrp_to_quat,
    quat_to_axis_angle,
    quat_to_dcm,
    quat_to_euler_zyx,
    quat_to_mrp,
    quat_to_rodrigues,
    rodrigues_to_quat,
)

S2 = np.sqrt(0.5)

# Hand-written active rotation matrices for 90° about each principal axis:
# Rz(90°): x̂->ŷ, ŷ->-x̂;  Ry(90°): ẑ->x̂, x̂->-ẑ;  Rx(90°): ŷ->ẑ, ẑ->-ŷ.
RZ90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
RY90 = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
RX90 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


class TestDCM:
    @pytest.mark.parametrize(
        ("q", "r"),
        [
            (np.array([S2, 0.0, 0.0, S2]), RZ90),
            (np.array([S2, 0.0, S2, 0.0]), RY90),
            (np.array([S2, S2, 0.0, 0.0]), RX90),
        ],
        ids=["z90", "y90", "x90"],
    )
    def test_principal_90deg_known_answers(self, q, r):
        np.testing.assert_allclose(quat_to_dcm(q), r, atol=1e-15)
        np.testing.assert_allclose(dcm_to_quat(r), q, atol=1e-15)

    def test_identity(self):
        np.testing.assert_allclose(
            quat_to_dcm([1.0, 0.0, 0.0, 0.0]), np.eye(3), atol=1e-15
        )

    def test_dcm_matches_rotate(self):
        q = Quaternion.from_axis_angle([1.0, 2.0, -0.5], 1.1)
        v = np.array([0.3, -1.2, 2.5])
        np.testing.assert_allclose(q.to_dcm() @ v, q.rotate(v), atol=1e-13)

    def test_roundtrip_180deg_all_axes(self):
        # 180° rotations exercise every branch of Shepperd's method (trace = -1).
        for axis in ([1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]):
            q = Quaternion.from_axis_angle(axis, np.pi)
            q2 = Quaternion.from_dcm(q.to_dcm())
            assert q.isclose(q2, atol=1e-12)

    def test_non_orthogonal_raises(self):
        with pytest.raises(ValueError, match="not a rotation matrix"):
            dcm_to_quat(np.eye(3) * 1.1)

    def test_reflection_raises(self):
        # det = -1 (improper rotation) must be rejected.
        with pytest.raises(ValueError, match="not a rotation matrix"):
            dcm_to_quat(np.diag([1.0, 1.0, -1.0]))

    def test_bad_shape_raises(self):
        with pytest.raises(ValueError, match=r"\(\.\.\., 3, 3\)"):
            dcm_to_quat(np.eye(4))

    def test_batched_dcm(self):
        qs = np.stack([[S2, 0.0, 0.0, S2], [1.0, 0.0, 0.0, 0.0]])
        rs = quat_to_dcm(qs)
        assert rs.shape == (2, 3, 3)
        back = dcm_to_quat(rs)
        np.testing.assert_allclose(back, qs, atol=1e-12)


class TestEulerZYX:
    def test_pure_yaw_90(self):
        # yaw = 90°: q = [cos45°, 0, 0, sin45°] (rotation about z only).
        q = euler_zyx_to_quat(np.pi / 2, 0.0, 0.0)
        np.testing.assert_allclose(q, [S2, 0.0, 0.0, S2], atol=1e-15)

    def test_pure_pitch_and_roll(self):
        np.testing.assert_allclose(
            euler_zyx_to_quat(0.0, np.pi / 2, 0.0), [S2, 0.0, S2, 0.0], atol=1e-15
        )
        np.testing.assert_allclose(
            euler_zyx_to_quat(0.0, 0.0, np.pi / 2), [S2, S2, 0.0, 0.0], atol=1e-15
        )

    def test_known_combined(self):
        # yaw=30°, pitch=20°, roll=10° -> compose elementary quaternions by hand:
        # q = qz(30°) ⊗ qy(20°) ⊗ qx(10°). Cross-checked against the DCM product
        # Rz@Ry@Rx built from hand-written elementary matrices below.
        y, p, r = np.radians([30.0, 20.0, 10.0])
        cz, sz = np.cos(y), np.sin(y)
        cy_, sy_ = np.cos(p), np.sin(p)
        cx, sx = np.cos(r), np.sin(r)
        rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        ry = np.array([[cy_, 0, sy_], [0, 1, 0], [-sy_, 0, cy_]])
        rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        q = euler_zyx_to_quat(y, p, r)
        np.testing.assert_allclose(quat_to_dcm(q), rz @ ry @ rx, atol=1e-14)

    def test_roundtrip(self):
        angles = (0.4, -0.7, 2.1)
        out = quat_to_euler_zyx(euler_zyx_to_quat(*angles))
        np.testing.assert_allclose(out, angles, atol=1e-12)

    def test_angle_ranges(self):
        rng = np.random.default_rng(7)
        for _ in range(50):
            q = rng.standard_normal(4)
            q /= np.linalg.norm(q)
            y, p, r = quat_to_euler_zyx(q)
            assert -np.pi <= y <= np.pi
            assert -np.pi / 2 <= p <= np.pi / 2
            assert -np.pi <= r <= np.pi


class TestGimbalLock:
    def test_exact_lock_warns_and_reconstructs(self):
        # pitch = +90° with nonzero yaw and roll: only yaw - roll is observable.
        q = euler_zyx_to_quat(0.3, np.pi / 2, 0.2)
        with pytest.warns(GimbalLockWarning):
            y, p, r = quat_to_euler_zyx(q)
        assert r == 0.0  # documented fallback
        assert p == pytest.approx(np.pi / 2, abs=1e-7)
        assert y == pytest.approx(0.3 - 0.2, abs=1e-9)  # yaw absorbs -roll at +90°
        # The returned triple must still reconstruct the same attitude.
        q2 = euler_zyx_to_quat(y, p, r)
        assert float(angle_between(q2, q)) < 1e-6

    def test_negative_lock(self):
        q = euler_zyx_to_quat(-0.5, -np.pi / 2, 0.25)
        with pytest.warns(GimbalLockWarning):
            y, p, r = quat_to_euler_zyx(q)
        q2 = euler_zyx_to_quat(y, p, r)
        assert float(angle_between(q2, q)) < 1e-6

    def test_near_lock_within_margin_warns(self):
        q = euler_zyx_to_quat(0.0, np.pi / 2 - 1e-9, 0.0)
        with pytest.warns(GimbalLockWarning):
            quat_to_euler_zyx(q)

    def test_just_outside_margin_no_warning(self):
        # 5 mrad from the singularity (sin margin 1.25e-5 > 1e-6): must NOT warn
        # and must still round-trip despite the 1/cos(pitch) conditioning.
        import warnings

        q = euler_zyx_to_quat(0.4, np.pi / 2 - 5e-3, -0.3)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            y, p, r = quat_to_euler_zyx(q)
        np.testing.assert_allclose([y, p, r], [0.4, np.pi / 2 - 5e-3, -0.3], atol=1e-9)


class TestAxisAngle:
    def test_known(self):
        q = axis_angle_to_quat([0.0, 0.0, 1.0], np.pi / 2)
        np.testing.assert_allclose(q, [S2, 0.0, 0.0, S2], atol=1e-15)
        axis, angle = quat_to_axis_angle(q)
        np.testing.assert_allclose(axis, [0.0, 0.0, 1.0], atol=1e-15)
        assert angle == pytest.approx(np.pi / 2)

    def test_axis_normalized_on_input(self):
        q1 = axis_angle_to_quat([0.0, 0.0, 10.0], 0.8)
        q2 = axis_angle_to_quat([0.0, 0.0, 1.0], 0.8)
        np.testing.assert_allclose(q1, q2, atol=1e-15)

    def test_zero_axis_raises(self):
        with pytest.raises(ValueError, match="nonzero"):
            axis_angle_to_quat([0.0, 0.0, 0.0], 1.0)

    def test_identity_gives_zero_angle(self):
        axis, angle = quat_to_axis_angle([1.0, 0.0, 0.0, 0.0])
        assert angle == 0.0
        np.testing.assert_allclose(axis, [1.0, 0.0, 0.0])

    def test_double_cover_resolved(self):
        # -q is the same rotation: angle must stay in [0, π].
        q = axis_angle_to_quat([1.0, 0.0, 0.0], 0.6)
        axis, angle = quat_to_axis_angle(-q)
        assert angle == pytest.approx(0.6)
        np.testing.assert_allclose(axis, [1.0, 0.0, 0.0], atol=1e-12)


class TestRodriguesMRP:
    def test_gibbs_known(self):
        # 90° about z: g = ẑ tan(45°) = [0, 0, 1].
        q = axis_angle_to_quat([0, 0, 1], np.pi / 2)
        np.testing.assert_allclose(quat_to_rodrigues(q), [0.0, 0.0, 1.0], atol=1e-14)

    def test_gibbs_roundtrip(self):
        q = Quaternion.from_axis_angle([1.0, -1.0, 0.5], 1.9)
        g = q.to_rodrigues()
        assert Quaternion.from_rodrigues(g).isclose(q, atol=1e-12)

    def test_gibbs_singular_at_180_raises(self):
        q = axis_angle_to_quat([1.0, 0.0, 0.0], np.pi)
        with pytest.raises(ValueError, match="singular at 180"):
            quat_to_rodrigues(q)

    def test_mrp_known(self):
        # 90° about z: p = ẑ tan(22.5°).
        q = axis_angle_to_quat([0, 0, 1], np.pi / 2)
        np.testing.assert_allclose(
            quat_to_mrp(q), [0.0, 0.0, np.tan(np.pi / 8)], atol=1e-14
        )

    def test_mrp_roundtrip(self):
        q = Quaternion.from_axis_angle([0.2, 0.9, -0.4], 2.5)
        assert Quaternion.from_mrp(q.to_mrp()).isclose(q, atol=1e-12)

    def test_mrp_handles_180(self):
        # MRP is finite at 180° (|p| = 1), unlike the Gibbs vector.
        q = axis_angle_to_quat([0.0, 1.0, 0.0], np.pi)
        p = quat_to_mrp(q)
        assert np.linalg.norm(p) == pytest.approx(1.0, abs=1e-12)
        assert Quaternion.from_mrp(p).isclose(Quaternion.from_array(q), atol=1e-12)

    def test_mrp_principal_set(self):
        # Double-cover flip keeps |p| <= 1 even when qw < 0.
        q = -axis_angle_to_quat([0.0, 0.0, 1.0], 0.3)
        assert np.linalg.norm(quat_to_mrp(q)) <= 1.0

    def test_mrp_from_shadow_input(self):
        np.testing.assert_allclose(mrp_to_quat([0.0, 0.0, 0.0]), [1, 0, 0, 0], atol=1e-15)
        np.testing.assert_allclose(
            rodrigues_to_quat([0.0, 0.0, 0.0]), [1, 0, 0, 0], atol=1e-15
        )
