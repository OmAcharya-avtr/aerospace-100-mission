"""Frames, quaternions, and the attitude solve."""

from __future__ import annotations

import numpy as np
import pytest

from skymatch.geometry import (
    ARCSEC,
    angle_between_dcm,
    angular_separation,
    davenport_attitude,
    dcm_from_quat,
    normalise,
    quat_from_dcm,
    radec_from_unit_vectors,
    random_rotation,
    skew,
    unit_vectors_from_radec,
)


class TestKnownAnswers:
    def test_radec_axes(self) -> None:
        # Eq. G1 by hand: (ra, dec) = (0, 0) -> +x; (90 deg, 0) -> +y;
        # (0, 90 deg) -> +z.
        got = unit_vectors_from_radec([0.0, np.pi / 2, 0.0], [0.0, 0.0, np.pi / 2])
        assert np.allclose(got, np.eye(3), atol=1e-15)

    def test_separation_right_angle(self) -> None:
        # +x to +y is exactly pi/2.
        assert angular_separation([1.0, 0, 0], [0, 1.0, 0])[0] == pytest.approx(np.pi / 2)

    def test_separation_ten_degrees_on_the_equator(self) -> None:
        # Two points on the celestial equator 10 deg apart in right ascension
        # are 10 deg apart on the sky, exactly.
        v = unit_vectors_from_radec(np.radians([0.0, 10.0]), [0.0, 0.0])
        assert np.degrees(angular_separation(v[0], v[1])[0]) == pytest.approx(10.0, abs=1e-12)

    def test_antipodal_is_pi(self) -> None:
        assert angular_separation([1.0, 0, 0], [-1.0, 0, 0])[0] == pytest.approx(np.pi)

    def test_identity_quaternion(self) -> None:
        assert np.allclose(dcm_from_quat([1.0, 0.0, 0.0, 0.0]), np.eye(3))

    def test_ninety_degrees_about_z(self) -> None:
        # q = [cos45, 0, 0, sin45] is a +90 deg rotation about z. With
        # v_new = A v_old and A the reference-to-body map, the reference +x
        # axis is seen at body -y... the convention check is that A maps the
        # rotated frame's axes: A @ [1,0,0] = [0, -1, 0] for the passive form
        # used throughout (v_cam = A r_inertial).
        a = dcm_from_quat([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])
        assert np.allclose(a @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-15)
        assert np.allclose(a @ np.array([0.0, 1.0, 0.0]), [-1.0, 0.0, 0.0], atol=1e-15)

    def test_skew_by_hand(self) -> None:
        # [v x] for v = (1, 2, 3); cross product with w must match.
        v = np.array([1.0, 2.0, 3.0])
        w = np.array([4.0, 5.0, 6.0])
        assert np.allclose(skew(v) @ w, np.cross(v, w))
        assert np.allclose(skew(v), [[0, -3, 2], [3, 0, -1], [-2, 1, 0]])

    def test_arcsec_constant(self) -> None:
        # 1 arcsec = pi / (180 * 3600) rad = 4.8481368e-6 rad.
        assert float(ARCSEC) == pytest.approx(4.84813681e-6, rel=1e-8)

    def test_attitude_from_two_known_vectors(self) -> None:
        # A 90 deg rotation about z recovered from the two axes it moves.
        a_true = dcm_from_quat([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])
        ref = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        assert angle_between_dcm(davenport_attitude(ref @ a_true.T, ref), a_true) < 1e-14


class TestAttitudeSolve:
    def test_returns_a_not_a_transpose(self) -> None:
        # The regression that Davenport's K is in Shuster's convention, whose
        # attitude matrix is the transpose of this package's. Without the sign
        # flip the answer is orthogonal, det +1, and the inverse rotation.
        rng = np.random.default_rng(3)
        a_true = random_rotation(rng)
        ref = normalise(rng.normal(size=(5, 3)))
        est = davenport_attitude(ref @ a_true.T, ref)
        assert angle_between_dcm(est, a_true) < 1e-13
        assert angle_between_dcm(est, a_true.T) > 1e-3

    def test_noise_free_exact_for_many_n(self) -> None:
        rng = np.random.default_rng(4)
        for n in (2, 3, 5, 9):
            a_true = random_rotation(rng)
            ref = normalise(rng.normal(size=(n, 3)))
            assert angle_between_dcm(davenport_attitude(ref @ a_true.T, ref), a_true) < 1e-13

    def test_weights_are_used(self) -> None:
        # One good observation and one wildly wrong one: weighting the wrong
        # one to nearly zero must move the answer towards the good one.
        rng = np.random.default_rng(5)
        a_true = random_rotation(rng)
        ref = normalise(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
        body = ref @ a_true.T
        body[2] = normalise(body[2] + np.array([0.4, 0.0, 0.0]))[0]
        equal = angle_between_dcm(davenport_attitude(body, ref), a_true)
        weighted = angle_between_dcm(
            davenport_attitude(body, ref, weights=[1.0, 1.0, 1e-6]), a_true
        )
        assert weighted < equal

    def test_collinear_raises(self) -> None:
        v = np.array([[1.0, 0.0, 0.0], [1.0, 1e-15, 0.0]])
        with pytest.raises(ValueError, match="not observable"):
            davenport_attitude(v, v)

    def test_angle_metric_resolves_below_the_trace_floor(self) -> None:
        # arccos((tr - 1)/2) cannot resolve below about 3e-8 rad. The
        # quaternion form can, and everything in validation depends on it.
        tiny = 1e-12
        a = dcm_from_quat([np.cos(tiny / 2), np.sin(tiny / 2), 0.0, 0.0])
        assert angle_between_dcm(a, np.eye(3)) == pytest.approx(tiny, rel=1e-4)


class TestValidation:
    @pytest.mark.parametrize(
        "bad", [[0.0, 0.0, 0.0], [np.nan, 0.0, 1.0], [np.inf, 0.0, 0.0]]
    )
    def test_normalise_rejects_degenerate_rows(self, bad: list[float]) -> None:
        with pytest.raises(ValueError):
            normalise([bad])

    def test_normalise_rejects_wrong_shape(self) -> None:
        with pytest.raises(ValueError, match=r"shape \(3,\) or \(N, 3\)"):
            normalise(np.zeros((4, 2)))

    def test_radec_rejects_mismatched_shapes(self) -> None:
        with pytest.raises(ValueError, match="same shape"):
            unit_vectors_from_radec([0.0, 1.0], [0.0])

    def test_radec_rejects_out_of_range_dec(self) -> None:
        with pytest.raises(ValueError, match=r"\[-pi/2, pi/2\]"):
            unit_vectors_from_radec([0.0], [2.0])

    def test_quat_rejects_zero(self) -> None:
        with pytest.raises(ValueError, match="finite and non-zero"):
            dcm_from_quat([0.0, 0.0, 0.0, 0.0])

    def test_quat_from_dcm_rejects_reflection(self) -> None:
        with pytest.raises(ValueError, match="proper rotation"):
            quat_from_dcm(np.diag([1.0, 1.0, -1.0]))

    def test_davenport_rejects_one_observation(self) -> None:
        with pytest.raises(ValueError, match="at least 2 observations"):
            davenport_attitude([[1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]])

    def test_davenport_rejects_bad_weights(self) -> None:
        v = np.eye(3)
        with pytest.raises(ValueError, match="length"):
            davenport_attitude(v, v, weights=[1.0])
        with pytest.raises(ValueError, match="non-negative"):
            davenport_attitude(v, v, weights=[1.0, -1.0, 1.0])
        with pytest.raises(ValueError, match="not all be zero"):
            davenport_attitude(v, v, weights=[0.0, 0.0, 0.0])

    def test_davenport_rejects_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="same shape"):
            davenport_attitude(np.eye(3), np.eye(3)[:2])


class TestRoundTrips:
    def test_radec_round_trip(self) -> None:
        rng = np.random.default_rng(6)
        ra = rng.random(200) * 2.0 * np.pi
        dec = np.arcsin(2.0 * rng.random(200) - 1.0)
        ra2, dec2 = radec_from_unit_vectors(unit_vectors_from_radec(ra, dec))
        assert np.allclose(dec, dec2, atol=1e-14)
        assert np.allclose(np.mod(ra - ra2 + np.pi, 2 * np.pi) - np.pi, 0.0, atol=1e-13)

    def test_quat_dcm_round_trip(self) -> None:
        rng = np.random.default_rng(7)
        for _ in range(50):
            a = random_rotation(rng)
            assert np.allclose(dcm_from_quat(quat_from_dcm(a)), a, atol=1e-13)

    def test_random_rotation_is_a_rotation(self) -> None:
        rng = np.random.default_rng(8)
        for _ in range(50):
            a = random_rotation(rng)
            assert np.allclose(a @ a.T, np.eye(3), atol=1e-14)
            assert np.linalg.det(a) == pytest.approx(1.0)
