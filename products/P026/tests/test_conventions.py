"""Conventions: quaternion <-> DCM, frame order, canonical sign, error vector."""

from __future__ import annotations

import numpy as np
import pytest

from wahbakit import (
    angle_between_dcm,
    attitude_error_vector,
    dcm_from_quat,
    is_rotation,
    quat_canonical,
    quat_conjugate,
    quat_from_dcm,
    quat_multiply,
    quat_normalize,
    rotation_vector_from_dcm,
    skew,
    unit_vectors,
)


def test_skew_is_cross_product():
    a = np.array([1.0, -2.0, 3.0])
    b = np.array([0.5, 4.0, -1.0])
    assert np.allclose(skew(a) @ b, np.cross(a, b))


def test_skew_is_antisymmetric():
    m = skew([0.3, -0.7, 2.0])
    assert np.allclose(m, -m.T)


def test_identity_quaternion_gives_identity_matrix():
    assert np.allclose(dcm_from_quat([1.0, 0.0, 0.0, 0.0]), np.eye(3))


def test_known_answer_90_deg_about_z():
    # Hand check: q = [cos45, 0, 0, sin45] rotates +x onto +y.
    # A = [[0,-1,0],[1,0,0],[0,0,1]] exactly.
    root = np.sqrt(0.5)
    dcm = dcm_from_quat([root, 0.0, 0.0, root])
    expected = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert np.allclose(dcm, expected, atol=1e-15)
    assert np.allclose(dcm @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-15)


def test_known_answer_180_deg_about_x():
    # q = [0, 1, 0, 0]: A = diag(1, -1, -1).
    assert np.allclose(dcm_from_quat([0.0, 1.0, 0.0, 0.0]), np.diag([1.0, -1.0, -1.0]))


def test_quat_from_dcm_round_trip():
    rng = np.random.default_rng(4)
    for _ in range(200):
        q = quat_canonical(rng.normal(size=4))
        assert np.allclose(quat_from_dcm(dcm_from_quat(q)), q, atol=1e-12)


def test_quat_from_dcm_all_four_shepperd_branches():
    # One rotation per branch: identity (trace), and 180 deg about x, y, z.
    for q in ([1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]):
        arr = np.array(q, dtype=float)
        assert np.allclose(dcm_from_quat(quat_from_dcm(dcm_from_quat(arr))), dcm_from_quat(arr))


def test_quat_multiply_matches_matrix_product():
    rng = np.random.default_rng(5)
    for _ in range(100):
        q1 = quat_normalize(rng.normal(size=4))
        q2 = quat_normalize(rng.normal(size=4))
        assert np.allclose(
            dcm_from_quat(quat_multiply(q2, q1)), dcm_from_quat(q2) @ dcm_from_quat(q1), atol=1e-13
        )


def test_conjugate_is_inverse_rotation():
    q = quat_normalize([0.3, -0.4, 0.5, 0.7])
    assert np.allclose(dcm_from_quat(quat_conjugate(q)), dcm_from_quat(q).T)


def test_canonical_sign_is_non_negative_scalar():
    q = quat_canonical([-0.5, 0.5, -0.5, 0.5])
    assert q[0] >= 0.0


def test_canonical_falls_through_to_first_non_zero_component():
    q = quat_canonical([0.0, -1.0, 0.0, 0.0])
    assert q[1] > 0.0


def test_is_rotation_rejects_reflection():
    assert not is_rotation(np.diag([1.0, 1.0, -1.0]))
    assert not is_rotation(2.0 * np.eye(3))
    assert is_rotation(np.eye(3))


def test_rotation_vector_small_angle_precision():
    # 1e-9 rad about z; the arccos((tr-1)/2) route would lose half the digits.
    angle = 1e-9
    dcm = dcm_from_quat([np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)])
    assert np.allclose(rotation_vector_from_dcm(dcm), [0.0, 0.0, angle], atol=1e-18)


def test_attitude_error_vector_is_body_frame():
    rng = np.random.default_rng(6)
    dcm_true = dcm_from_quat(rng.normal(size=4))
    delta = np.array([1e-4, -2e-4, 3e-4])
    dcm_est = (np.eye(3) + skew(delta)) @ dcm_true
    # Re-orthogonalise so the input is a proper rotation.
    u, _, vt = np.linalg.svd(dcm_est)
    dcm_est = u @ vt
    assert np.allclose(attitude_error_vector(dcm_est, dcm_true), delta, atol=1e-8)


def test_angle_between_dcm_is_symmetric_and_zero_for_equal():
    rng = np.random.default_rng(7)
    a = dcm_from_quat(rng.normal(size=4))
    b = dcm_from_quat(rng.normal(size=4))
    assert angle_between_dcm(a, a) == pytest.approx(0.0, abs=1e-15)
    assert angle_between_dcm(a, b) == pytest.approx(angle_between_dcm(b, a), abs=1e-12)


@pytest.mark.parametrize(
    ("bad", "message"),
    [
        ([[0.0, 0.0, 0.0]], "norm"),
        ([[1.0, 0.0]], "shape"),
        ([[np.nan, 0.0, 1.0]], "non-finite"),
    ],
)
def test_unit_vectors_rejects_bad_input(bad, message):
    with pytest.raises(ValueError, match=message):
        unit_vectors(bad)


def test_quat_normalize_rejects_wrong_length_and_zero():
    with pytest.raises(ValueError, match="4 components"):
        quat_normalize([1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="norm"):
        quat_normalize([0.0, 0.0, 0.0, 0.0])


def test_quat_from_dcm_rejects_non_rotation():
    with pytest.raises(ValueError, match="not a proper rotation"):
        quat_from_dcm(np.diag([1.0, 1.0, -1.0]))
    with pytest.raises(ValueError, match="shape"):
        quat_from_dcm(np.eye(2))


def test_scipy_convention_agreement():
    scipy_transform = pytest.importorskip("scipy.spatial.transform")
    rng = np.random.default_rng(8)
    for _ in range(50):
        q = quat_canonical(rng.normal(size=4))
        scipy_matrix = scipy_transform.Rotation.from_quat(np.roll(q, -1)).as_matrix()
        assert np.allclose(dcm_from_quat(q), scipy_matrix, atol=1e-14)
