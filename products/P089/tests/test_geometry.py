"""SO(3) algebra, including an independent cross-check against scipy.

``scipy.spatial.transform.Rotation`` is installed in the measured environment
and is an implementation this package did not write, so the agreement checks
here are a genuine independent cross-check and are labelled as such. The
``scipy`` dependency exists only for this and for ``scipy.special.erf`` in
:mod:`aligncal.benchmark`; nothing in the estimators calls it.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from aligncal.geometry import (
    SMALL_ANGLE_RAD,
    angle_between,
    is_rotation,
    rotation_angle,
    rotation_from_vector,
    rotation_to_vector,
    skew,
    unskew,
)


def test_skew_matches_cross_product():
    """Known answer, by hand: [e1]x e2 = e3."""
    assert np.allclose(skew(np.array([1.0, 0.0, 0.0])) @ np.array([0.0, 1.0, 0.0]),
                       np.array([0.0, 0.0, 1.0]), atol=0.0, rtol=0.0)


def test_skew_hand_computed_entries():
    """[a]x for a = (1, 2, 3) written out by hand."""
    expected = np.array([[0.0, -3.0, 2.0], [3.0, 0.0, -1.0], [-2.0, 1.0, 0.0]])
    assert np.array_equal(skew(np.array([1.0, 2.0, 3.0])), expected)


def test_skew_is_antisymmetric():
    rng = np.random.default_rng(0)
    for _ in range(25):
        vector = rng.normal(size=3)
        matrix = skew(vector)
        assert np.max(np.abs(matrix + matrix.T)) == 0.0


def test_skew_broadcasts():
    block = skew(np.arange(30.0).reshape(10, 3))
    assert block.shape == (10, 3, 3)
    assert np.array_equal(block[4], skew(np.array([12.0, 13.0, 14.0])))


def test_skew_rejects_wrong_shape():
    with pytest.raises(ValueError, match="last axis of length 3"):
        skew(np.zeros((4, 2)))


def test_unskew_inverts_skew():
    rng = np.random.default_rng(1)
    for _ in range(25):
        vector = rng.normal(size=3)
        assert np.max(np.abs(unskew(skew(vector)) - vector)) == 0.0


def test_unskew_takes_the_antisymmetric_part():
    """A symmetric matrix carries no rotation, so unskew must return zero."""
    symmetric = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]])
    assert np.array_equal(unskew(symmetric), np.zeros(3))


def test_unskew_rejects_wrong_shape():
    with pytest.raises(ValueError, match=r"trailing \(3, 3\)"):
        unskew(np.zeros((3, 4)))


def test_identity_three_of_the_geometry_docstring():
    """[a]x^T [a]x = |a|^2 I - a a^T, which is equation (3) and the whole product."""
    rng = np.random.default_rng(2)
    worst = 0.0
    for _ in range(200):
        vector = rng.normal(size=3) * rng.uniform(1.0e-6, 10.0)
        left = skew(vector).T @ skew(vector)
        right = float(vector @ vector) * np.eye(3) - np.outer(vector, vector)
        worst = max(worst, float(np.max(np.abs(left - right))))
    assert worst < 1.0e-13


def test_skew_annihilates_its_own_vector():
    """[a]x a = 0 exactly: the reason the rotation axis is unobservable."""
    rng = np.random.default_rng(3)
    for _ in range(50):
        vector = rng.normal(size=3)
        assert np.max(np.abs(skew(vector) @ vector)) < 1.0e-16 * float(vector @ vector)


def test_rotation_from_vector_is_a_rotation():
    rng = np.random.default_rng(4)
    for _ in range(100):
        theta = rng.normal(size=3) * rng.uniform(0.0, 1.0)
        assert is_rotation(rotation_from_vector(theta))


def test_rotation_from_vector_known_answer_quarter_turn():
    """A quarter turn about z takes e1 to e2. Hand-computed."""
    matrix = rotation_from_vector(np.array([0.0, 0.0, np.pi / 2.0]))
    assert np.max(np.abs(matrix @ np.array([1.0, 0.0, 0.0]) - np.array([0.0, 1.0, 0.0]))) < 1e-15


def test_rotation_from_vector_known_answer_half_turn():
    """A half turn about x maps (0, 1, 0) to (0, -1, 0). Hand-computed."""
    matrix = rotation_from_vector(np.array([np.pi, 0.0, 0.0]))
    assert np.max(np.abs(matrix @ np.array([0.0, 1.0, 0.0]) - np.array([0.0, -1.0, 0.0]))) < 1e-15


def test_rotation_from_vector_zero_is_identity():
    assert np.array_equal(rotation_from_vector(np.zeros(3)), np.eye(3))


def test_rotation_from_vector_small_angle_branch():
    """Below SMALL_ANGLE_RAD the Taylor branch is used; it must still be a rotation."""
    theta = np.array([1.0, -2.0, 0.5]) * (SMALL_ANGLE_RAD / 10.0)
    matrix = rotation_from_vector(theta)
    assert is_rotation(matrix, tolerance=1.0e-14)
    assert np.max(np.abs(unskew(matrix) - theta)) < 1.0e-22


def test_rotation_from_vector_rejects_bad_input():
    with pytest.raises(ValueError, match="3 components"):
        rotation_from_vector(np.zeros(4))
    with pytest.raises(ValueError, match="finite"):
        rotation_from_vector(np.array([np.nan, 0.0, 0.0]))


def test_rotation_angle_known_answers():
    assert rotation_angle(np.eye(3)) == 0.0
    quarter = rotation_from_vector(np.array([0.0, 0.0, np.pi / 2.0]))
    assert abs(rotation_angle(quarter) - np.pi / 2.0) < 1.0e-15


def test_rotation_angle_rejects_wrong_shape():
    with pytest.raises(ValueError, match=r"\(3, 3\) matrix"):
        rotation_angle(np.zeros((2, 2)))


def test_rotation_to_vector_round_trip():
    """The factor in the logarithm is p / sin p, not p / (2 sin p)."""
    rng = np.random.default_rng(5)
    worst = 0.0
    for _ in range(500):
        axis = rng.normal(size=3)
        axis = axis / np.linalg.norm(axis)
        theta = axis * rng.uniform(0.0, np.pi * 0.999)
        worst = max(
            worst, float(np.max(np.abs(rotation_to_vector(rotation_from_vector(theta)) - theta)))
        )
    assert worst < 1.0e-10


def test_rotation_to_vector_near_pi_branch():
    """The symmetric-part branch at p close to pi recovers the axis and its sign."""
    axis = np.array([0.3, -0.5, 0.81])
    axis = axis / np.linalg.norm(axis)
    for epsilon in (1.0e-7, 1.0e-9, 1.0e-12):
        theta = axis * (np.pi - epsilon)
        recovered = rotation_to_vector(rotation_from_vector(theta))
        assert np.max(np.abs(recovered - theta)) < 1.0e-6


def test_rotation_to_vector_rejects_bad_input():
    with pytest.raises(ValueError, match=r"\(3, 3\) matrix"):
        rotation_to_vector(np.zeros(3))
    with pytest.raises(ValueError, match="finite"):
        rotation_to_vector(np.full((3, 3), np.nan))


def test_is_rotation_rejects_non_rotations():
    assert not is_rotation(np.zeros((3, 3)))
    assert not is_rotation(2.0 * np.eye(3))
    assert not is_rotation(np.diag([1.0, 1.0, -1.0]))  # det = -1
    assert not is_rotation(np.zeros((2, 2)))
    assert not is_rotation(np.full((3, 3), np.inf))


def test_angle_between_known_answers():
    assert angle_between(np.array([1.0, 0, 0]), np.array([0, 1.0, 0])) == pytest.approx(
        np.pi / 2.0, rel=1.0e-12, abs=0.0
    )
    assert angle_between(np.array([1.0, 0, 0]), np.array([-2.0, 0, 0])) == pytest.approx(
        np.pi, rel=1.0e-12, abs=1.0e-15
    )
    assert angle_between(np.array([1.0, 1.0, 0]), np.array([3.0, 3.0, 0])) < 1.0e-8


def test_angle_between_rejects_zero_vector():
    with pytest.raises(ValueError, match="zero-length"):
        angle_between(np.zeros(3), np.array([1.0, 0.0, 0.0]))


def test_angle_between_rejects_wrong_size():
    with pytest.raises(ValueError, match="two 3-vectors"):
        angle_between(np.zeros(2), np.zeros(3))


# ---------------------------------------------------------------------------
# Independent cross-check against scipy.spatial.transform, an implementation
# this package did not write.
# ---------------------------------------------------------------------------


def test_cross_check_rotation_matrix_against_scipy():
    rng = np.random.default_rng(6)
    worst = 0.0
    for _ in range(500):
        axis = rng.normal(size=3)
        axis = axis / np.linalg.norm(axis)
        theta = axis * rng.uniform(0.0, np.pi)
        worst = max(
            worst,
            float(
                np.max(
                    np.abs(
                        rotation_from_vector(theta) - Rotation.from_rotvec(theta).as_matrix()
                    )
                )
            ),
        )
    assert worst < 1.0e-14


def test_cross_check_logarithm_against_scipy():
    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(500):
        axis = rng.normal(size=3)
        axis = axis / np.linalg.norm(axis)
        theta = axis * rng.uniform(0.0, np.pi * 0.999)
        matrix = Rotation.from_rotvec(theta).as_matrix()
        worst = max(worst, float(np.max(np.abs(rotation_to_vector(matrix) - theta))))
    assert worst < 1.0e-10


def test_cross_check_composition_against_scipy():
    """exp([a]x) exp([b]x) must equal scipy's composition of the same two rotations."""
    rng = np.random.default_rng(8)
    worst = 0.0
    for _ in range(200):
        first = rng.normal(size=3) * 0.4
        second = rng.normal(size=3) * 0.4
        mine = rotation_from_vector(first) @ rotation_from_vector(second)
        theirs = (Rotation.from_rotvec(first) * Rotation.from_rotvec(second)).as_matrix()
        worst = max(worst, float(np.max(np.abs(mine - theirs))))
    assert worst < 1.0e-14
