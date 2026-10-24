"""Vector geometry: validation, projections and the small-angle path."""

from __future__ import annotations

import numpy as np
import pytest

from pointahead.geometry import (
    C_LIGHT,
    angle_between,
    as_vector,
    norm,
    orthonormal_basis,
    radial,
    rotation_matrix,
    transverse,
    unit,
)


def test_speed_of_light_is_the_si_defining_value() -> None:
    assert C_LIGHT == 299792458.0


@pytest.mark.parametrize(
    "value",
    [[1.0, 2.0], [1.0, 2.0, 3.0, 4.0], np.zeros((2, 2))],
)
def test_as_vector_rejects_wrong_length(value: object) -> None:
    with pytest.raises(ValueError, match="exactly 3 components"):
        as_vector(value, "sample")


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_as_vector_rejects_non_finite(bad: float) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        as_vector([1.0, bad, 3.0], "sample")


def test_as_vector_rejects_non_numeric() -> None:
    with pytest.raises(TypeError, match="length-3 sequence"):
        as_vector(["a", "b", "c"], "sample")


def test_unit_rejects_zero_vector() -> None:
    with pytest.raises(ValueError, match="zero norm"):
        unit([0.0, 0.0, 0.0], "sample")


def test_unit_norm_is_one() -> None:
    assert norm(unit([3.0, 4.0, 0.0])) == pytest.approx(1.0, abs=1e-15)


def test_norm_known_answer() -> None:
    # 3-4-5 right triangle: |[3, 4, 0]| = 5 exactly in binary floating point.
    assert norm([3.0, 4.0, 0.0]) == 5.0


def test_transverse_removes_the_radial_part() -> None:
    v = np.array([3.0, 4.0, 5.0])
    n = np.array([0.0, 0.0, 2.0])
    perpendicular = transverse(v, n)
    assert perpendicular[2] == pytest.approx(0.0, abs=1e-15)
    assert radial(v, n) == pytest.approx(5.0, rel=1e-15)


def test_transverse_plus_radial_reconstructs_the_vector() -> None:
    v = np.array([-1.5, 2.25, 7.0])
    n = unit([1.0, 1.0, 1.0])
    assert np.allclose(transverse(v, n) + radial(v, n) * n, v, atol=1e-13)


def test_angle_between_resolves_a_nanoradian() -> None:
    # arccos of a normalised dot product cannot: at 1e-9 rad the dot product is
    # 1 - 5e-19, which rounds to exactly 1.0 in float64 and returns 0.0.
    a = np.array([1.0, 0.0, 0.0])
    b = unit([1.0, 1.0e-9, 0.0])
    assert angle_between(a, b) == pytest.approx(1.0e-9, rel=1e-9)
    assert float(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))) == 0.0


def test_angle_between_known_answers() -> None:
    assert angle_between([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]) == pytest.approx(np.pi / 2)
    assert angle_between([1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]) == pytest.approx(np.pi)
    assert angle_between([1.0, 1.0, 0.0], [1.0, 0.0, 0.0]) == pytest.approx(np.pi / 4)


def test_angle_between_rejects_zero_vectors() -> None:
    with pytest.raises(ValueError, match="non-zero"):
        angle_between([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])


def test_rotation_matrix_is_orthogonal_with_unit_determinant() -> None:
    matrix = rotation_matrix([1.0, 2.0, 3.0], 0.7)
    assert np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-14)
    assert float(np.linalg.det(matrix)) == pytest.approx(1.0, abs=1e-14)


def test_rotation_matrix_quarter_turn_known_answer() -> None:
    # A quarter turn about +z sends +x to +y.
    matrix = rotation_matrix([0.0, 0.0, 1.0], np.pi / 2)
    assert np.allclose(matrix @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-15)


def test_rotation_matrix_rejects_non_finite_angle() -> None:
    with pytest.raises(ValueError, match="angle_rad must be finite"):
        rotation_matrix([0.0, 0.0, 1.0], np.nan)


def test_orthonormal_basis_is_right_handed_and_orthonormal() -> None:
    n, e1, e2 = orthonormal_basis([0.3, -0.4, 0.5])
    for a in (n, e1, e2):
        assert norm(a) == pytest.approx(1.0, abs=1e-14)
    assert float(np.dot(n, e1)) == pytest.approx(0.0, abs=1e-15)
    assert float(np.dot(n, e2)) == pytest.approx(0.0, abs=1e-15)
    assert float(np.dot(e1, e2)) == pytest.approx(0.0, abs=1e-15)
    assert np.allclose(np.cross(e1, e2), n, atol=1e-14)


def test_orthonormal_basis_is_deterministic() -> None:
    first = orthonormal_basis([1.0, 0.0, 0.0])
    second = orthonormal_basis([2.0, 0.0, 0.0])
    for a, b in zip(first, second, strict=True):
        assert np.allclose(a, b, atol=0.0)
