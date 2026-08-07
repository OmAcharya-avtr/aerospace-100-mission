"""Tests for the analytic Cartesian gradients of Zernike modes."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from zernkit import (
    zernike_cartesian,
    zernike_gradient,
    zernike_gradient_noll,
    zernike_gradient_osa,
    zernike_slope_matrix,
)


def _legal_nm(max_n: int = 8) -> list[tuple[int, int]]:
    return [(n, m) for n in range(max_n + 1) for m in range(-n, n + 1, 2)]


def _central_difference(
    n: int, m: int, x: np.ndarray, y: np.ndarray, h: float
) -> tuple[np.ndarray, np.ndarray]:
    """4th-order central difference of Z_n^m, error O(h^4)."""

    def f(xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
        return zernike_cartesian(n, m, xx, yy)

    dx = (
        -f(x + 2 * h, y) + 8 * f(x + h, y) - 8 * f(x - h, y) + f(x - 2 * h, y)
    ) / (12 * h)
    dy = (
        -f(x, y + 2 * h) + 8 * f(x, y + h) - 8 * f(x, y - h) + f(x, y - 2 * h)
    ) / (12 * h)
    return dx, dy


# --- known answers --------------------------------------------------------


def test_piston_gradient_is_zero() -> None:
    gx, gy = zernike_gradient(0, 0, np.array([0.0, 0.4]), np.array([0.0, -0.3]))
    assert np.allclose(gx, 0.0)
    assert np.allclose(gy, 0.0)


def test_tilt_gradient_is_constant() -> None:
    """Noll j=2 is Z = 2x exactly, so dZ/dx = 2 and dZ/dy = 0 everywhere."""
    x = np.array([0.0, 0.1, -0.9, 0.5])
    y = np.array([0.0, -0.5, 0.2, 0.5])
    gx, gy = zernike_gradient_noll(2, x, y)
    assert np.allclose(gx, 2.0)
    assert np.allclose(gy, 0.0, atol=1e-14)
    gx, gy = zernike_gradient_noll(3, x, y)  # Z = 2y
    assert np.allclose(gx, 0.0, atol=1e-14)
    assert np.allclose(gy, 2.0)


def test_defocus_gradient_hand_calculation() -> None:
    """Z4 = sqrt(3)(2(x^2+y^2) - 1) => dZ/dx = 4 sqrt(3) x, dZ/dy = 4 sqrt(3) y."""
    x = np.array([0.0, 0.3, -0.6])
    y = np.array([0.0, 0.4, 0.1])
    gx, gy = zernike_gradient_noll(4, x, y)
    assert np.allclose(gx, 4 * np.sqrt(3) * x)
    assert np.allclose(gy, 4 * np.sqrt(3) * y)


def test_astigmatism_gradient_hand_calculation() -> None:
    """Z6 = sqrt(6) rho^2 cos(2 theta) = sqrt(6)(x^2 - y^2)."""
    x = np.array([0.2, -0.5, 0.7])
    y = np.array([0.5, 0.1, -0.2])
    gx, gy = zernike_gradient_noll(6, x, y)
    assert np.allclose(gx, 2 * np.sqrt(6) * x)
    assert np.allclose(gy, -2 * np.sqrt(6) * y)
    # Z5 = sqrt(6) rho^2 sin(2 theta) = 2 sqrt(6) x y
    gx, gy = zernike_gradient_noll(5, x, y)
    assert np.allclose(gx, 2 * np.sqrt(6) * y)
    assert np.allclose(gy, 2 * np.sqrt(6) * x)


def test_gradient_at_origin_is_finite_and_correct() -> None:
    """The 1/rho factor cancels analytically; no nan at rho = 0.

    Near the origin ``Z ~ N c_1 rho cos/sin(theta)`` where ``c_1`` is the
    coefficient of ``rho^1`` in ``R_n^|m|``. That term exists only for
    ``|m| = 1``, so exactly the ``|m| = 1`` modes have a non-zero slope at the
    pupil centre. Hand check, Noll j=7 (n=3, m=-1):
    ``Z7 = sqrt(8)(3 rho^3 - 2 rho) sin(theta) = sqrt(8)(3 rho^2 - 2) y``, so
    ``dZ/dy(0,0) = -2 sqrt(8) = -5.656854...`` and ``dZ/dx(0,0) = 0``.
    """
    from zernkit import normalization, radial_coefficients

    for n, m in _legal_nm(9):
        gx, gy = zernike_gradient(n, m, 0.0, 0.0)
        assert np.isfinite(gx) and np.isfinite(gy)
        if abs(m) == 1:
            slope = normalization(n, m) * radial_coefficients(n, m)[1]
            expected = (slope, 0.0) if m > 0 else (0.0, slope)
            assert gx == pytest.approx(expected[0], abs=1e-12)
            assert gy == pytest.approx(expected[1], abs=1e-12)
        else:
            assert gx == pytest.approx(0.0, abs=1e-12)
            assert gy == pytest.approx(0.0, abs=1e-12)

    gx, gy = zernike_gradient_noll(7, 0.0, 0.0)
    assert gx == pytest.approx(0.0, abs=1e-12)
    assert gy == pytest.approx(-2 * np.sqrt(8.0), abs=1e-12)


def test_osa_gradient_matches_noll_gradient() -> None:
    from zernkit import noll_to_osa

    x, y = 0.31, -0.42
    for j in range(1, 25):
        gx_n, gy_n = zernike_gradient_noll(j, x, y)
        gx_o, gy_o = zernike_gradient_osa(noll_to_osa(j), x, y)
        assert gx_n == pytest.approx(gx_o)
        assert gy_n == pytest.approx(gy_o)


# --- finite-difference verification --------------------------------------


def test_gradients_match_high_accuracy_finite_differences() -> None:
    rng = np.random.default_rng(20260807)
    r = 0.85 * np.sqrt(rng.random(60))
    t = 2 * np.pi * rng.random(60)
    x, y = r * np.cos(t), r * np.sin(t)
    h = 1e-3
    worst = 0.0
    for n, m in _legal_nm(8):
        gx, gy = zernike_gradient(n, m, x, y)
        fx, fy = _central_difference(n, m, x, y, h)
        scale = max(1.0, float(np.max(np.abs(gx))), float(np.max(np.abs(gy))))
        worst = max(worst, float(np.max(np.abs(gx - fx))) / scale)
        worst = max(worst, float(np.max(np.abs(gy - fy))) / scale)
    # O(h^4) = 1e-12 truncation, plus ~1e-16/h^1 round-off; 1e-8 is generous.
    assert worst < 1e-8


# --- slope matrix ---------------------------------------------------------


def test_slope_matrix_shape_and_content() -> None:
    x = np.array([0.1, -0.2, 0.3])
    y = np.array([0.0, 0.5, -0.4])
    indices = [(1, 1), (2, 0)]
    mat = zernike_slope_matrix(indices, x, y)
    assert mat.shape == (6, 2)
    assert np.allclose(mat[:3, 0], 2.0)
    assert np.allclose(mat[3:, 1], 4 * np.sqrt(3) * y)


def test_slope_matrix_reproduces_a_known_slope_field() -> None:
    x = np.array([0.1, -0.2, 0.3, 0.05])
    y = np.array([0.0, 0.5, -0.4, 0.15])
    indices = [(1, 1), (1, -1), (2, 0)]
    coeffs = np.array([0.3, -0.2, 0.7])
    mat = zernike_slope_matrix(indices, x, y)
    slopes = mat @ coeffs
    expect_x = 0.3 * 2.0 + 0.7 * 4 * np.sqrt(3) * x
    expect_y = -0.2 * 2.0 + 0.7 * 4 * np.sqrt(3) * y
    assert np.allclose(slopes[:4], expect_x)
    assert np.allclose(slopes[4:], expect_y)


def test_slope_matrix_validation() -> None:
    with pytest.raises(ValueError, match="at least one"):
        zernike_slope_matrix([], [0.0], [0.0])
    with pytest.raises(ValueError, match="same size"):
        zernike_slope_matrix([(1, 1)], [0.0, 0.1], [0.0])


def test_gradient_rejects_illegal_indices() -> None:
    with pytest.raises(ValueError):
        zernike_gradient(3, 2, 0.1, 0.1)


# --- property-based -------------------------------------------------------

_NM = st.integers(min_value=0, max_value=10).flatmap(
    lambda n: st.tuples(st.just(n), st.sampled_from(list(range(-n, n + 1, 2))))
)


@given(
    _NM,
    st.floats(min_value=-0.7, max_value=0.7),
    st.floats(min_value=-0.7, max_value=0.7),
)
@settings(max_examples=200, deadline=None)
def test_property_gradient_matches_finite_difference(
    nm: tuple[int, int], x: float, y: float
) -> None:
    n, m = nm
    h = 1e-3
    gx, gy = zernike_gradient(n, m, x, y)
    fx, fy = _central_difference(n, m, np.array(x), np.array(y), h)
    scale = max(1.0, abs(float(gx)), abs(float(gy)))
    assert abs(float(gx) - float(fx)) / scale < 1e-7
    assert abs(float(gy) - float(fy)) / scale < 1e-7
