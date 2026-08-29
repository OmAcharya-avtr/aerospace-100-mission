"""Zernike basis: indexing, normalisation, closed forms and analytic gradients."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from wavelab.zernike import (
    nm_to_noll,
    nm_to_osa,
    noll_to_nm,
    noll_to_osa,
    osa_to_nm,
    radial_coefficients,
    zernike_basis,
    zernike_gradient_basis,
    zernike_gradient_noll,
    zernike_noll,
    zernike_norm,
)

# Noll (1976) JOSA 66(3) 207, Table I, transcribed by hand:
#   j :  1     2     3     4     5     6     7     8     9    10    11
#   n :  0     1     1     2     2     2     3     3     3     3     4
#   m :  0    +1    -1     0    -2    +2    -1    +1    -3    +3     0
NOLL_TABLE_I = [
    (1, 0, 0),
    (2, 1, 1),
    (3, 1, -1),
    (4, 2, 0),
    (5, 2, -2),
    (6, 2, 2),
    (7, 3, -1),
    (8, 3, 1),
    (9, 3, -3),
    (10, 3, 3),
    (11, 4, 0),
]


@pytest.mark.parametrize(("j", "n", "m"), NOLL_TABLE_I)
def test_noll_table_i(j: int, n: int, m: int) -> None:
    """Known answer: reproduce Noll's own published index table."""
    assert noll_to_nm(j) == (n, m)
    assert nm_to_noll(n, m) == j


# OSA/ANSI ordering (Thibos et al. 2002): j = 0..5 -> (0,0) (1,-1) (1,1)
# (2,-2) (2,0) (2,2). Hand-checked against j = (n(n+2)+m)/2.
@pytest.mark.parametrize(
    ("j", "nm"),
    [(0, (0, 0)), (1, (1, -1)), (2, (1, 1)), (3, (2, -2)), (4, (2, 0)), (5, (2, 2))],
)
def test_osa_table(j: int, nm: tuple[int, int]) -> None:
    assert osa_to_nm(j) == nm
    assert nm_to_osa(*nm) == j


def test_noll_and_osa_disagree_from_tip_tilt() -> None:
    """The two conventions already swap tip and tilt; documented, not a bug."""
    assert noll_to_osa(1) == 0
    assert noll_to_osa(2) == 2
    assert noll_to_osa(3) == 1
    assert noll_to_osa(4) == 4


@settings(max_examples=200, deadline=None)
@given(st.integers(min_value=1, max_value=5000))
def test_noll_roundtrip(j: int) -> None:
    """Property: noll_to_nm and nm_to_noll are exact mutual inverses."""
    assert nm_to_noll(*noll_to_nm(j)) == j


def test_radial_polynomial_is_one_at_the_rim() -> None:
    """Known answer: R_n^m(1) = 1 for every legal (n, m) (Born & Wolf, Sec. 9.2)."""
    for j in range(1, 46):
        n, m = noll_to_nm(j)
        assert radial_coefficients(n, m).sum() == pytest.approx(1.0, abs=1e-12)


def test_defocus_closed_form() -> None:
    """Known answer: Z_4 (Noll) = sqrt(3) (2 rho^2 - 1), hand-evaluated at rho = 0, 0.5, 1."""
    rho = np.array([0.0, 0.5, 1.0])
    got = zernike_noll(4, rho, np.zeros_like(rho))
    want = np.sqrt(3.0) * (2.0 * rho**2 - 1.0)
    assert got == pytest.approx(want, abs=1e-13)
    # At rho = 0.5 that is sqrt(3) * (-0.5) = -0.8660254037844386.
    assert got[1] == pytest.approx(-0.8660254037844386, abs=1e-13)


def test_tilt_closed_form() -> None:
    """Known answer: Z_2 = 2 x, Z_3 = 2 y in the Noll normalisation."""
    x = np.array([0.0, 0.3, -0.7])
    y = np.array([0.2, -0.4, 0.1])
    assert zernike_noll(2, x, y) == pytest.approx(2.0 * x, abs=1e-14)
    assert zernike_noll(3, x, y) == pytest.approx(2.0 * y, abs=1e-14)


def test_normalisation_values() -> None:
    """Noll (1976) Eq. 2: sqrt(2(n+1)) for m != 0, sqrt(n+1) for m = 0."""
    assert zernike_norm(0, 0) == pytest.approx(1.0)
    assert zernike_norm(1, 1) == pytest.approx(2.0)
    assert zernike_norm(2, 0) == pytest.approx(np.sqrt(3.0))
    assert zernike_norm(4, 2) == pytest.approx(np.sqrt(10.0))


def test_orthonormality_on_a_fine_disc() -> None:
    """Discrete orthonormality under the 1/pi weight, to the sampling accuracy."""
    n = 512
    ax = np.linspace(-1.0, 1.0, n)
    gx, gy = np.meshgrid(ax, ax, indexing="xy")
    inside = gx**2 + gy**2 <= 1.0
    basis = zernike_basis(list(range(1, 12)), gx[inside], gy[inside])
    gram = basis.T @ basis / basis.shape[0]
    assert np.max(np.abs(gram - np.eye(11))) < 6e-3


@pytest.mark.parametrize("j", [2, 3, 4, 5, 8, 11, 15, 21])
def test_gradient_matches_central_difference(j: int) -> None:
    rng = np.random.default_rng(3)
    r = np.sqrt(rng.random(60)) * 0.9
    t = rng.random(60) * 2 * np.pi
    x, y = r * np.cos(t), r * np.sin(t)
    h = 1e-6
    gx, gy = zernike_gradient_noll(j, x, y)
    fx = (zernike_noll(j, x + h, y) - zernike_noll(j, x - h, y)) / (2 * h)
    fy = (zernike_noll(j, x, y + h) - zernike_noll(j, x, y - h)) / (2 * h)
    assert np.max(np.abs(gx - fx)) < 1e-6
    assert np.max(np.abs(gy - fy)) < 1e-6


def test_gradient_at_the_origin_is_finite() -> None:
    """The 1/rho in the chain rule is removable; no NaN at rho = 0."""
    z = np.zeros(1)
    for j in range(2, 22):
        gx, gy = zernike_gradient_noll(j, z, z)
        assert np.all(np.isfinite(gx))
        assert np.all(np.isfinite(gy))
    # Tip: dZ_2/dx = 2 exactly everywhere, including the origin.
    assert zernike_gradient_noll(2, z, z)[0][0] == pytest.approx(2.0, abs=1e-14)


@settings(max_examples=50, deadline=None)
@given(
    st.floats(min_value=-2.0, max_value=2.0),
    st.floats(min_value=-2.0, max_value=2.0),
    st.floats(min_value=-0.9, max_value=0.9),
    st.floats(min_value=-0.9, max_value=0.9),
)
def test_basis_linearity(a: float, b: float, x: float, y: float) -> None:
    """Property: the expansion is linear in its coefficients."""
    xs, ys = np.array([x]), np.array([y])
    lhs = a * zernike_noll(5, xs, ys) + b * zernike_noll(9, xs, ys)
    basis = zernike_basis([5, 9], xs, ys)
    rhs = basis @ np.array([a, b])
    assert lhs == pytest.approx(rhs, abs=1e-12)


def test_gradient_basis_shapes_and_agreement() -> None:
    x = np.linspace(-0.5, 0.5, 7)
    y = np.linspace(0.2, -0.2, 7)
    gx, gy = zernike_gradient_basis([2, 4, 7], x, y)
    assert gx.shape == (7, 3)
    assert gy.shape == (7, 3)
    for k, j in enumerate([2, 4, 7]):
        ex, ey = zernike_gradient_noll(j, x, y)
        assert gx[:, k] == pytest.approx(ex)
        assert gy[:, k] == pytest.approx(ey)


@pytest.mark.parametrize("bad", [0, -1, 1.5, True, "4"])
def test_noll_index_validation(bad: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        noll_to_nm(bad)  # type: ignore[arg-type]


def test_nm_validation() -> None:
    with pytest.raises(ValueError, match=r"\|m\| <= n"):
        nm_to_noll(2, 3)
    with pytest.raises(ValueError, match="must be even"):
        nm_to_noll(2, 1)
    with pytest.raises(ValueError, match="n must be >= 0"):
        nm_to_noll(-1, 0)


def test_shape_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="same shape"):
        zernike_noll(4, np.zeros(3), np.zeros(4))
    with pytest.raises(ValueError, match="same shape"):
        zernike_gradient_noll(4, np.zeros(3), np.zeros(4))
    with pytest.raises(ValueError, match="same number of elements"):
        zernike_basis([2], np.zeros(3), np.zeros(4))


def test_empty_mode_list_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        zernike_basis([], np.zeros(3), np.zeros(3))
    with pytest.raises(ValueError, match="at least one"):
        zernike_gradient_basis([], np.zeros(3), np.zeros(3))


def test_osa_index_validation() -> None:
    with pytest.raises(ValueError, match="0-based"):
        osa_to_nm(-1)
    with pytest.raises(TypeError):
        osa_to_nm(1.0)  # type: ignore[arg-type]
