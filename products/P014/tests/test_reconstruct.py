"""Regularised least-squares reconstruction: known answers, identities, noise propagation."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from wavelab.geometry import SubapertureGeometry, build_geometry_matrices
from wavelab.reconstruct import (
    ModalReconstructor,
    ZonalReconstructor,
    noise_propagation_coefficient,
    piston_remove,
    regularised_pinv,
)
from wavelab.zernike import zernike_gradient_noll, zernike_noll

MODES = tuple(range(2, 22))

# Quadratic phase: Noll j = 2, 3 (tilt), 4 (defocus), 5, 6 (astigmatism) are the
# complete set of modes that are polynomials of degree <= 2 in (x, y). Both the
# Southwell trapezoidal relation and the Fried edge-average relation are exact
# for such a phase (see wavelab.geometry module docstring), so noise-free
# reconstruction must be exact to floating-point round-off.
QUADRATIC = {2: 0.7, 3: -0.4, 4: 1.3, 5: 0.6, 6: -0.9}


@pytest.fixture(name="geom")
def _geom() -> SubapertureGeometry:
    return SubapertureGeometry(n_sub=8, diameter=1.0)


def _quadratic_slopes(geom: SubapertureGeometry) -> np.ndarray:
    cx, cy = geom.subaperture_centres()
    ux = np.zeros_like(cx)
    uy = np.zeros_like(cy)
    for j, c in QUADRATIC.items():
        gx, gy = zernike_gradient_noll(j, cx, cy)
        ux += c * gx
        uy += c * gy
    return np.concatenate([ux, uy]) * geom.scaled_slope_factor


def _quadratic_phase(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.zeros_like(x)
    for j, c in QUADRATIC.items():
        p += c * zernike_noll(j, x, y)
    return p - p.mean()


def test_southwell_is_exact_for_quadratic_phase(geom: SubapertureGeometry) -> None:
    """Known answer: Southwell's trapezoidal relation is exact for quadratic phase."""
    rec = ZonalReconstructor(geom, "southwell", "tsvd", 1e-6)
    px, py = rec.phase_points()
    est = rec.reconstruct(_quadratic_slopes(geom))
    err = np.max(np.abs(est - _quadratic_phase(px, py)))
    assert err < 1e-12, f"Southwell noise-free error {err:.3e} rad exceeds 1e-12 rad"


def test_fried_is_exact_up_to_its_null_space(geom: SubapertureGeometry) -> None:
    """Fried is exact only modulo piston + waffle, which it cannot observe.

    The residual is projected onto the complement of the two right singular
    vectors with numerically zero singular values.
    """
    rec = ZonalReconstructor(geom, "fried", "tsvd", 1e-6)
    px, py = rec.phase_points()
    a = build_geometry_matrices(geom, "fried").a
    _, s, vt = np.linalg.svd(a)
    null = vt[np.count_nonzero(s > 1e-9 * s[0]) :]
    assert null.shape[0] == 2
    resid = rec.reconstruct(_quadratic_slopes(geom)) - _quadratic_phase(px, py)
    observable = resid - null.T @ (null @ resid)
    assert np.max(np.abs(observable)) < 1e-12
    # And the unobservable part is genuinely non-zero -- this is a real blind spot.
    assert np.max(np.abs(null @ resid)) > 1e-3


def test_modal_reconstruction_is_exact_noise_free(geom: SubapertureGeometry) -> None:
    """Known answer: a pure Zernike input is recovered to numerical tolerance."""
    rec = ModalReconstructor(geom, MODES, "tsvd", 1e-8)
    rng = np.random.default_rng(0)
    a = rng.normal(size=len(MODES))
    err = np.max(np.abs(rec.reconstruct(rec.forward(a)) - a))
    assert err < 1e-12, f"modal noise-free error {err:.3e} rad exceeds 1e-12 rad"


def test_modal_matches_zonal_slopes(geom: SubapertureGeometry) -> None:
    """The modal forward model reproduces the analytic Zernike slopes."""
    rec = ModalReconstructor(geom, MODES)
    a = np.zeros(len(MODES))
    for j, c in QUADRATIC.items():
        a[MODES.index(j)] = c
    assert rec.forward(a) == pytest.approx(_quadratic_slopes(geom), abs=1e-14)


@settings(max_examples=25, deadline=None)
@given(
    st.floats(min_value=-3.0, max_value=3.0, allow_nan=False),
    st.floats(min_value=-3.0, max_value=3.0, allow_nan=False),
)
def test_pure_tilt_is_recovered(ax: float, ay: float) -> None:
    """Property: a pure tip/tilt in, the same tip/tilt out, for both geometries."""
    geom = SubapertureGeometry(6, 1.0)
    modes = tuple(range(2, 12))
    rec = ModalReconstructor(geom, modes, "tsvd", 1e-8)
    a = np.zeros(len(modes))
    a[0], a[1] = ax, ay
    got = rec.reconstruct(rec.forward(a))
    assert got == pytest.approx(a, abs=1e-10)


@settings(max_examples=25, deadline=None)
@given(
    st.floats(min_value=-5.0, max_value=5.0, allow_nan=False),
    st.floats(min_value=-5.0, max_value=5.0, allow_nan=False),
)
def test_reconstruction_is_linear(alpha: float, beta: float) -> None:
    """Property: reconstruction is a linear operator on the slope vector."""
    geom = SubapertureGeometry(6, 1.0)
    rng = np.random.default_rng(11)
    u1 = rng.normal(size=geom.n_slopes)
    u2 = rng.normal(size=geom.n_slopes)
    for name in ("southwell", "fried"):
        rec = ZonalReconstructor(geom, name, "tikhonov", 1e-3)
        lhs = rec.reconstruct(alpha * u1 + beta * u2)
        rhs = alpha * rec.reconstruct(u1) + beta * rec.reconstruct(u2)
        assert lhs == pytest.approx(rhs, abs=1e-9)


def test_reconstruction_output_is_piston_free(geom: SubapertureGeometry) -> None:
    rng = np.random.default_rng(5)
    for name in ("southwell", "fried"):
        rec = ZonalReconstructor(geom, name)
        est = rec.reconstruct(rng.normal(size=(4, geom.n_slopes)))
        assert np.max(np.abs(est.mean(axis=1))) < 1e-12


def test_noise_propagation_matches_monte_carlo(geom: SubapertureGeometry) -> None:
    """The analytic a_NP predicts the Monte-Carlo error variance.

    a_NP = ||P R||_F^2 / M, so <|p_err|^2> = sigma_u^2 a_NP exactly for white
    noise. This is an algebraic identity, so agreement is limited only by
    Monte-Carlo sampling.
    """
    rng = np.random.default_rng(17)
    sigma = 0.05
    for name in ("southwell", "fried"):
        rec = ZonalReconstructor(geom, name, "tikhonov", 1e-2)
        a_np = rec.noise_propagation()
        noise = sigma * rng.standard_normal((4000, geom.n_slopes))
        err = piston_remove(rec.reconstruct(noise))
        measured = np.mean(err**2)
        assert measured == pytest.approx(sigma**2 * a_np, rel=0.06)


def test_noise_propagation_grows_with_array_size() -> None:
    """Fried (1977) / Hudgin (1977): least-squares noise propagation grows with N."""
    vals = [
        ZonalReconstructor(SubapertureGeometry(n, 1.0), "southwell", "tsvd", 1e-6).noise_propagation()
        for n in (4, 8, 16)
    ]
    assert vals[0] < vals[1] < vals[2]


def test_dropout_increases_noise_propagation(geom: SubapertureGeometry) -> None:
    rec = ZonalReconstructor(geom, "southwell", "tikhonov", 1e-2)
    full = rec.noise_propagation()
    avail = np.ones(geom.n_valid_sub, dtype=bool)
    avail[::5] = False
    assert rec.noise_propagation(avail) > full


def test_dropped_columns_are_exactly_zero(geom: SubapertureGeometry) -> None:
    rec = ZonalReconstructor(geom, "fried", "tikhonov", 1e-3)
    avail = np.ones(geom.n_valid_sub, dtype=bool)
    avail[3] = False
    r = rec.matrix(avail)
    assert np.all(r[:, 3] == 0.0)
    assert np.all(r[:, geom.n_valid_sub + 3] == 0.0)


def test_tsvd_and_tikhonov_agree_in_the_well_posed_limit(geom: SubapertureGeometry) -> None:
    rng = np.random.default_rng(2)
    u = rng.normal(size=geom.n_slopes) * 0.01
    a = ZonalReconstructor(geom, "southwell", "tsvd", 1e-8).reconstruct(u)
    b = ZonalReconstructor(geom, "southwell", "tikhonov", 1e-8).reconstruct(u)
    assert a == pytest.approx(b, abs=1e-6 * np.max(np.abs(a)))


def test_prior_weighted_tikhonov_shrinks_high_orders_more(geom: SubapertureGeometry) -> None:
    """A coloured prior must shrink the low-variance modes harder than a white one."""
    prior = np.linspace(4.0, 0.2, len(MODES))
    rng = np.random.default_rng(4)
    u = rng.normal(size=(64, geom.n_slopes))
    white = ModalReconstructor(geom, MODES, "tikhonov", 0.2).reconstruct(u)
    coloured = ModalReconstructor(geom, MODES, "tikhonov", 0.2, prior_std=prior).reconstruct(u)
    # Ratio of output RMS, low orders versus high orders.
    r_white = white[:, -5:].std() / white[:, :5].std()
    r_col = coloured[:, -5:].std() / coloured[:, :5].std()
    assert r_col < r_white


def test_regularised_pinv_known_answer() -> None:
    """Hand calculation: for a = diag(2, 1e-9), TSVD at rcond 1e-3 zeroes the tiny mode."""
    a = np.diag([2.0, 1e-9])
    inv = regularised_pinv(a, "tsvd", 1e-3)
    assert inv[0, 0] == pytest.approx(0.5)
    assert inv[1, 1] == 0.0
    # Tikhonov with lambda = 1e-3 * 2 = 2e-3 gives s/(s^2+lam^2) = 1e-9/4e-6.
    tik = regularised_pinv(a, "tikhonov", 1e-3)
    assert tik[1, 1] == pytest.approx(1e-9 / (1e-18 + 4e-6), rel=1e-9)


def test_noise_propagation_coefficient_known_answer() -> None:
    """Hand calculation: for R = I_3 with piston removed, a_NP = (1 - 1/3) = 2/3."""
    assert noise_propagation_coefficient(np.eye(3)) == pytest.approx(2.0 / 3.0)


def test_input_validation(geom: SubapertureGeometry) -> None:
    rec = ZonalReconstructor(geom, "southwell")
    with pytest.raises(ValueError, match="slope entries"):
        rec.reconstruct(np.zeros(7))
    with pytest.raises(ValueError, match="non-finite"):
        rec.reconstruct(np.full(geom.n_slopes, np.nan))
    with pytest.raises(ValueError, match="1-D or 2-D"):
        rec.reconstruct(np.zeros((2, 2, geom.n_slopes)))
    with pytest.raises(ValueError, match="sub_available"):
        rec.reconstruct(np.zeros(geom.n_slopes), np.ones(3, dtype=bool))
    with pytest.raises(ValueError, match="reg must be finite"):
        ZonalReconstructor(geom, "southwell", "tsvd", 0.0)
    with pytest.raises(ValueError, match="tsvd"):
        ZonalReconstructor(geom, "southwell", "ridge")
    with pytest.raises(ValueError, match="piston"):
        ModalReconstructor(geom, (1, 2, 3))
    with pytest.raises(ValueError, match="duplicates"):
        ModalReconstructor(geom, (2, 2))
    with pytest.raises(ValueError, match="at least one mode"):
        ModalReconstructor(geom, ())
    with pytest.raises(ValueError, match="prior_std"):
        ModalReconstructor(geom, MODES, prior_std=np.ones(3))
    with pytest.raises(ValueError, match="prior_std"):
        ModalReconstructor(geom, MODES, prior_std=np.zeros(len(MODES)))
    with pytest.raises(ValueError, match="2-D"):
        regularised_pinv(np.zeros(4))
    with pytest.raises(ValueError, match="non-finite"):
        regularised_pinv(np.array([[np.inf, 0.0], [0.0, 1.0]]))
    with pytest.raises(ValueError, match="2-D"):
        noise_propagation_coefficient(np.zeros(4))
