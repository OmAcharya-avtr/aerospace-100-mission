"""Tests for least-squares Zernike wavefront fitting."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from zernkit import (
    fit_wavefront,
    mode_list,
    noll_to_nm,
    unit_disc_grid,
    zernike_design_matrix,
)


def _disc_samples(n_pix: int = 64) -> tuple[np.ndarray, np.ndarray]:
    x, y, mask = unit_disc_grid(n_pix)
    return x[mask], y[mask]


def _synthesize(x: np.ndarray, y: np.ndarray, coeffs: np.ndarray, n_modes: int) -> np.ndarray:
    indices = mode_list(n_modes, indexing="noll")
    return zernike_design_matrix(indices, x, y) @ coeffs


# --- recovery -------------------------------------------------------------


def test_recovers_injected_coefficients_noise_free() -> None:
    x, y = _disc_samples(80)
    rng = np.random.default_rng(11)
    n_modes = 15
    truth = rng.normal(size=n_modes)
    w = _synthesize(x, y, truth, n_modes)
    fit = fit_wavefront(x, y, w, n_modes)
    assert np.allclose(fit.coefficients, truth, atol=1e-10)
    assert fit.residual_rms < 1e-12
    assert fit.variance_explained == pytest.approx(1.0)
    assert fit.n_dropped == 0
    assert fit.n_used == x.size


def test_fitted_indices_are_reported_in_both_conventions() -> None:
    x, y = _disc_samples(40)
    w = np.zeros_like(x)
    fit = fit_wavefront(x, y, w, 6)
    assert fit.noll_indices == [1, 2, 3, 4, 5, 6]
    assert fit.osa_indices == [0, 2, 1, 4, 3, 5]
    assert fit.indices == [noll_to_nm(j) for j in range(1, 7)]


def test_osa_ordering_produces_a_permuted_coefficient_vector() -> None:
    """Same wavefront, two conventions: the coefficients are permuted, not different."""
    x, y = _disc_samples(60)
    rng = np.random.default_rng(3)
    truth = rng.normal(size=15)
    w = _synthesize(x, y, truth, 15)
    fit_noll = fit_wavefront(x, y, w, 15, indexing="noll")
    fit_osa = fit_wavefront(x, y, w, 15, indexing="osa")
    lookup = dict(zip(fit_osa.indices, fit_osa.coefficients, strict=True))
    for nm, c in zip(fit_noll.indices, fit_noll.coefficients, strict=True):
        assert lookup[nm] == pytest.approx(c, abs=1e-9)


def test_coefficient_accessor() -> None:
    x, y = _disc_samples(60)
    truth = np.zeros(10)
    truth[3] = 0.4  # Noll j=4 -> defocus (2, 0)
    w = _synthesize(x, y, truth, 10)
    fit = fit_wavefront(x, y, w, 10)
    assert fit.coefficient(2, 0) == pytest.approx(0.4, abs=1e-10)
    with pytest.raises(KeyError):
        fit.coefficient(6, 0)


def test_residual_rms_and_variance_explained_on_truncated_fit() -> None:
    x, y = _disc_samples(80)
    truth = np.zeros(15)
    truth[3] = 1.0  # defocus
    truth[10] = 0.5  # Noll j=11, primary spherical
    w = _synthesize(x, y, truth, 15)
    fit = fit_wavefront(x, y, w, 4)  # spherical not in the basis -> leftover
    assert fit.residual_rms > 0.4
    assert 0.0 < fit.variance_explained < 1.0
    full = fit_wavefront(x, y, w, 15)
    assert full.residual_rms < 1e-12


def test_condition_number_is_modest_on_a_dense_disc() -> None:
    x, y = _disc_samples(80)
    fit = fit_wavefront(x, y, np.zeros_like(x), 21)
    assert 1.0 <= fit.condition_number < 5.0


def test_unnormalized_fit_gives_scaled_coefficients() -> None:
    """Same wavefront, different normalisation => coefficients differ by N_n^m."""
    from zernkit import normalization

    x, y = _disc_samples(60)
    rng = np.random.default_rng(5)
    truth = rng.normal(size=10)
    w = _synthesize(x, y, truth, 10)
    fit = fit_wavefront(x, y, w, 10, normalized=False)
    for (n, m), c_un, c_norm in zip(fit.indices, fit.coefficients, truth, strict=True):
        assert c_un == pytest.approx(c_norm * normalization(n, m), abs=1e-9)


# --- outside-disc policy --------------------------------------------------


def test_outside_raise_is_the_default() -> None:
    x, y, _ = unit_disc_grid(21)
    w = np.zeros_like(x)
    with pytest.raises(ValueError, match="outside the unit disc"):
        fit_wavefront(x, y, w, 6)


def test_outside_drop_excludes_corners() -> None:
    x, y, mask = unit_disc_grid(41)
    rng = np.random.default_rng(2)
    truth = rng.normal(size=10)
    full = np.zeros_like(x)
    full[mask] = _synthesize(x[mask], y[mask], truth, 10)
    fit = fit_wavefront(x, y, full, 10, outside="drop")
    assert fit.n_dropped == int((~mask).sum())
    assert fit.n_used == int(mask.sum())
    assert np.allclose(fit.coefficients, truth, atol=1e-9)


def test_outside_extrapolate_keeps_everything() -> None:
    x, y, _ = unit_disc_grid(21)
    truth = np.zeros(6)
    truth[1] = 1.0
    w = _synthesize(x, y, truth, 6)
    fit = fit_wavefront(x, y, w, 6, outside="extrapolate")
    assert fit.n_used == x.size
    assert fit.n_dropped == 0
    assert np.allclose(fit.coefficients, truth, atol=1e-10)


def test_outside_tolerance_admits_exact_rim_points() -> None:
    theta = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    x, y = np.cos(theta), np.sin(theta)
    rho = np.hypot(x, y)
    assert rho.max() >= 1.0  # round-off can push it a hair over
    fit_wavefront(x, y, np.zeros_like(x), 3)  # must not raise


def test_all_samples_outside_raises() -> None:
    x = np.array([2.0, 3.0, 4.0, 5.0])
    y = np.zeros(4)
    with pytest.raises(ValueError, match="all samples lie outside"):
        fit_wavefront(x, y, np.zeros(4), 3, outside="drop")


# --- input validation -----------------------------------------------------


def test_size_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same number of samples"):
        fit_wavefront([0.0, 0.1], [0.0], [0.0, 0.1], 3)


def test_non_finite_wavefront_raises() -> None:
    x, y = _disc_samples(20)
    w = np.zeros_like(x)
    w[0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        fit_wavefront(x, y, w, 3)


def test_underdetermined_raises() -> None:
    x = np.array([0.0, 0.1, 0.2])
    y = np.array([0.0, 0.0, 0.1])
    with pytest.raises(ValueError, match="under-determined"):
        fit_wavefront(x, y, np.zeros(3), 10)


def test_unknown_policy_raises() -> None:
    x, y = _disc_samples(20)
    with pytest.raises(ValueError, match="outside must be one of"):
        fit_wavefront(x, y, np.zeros_like(x), 3, outside="clip")


def test_missing_mode_specification_raises() -> None:
    x, y = _disc_samples(20)
    with pytest.raises(ValueError, match="either n_modes or"):
        fit_wavefront(x, y, np.zeros_like(x))


def test_explicit_indices_are_honoured() -> None:
    x, y = _disc_samples(50)
    indices = [(2, 0), (4, 0)]
    w = 0.3 * np.sqrt(3) * (2 * (x**2 + y**2) - 1)
    fit = fit_wavefront(x, y, w, indices=indices)
    assert fit.indices == indices
    assert fit.coefficient(2, 0) == pytest.approx(0.3, abs=1e-9)
    assert fit.coefficient(4, 0) == pytest.approx(0.0, abs=1e-9)


def test_mode_list_errors() -> None:
    with pytest.raises(ValueError, match="n_modes must be >= 1"):
        mode_list(0)
    with pytest.raises(ValueError, match="indexing must be"):
        mode_list(3, indexing="fringe")
    with pytest.raises(TypeError):
        mode_list(3.0)  # type: ignore[arg-type]
    assert mode_list(3, indexing="ansi") == mode_list(3, indexing="osa")


def test_design_matrix_validation() -> None:
    with pytest.raises(ValueError, match="at least one"):
        zernike_design_matrix([], [0.0], [0.0])
    with pytest.raises(ValueError, match="same size"):
        zernike_design_matrix([(0, 0)], [0.0, 0.1], [0.0])


def test_empty_input_raises() -> None:
    with pytest.raises(ValueError, match="no samples"):
        fit_wavefront([], [], [], 3)


# --- property-based -------------------------------------------------------


@given(
    st.lists(
        st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=15,
    )
)
@settings(max_examples=40, deadline=None)
def test_property_fit_recovers_injected_coefficients(coeffs: list[float]) -> None:
    """Noise-free synthesis then fit must return the injected vector."""
    x, y = _disc_samples(56)
    truth = np.asarray(coeffs, dtype=float)
    w = _synthesize(x, y, truth, truth.size)
    fit = fit_wavefront(x, y, w, truth.size)
    assert np.allclose(fit.coefficients, truth, atol=1e-9, rtol=1e-9)


@given(
    st.floats(min_value=-3.0, max_value=3.0),
    st.integers(min_value=1, max_value=21),
)
@settings(max_examples=60, deadline=None)
def test_property_single_mode_isolation(amplitude: float, j: int) -> None:
    """Injecting one mode must leave every other fitted coefficient at zero."""
    x, y = _disc_samples(56)
    truth = np.zeros(21)
    truth[j - 1] = amplitude
    w = _synthesize(x, y, truth, 21)
    fit = fit_wavefront(x, y, w, 21)
    assert np.allclose(fit.coefficients, truth, atol=1e-9)
