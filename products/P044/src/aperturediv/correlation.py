"""Spatial correlation between apertures, as a first-class input.

The textbook diversity result — ``L`` apertures buy diversity order ``L`` —
holds for *independent* branches. Apertures on one optical bench are not
independent, and the correlation between them is what destroys the quoted
gain. This module therefore makes the inter-aperture correlation an explicit
input everywhere, and provides the exact lognormal relation between the
correlation of the log-irradiance field (which is what a turbulence model
gives) and the correlation of the irradiance itself (which is what a
combiner sees).

Convention
----------
Correlation is specified on the **log-irradiance** field. For a unit-mean
lognormal irradiance with ``ln I = -s^2/2 + s Z``, ``Z`` jointly Gaussian
with correlation matrix ``R``, the irradiance correlation is exactly

``corr(I_j, I_k) = (exp(R_jk s^2) - 1) / (exp(s^2) - 1)``

derived from ``E[I_j I_k] = exp(R_jk s^2)`` for unit-mean factors. That
relation is implemented in :func:`log_to_irradiance_correlation` and
checked against Monte Carlo in
``validation/validate_correlation.py``. It is always weaker than the
log-domain correlation it comes from, which is the first reason quoted
diversity gains are optimistic.

Correlation models are the same two shapes as
:mod:`aperturediv.aperture`: Gaussian ``exp(-(d/rho_c)^2)`` and exponential
``exp(-d/rho_c)``. ``rho_c`` is the correlation scale in metres.

Nothing here is flight-qualified or certified.
"""

from __future__ import annotations

import numpy as np
from scipy import special, stats

from .aperture import CorrelationModel, normalised_covariance
from .channel import lognormal_sigma_log

__all__ = [
    "correlation_matrix",
    "sample_correlated_gamma_gamma",
    "equispaced_positions",
    "irradiance_correlation_matrix",
    "log_to_irradiance_correlation",
    "nearest_psd",
    "sample_correlated_lognormal",
]


def equispaced_positions(n_apertures: int, spacing_m: float) -> np.ndarray:
    """Positions in metres of ``n`` apertures on a line at fixed pitch.

    Returns an array of shape ``(n,)`` starting at 0. A line array is the
    geometry that makes the separation-dependence legible; arbitrary
    positions may be passed directly to :func:`correlation_matrix`.
    """
    n = int(n_apertures)
    if n < 1:
        raise ValueError(f"n_apertures must be >= 1, got {n!r}")
    pitch = float(spacing_m)
    if not np.isfinite(pitch) or pitch < 0.0:
        raise ValueError(f"spacing_m must be finite and >= 0, got {pitch!r}")
    return np.arange(n, dtype=float) * pitch


def correlation_matrix(
    positions_m: np.ndarray,
    correlation_scale_m: float,
    model: CorrelationModel = "gaussian",
) -> np.ndarray:
    """Log-irradiance correlation matrix for apertures at given positions.

    Parameters
    ----------
    positions_m
        Shape ``(n,)`` for a line array or ``(n, k)`` for ``k``-dimensional
        positions, in metres.
    correlation_scale_m
        Correlation scale ``rho_c`` in metres.
    model
        ``"gaussian"`` or ``"exponential"``.

    Returns
    -------
    numpy.ndarray
        Shape ``(n, n)``, unit diagonal, symmetric, dimensionless.
    """
    pos = np.asarray(positions_m, dtype=float)
    if pos.ndim == 1:
        pos = pos[:, None]
    if pos.ndim != 2:
        raise ValueError("positions_m must have shape (n,) or (n, k)")
    if not np.all(np.isfinite(pos)):
        raise ValueError("positions_m must be finite")
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    r = normalised_covariance(d, correlation_scale_m, model)
    np.fill_diagonal(r, 1.0)
    return np.asarray(r, dtype=float)


def log_to_irradiance_correlation(
    log_correlation: np.ndarray | float, si: float
) -> np.ndarray:
    """Irradiance correlation implied by a log-irradiance correlation.

    ``corr(I_j, I_k) = (exp(R_jk s^2) - 1) / (exp(s^2) - 1)``
    with ``s^2 = ln(1 + si)``. Exact for the lognormal model; the result is
    always ``<= R_jk`` for ``R_jk`` in ``[0, 1]``.
    """
    r = np.asarray(log_correlation, dtype=float)
    if np.any(r < -1.0) or np.any(r > 1.0):
        raise ValueError("log_correlation entries must lie in [-1, 1]")
    s2 = lognormal_sigma_log(si) ** 2
    return np.expm1(r * s2) / np.expm1(s2)


def irradiance_correlation_matrix(
    positions_m: np.ndarray,
    correlation_scale_m: float,
    si: float,
    model: CorrelationModel = "gaussian",
) -> np.ndarray:
    """Convenience: geometry and ``si`` straight to irradiance correlation."""
    r = correlation_matrix(positions_m, correlation_scale_m, model)
    out = log_to_irradiance_correlation(r, si)
    np.fill_diagonal(out, 1.0)
    return out


def nearest_psd(matrix: np.ndarray, *, floor: float = 0.0) -> np.ndarray:
    """Clip negative eigenvalues and renormalise to a unit diagonal.

    An exponential correlation model on an arbitrary point set can return a
    matrix that is positive semi-definite only to within rounding; this
    makes the Cholesky factorisation in
    :func:`sample_correlated_lognormal` well defined without silently
    changing a well-conditioned input (the operation is the identity, to
    rounding, when the input is already PSD with a unit diagonal).
    """
    m = np.asarray(matrix, dtype=float)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError("matrix must be square")
    m = 0.5 * (m + m.T)
    w, v = np.linalg.eigh(m)
    w = np.clip(w, floor, None)
    out = v @ np.diag(w) @ v.T
    d = np.sqrt(np.clip(np.diag(out), 1e-300, None))
    out = out / np.outer(d, d)
    out = 0.5 * (out + out.T)
    np.fill_diagonal(out, 1.0)
    return out


def sample_correlated_lognormal(
    n_samples: int,
    si: float,
    log_correlation: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw correlated unit-mean lognormal irradiance across apertures.

    A Gaussian copula in the log domain: ``Z ~ N(0, R)`` by Cholesky, then
    ``I = exp(s Z - s^2/2)``. Every marginal is exactly the unit-mean
    lognormal of :mod:`aperturediv.channel`, and the log-domain correlation
    is exactly ``R``.

    Parameters
    ----------
    n_samples
        Number of independent realisations.
    si
        Marginal scintillation index, identical across apertures.
    log_correlation
        Shape ``(L, L)`` log-irradiance correlation matrix.
    rng
        ``numpy.random.Generator``.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_samples, L)``, unit mean per column in expectation.
    """
    n = int(n_samples)
    if n < 1:
        raise ValueError(f"n_samples must be >= 1, got {n!r}")
    r = np.asarray(log_correlation, dtype=float)
    if r.ndim != 2 or r.shape[0] != r.shape[1]:
        raise ValueError("log_correlation must be a square matrix")
    if not np.allclose(np.diag(r), 1.0, atol=1e-10):
        raise ValueError("log_correlation must have a unit diagonal")
    s = lognormal_sigma_log(si)
    try:
        chol = np.linalg.cholesky(r)
    except np.linalg.LinAlgError:
        chol = np.linalg.cholesky(nearest_psd(r, floor=1e-12))
    z = rng.standard_normal((n, r.shape[0])) @ chol.T
    return np.exp(s * z - 0.5 * s * s)


def sample_correlated_gamma_gamma(
    n_samples: int,
    alpha: float,
    beta: float,
    large_scale_correlation: np.ndarray,
    rng: np.random.Generator,
    small_scale_correlation: np.ndarray | None = None,
) -> np.ndarray:
    """Draw correlated unit-mean gamma-gamma irradiance across apertures.

    A Gaussian copula is applied separately to each of the two unit-mean
    gamma factors of the gamma-gamma model, so every marginal is exactly the
    gamma-gamma of :mod:`aperturediv.channel` and its scintillation index is
    exactly ``1/alpha + 1/beta + 1/(alpha beta)``.

    The two factors carry **separate** correlation matrices because they
    describe different eddy sizes: the large-scale factor ``alpha`` stays
    correlated over separations comparable with the beam, the small-scale
    factor ``beta`` decorrelates over a much shorter distance. The default
    ``small_scale_correlation=None`` means the identity — independent
    small-scale fading across apertures — which is the usual modelling
    choice for apertures separated by more than a few centimetres. Passing
    a matrix overrides it.

    Parameters
    ----------
    n_samples
        Number of independent realisations.
    alpha, beta
        Gamma-gamma shape parameters, > 0.
    large_scale_correlation
        Shape ``(L, L)`` Gaussian-copula correlation for the ``alpha``
        factor, unit diagonal.
    rng
        ``numpy.random.Generator``.
    small_scale_correlation
        Optional ``(L, L)`` correlation for the ``beta`` factor; identity if
        omitted.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_samples, L)``.
    """
    n = int(n_samples)
    if n < 1:
        raise ValueError(f"n_samples must be >= 1, got {n!r}")
    a, b = float(alpha), float(beta)
    for name, v in (("alpha", a), ("beta", b)):
        if not np.isfinite(v) or v <= 0.0:
            raise ValueError(f"{name} must be finite and > 0, got {v!r}")
    r_big = np.asarray(large_scale_correlation, dtype=float)
    if r_big.ndim != 2 or r_big.shape[0] != r_big.shape[1]:
        raise ValueError("large_scale_correlation must be a square matrix")
    if not np.allclose(np.diag(r_big), 1.0, atol=1e-10):
        raise ValueError("large_scale_correlation must have a unit diagonal")
    n_ap = r_big.shape[0]
    r_small = np.eye(n_ap) if small_scale_correlation is None else np.asarray(
        small_scale_correlation, dtype=float
    )
    if r_small.shape != r_big.shape:
        raise ValueError("small_scale_correlation must match large_scale_correlation in shape")

    def _gamma_copula(r: np.ndarray, shape: float) -> np.ndarray:
        try:
            chol = np.linalg.cholesky(r)
        except np.linalg.LinAlgError:
            chol = np.linalg.cholesky(nearest_psd(r, floor=1e-12))
        z = rng.standard_normal((n, n_ap)) @ chol.T
        u = np.clip(special.ndtr(z), 1e-15, 1.0 - 1e-15)
        return stats.gamma.ppf(u, a=shape, scale=1.0 / shape)

    return _gamma_copula(r_big, a) * _gamma_copula(r_small, b)
