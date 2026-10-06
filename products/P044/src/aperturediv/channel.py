"""Irradiance statistics for an atmospheric optical channel.

Two fading models are provided, both normalised to unit mean irradiance
(``E[I] = 1``) so that the mean received power is carried by the link budget
and the model carries only the fluctuation.

Units
-----
Irradiance ``I`` is dimensionless throughout: it is the instantaneous
irradiance divided by its own ensemble mean. The scintillation index
``si`` is also dimensionless.

Models and sources
------------------
Lognormal
    ``ln I ~ N(-s^2/2, s^2)``. The ``-s^2/2`` offset is what makes
    ``E[I] = 1``. The scintillation index of a lognormal irradiance is
    ``si = exp(s^2) - 1`` exactly, hence ``s^2 = ln(1 + si)``.
    Validity: weak fluctuation, Rytov variance below roughly 1. The model
    is the standard weak-turbulence description in Andrews & Phillips,
    *Laser Beam Propagation through Random Media* (SPIE Press, 2005).

Gamma-gamma
    ``I = I_x * I_y`` with both factors unit-mean gamma, shapes ``alpha``
    (large-scale) and ``beta`` (small-scale). Introduced for optical
    scintillation by Al-Habash, Andrews & Phillips, *Optical Engineering*
    40(8), 2001, and used throughout Andrews & Phillips (2005).
    Validity: weak through strong fluctuation.

The plane-wave mapping from Rytov variance to ``(alpha, beta)`` used in
:func:`gamma_gamma_params_from_rytov` is the widely reproduced form of that
literature. This module does not quote a page number for it; instead
``validation/validate_channel_stats.py`` checks the mapping for internal
consistency against the model's own closed-form scintillation index and
against Monte Carlo moments, which is a check rather than an assertion.

Nothing here is flight-qualified or certified.
"""

from __future__ import annotations

import numpy as np
from scipy import integrate, special

__all__ = [
    "gamma_gamma_cdf",
    "gamma_gamma_moment",
    "gamma_gamma_params_from_rytov",
    "gamma_gamma_pdf",
    "gamma_gamma_scintillation_index",
    "lognormal_cdf",
    "lognormal_moment",
    "lognormal_pdf",
    "lognormal_quantile",
    "lognormal_scintillation_index",
    "lognormal_sigma_log",
    "sample_gamma_gamma",
    "sample_lognormal",
]

_LOG_LOWER = -60.0  # lower limit in ln(I) for cdf quadrature


def _check_si(si: float) -> float:
    si = float(si)
    if not np.isfinite(si):
        raise ValueError(f"scintillation index must be finite, got {si!r}")
    if si <= 0.0:
        raise ValueError(f"scintillation index must be > 0, got {si!r}")
    return si


def _as_positive_array(x: np.ndarray | float, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if np.any(~np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    if np.any(arr < 0.0):
        raise ValueError(f"{name} must be >= 0")
    return arr


def lognormal_sigma_log(si: float) -> float:
    """Standard deviation of ``ln I`` for a unit-mean lognormal irradiance.

    Parameters
    ----------
    si
        Scintillation index (dimensionless, > 0).

    Returns
    -------
    float
        ``s = sqrt(ln(1 + si))`` (dimensionless, natural-log units).
    """
    return float(np.sqrt(np.log1p(_check_si(si))))


def lognormal_scintillation_index(sigma_log: float) -> float:
    """Scintillation index implied by a log-irradiance standard deviation.

    Inverse of :func:`lognormal_sigma_log`: ``si = exp(s^2) - 1``.
    """
    s = float(sigma_log)
    if s <= 0.0:
        raise ValueError(f"sigma_log must be > 0, got {s!r}")
    return float(np.expm1(s * s))


def lognormal_pdf(irradiance: np.ndarray | float, si: float) -> np.ndarray:
    """Probability density of a unit-mean lognormal irradiance.

    ``f(I) = 1/(I s sqrt(2 pi)) exp(-(ln I + s^2/2)^2 / (2 s^2))``.

    Parameters
    ----------
    irradiance
        Normalised irradiance ``I >= 0`` (dimensionless).
    si
        Scintillation index (dimensionless).

    Returns
    -------
    numpy.ndarray
        Density in units of inverse normalised irradiance.
    """
    i = _as_positive_array(irradiance, "irradiance")
    s = lognormal_sigma_log(si)
    out = np.zeros_like(i, dtype=float)
    pos = i > 0.0
    li = np.log(i[pos])
    out[pos] = np.exp(-((li + 0.5 * s * s) ** 2) / (2.0 * s * s)) / (
        i[pos] * s * np.sqrt(2.0 * np.pi)
    )
    return out


def lognormal_cdf(irradiance: np.ndarray | float, si: float) -> np.ndarray:
    """Cumulative distribution of a unit-mean lognormal irradiance.

    ``F(I) = Phi((ln I + s^2/2) / s)``.
    """
    i = _as_positive_array(irradiance, "irradiance")
    s = lognormal_sigma_log(si)
    out = np.zeros_like(i, dtype=float)
    pos = i > 0.0
    out[pos] = special.ndtr((np.log(i[pos]) + 0.5 * s * s) / s)
    return out


def lognormal_quantile(p: np.ndarray | float, si: float) -> np.ndarray:
    """Inverse CDF of a unit-mean lognormal irradiance.

    Parameters
    ----------
    p
        Probability in (0, 1).
    si
        Scintillation index.
    """
    pa = np.asarray(p, dtype=float)
    if np.any(pa <= 0.0) or np.any(pa >= 1.0):
        raise ValueError("p must lie strictly in (0, 1)")
    s = lognormal_sigma_log(si)
    return np.exp(s * special.ndtri(pa) - 0.5 * s * s)


def lognormal_moment(order: int, si: float) -> float:
    """``E[I^n]`` for a unit-mean lognormal irradiance.

    ``E[I^n] = exp(n(n-1) s^2 / 2)``.
    """
    n = int(order)
    s2 = np.log1p(_check_si(si))
    return float(np.exp(0.5 * n * (n - 1) * s2))


def sample_lognormal(
    size: int | tuple[int, ...], si: float, rng: np.random.Generator
) -> np.ndarray:
    """Draw unit-mean lognormal irradiance samples."""
    s = lognormal_sigma_log(si)
    z = rng.standard_normal(size)
    return np.exp(s * z - 0.5 * s * s)


def gamma_gamma_params_from_rytov(rytov_variance: float) -> tuple[float, float]:
    """Plane-wave gamma-gamma shape parameters from the Rytov variance.

    ``alpha = 1 / (exp(0.49 sr / (1 + 1.11 sr^(6/5))^(7/6)) - 1)``
    ``beta  = 1 / (exp(0.51 sr / (1 + 0.69 sr^(6/5))^(5/6)) - 1)``

    with ``sr`` the Rytov variance ``sigma_R^2`` (dimensionless). This is
    the plane-wave form of Al-Habash, Andrews & Phillips (2001) and
    Andrews & Phillips (2005), written with ``sr^(6/5) = (sigma_R^2)^(6/5)``
    so that the exponent matches the ``sigma_R^(12/5)`` of the papers.

    Validity: the mapping is a fit over weak-to-strong fluctuation for a
    plane wave and an unbounded (point) receiver; it carries no aperture
    averaging, which is applied separately by
    :func:`aperturediv.aperture.aperture_averaging_factor`.

    Returns
    -------
    tuple of float
        ``(alpha, beta)``, both > 0 and ``alpha >= beta``.
    """
    sr = float(rytov_variance)
    if not np.isfinite(sr) or sr <= 0.0:
        raise ValueError(f"rytov_variance must be finite and > 0, got {sr!r}")
    p = sr ** (6.0 / 5.0)
    alpha = 1.0 / np.expm1(0.49 * sr / (1.0 + 1.11 * p) ** (7.0 / 6.0))
    beta = 1.0 / np.expm1(0.51 * sr / (1.0 + 0.69 * p) ** (5.0 / 6.0))
    return float(alpha), float(beta)


def _check_shapes(alpha: float, beta: float) -> tuple[float, float]:
    a, b = float(alpha), float(beta)
    for name, v in (("alpha", a), ("beta", b)):
        if not np.isfinite(v) or v <= 0.0:
            raise ValueError(f"{name} must be finite and > 0, got {v!r}")
    return a, b


def gamma_gamma_scintillation_index(alpha: float, beta: float) -> float:
    """Closed-form scintillation index of the gamma-gamma model.

    ``si = 1/alpha + 1/beta + 1/(alpha beta)``, which follows from
    ``E[I^2] = (alpha+1)(beta+1)/(alpha beta)`` with ``E[I] = 1``.
    """
    a, b = _check_shapes(alpha, beta)
    return float(1.0 / a + 1.0 / b + 1.0 / (a * b))


def gamma_gamma_moment(order: int, alpha: float, beta: float) -> float:
    """``E[I^n] = Gamma(a+n)Gamma(b+n) / (Gamma(a)Gamma(b) (a b)^n)``."""
    a, b = _check_shapes(alpha, beta)
    n = int(order)
    log_m = (
        special.gammaln(a + n)
        + special.gammaln(b + n)
        - special.gammaln(a)
        - special.gammaln(b)
        - n * np.log(a * b)
    )
    return float(np.exp(log_m))


def _gg_log_pdf(log_i: np.ndarray, a: float, b: float) -> np.ndarray:
    """log f(I) evaluated at ``I = exp(log_i)``, stable for large ``I``."""
    ab = a * b
    nu = a - b
    z = 2.0 * np.sqrt(ab * np.exp(log_i))
    # kve(nu, z) = kv(nu, z) * exp(z), so log kv = log kve - z.
    log_kv = np.log(special.kve(nu, z)) - z
    return (
        np.log(2.0)
        + 0.5 * (a + b) * np.log(ab)
        - special.gammaln(a)
        - special.gammaln(b)
        + (0.5 * (a + b) - 1.0) * log_i
        + log_kv
    )


def gamma_gamma_pdf(irradiance: np.ndarray | float, alpha: float, beta: float) -> np.ndarray:
    """Probability density of the unit-mean gamma-gamma irradiance.

    ``f(I) = 2 (ab)^((a+b)/2) / (Gamma(a)Gamma(b)) I^((a+b)/2 - 1)
             K_(a-b)(2 sqrt(a b I))``

    Source: Al-Habash, Andrews & Phillips, *Optical Engineering* 40(8),
    2001; also Andrews & Phillips (2005). ``K`` is the modified Bessel
    function of the second kind. Evaluated in log space through
    ``scipy.special.kve`` so that it does not underflow at large ``I``.
    """
    a, b = _check_shapes(alpha, beta)
    i = _as_positive_array(irradiance, "irradiance")
    out = np.zeros_like(i, dtype=float)
    pos = i > 0.0
    if np.any(pos):
        out[pos] = np.exp(_gg_log_pdf(np.log(i[pos]), a, b))
    return out


def gamma_gamma_cdf(
    irradiance: np.ndarray | float, alpha: float, beta: float, *, rtol: float = 1e-10
) -> np.ndarray:
    """Cumulative distribution of the gamma-gamma irradiance by quadrature.

    The integral is taken in ``t = ln I`` so that the integrand
    ``f(e^t) e^t ~ e^(t min(a,b))`` vanishes at the lower limit and the
    ``I -> 0`` singularity of the density is removed. Accuracy is checked
    against Monte Carlo in ``validation/validate_channel_stats.py``.
    """
    a, b = _check_shapes(alpha, beta)
    i = _as_positive_array(irradiance, "irradiance")
    flat = np.atleast_1d(i).ravel()
    out = np.zeros_like(flat, dtype=float)

    def integrand(t: float) -> float:
        return float(np.exp(_gg_log_pdf(np.asarray(t), a, b) + t))

    for k, val in enumerate(flat):
        if val <= 0.0:
            continue
        upper = float(np.log(val))
        lower = min(_LOG_LOWER, upper - 1.0)
        val_int, _ = integrate.quad(integrand, lower, upper, epsabs=1e-13, epsrel=rtol, limit=400)
        out[k] = min(max(val_int, 0.0), 1.0)
    return out.reshape(np.shape(i)) if np.shape(i) else out[0]


def sample_gamma_gamma(
    size: int | tuple[int, ...], alpha: float, beta: float, rng: np.random.Generator
) -> np.ndarray:
    """Draw unit-mean gamma-gamma irradiance samples as a product of gammas.

    ``I = X Y`` with ``X ~ Gamma(alpha, 1/alpha)``, ``Y ~ Gamma(beta, 1/beta)``,
    both unit mean, so ``E[I] = 1``.
    """
    a, b = _check_shapes(alpha, beta)
    x = rng.gamma(shape=a, scale=1.0 / a, size=size)
    y = rng.gamma(shape=b, scale=1.0 / b, size=size)
    return x * y
