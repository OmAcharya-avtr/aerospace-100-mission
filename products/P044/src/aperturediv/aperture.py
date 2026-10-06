"""Aperture averaging: when one large aperture beats several small ones.

A receiver of finite diameter integrates the irradiance over its own
collecting area, so the fluctuation it sees is smaller than the fluctuation
at a point. The reduction is the *aperture averaging factor*

``A(D) = sigma_I^2(D) / sigma_I^2(0)``

the ratio of the scintillation index measured by an aperture of diameter
``D`` to the point scintillation index.

Definition used here
--------------------
``A`` is computed from its defining integral: the normalised irradiance
covariance ``b(rho)`` weighted by the autocorrelation of a uniformly
illuminated circular aperture,

``A(D) = (16 / (pi D^2)) * int_0^D rho * b(rho) * W(rho/D) drho``
``W(u) = arccos(u) - u sqrt(1 - u^2)``

The prefactor is fixed by the identity ``int_0^1 u W(u) du = pi/16``, so
``A = 1`` whenever ``b`` is constant over the aperture (a fully correlated
irradiance field). The weighting ``W`` is the standard circle-circle
overlap kernel. This form is the aperture-averaging integral of Andrews &
Phillips, *Laser Beam Propagation through Random Media* (SPIE Press, 2005),
Chapter 10 ("Aperture averaging"); the normalisation identity is re-derived
and checked numerically in ``validation/validate_aperture_averaging.py``
rather than taken on trust.

Covariance model
----------------
This module takes the normalised irradiance covariance as an explicit
input, not as a buried assumption. Two shapes are offered:

``"gaussian"``   ``b(rho) = exp(-(rho / rho_c)^2)``
``"exponential"`` ``b(rho) = exp(-rho / rho_c)``

``rho_c`` is the irradiance correlation scale in metres. For weak
plane-wave turbulence it is of the order of the Fresnel scale
``sqrt(lambda L)`` (:func:`fresnel_scale`); treating it as a free input is
deliberate, because the correlation scale is the quantity a link designer
actually measures or bounds, and the aperture-averaging result is far more
sensitive to it than to the shape of ``b``.

Two closed-form limits of the Gaussian case are derived in the
documentation of :func:`aperture_averaging_small_d` and
:func:`aperture_averaging_large_d` and are used as the validation targets.

Nothing here is flight-qualified or certified.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy import integrate

__all__ = [
    "CORRELATION_MODELS",
    "aperture_averaging_factor",
    "aperture_averaging_large_d",
    "aperture_averaging_small_d",
    "circular_aperture_weight",
    "effective_scintillation_index",
    "equal_area_diameter",
    "fresnel_scale",
    "normalised_covariance",
]

CORRELATION_MODELS: tuple[str, ...] = ("gaussian", "exponential")
CorrelationModel = Literal["gaussian", "exponential"]

_W_NORM = np.pi / 16.0  # int_0^1 u (arccos u - u sqrt(1-u^2)) du


def fresnel_scale(wavelength_m: float, path_length_m: float) -> float:
    """Fresnel scale ``sqrt(lambda L)`` in metres.

    Parameters
    ----------
    wavelength_m
        Optical wavelength in metres (e.g. 1.55e-6).
    path_length_m
        Propagation path length in metres.
    """
    lam = float(wavelength_m)
    length = float(path_length_m)
    if not np.isfinite(lam) or lam <= 0.0:
        raise ValueError(f"wavelength_m must be finite and > 0, got {lam!r}")
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError(f"path_length_m must be finite and > 0, got {length!r}")
    return float(np.sqrt(lam * length))


def circular_aperture_weight(u: np.ndarray | float) -> np.ndarray:
    """Circle-circle overlap kernel ``W(u) = arccos(u) - u sqrt(1 - u^2)``.

    Defined for ``0 <= u <= 1`` (``u = rho / D``); zero outside. Satisfies
    ``int_0^1 u W(u) du = pi / 16``, checked in the validation suite.
    """
    ua = np.asarray(u, dtype=float)
    clipped = np.clip(ua, 0.0, 1.0)
    w = np.arccos(clipped) - clipped * np.sqrt(np.clip(1.0 - clipped * clipped, 0.0, None))
    return np.where((ua >= 0.0) & (ua <= 1.0), w, 0.0)


def normalised_covariance(
    separation_m: np.ndarray | float,
    correlation_scale_m: float,
    model: CorrelationModel = "gaussian",
) -> np.ndarray:
    """Normalised irradiance covariance ``b(rho) / b(0)``, dimensionless.

    Parameters
    ----------
    separation_m
        Separation ``rho`` in metres, ``>= 0``.
    correlation_scale_m
        Correlation scale ``rho_c`` in metres, ``> 0``.
    model
        ``"gaussian"`` or ``"exponential"``.
    """
    rho = np.asarray(separation_m, dtype=float)
    if np.any(rho < 0.0):
        raise ValueError("separation_m must be >= 0")
    rc = float(correlation_scale_m)
    if not np.isfinite(rc) or rc <= 0.0:
        raise ValueError(f"correlation_scale_m must be finite and > 0, got {rc!r}")
    if model == "gaussian":
        return np.exp(-((rho / rc) ** 2))
    if model == "exponential":
        return np.exp(-rho / rc)
    raise ValueError(f"model must be one of {CORRELATION_MODELS}, got {model!r}")


def aperture_averaging_factor(
    diameter_m: float,
    correlation_scale_m: float,
    model: CorrelationModel = "gaussian",
) -> float:
    """Aperture averaging factor ``A(D)`` by quadrature of its definition.

    ``A(D) = (16 / pi) int_0^1 u b(u D) W(u) du``, the change of variable
    ``u = rho / D`` of the integral in this module's docstring.

    Parameters
    ----------
    diameter_m
        Aperture diameter ``D`` in metres, ``>= 0``. ``D = 0`` returns 1.
    correlation_scale_m
        Irradiance correlation scale ``rho_c`` in metres.
    model
        Covariance shape, see :func:`normalised_covariance`.

    Returns
    -------
    float
        ``A`` in ``(0, 1]``, dimensionless.
    """
    d = float(diameter_m)
    if not np.isfinite(d) or d < 0.0:
        raise ValueError(f"diameter_m must be finite and >= 0, got {d!r}")
    if d == 0.0:
        normalised_covariance(0.0, correlation_scale_m, model)
        return 1.0

    def integrand(u: float) -> float:
        bu = float(normalised_covariance(u * d, correlation_scale_m, model))
        wu = float(np.arccos(u) - u * np.sqrt(max(1.0 - u * u, 0.0)))
        return u * bu * wu

    val, _ = integrate.quad(integrand, 0.0, 1.0, epsabs=1e-13, epsrel=1e-11, limit=400)
    return float(val / _W_NORM)


def aperture_averaging_small_d(diameter_m: float, correlation_scale_m: float) -> float:
    """Small-aperture limit of the Gaussian-covariance ``A(D)``.

    Expanding ``exp(-(rho/rho_c)^2) ~ 1 - (rho/rho_c)^2`` and using
    ``int_0^1 u^3 W(u) du = pi/64`` gives

    ``A ~ 1 - D^2 / (4 rho_c^2)``   for ``D << rho_c``.

    Derived in this docstring, not quoted; checked against
    :func:`aperture_averaging_factor` in the validation suite.
    """
    d = float(diameter_m)
    rc = float(correlation_scale_m)
    if rc <= 0.0:
        raise ValueError("correlation_scale_m must be > 0")
    return float(1.0 - d * d / (4.0 * rc * rc))


def aperture_averaging_large_d(
    diameter_m: float, correlation_scale_m: float, *, order: int = 2
) -> float:
    """Large-aperture limit of the Gaussian-covariance ``A(D)``.

    For ``D >> rho_c`` the kernel is flat over the support of ``b``. Using
    ``W(u) = pi/2 - 2u + O(u^3)`` and
    ``int_0^inf rho^k exp(-(rho/rho_c)^2) drho`` for ``k = 1, 2``:

    ``A ~ 4 (rho_c/D)^2 - (8/sqrt(pi)) (rho_c/D)^3``

    so the leading term is ``A D^2 / rho_c^2 -> 4``. Both terms are derived
    in this docstring and checked against the quadrature in
    ``validation/validate_aperture_averaging.py``.

    Parameters
    ----------
    diameter_m
        Aperture diameter in metres.
    correlation_scale_m
        Correlation scale in metres.
    order
        1 for the leading term alone, 2 to include the ``(rho_c/D)^3``
        correction. Default 2.
    """
    d = float(diameter_m)
    rc = float(correlation_scale_m)
    if d <= 0.0 or rc <= 0.0:
        raise ValueError("diameter_m and correlation_scale_m must be > 0")
    if order not in (1, 2):
        raise ValueError(f"order must be 1 or 2, got {order!r}")
    x = rc / d
    a = 4.0 * x * x
    if order == 2:
        a -= 8.0 / np.sqrt(np.pi) * x**3
    return float(a)


def effective_scintillation_index(
    point_si: float,
    diameter_m: float,
    correlation_scale_m: float,
    model: CorrelationModel = "gaussian",
) -> float:
    """Scintillation index seen by an aperture of diameter ``D``.

    ``si(D) = A(D) * si(0)``. This is the standard engineering use of the
    averaging factor: the *shape* of the fading distribution is kept and
    only its scintillation index is reduced. That is an approximation —
    aperture averaging does not strictly preserve the lognormal or
    gamma-gamma family — and it is listed as a limitation in the README.
    """
    si0 = float(point_si)
    if not np.isfinite(si0) or si0 <= 0.0:
        raise ValueError(f"point_si must be finite and > 0, got {si0!r}")
    return float(si0 * aperture_averaging_factor(diameter_m, correlation_scale_m, model))


def equal_area_diameter(total_diameter_m: float, n_apertures: int) -> float:
    """Diameter of each of ``n`` apertures that together match one of ``D``.

    Equal collecting area means ``n (d/2)^2 = (D/2)^2``, so
    ``d = D / sqrt(n)``. This is the fair-comparison convention used by
    ``examples/one_big_vs_many_small.py``: the total glass is held fixed,
    so splitting it buys decorrelation and loses aperture averaging.
    """
    d = float(total_diameter_m)
    n = int(n_apertures)
    if not np.isfinite(d) or d <= 0.0:
        raise ValueError(f"total_diameter_m must be finite and > 0, got {d!r}")
    if n < 1:
        raise ValueError(f"n_apertures must be >= 1, got {n!r}")
    return float(d / np.sqrt(n))
