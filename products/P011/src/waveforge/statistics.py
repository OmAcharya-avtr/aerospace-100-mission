"""Kolmogorov turbulence statistics: Zernike variances, r0, and time constants.

Every expression carries its source, units, assumptions and validity range.

Base spectrum
-------------
Kolmogorov phase power spectral density over a horizontal path of Fried
parameter ``r0``::

    Phi_phi(kappa) = 0.49 * r0^(-5/3) * kappa^(-11/3)      [rad^2 m^2]

with ``kappa`` the angular spatial frequency in rad/m (Roddier 1981,
*Progress in Optics* **19**, 281-376, Eq. 3.14).  In cyclic spatial frequency
``f = kappa / (2 pi)`` [cycles/m] this is::

    Phi_phi(f) = 0.49 (2 pi)^(-5/3) r0^(-5/3) f^(-11/3)
               = 0.022903 r0^(-5/3) f^(-11/3)               [rad^2 m^2]

which is the ``0.023 r0^(-5/3) f^(-11/3)`` quoted by Hardy (1998), *Adaptive
Optics for Astronomical Telescopes*, Oxford University Press, Eq. 3.55.

*Assumptions:* isotropic, homogeneous, Kolmogorov inertial subrange; near
field (no scintillation); the outer scale ``L0`` is infinite.  *Validity:*
``1/L0 << f << 1/l0`` (inner scale).  The ``f -> 0`` divergence is the reason
piston, and to a lesser degree tip/tilt, are ill-defined for an infinite outer
scale; both are handled explicitly rather than ignored.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gamma, gammaln

from .zernike import noll_to_nm

__all__ = [
    "KOLMOGOROV_PSD_CYCLIC",
    "KOLMOGOROV_SF_CONSTANT",
    "NOLL_RESIDUAL_TABLE",
    "fried_parameter_from_cn2",
    "greenwood_frequency",
    "greenwood_time_constant",
    "kolmogorov_psd_cyclic",
    "noll_residual_variance",
    "noll_residual_asymptote",
    "phase_structure_function",
    "total_phase_variance",
    "zernike_variance",
]

#: PSD constant in cyclic spatial frequency, ``0.49 (2 pi)^(-5/3)``.
KOLMOGOROV_PSD_CYCLIC: float = 0.49 * (2.0 * np.pi) ** (-5.0 / 3.0)

#: Structure-function constant ``2 (24/5 Gamma(6/5))^(5/6) = 6.883877``, of
#: which the ``6.88`` usually quoted in the literature is the rounded form.
KOLMOGOROV_SF_CONSTANT: float = float(2.0 * (24.0 / 5.0 * gamma(6.0 / 5.0)) ** (5.0 / 6.0))

#: Noll (1976) Table IV: residual mean-square phase after removing the first
#: ``J`` Zernike modes, in units of ``(D/r0)^(5/3)`` [rad^2].  Published
#: reference data only — :func:`noll_residual_variance` computes its own.
NOLL_RESIDUAL_TABLE: dict[int, float] = {
    1: 1.0299,
    2: 0.582,
    3: 0.134,
    4: 0.111,
    5: 0.0880,
    6: 0.0648,
    7: 0.0587,
    8: 0.0525,
    9: 0.0463,
    10: 0.0401,
    11: 0.0377,
    12: 0.0352,
    13: 0.0328,
    14: 0.0304,
    15: 0.0279,
    16: 0.0267,
    17: 0.0255,
    18: 0.0243,
    19: 0.0232,
    20: 0.0220,
    21: 0.0208,
}


def kolmogorov_psd_cyclic(f_cyc_per_m: np.ndarray | float, r0_m: float) -> np.ndarray:
    """Kolmogorov phase PSD in cyclic spatial frequency.

    Parameters
    ----------
    f_cyc_per_m:
        Spatial frequency magnitude [cycles/m].  Zero entries return ``inf``.
    r0_m:
        Fried parameter [m], ``> 0``.

    Returns
    -------
    numpy.ndarray
        ``0.022903 r0^(-5/3) f^(-11/3)`` [rad^2 m^2].
    """
    if not np.isfinite(r0_m) or r0_m <= 0.0:
        raise ValueError(f"r0_m must be finite and > 0, got {r0_m!r}")
    f = np.asarray(f_cyc_per_m, dtype=float)
    if np.any(f < 0.0):
        raise ValueError("spatial frequency must be non-negative")
    with np.errstate(divide="ignore"):
        return KOLMOGOROV_PSD_CYCLIC * r0_m ** (-5.0 / 3.0) * f ** (-11.0 / 3.0)


def phase_structure_function(r_m: np.ndarray | float, r0_m: float) -> np.ndarray:
    """Kolmogorov phase structure function ``D_phi(r) = 6.8839 (r/r0)^(5/3)``.

    Source: D. L. Fried, "Statistical geometry of atmospheric turbulence",
    *J. Opt. Soc. Am.* **55**, 1427-1435 (1965); Roddier 1981 Eq. 3.19.  The
    constant :data:`KOLMOGOROV_SF_CONSTANT` is used at full precision
    (6.883877), not the rounded 6.88 that most texts quote; the two differ by
    0.056 %.

    Units: rad^2.  Validity: ``l0 << r << L0``; infinite outer scale.
    """
    if not np.isfinite(r0_m) or r0_m <= 0.0:
        raise ValueError(f"r0_m must be finite and > 0, got {r0_m!r}")
    r = np.asarray(r_m, dtype=float)
    if np.any(r < 0.0):
        raise ValueError("separation r must be non-negative")
    return KOLMOGOROV_SF_CONSTANT * (r / r0_m) ** (5.0 / 3.0)


def fried_parameter_from_cn2(cn2_path_integral: float, wavelength_m: float) -> float:
    """Fried parameter from the path-integrated ``Cn^2``.

    ``r0 = [0.423 k^2 int Cn^2(z) dz]^(-3/5)`` with ``k = 2 pi / lambda``
    (Fried 1965; Roddier 1981 Eq. 3.20; plane-wave, zenith path).

    Parameters
    ----------
    cn2_path_integral:
        ``int Cn^2 dz`` [m^(1/3)], ``> 0``.
    wavelength_m:
        Optical wavelength [m], ``> 0``.

    Returns
    -------
    float
        ``r0`` [m].  Assumptions: plane wave, near field, Kolmogorov spectrum.
    """
    if not np.isfinite(cn2_path_integral) or cn2_path_integral <= 0.0:
        raise ValueError(f"cn2_path_integral must be finite and > 0, got {cn2_path_integral!r}")
    if not np.isfinite(wavelength_m) or wavelength_m <= 0.0:
        raise ValueError(f"wavelength_m must be finite and > 0, got {wavelength_m!r}")
    k = 2.0 * np.pi / wavelength_m
    return float((0.423 * k**2 * cn2_path_integral) ** (-3.0 / 5.0))


def greenwood_frequency(r0_m: float, wind_speed_m_s: float) -> float:
    """Greenwood frequency for a single frozen-flow layer.

    ``f_G = 0.427 v / r0`` [Hz].  Source: D. P. Greenwood, "Bandwidth
    specification for adaptive optics systems", *J. Opt. Soc. Am.* **67**,
    390-393 (1977); the single-layer reduction of his Eq. 4 with the
    ``0.427`` constant as given by Hardy (1998) Eq. 9.29.

    Assumptions: Taylor frozen-flow, single layer, Kolmogorov spectrum, and a
    first-order (integrator) correction loop.
    """
    if not np.isfinite(r0_m) or r0_m <= 0.0:
        raise ValueError(f"r0_m must be finite and > 0, got {r0_m!r}")
    if not np.isfinite(wind_speed_m_s) or wind_speed_m_s < 0.0:
        raise ValueError(f"wind_speed_m_s must be finite and >= 0, got {wind_speed_m_s!r}")
    return float(0.427 * wind_speed_m_s / r0_m)


def greenwood_time_constant(r0_m: float, wind_speed_m_s: float) -> float:
    """Atmospheric coherence time ``tau0 = 0.314 r0 / v`` [s].

    Source: D. L. Fried, "Time-delay-induced mean-square error in adaptive
    optics", *J. Opt. Soc. Am. A* **7**, 1224-1225 (1990); Hardy (1998)
    Eq. 9.31.  Defined so that a pure delay ``tau`` costs a residual variance
    ``(tau / tau0)^(5/3)`` rad^2.

    Assumptions: frozen flow, single layer, Kolmogorov, perfect correction of
    everything except the delay.  Requires ``v > 0``.
    """
    if not np.isfinite(r0_m) or r0_m <= 0.0:
        raise ValueError(f"r0_m must be finite and > 0, got {r0_m!r}")
    if not np.isfinite(wind_speed_m_s) or wind_speed_m_s <= 0.0:
        raise ValueError(f"wind_speed_m_s must be finite and > 0, got {wind_speed_m_s!r}")
    return float(0.314 * r0_m / wind_speed_m_s)


# --- Zernike coefficient variances -------------------------------------------
#
# Derivation (Noll 1976, Eqs. 8 and 18).  The Fourier transform of the
# orthonormal mode Z_j over a circular aperture of radius R is
#
#     Q_j(k) = sqrt(n+1) J_{n+1}(2 pi k R) / (pi k R) * (angular factor)
#
# whose squared angular factor integrates to 2 pi over theta for every m.
# Hence
#
#     <a_j^2> = 2 pi (n+1) int_0^inf Phi_phi(k) [J_{n+1}(2 pi k R)/(pi k R)]^2 k dk
#
# and with the Weber-Schafheitlin integral
#
#     int_0^inf u^(-14/3) J_mu(u)^2 du
#         = Gamma(14/3) Gamma(mu - 7/3 + 1/2)
#           / [2^(14/3) Gamma(17/6)^2 Gamma(mu + 7/3 + 1/2)]
#
# this closes to <a_j^2> = C (n+1) Gamma(n - 5/6) / Gamma(n + 23/6) (D/r0)^(5/3)
# with the constant C assembled below.  Logs are used so that large n does not
# overflow the Gamma function.

_LN_C = (
    np.log(
        2.0
        * np.pi
        * KOLMOGOROV_PSD_CYCLIC
        * (2.0 * np.pi) ** (11.0 / 3.0)
        / (np.pi**2 * 2.0 ** (5.0 / 3.0))
    )
    + gammaln(14.0 / 3.0)
    - (14.0 / 3.0) * np.log(2.0)
    - 2.0 * gammaln(17.0 / 6.0)
)


def _variance_for_order(n: int) -> float:
    """``<a_j^2>`` in units of ``(D/r0)^(5/3)`` for any mode of radial order n."""
    if n < 1:
        raise ValueError("piston (n = 0) has infinite variance for an infinite outer scale")
    return float(np.exp(_LN_C + np.log(n + 1) + gammaln(n - 5.0 / 6.0) - gammaln(n + 23.0 / 6.0)))


def zernike_variance(j: int, d_over_r0: float = 1.0) -> float:
    """Kolmogorov variance of the Noll coefficient ``a_j`` [rad^2].

    ``<a_j^2> = C (n+1) Gamma(n-5/6)/Gamma(n+23/6) (D/r0)^(5/3)``, derived
    above from Noll (1976) Eqs. 8 and 18.  Depends only on the radial order
    ``n``, so all ``n+1`` modes of an order share a variance.

    Raises for ``j = 1`` (piston): its variance diverges for an infinite outer
    scale and is not a meaningful number.
    """
    n, _ = noll_to_nm(j)
    if not np.isfinite(d_over_r0) or d_over_r0 <= 0.0:
        raise ValueError(f"d_over_r0 must be finite and > 0, got {d_over_r0!r}")
    return _variance_for_order(n) * float(d_over_r0) ** (5.0 / 3.0)


def total_phase_variance(d_over_r0: float = 1.0, n_max: int = 20000) -> float:
    """Piston-removed Kolmogorov phase variance over a circular pupil [rad^2].

    Summing ``(n+1) <a^2>(n)`` over radial orders converges to Noll's
    ``Delta_1 = 1.0299 (D/r0)^(5/3)``.  ``n_max`` truncates the sum; the tail
    falls as ``n^(-11/3)`` so 20000 orders is far past double precision.
    """
    if int(n_max) != n_max or n_max < 1:
        raise ValueError(f"n_max must be an integer >= 1, got {n_max!r}")
    orders = np.arange(1, int(n_max) + 1)
    per_order = np.exp(
        _LN_C + np.log(orders + 1) + gammaln(orders - 5.0 / 6.0) - gammaln(orders + 23.0 / 6.0)
    )
    return float(np.sum((orders + 1) * per_order)) * float(d_over_r0) ** (5.0 / 3.0)


def noll_residual_variance(j_max: int, d_over_r0: float = 1.0) -> float:
    """Residual phase variance after perfect correction of Noll modes 1..j_max.

    ``Delta_J = Delta_1 - sum_{j=2}^{J} <a_j^2>`` [rad^2].  Compare with
    :data:`NOLL_RESIDUAL_TABLE` (Noll 1976 Table IV).
    """
    if int(j_max) != j_max or j_max < 1:
        raise ValueError(f"j_max must be an integer >= 1, got {j_max!r}")
    total = total_phase_variance(1.0)
    removed = sum(zernike_variance(j, 1.0) for j in range(2, int(j_max) + 1))
    return (total - removed) * float(d_over_r0) ** (5.0 / 3.0)


def noll_residual_asymptote(j_max: int, d_over_r0: float = 1.0) -> float:
    """Noll's large-``J`` asymptote ``Delta_J ~ 0.2944 J^(-sqrt(3)/2)``.

    Source: Noll (1976), Eq. 32.  Valid for ``J`` large (a few tens upward);
    it is quoted here as a reference curve, not used internally.
    """
    if int(j_max) != j_max or j_max < 1:
        raise ValueError(f"j_max must be an integer >= 1, got {j_max!r}")
    return float(0.2944 * j_max ** (-np.sqrt(3.0) / 2.0)) * float(d_over_r0) ** (5.0 / 3.0)
