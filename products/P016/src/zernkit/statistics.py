"""Kolmogorov-turbulence Zernike coefficient statistics and Noll residual variances.

Reference
---------
R. J. Noll, "Zernike polynomials and atmospheric turbulence", *Journal of the
Optical Society of America* **66** (3), 207-211 (1976). The residual-variance
values reproduced in :data:`NOLL_TABLE_IV` are the published ones from that
paper's table of residual mean-square wavefront error after removal of the
first ``J`` Zernike terms. No page or table number beyond the paper's own
labelling is asserted here.

Model and assumptions
---------------------
* Kolmogorov turbulence (infinite outer scale, ``-11/3`` power law), phase
  power spectral density in *cyclic* spatial frequency ``k`` [cycles/m]::

      Phi(k) = C_psd * r0^(-5/3) * k^(-11/3)      [rad^2 m^2]

  Noll's Eq. (4) quotes ``C_psd = 0.023``. The unrounded value follows from
  the standard angular-frequency form ``Phi_phi(kappa) = 0.490 r0^(-5/3)
  kappa^(-11/3)`` (F. Roddier, "The effects of atmospheric turbulence in
  optical astronomy", *Progress in Optics* **XIX**, 281-376, 1981; also
  J. W. Hardy, *Adaptive Optics for Astronomical Telescopes*, OUP 1998) via
  ``C_psd = 0.490 / (2 pi)^(5/3)``. Both are exposed; the default is Noll's
  0.023 so that results are directly comparable with his table.
* Circular unobscured pupil of diameter ``D``; near-field (no scintillation);
  phase only.
* Results scale as ``(D / r0)^(5/3)`` and are in **rad^2** when ``r0`` is the
  Fried parameter at the wavelength of interest.

Coefficient variance
--------------------
Projecting the phase spectrum onto the Zernike modes (Noll's Eqs. 8 and 30,
with the Fourier transform of ``Z_j`` over a circular aperture proportional to
``J_{n+1}(2 pi k R) / (pi k R)``) and evaluating the resulting Bessel integral
with the Weber-Schafheitlin formula
``int_0^inf t^-L J_v(t)^2 dt = Gamma(L) Gamma(v - L/2 + 1/2) /
[2^L Gamma((L+1)/2)^2 Gamma(v + L/2 + 1/2)]`` (Gradshteyn & Ryzhik 6.574.2)
with ``L = 14/3``, ``v = n + 1`` gives a variance that depends only on the
radial degree ``n``::

    <a_j^2> = 8 C_psd pi^(8/3) (n+1) (D/r0)^(5/3)
              * Gamma(14/3) Gamma(n - 5/6)
              / [2^(14/3) Gamma(17/6)^2 Gamma(n + 23/6)]

Every mode of a given radial order therefore has the same variance -- a
property worth checking against the published table, since it makes
``Delta_J`` a step function of ``J`` within each order.

Residual variance
-----------------
``Delta_J = sum_{j > J} <a_j^2>``, the mean-square wavefront error left after
the first ``J`` Noll modes are removed perfectly. Computed here by direct
summation over radial orders up to ``n_max`` (default 200000); the neglected
tail falls as ``n^(-5/3)`` and is below ``1e-9 (D/r0)^(5/3)`` at that cutoff.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gamma, gammaln

from .indexing import noll_to_nm

__all__ = [
    "NOLL_PSD_CONSTANT",
    "KOLMOGOROV_PSD_CONSTANT",
    "NOLL_TABLE_IV",
    "coefficient_variance",
    "coefficient_variance_noll",
    "residual_variance",
    "residual_variance_asymptotic",
]

#: Kolmogorov phase-PSD constant as quoted in Noll (1976), Eq. (4), for the
#: cyclic spatial-frequency convention. Rounded to two significant figures in
#: the source, which is the dominant cause of the sub-percent offset between
#: computed and tabulated residual variances.
NOLL_PSD_CONSTANT: float = 0.023

#: Unrounded equivalent, from Phi_phi(kappa) = 0.490 r0^(-5/3) kappa^(-11/3)
#: (Roddier 1981; Hardy 1998) converted to cycles/m: 0.490 / (2 pi)^(5/3).
KOLMOGOROV_PSD_CONSTANT: float = 0.490 / (2.0 * np.pi) ** (5.0 / 3.0)

#: Published residual mean-square error Delta_J in units of (D/r0)^(5/3), after
#: removal of the first J Zernike terms in Noll's ordering.
#: Source: Noll (1976), JOSA 66(3), 207-211, residual-error table. Reproduced
#: verbatim for comparison; NOT used in any computation.
NOLL_TABLE_IV: dict[int, float] = {
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

# Constant prefactor of the coefficient-variance expression, excluding C_psd,
# (n+1) and the Gamma ratio in n.
_PREFACTOR = 8.0 * np.pi ** (8.0 / 3.0) * gamma(14.0 / 3.0) / (
    2.0 ** (14.0 / 3.0) * gamma(17.0 / 6.0) ** 2
)


def _check_d_over_r0(d_over_r0: float) -> float:
    value = float(d_over_r0)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"d_over_r0 must be finite and > 0, got {d_over_r0!r}")
    return value


def coefficient_variance(
    n: int,
    d_over_r0: float = 1.0,
    psd_constant: float = NOLL_PSD_CONSTANT,
) -> float:
    """Kolmogorov variance of any Zernike coefficient of radial degree ``n``.

    Parameters
    ----------
    n : int
        Radial degree, ``n >= 1``. Piston (``n = 0``) is excluded: its variance
        diverges for an infinite outer scale, which is why Noll's series starts
        at ``j = 2``.
    d_over_r0 : float, optional
        Telescope diameter over Fried parameter, ``D / r0`` (dimensionless),
        both at the same wavelength. Default 1.0.
    psd_constant : float, optional
        ``C_psd`` in the phase PSD; default :data:`NOLL_PSD_CONSTANT`.

    Returns
    -------
    float
        ``<a_j^2>`` in rad^2 (if ``r0`` is the Fried parameter in the same
        units as ``D`` and the phase is in radians).

    Notes
    -----
    Independent of ``m``: for example every ``n = 2`` mode (defocus and both
    astigmatisms) has the same variance.
    """
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)):
        raise TypeError(f"n must be an integer, got {n!r}")
    if n < 1:
        raise ValueError(
            f"n must be >= 1; the piston variance diverges for Kolmogorov turbulence, got n={n}"
        )
    scale = _check_d_over_r0(d_over_r0) ** (5.0 / 3.0)
    ratio = np.exp(gammaln(n - 5.0 / 6.0) - gammaln(n + 23.0 / 6.0))
    return float(psd_constant * _PREFACTOR * (n + 1) * ratio * scale)


def coefficient_variance_noll(
    j: int,
    d_over_r0: float = 1.0,
    psd_constant: float = NOLL_PSD_CONSTANT,
) -> float:
    """Kolmogorov variance of the Noll-indexed coefficient ``a_j`` (``j >= 2``)."""
    n, _ = noll_to_nm(j)
    if n < 1:
        raise ValueError("j = 1 is piston; its Kolmogorov variance diverges (use j >= 2)")
    return coefficient_variance(n, d_over_r0=d_over_r0, psd_constant=psd_constant)


def residual_variance(
    j_removed: int,
    d_over_r0: float = 1.0,
    psd_constant: float = NOLL_PSD_CONSTANT,
    n_max: int = 200_000,
) -> float:
    """Residual phase variance ``Delta_J`` after removing the first ``J`` Noll modes.

    ``Delta_J = sum_{j > J} <a_j^2>``, evaluated by summing complete radial
    orders and adding back the modes of the partially removed order.

    Parameters
    ----------
    j_removed : int
        ``J``, the number of Noll modes removed (``J >= 1``; ``J = 1`` is
        piston-removed only).
    d_over_r0 : float, optional
        ``D / r0`` (dimensionless).
    psd_constant : float, optional
        ``C_psd`` in the phase PSD.
    n_max : int, optional
        Radial-order cutoff of the summation (default 200000). The truncated
        tail scales as ``n_max^(-5/3)``.

    Returns
    -------
    float
        ``Delta_J`` in rad^2 for the given ``D / r0``.

    Notes
    -----
    Compare with :data:`NOLL_TABLE_IV` for ``J <= 21``. Agreement is at the
    0.5-1 % level; the residual gap is dominated by the two-significant-figure
    rounding of ``C_psd`` in Noll's Eq. (4) and by rounding in the published
    table itself (see ``validation/VALIDATION.md``).
    """
    if isinstance(j_removed, bool) or not isinstance(j_removed, (int, np.integer)):
        raise TypeError(f"j_removed must be an integer, got {j_removed!r}")
    if j_removed < 1:
        raise ValueError(f"j_removed must be >= 1 (piston is always removed), got {j_removed}")
    if isinstance(n_max, bool) or not isinstance(n_max, (int, np.integer)) or n_max < 10:
        raise ValueError(f"n_max must be an integer >= 10, got {n_max!r}")

    j_removed = int(j_removed)
    n_cut, _ = noll_to_nm(j_removed)
    scale = _check_d_over_r0(d_over_r0) ** (5.0 / 3.0)

    if n_cut + 1 > n_max:
        raise ValueError(
            f"n_max={n_max} is below the radial order n={n_cut} implied by j_removed={j_removed}"
        )

    # Complete radial orders strictly above n_cut: (n + 1) modes each.
    orders = np.arange(n_cut + 1, int(n_max) + 1, dtype=np.float64)
    per_mode = (
        psd_constant
        * _PREFACTOR
        * (orders + 1.0)
        * np.exp(gammaln(orders - 5.0 / 6.0) - gammaln(orders + 23.0 / 6.0))
    )
    total = float(np.sum((orders + 1.0) * per_mode))

    # Modes of order n_cut that survive because j_removed stopped part-way.
    j_end_of_order = (n_cut + 1) * (n_cut + 2) // 2
    n_surviving = j_end_of_order - j_removed
    if n_surviving > 0 and n_cut >= 1:
        total += n_surviving * coefficient_variance(n_cut, 1.0, psd_constant)
    return total * scale


def residual_variance_asymptotic(j_removed: int, d_over_r0: float = 1.0) -> float:
    """Noll's large-``J`` asymptote ``Delta_J ~= 0.2944 J^(-sqrt(3)/2) (D/r0)^(5/3)``.

    Source: Noll (1976), the asymptotic expression quoted for large numbers of
    corrected modes. Accurate to about 1-2 % against the tabulated values
    already by ``J ~ 20``; not valid for small ``J``.
    """
    if isinstance(j_removed, bool) or not isinstance(j_removed, (int, np.integer)):
        raise TypeError(f"j_removed must be an integer, got {j_removed!r}")
    if j_removed < 1:
        raise ValueError(f"j_removed must be >= 1, got {j_removed}")
    scale = _check_d_over_r0(d_over_r0) ** (5.0 / 3.0)
    return float(0.2944 * float(j_removed) ** (-np.sqrt(3.0) / 2.0) * scale)
