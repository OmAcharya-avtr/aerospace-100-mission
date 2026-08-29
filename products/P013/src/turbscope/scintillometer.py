r"""Scintillometer-style irradiance-variance forward model and inversion.

A line-of-sight scintillometer measures the normalised intensity variance
(scintillation index) ``sigma_I^2 = <I^2>/<I>^2 - 1`` of a beam propagated
over a path of length ``L``, and infers the path-averaged Cn^2 from it. Two
regimes are modelled:

1. **Weak fluctuations (Rytov theory).** ``sigma_I^2 ~= sigma_R^2``, the
   Rytov variance, which is *linear* in Cn2 and therefore invertible in
   closed form. This is implemented in :func:`rytov_variance` /
   :func:`invert_cn2_weak` and is the mandated classical baseline this
   product benchmarks its learned model against (``MODEL_CARD.md``).
2. **Strong fluctuations (saturation).** As turbulence or path length grow,
   real scintillation stops growing linearly, overshoots (the "focusing"
   regime), and saturates toward an asymptotic value of order 1
   (Andrews & Phillips 2005, Ch. 9, describe this qualitative shape). Because
   the curve rises, peaks, and then falls toward its asymptote, a *single*
   measured ``sigma_I^2`` value in that overshoot band corresponds to
   **more than one** possible ``sigma_R^2`` (hence Cn2) -- the inversion
   becomes genuinely multi-valued. :func:`scintillation_index_full` and
   :func:`invert_cn2_all_roots` model and demonstrate this.

Honesty note on the saturation model
-------------------------------------
:func:`scintillation_index_full` is a **heuristic bridging function built for
this product**, not a specific published curve fit. It is constructed to
reproduce the well-documented *qualitative* shape of scintillation from weak
to strong turbulence (linear Rytov growth at small sigma_R^2; a "focusing"
overshoot above the asymptote around sigma_R^2 of order 1-3; saturation
toward an O(1) asymptote at large sigma_R^2), as described in Andrews &
Phillips (2005) Ch. 9. It reproduces the correct qualitative physics and the
correct weak-limit identity with the Rytov variance; it is **not** claimed to
reproduce any specific published strong-fluctuation numerical curve, and no
equation number from any source is attached to it. Do not use it for
quantitative strong-turbulence engineering predictions -- use it only for
what it is validated for here: demonstrating and quantifying the existence
and shape of the multi-valued inversion (``validation/VALIDATION.md`` S2).

References
----------
Tatarski, V. I. (1961), *Wave Propagation in a Turbulent Medium*, McGraw-Hill.
Andrews, L. C. and Phillips, R. L. (2005), *Laser Beam Propagation through
    Random Media*, 2nd ed., SPIE Press, Ch. 1, 5 and 9.
Wang, T., Ochs, G. R. and Lawrence, R. S. (1978), "Wind measurements by the
    temporal cross-correlation of the optical scintillations", *Appl. Opt.*
    20(23), 4073-4081.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq

from .constants import (
    OPTICAL_WAVELENGTH_RANGE_M,
    PLANE_WAVE_RYTOV_COEFF,
    SPHERICAL_WAVE_RYTOV_COEFF,
    WAVE_TYPES,
    WEAK_REGIME_MAX_SIGMA_R2,
)

__all__ = [
    "SATURATION_ASYMPTOTE",
    "SATURATION_BUMP_HEIGHT",
    "SATURATION_BUMP_LOCATION",
    "MultiValuedInversion",
    "invert_cn2_all_roots",
    "invert_cn2_weak",
    "is_weak_regime",
    "rytov_variance",
    "saturation_peak",
    "scintillation_index_full",
    "scintillation_index_weak",
    "wave_number",
]

# --- heuristic saturation-model parameters (see module docstring) ---------
SATURATION_ASYMPTOTE: float = 1.0
"""Asymptotic sigma_I^2 as sigma_R^2 -> infinity in the heuristic saturation
model (dimensionless). Chosen to match the well-documented order-unity
saturation level of unbounded-wave scintillation (Andrews & Phillips 2005)."""

SATURATION_BUMP_LOCATION: float = 1.5
"""sigma_R^2 at which the heuristic "focusing" overshoot peaks."""

SATURATION_BUMP_HEIGHT: float = 0.5
"""Peak height of the heuristic overshoot bump added on top of the monotone
saturating term (dimensionless, added to sigma_I^2)."""


def _validate_wavelength(wavelength_m: float) -> float:
    lam = float(wavelength_m)
    if not np.isfinite(lam) or lam <= 0.0:
        raise ValueError(f"wavelength_m must be finite and > 0 (got {wavelength_m!r}).")
    lo, hi = OPTICAL_WAVELENGTH_RANGE_M
    if not (lo <= lam <= hi):
        raise ValueError(
            f"wavelength_m = {lam:g} is outside the supported {lo:g}-{hi:g} m "
            "optical/near-IR range."
        )
    return lam


def _validate_path_length(path_length_m: float) -> float:
    length = float(path_length_m)
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError(f"path_length_m must be finite and > 0 (got {path_length_m!r}).")
    return length


def _validate_cn2(cn2_path: ArrayLike) -> NDArray[np.float64]:
    c = np.asarray(cn2_path, dtype=float)
    if not np.all(np.isfinite(c)):
        raise ValueError("cn2_path must be finite.")
    if np.any(c < 0.0):
        raise ValueError("cn2_path must be >= 0 m^-2/3 (it is a variance-like quantity).")
    return c


def _validate_wave_type(wave_type: str) -> str:
    if wave_type not in WAVE_TYPES:
        raise ValueError(f"wave_type must be one of {WAVE_TYPES} (got {wave_type!r}).")
    return wave_type


def _rytov_coeff(wave_type: str) -> float:
    return PLANE_WAVE_RYTOV_COEFF if wave_type == "plane" else SPHERICAL_WAVE_RYTOV_COEFF


def wave_number(wavelength_m: float) -> float:
    """Optical wavenumber ``k = 2*pi / wavelength_m``, rad/m.

    Parameters
    ----------
    wavelength_m : float
        Wavelength, m (0.1-100 um).

    Returns
    -------
    float
        Wavenumber, rad/m.
    """
    lam = _validate_wavelength(wavelength_m)
    return float(2.0 * np.pi / lam)


def rytov_variance(
    cn2_path: ArrayLike,
    path_length_m: float,
    wavelength_m: float,
    wave_type: str = "spherical",
) -> NDArray[np.float64]:
    r"""Rytov (weak-fluctuation) variance for a homogeneous path.

    .. math::

        \sigma_R^2 = C_w\, C_n^2\, k^{7/6}\, L^{11/6}

    with :math:`C_w` = 1.23 (plane wave) or 0.50 (spherical wave).

    Parameters
    ----------
    cn2_path : array_like
        Path-averaged Cn^2, m^-2/3 (constant along the path). >= 0.
    path_length_m : float
        Path length L, m (> 0).
    wavelength_m : float
        Wavelength, m (0.1-100 um).
    wave_type : {"plane", "spherical"}
        Wavefront geometry. Default "spherical" (small-aperture / point-source
        scintillometer, e.g. Wang, Ochs & Lawrence 1978).

    Returns
    -------
    ndarray
        Rytov variance, dimensionless, same shape as ``cn2_path``.

    Notes
    -----
    Source: Tatarski (1961); Andrews & Phillips (2005) Ch. 1, 5. Assumes a
    homogeneous, isotropic Kolmogorov turbulence spectrum and a
    horizontally/along-path-uniform Cn2. Formally valid for ``k*L >> 1``;
    physically meaningful as a *weak-fluctuation* prediction only while
    :func:`is_weak_regime` is True -- see :data:`turbscope.constants.WEAK_REGIME_MAX_SIGMA_R2`.

    Raises
    ------
    ValueError
        On negative/non-finite Cn2, non-positive/non-finite path length,
        wavelength out of range, or an unknown ``wave_type``.
    """
    c = _validate_cn2(cn2_path)
    length = _validate_path_length(path_length_m)
    k = wave_number(wavelength_m)
    coeff = _rytov_coeff(_validate_wave_type(wave_type))
    return np.asarray(coeff * c * k ** (7.0 / 6.0) * length ** (11.0 / 6.0), dtype=float)


def scintillation_index_weak(
    cn2_path: ArrayLike,
    path_length_m: float,
    wavelength_m: float,
    wave_type: str = "spherical",
) -> NDArray[np.float64]:
    """Weak-fluctuation scintillation index, ``sigma_I^2 ~= sigma_R^2``.

    The weak-fluctuation identity between the log-amplitude/Rytov variance and
    the normalised intensity variance (Andrews & Phillips 2005 Ch. 1). Same
    signature and validity as :func:`rytov_variance`; provided as a distinct
    name because it is the physically *measured* quantity a scintillometer
    reports, of which :func:`rytov_variance` is the weak-theory prediction.
    """
    return rytov_variance(cn2_path, path_length_m, wavelength_m, wave_type)


def scintillation_index_full(rytov_var: ArrayLike) -> NDArray[np.float64]:
    r"""Heuristic weak-to-saturated scintillation-index bridging function.

    .. math::

        \sigma_I^2(\sigma_R^2) = \frac{S_\infty \sigma_R^2}{\sigma_R^2 + S_\infty}
            + h \left(\frac{\sigma_R^2}{x_p}\right)^2
              \exp\!\left[2\left(1 - \frac{\sigma_R^2}{x_p}\right)\right]

    with :math:`S_\infty` = :data:`SATURATION_ASYMPTOTE`, :math:`x_p` =
    :data:`SATURATION_BUMP_LOCATION`, :math:`h` = :data:`SATURATION_BUMP_HEIGHT`.

    **This is a heuristic model built for this product, not a published curve
    fit -- see the module docstring "Honesty note" before using it.** By
    construction: :math:`\sigma_I^2(x) \to x` as :math:`x \to 0` (matching the
    weak Rytov identity exactly, verified in ``tests/test_scintillometer.py``
    and to < 3 % for :math:`\sigma_R^2 < 0.05` in
    ``validation/round_trip_recovery.py``), it has a single local maximum
    above :math:`S_\infty` near :math:`x = x_p` (the "focusing" overshoot),
    and it decays to the asymptote :math:`S_\infty` as
    :math:`\sigma_R^2 \to \infty`.

    Parameters
    ----------
    rytov_var : array_like
        Rytov variance sigma_R^2 (>= 0), dimensionless.

    Returns
    -------
    ndarray
        Scintillation index sigma_I^2, dimensionless, same shape.

    Raises
    ------
    ValueError
        If ``rytov_var`` is negative or non-finite.
    """
    x = np.asarray(rytov_var, dtype=float)
    if not np.all(np.isfinite(x)):
        raise ValueError("rytov_var must be finite.")
    if np.any(x < 0.0):
        raise ValueError("rytov_var must be >= 0.")
    s_inf = SATURATION_ASYMPTOTE
    x_p = SATURATION_BUMP_LOCATION
    h = SATURATION_BUMP_HEIGHT
    monotone = s_inf * x / (x + s_inf)
    bump = h * (x / x_p) ** 2 * np.exp(2.0 * (1.0 - x / x_p))
    return np.asarray(monotone + bump, dtype=float)


def saturation_peak() -> tuple[float, float]:
    """Location and height of the heuristic saturation curve's local maximum.

    Found numerically (bounded scalar minimisation is unnecessary here; a
    fine grid search plus a local refinement is exact enough for reporting
    purposes and is what ``validation/saturation_regime.py`` uses).

    Returns
    -------
    (x_peak, sigma_I2_peak) : tuple of float
        Rytov variance at the peak and the peak scintillation index.
    """
    grid = np.geomspace(1e-3, 50.0, 20_000)
    vals = scintillation_index_full(grid)
    i = int(np.argmax(vals))
    lo = grid[max(i - 1, 0)]
    hi = grid[min(i + 1, grid.size - 1)]
    refined = np.linspace(lo, hi, 20_000)
    rvals = scintillation_index_full(refined)
    j = int(np.argmax(rvals))
    return float(refined[j]), float(rvals[j])


def is_weak_regime(rytov_var: ArrayLike) -> NDArray[np.bool_]:
    """True where ``rytov_var <= WEAK_REGIME_MAX_SIGMA_R2``."""
    x = np.asarray(rytov_var, dtype=float)
    if not np.all(np.isfinite(x)):
        raise ValueError("rytov_var must be finite.")
    if np.any(x < 0.0):
        raise ValueError("rytov_var must be >= 0.")
    return np.asarray(x <= WEAK_REGIME_MAX_SIGMA_R2)


def invert_cn2_weak(
    sigma_i2_measured: float,
    path_length_m: float,
    wavelength_m: float,
    wave_type: str = "spherical",
) -> float:
    r"""Classical closed-form single-sensor inversion (the mandated baseline).

    Inverts the weak-fluctuation identity ``sigma_I^2 = C_w Cn2 k^(7/6) L^(11/6)``
    for Cn2:

    .. math::

        C_n^2 = \frac{\sigma_I^2}{C_w\, k^{7/6}\, L^{11/6}}

    This is applied to whatever ``sigma_i2_measured`` value is supplied,
    **without checking the weak-regime assumption** -- exactly what an
    instrument's onboard firmware or a naive analysis script does. Applying
    it to a measurement taken in the saturated regime silently returns a
    single, systematically wrong value; see :func:`invert_cn2_all_roots` and
    ``validation/saturation_regime.py`` for the quantified failure. Use
    :func:`is_weak_regime` on the *result* (via :func:`rytov_variance` of the
    returned Cn2) to check after the fact whether the weak assumption was
    self-consistent.

    Parameters
    ----------
    sigma_i2_measured : float
        Measured scintillation index (>= 0), dimensionless.
    path_length_m : float
        Path length, m (> 0).
    wavelength_m : float
        Wavelength, m.
    wave_type : {"plane", "spherical"}
        Wavefront geometry, must match how the instrument was modelled.

    Returns
    -------
    float
        Inverted Cn2_path, m^-2/3. Always a single value (this formula is
        linear and monotone; the multi-valuedness only appears in the *true*
        strong-fluctuation forward model, not in this inversion formula
        itself).

    Raises
    ------
    ValueError
        On a negative/non-finite measurement or invalid geometry/wavelength.
    """
    s = float(sigma_i2_measured)
    if not np.isfinite(s) or s < 0.0:
        raise ValueError(f"sigma_i2_measured must be finite and >= 0 (got {sigma_i2_measured!r}).")
    length = _validate_path_length(path_length_m)
    k = wave_number(wavelength_m)
    coeff = _rytov_coeff(_validate_wave_type(wave_type))
    return float(s / (coeff * k ** (7.0 / 6.0) * length ** (11.0 / 6.0)))


@dataclass(frozen=True)
class MultiValuedInversion:
    """Result of inverting a measured sigma_I^2 through the *full* saturation
    curve, which may have more than one root.

    Attributes
    ----------
    sigma_i2_measured : float
        The measurement that was inverted.
    rytov_roots : tuple of float
        All ``sigma_R^2`` roots found on the search bracket, ascending.
    cn2_roots : tuple of float
        The corresponding Cn2_path values, m^-2/3, same order.
    is_multivalued : bool
        True if more than one root was found.
    """

    sigma_i2_measured: float
    rytov_roots: tuple[float, ...]
    cn2_roots: tuple[float, ...]
    is_multivalued: bool


def invert_cn2_all_roots(
    sigma_i2_measured: float,
    path_length_m: float,
    wavelength_m: float,
    wave_type: str = "spherical",
    rytov_search_max: float = 200.0,
    n_grid: int = 4000,
) -> MultiValuedInversion:
    """Invert a measurement through the full saturation curve, finding *all*
    roots on ``[0, rytov_search_max]``.

    Demonstrates and quantifies the saturation failure mode: for a
    ``sigma_i2_measured`` between the saturation asymptote and the local
    maximum of :func:`scintillation_index_full`, there are two (or more)
    ``sigma_R^2`` values consistent with the same measurement, and therefore
    two candidate Cn2 values -- the inversion is genuinely ill-posed. This is
    quantified for concrete cases in ``validation/saturation_regime.py``.

    Parameters
    ----------
    sigma_i2_measured : float
        Measured scintillation index (>= 0).
    path_length_m, wavelength_m, wave_type
        As in :func:`rytov_variance`.
    rytov_search_max : float
        Upper end of the sigma_R^2 search bracket (>= 0).
    n_grid : int
        Number of grid points used to bracket sign changes before refining
        each root with Brent's method (>= 10).

    Returns
    -------
    MultiValuedInversion

    Raises
    ------
    ValueError
        On invalid inputs.
    """
    s = float(sigma_i2_measured)
    if not np.isfinite(s) or s < 0.0:
        raise ValueError(f"sigma_i2_measured must be finite and >= 0 (got {sigma_i2_measured!r}).")
    rmax = float(rytov_search_max)
    if not np.isfinite(rmax) or rmax <= 0.0:
        raise ValueError("rytov_search_max must be finite and > 0.")
    n = int(n_grid)
    if n < 10:
        raise ValueError("n_grid must be >= 10.")
    length = _validate_path_length(path_length_m)
    k = wave_number(wavelength_m)
    coeff = _rytov_coeff(_validate_wave_type(wave_type))

    grid = np.linspace(0.0, rmax, n)
    resid = scintillation_index_full(grid) - s
    roots: list[float] = []
    sign = np.sign(resid)
    # Treat an exact zero on the grid as its own bracket.
    zero_idx = np.where(resid == 0.0)[0]
    for idx in zero_idx:
        roots.append(float(grid[idx]))
    change_idx = np.where(np.diff(sign) != 0)[0]
    for i in change_idx:
        a, b = grid[i], grid[i + 1]
        if resid[i] == 0.0 or resid[i + 1] == 0.0:
            continue
        root = brentq(lambda x: float(scintillation_index_full(np.array([x]))[0]) - s, a, b)
        roots.append(float(root))
    roots = sorted(set(round(r, 10) for r in roots))
    cn2_roots = tuple(r / (coeff * k ** (7.0 / 6.0) * length ** (11.0 / 6.0)) for r in roots)
    return MultiValuedInversion(
        sigma_i2_measured=s,
        rytov_roots=tuple(roots),
        cn2_roots=cn2_roots,
        is_multivalued=len(roots) > 1,
    )
