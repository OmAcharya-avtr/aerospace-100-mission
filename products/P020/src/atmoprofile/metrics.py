r"""Atmospheric turbulence metrics from a Cn^2 profile.

Every function here evaluates one weighted integral along the vertical
(:mod:`atmoprofile.integrals`) and then applies an explicit, stated power of
sec(zeta).  The zenith dependence is never hidden inside a coefficient.

Summary of the exponents (derivations are in the individual docstrings; the
values are also machine-readable in
:data:`atmoprofile.constants.EXPONENT_SEC_ZENITH` and
:data:`atmoprofile.constants.EXPONENT_WAVELENGTH`):

======================  ==================  ====================
quantity                sec(zeta) exponent  wavelength exponent
======================  ==================  ====================
r0 (plane, spherical)   -3/5                +6/5
theta0                  -8/5                +6/5
f_Greenwood             +3/5                -6/5
Rytov variance          +11/6               -7/6
scintillation index     +11/6               -7/6
======================  ==================  ====================

References (work-level only; no page, equation or table numbers are quoted
anywhere in this package - see the README "Citation policy"):

* D. L. Fried, JOSA 56(10), 1372-1379, 1966 (r0).
* D. L. Fried, JOSA 72(1), 52-61, 1982 (isoplanatism).
* D. P. Greenwood, JOSA 67(3), 390-393, 1977 (bandwidth / f_G).
* L. C. Andrews and R. L. Phillips, "Laser Beam Propagation through Random
  Media", 2nd ed., SPIE Press, 2005 (slant-path Rytov variance, scintillation).
* J. W. Hardy, "Adaptive Optics for Astronomical Telescopes", OUP, 1998.
* F. Roddier, Progress in Optics XIX, 281-376, 1981.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import asdict, dataclass

from ._validate import check_choice, check_wavelength, check_zenith
from .constants import C_FRIED, C_GREENWOOD, C_ISOPLANATIC, C_RYTOV
from .integrals import (
    effective_turbulence_height,
    turbulence_moment,
    weighted_integral,
    wind_weighted_moment,
)
from .profiles import Cn2Profile
from .wind import WindProfile

__all__ = [
    "WEAK_FLUCTUATION_LIMIT",
    "fried_parameter",
    "isoplanatic_angle",
    "greenwood_frequency",
    "rytov_variance",
    "scintillation_index",
    "coherence_length_to_seeing",
    "TurbulenceSummary",
    "summarize",
]

#: Rytov variance above which the weak-fluctuation (first-order Rytov) theory
#: used by this package stops being valid; beyond it the scintillation index
#: saturates and eventually decreases, which requires the strong-fluctuation
#: theory of Andrews & Phillips (2005) that is deliberately not implemented
#: here.  Exceeding the limit raises a ``UserWarning``, never a silent result.
WEAK_FLUCTUATION_LIMIT: float = 1.0

_WAVE_KINDS: tuple[str, ...] = ("plane", "spherical")
_PATHS: tuple[str, ...] = ("downlink", "uplink")


def _wavenumber(wavelength_m: float) -> float:
    """Optical wavenumber k = 2 pi / lambda, rad/m."""
    return 2.0 * math.pi / wavelength_m


def _secant(zenith_rad: float) -> float:
    """sec(zeta) = 1 / cos(zeta), the plane-parallel airmass factor."""
    return 1.0 / math.cos(zenith_rad)


def fried_parameter(
    profile: Cn2Profile,
    wavelength_m: float,
    *,
    zenith_rad: float = 0.0,
    h_ground: float = 0.0,
    h_top: float | None = None,
    wave: str = "plane",
    path: str = "downlink",
    method: str = "quad",
    n_nodes: int = 2001,
) -> float:
    r"""Fried coherence length r0.

    **Weighting integral**

    Plane wave (a source at infinity, e.g. a star or a satellite downlink
    beacon whose beam is much wider than the aperture):

    .. math::

        r_0 = \left[0.423\,k^2\,\sec\zeta
              \int_{h_0}^{H} C_n^2(h)\,dh\right]^{-3/5}

    Spherical wave (a point source at a finite distance), with
    :math:`u = h - h_0` and :math:`L = H - h_0`:

    .. math::

        r_0^{sph} = \left[0.423\,k^2\,\sec\zeta \int_{h_0}^{H} C_n^2(h)\,
                    W_{sph}(u)\,dh\right]^{-3/5},\qquad
        W_{sph}(u) = \begin{cases}
            (1 - u/L)^{5/3} & \text{downlink (source at } H\text{)}\\
            (u/L)^{5/3}     & \text{uplink (source at } h_0\text{)}
        \end{cases}

    The spherical weight is (distance from the source / total distance)^(5/3):
    turbulence close to the point source distorts a wavefront that has not yet
    expanded, so it contributes least to the transverse coherence measured at
    the receiver.

    **Constants** ``0.423 = 2.914 / 6.884``, from equating the Kolmogorov
    plane-wave phase structure function ``D(r) = 2.914 k^2 r^(5/3) mu_0`` with
    Fried's definition ``D(r) = 6.884 (r/r0)^(5/3)``; see
    :mod:`atmoprofile.constants` for the arithmetic.  Source: Fried (1966);
    Andrews & Phillips (2005); Hardy (1998).

    **Units** ``wavelength_m`` m; ``zenith_rad`` rad; altitudes m; Cn^2
    m^(-2/3); returns r0 in **metres**.

    **Zenith dependence** the only zenith factor is the path-length element
    ``dz = sec(zeta) dh``, so ``r0 ~ sec(zeta)^(-3/5) = cos(zeta)^(3/5)``.
    The fractional coordinate ``u/L`` of the spherical weight is unchanged by
    the slant, so the exponent is the same for both wave types.

    **Wavelength dependence** ``r0 ~ k^(-6/5) ~ lambda^(6/5)``.

    **Assumptions**

    * Kolmogorov spectrum with an infinite outer scale and a negligible inner
      scale.  A finite outer scale (10-100 m in practice) *increases* the
      effective coherence length; this package does not apply a von Karman
      correction, so r0 here is the Kolmogorov value.
    * Horizontally homogeneous, isotropic, frozen turbulence; near-field
      (geometrical-optics) phase perturbation only.
    * Plane-parallel atmosphere: ``sec(zeta)`` airmass, valid to roughly
      60 deg (a ``UserWarning`` is issued beyond that).
    * Weak-fluctuation regime; r0 is a phase statistic and remains defined
      under strong scintillation, but the AO interpretations of it do not.

    Parameters
    ----------
    profile:
        Cn^2 profile, m^(-2/3).
    wavelength_m:
        Optical wavelength, m (100 nm - 20 um).
    zenith_rad:
        Zenith angle, rad, in [0, pi/2).
    h_ground, h_top:
        Path endpoints in metres; ``h_top=None`` uses the profile's top.
    wave:
        ``"plane"`` (default) or ``"spherical"``.
    path:
        ``"downlink"`` (default) or ``"uplink"``; used only for a spherical
        wave, where the geometry is not symmetric.
    method, n_nodes:
        Quadrature controls, see :mod:`atmoprofile.integrals`.

    Returns
    -------
    float
        Fried parameter r0 in metres.

    Raises
    ------
    ValueError
        On an out-of-band wavelength, a zenith angle outside [0, pi/2), an
        integration range outside the profile's validity, or a zero integral.
    """
    lam = check_wavelength(wavelength_m)
    zen = check_zenith(zenith_rad)
    kind = check_choice("wave", wave, _WAVE_KINDS)
    direction = check_choice("path", path, _PATHS)
    k = _wavenumber(lam)

    if kind == "plane":
        mu = turbulence_moment(
            profile, 0.0, h_ground=h_ground, h_top=h_top, method=method, n_nodes=n_nodes
        )
    else:
        top = profile.h_max if h_top is None else float(h_top)
        length = top - float(h_ground)
        h0 = float(h_ground)
        if direction == "downlink":

            def wgt(h: float) -> float:
                return max(1.0 - (h - h0) / length, 0.0) ** (5.0 / 3.0)

        else:

            def wgt(h: float) -> float:
                return max((h - h0) / length, 0.0) ** (5.0 / 3.0)

        mu = weighted_integral(
            profile, wgt, h_ground=h_ground, h_top=h_top, method=method, n_nodes=n_nodes
        )

    if mu <= 0.0:
        raise ValueError(
            "the Cn^2 path integral evaluated to zero or negative; r0 is undefined "
            "(check the integration limits and the profile)"
        )
    return float((C_FRIED * k**2 * _secant(zen) * mu) ** (-0.6))


def isoplanatic_angle(
    profile: Cn2Profile,
    wavelength_m: float,
    *,
    zenith_rad: float = 0.0,
    h_ground: float = 0.0,
    h_top: float | None = None,
    method: str = "quad",
    n_nodes: int = 2001,
) -> float:
    r"""Isoplanatic angle theta0 (rad).

    **Weighting integral**

    .. math::

        \theta_0 = \left[2.914\,k^2\,\sec^{8/3}\!\zeta
                   \int_{h_0}^{H} C_n^2(h)\,(h-h_0)^{5/3}\,dh\right]^{-3/5}

    **Constant** 2.914 is the Kolmogorov plane-wave structure-function
    coefficient (Fried 1982; Roddier 1981; Andrews & Phillips 2005).  It is not
    independent of the 0.423 in :func:`fried_parameter`: the two together give
    ``theta0 = 0.314 r0 / h_bar`` with
    ``h_bar = [mu_(5/3)/mu_0]^(3/5)``, and this package reproduces the 0.314 to
    machine precision (a regression test asserts it).

    **Units** returns radians.  ``mu_(5/3)`` has units m^(2) (m^(-2/3) * m^(5/3)
    * m).

    **Zenith dependence** ``sec^(8/3)`` inside the bracket, i.e.
    ``theta0 ~ sec(zeta)^(-8/5) = cos(zeta)^(8/5)``.  Derivation: the moment arm
    along the slant path is ``z = (h-h_0) sec(zeta)``, contributing
    ``sec^(5/3)``, and the path element contributes another ``sec``; total
    ``sec^(8/3)`` inside a ``-3/5`` power.

    **Wavelength dependence** ``theta0 ~ lambda^(6/5)``, the same as r0.

    **Assumptions**

    * Same Kolmogorov, plane-parallel and homogeneity assumptions as
      :func:`fried_parameter`.
    * The angular decorrelation is defined by a 1 rad^2 mean-square wavefront
      difference between two directions (Fried's definition); other
      conventions (e.g. Marechal-based) differ by an O(1) factor.
    * Plane-wave (infinite-source) geometry: this is the astronomical /
      downlink isoplanatic angle, not the uplink point-ahead angle.
    * Finite-aperture and outer-scale effects are ignored.

    Returns
    -------
    float
        Isoplanatic angle in radians.
    """
    lam = check_wavelength(wavelength_m)
    zen = check_zenith(zenith_rad)
    k = _wavenumber(lam)
    mu53 = turbulence_moment(
        profile, 5.0 / 3.0, h_ground=h_ground, h_top=h_top, method=method, n_nodes=n_nodes
    )
    if mu53 <= 0.0:
        raise ValueError(
            "the 5/3 moment evaluated to zero or negative; theta0 is undefined "
            "(a profile concentrated exactly at the observer has no anisoplanatism)"
        )
    return float((C_ISOPLANATIC * k**2 * _secant(zen) ** (8.0 / 3.0) * mu53) ** (-0.6))


def greenwood_frequency(
    profile: Cn2Profile,
    wind: WindProfile,
    wavelength_m: float,
    *,
    zenith_rad: float = 0.0,
    h_ground: float = 0.0,
    h_top: float | None = None,
    method: str = "quad",
    n_nodes: int = 2001,
) -> float:
    r"""Greenwood frequency f_G (Hz) - the AO closed-loop bandwidth scale.

    **Weighting integral**

    .. math::

        f_G = \left[0.102\,k^2\,\sec\zeta
              \int_{h_0}^{H} C_n^2(h)\,v^{5/3}(h)\,dh\right]^{3/5}

    An equivalent published form is
    ``f_G = 2.31 lambda^(-6/5) [sec(zeta) int Cn^2 v^(5/3) dh]^(3/5)``; the two
    agree because ``0.102^(3/5) (2 pi)^(6/5) = 2.3067`` (evaluated, not
    asserted, in :mod:`atmoprofile.constants`; a test checks the two forms
    against each other).

    **Constant** 0.102 from Greenwood (1977); also given by Hardy (1998).

    **Units** ``v`` m/s, Cn^2 m^(-2/3), altitude m; the bracket is s^(-5/3) so
    the result is in **Hz**.

    **Zenith dependence** ``sec(zeta)`` to the first power inside the bracket,
    i.e. ``f_G ~ sec(zeta)^(3/5)``.  This assumes the wind profile supplied is
    already the component *transverse to the line of sight*, so the only zenith
    factor is the path-length element ``dz = sec(zeta) dh``.  If instead one
    models the apparent layer-crossing speed as growing like ``sec(zeta)``, an
    extra ``sec^(5/3)`` appears inside the bracket and the exponent becomes
    8/5.  This package implements the first (Greenwood's published) convention
    and does not silently choose the second; the second is larger by exactly
    sec(zeta), i.e. 41.4 % at zeta = 45 deg, and the choice is listed under
    Limitations in the README.

    **Wavelength dependence** ``f_G ~ lambda^(-6/5)``.

    **Assumptions**

    * Taylor frozen-flow: layers are advected rigidly, turbulence does not
      evolve on a crossing timescale.
    * Greenwood's definition of the bandwidth: the closed-loop 3 dB bandwidth
      of a first-order servo at which the residual mean-square tracking phase
      error equals 1 rad^2.  Other bandwidth definitions (Tyler frequency for
      tilt, e.g.) are different quantities.
    * Kolmogorov spectrum, plane-parallel atmosphere, weak fluctuations.
    * The wind model is climatological unless the caller supplies measurements.

    Parameters
    ----------
    wind:
        Transverse wind-speed profile, m/s (e.g.
        :func:`atmoprofile.wind.bufton_wind`).

    Returns
    -------
    float
        Greenwood frequency in Hz.
    """
    lam = check_wavelength(wavelength_m)
    zen = check_zenith(zenith_rad)
    k = _wavenumber(lam)
    mu_v = wind_weighted_moment(
        profile,
        wind,
        power=5.0 / 3.0,
        h_ground=h_ground,
        h_top=h_top,
        method=method,
        n_nodes=n_nodes,
    )
    if mu_v < 0.0:  # pragma: no cover - defensive; wind and Cn^2 are non-negative
        raise ValueError("the wind-weighted moment is negative; check the profiles")
    return float((C_GREENWOOD * k**2 * _secant(zen) * mu_v) ** 0.6)


def rytov_variance(
    profile: Cn2Profile,
    wavelength_m: float,
    *,
    zenith_rad: float = 0.0,
    h_ground: float = 0.0,
    h_top: float | None = None,
    wave: str = "plane",
    method: str = "quad",
    n_nodes: int = 2001,
    warn_strong: bool = True,
) -> float:
    r"""Rytov variance sigma_R^2 (dimensionless) for a slant path.

    **Weighting integral**, with :math:`u = h - h_0` and :math:`L = H - h_0`:

    .. math::

        \sigma_R^2(\text{plane}) &= 2.25\,k^{7/6}\sec^{11/6}\!\zeta
            \int_{h_0}^{H} C_n^2(h)\,u^{5/6}\,dh \\
        \sigma_R^2(\text{spherical}) &= 2.25\,k^{7/6}\sec^{11/6}\!\zeta
            \int_{h_0}^{H} C_n^2(h)\,u^{5/6}\left(1-\frac{u}{L}\right)^{5/6} dh

    The plane-wave weight ``u^(5/6)`` measures distance *from the receiver*, so
    high layers dominate a downlink - the standard scintillation result.  The
    spherical weight is symmetric under ``u -> L-u``, so the spherical result is
    the same for an uplink and a downlink through the same profile
    (reciprocity); no direction argument is therefore needed.

    **Constant** 2.25 from Andrews & Phillips (2005) for slant paths.
    Consistency with the textbook horizontal-path forms is exact arithmetic,
    not a separate claim:

    * plane, homogeneous path of length L: ``2.25 * (6/11) = 1.227``, i.e.
      ``sigma_R^2 = 1.23 Cn^2 k^(7/6) L^(11/6)``;
    * spherical: ``2.25 * B(11/6, 11/6) = 0.4966``, i.e.
      ``sigma_R^2 = 0.50 Cn^2 k^(7/6) L^(11/6)``, and the classic ratio
      spherical/plane = 0.4046 ~ 0.4.

    Both identities are evaluated numerically in ``validation/``.

    **Units** dimensionless.  ``k^(7/6)`` has units m^(-7/6); the plane-wave
    integral has units m^(-2/3) m^(5/6) m = m^(7/6).

    **Zenith dependence** ``u -> u sec(zeta)`` in the moment arm gives
    ``sec^(5/6)`` and the path element gives ``sec``: total ``sec^(11/6)``, i.e.
    ``sigma_R^2 ~ sec(zeta)^(11/6)``.

    **Wavelength dependence** ``sigma_R^2 ~ k^(7/6) ~ lambda^(-7/6)``.

    **Validity** first-order Rytov (weak fluctuation) theory:
    ``sigma_R^2 < 1``.  Beyond that the true irradiance variance saturates and
    then falls, and this expression over-predicts without bound; a
    ``UserWarning`` is raised (suppress with ``warn_strong=False`` if the value
    is wanted purely as a regime indicator).  Also assumes a point receiver
    (no aperture averaging), zero inner scale, infinite outer scale, and an
    unbounded (untruncated) beam.

    Returns
    -------
    float
        Rytov variance, dimensionless.
    """
    lam = check_wavelength(wavelength_m)
    zen = check_zenith(zenith_rad)
    kind = check_choice("wave", wave, _WAVE_KINDS)
    k = _wavenumber(lam)
    h0 = float(h_ground)

    if kind == "plane":
        mu = turbulence_moment(
            profile, 5.0 / 6.0, h_ground=h_ground, h_top=h_top, method=method, n_nodes=n_nodes
        )
    else:
        top = profile.h_max if h_top is None else float(h_top)
        length = top - h0

        def wgt(h: float) -> float:
            u = max(h - h0, 0.0)
            return u ** (5.0 / 6.0) * max(1.0 - u / length, 0.0) ** (5.0 / 6.0)

        mu = weighted_integral(
            profile, wgt, h_ground=h_ground, h_top=h_top, method=method, n_nodes=n_nodes
        )

    sigma2 = float(C_RYTOV * k ** (7.0 / 6.0) * _secant(zen) ** (11.0 / 6.0) * mu)
    if warn_strong and sigma2 >= WEAK_FLUCTUATION_LIMIT:
        warnings.warn(
            f"Rytov variance {sigma2:.3g} >= {WEAK_FLUCTUATION_LIMIT:g}: first-order "
            "Rytov theory is outside its validity range here. The true irradiance "
            "variance saturates and then decreases in this regime; the returned value "
            "is an extrapolation and should be used only as a regime indicator.",
            UserWarning,
            stacklevel=2,
        )
    return sigma2


def scintillation_index(
    profile: Cn2Profile,
    wavelength_m: float,
    *,
    zenith_rad: float = 0.0,
    h_ground: float = 0.0,
    h_top: float | None = None,
    wave: str = "plane",
    method: str = "quad",
    n_nodes: int = 2001,
    warn_strong: bool = True,
) -> float:
    r"""Scintillation index sigma_I^2 in the **weak-fluctuation** regime.

    **Definition and weighting integral**

    .. math::

        \sigma_I^2 = \frac{\langle I^2\rangle - \langle I\rangle^2}
                          {\langle I\rangle^2} \;\simeq\; \sigma_R^2
        \qquad (\sigma_R^2 < 1)

    i.e. for a point receiver in the weak regime the normalised irradiance
    variance equals the Rytov variance evaluated by :func:`rytov_variance`, and
    therefore carries the same weighting integral, the same ``sec(zeta)^(11/6)``
    and the same ``lambda^(-7/6)``.  Source: Andrews & Phillips (2005).

    **Units** dimensionless.

    **Validity - read this before using the number**

    * Weak fluctuations only, ``sigma_R^2 < 1``.  In the strong regime the true
      index rises to a peak above unity and then *decreases* towards 1 (the
      saturation of scintillation); this function does not model that and
      warns instead.
    * Point receiver: no aperture averaging.  A real receiver of diameter D
      reduces the index substantially once D exceeds the Fresnel scale
      sqrt(L/k); the aperture-averaging factor is not implemented here.
    * Zero inner scale and infinite outer scale; unbounded plane or spherical
      wave, no beam truncation, no tracked-beam (jitter) contribution.

    Returns
    -------
    float
        Scintillation index (dimensionless).
    """
    return rytov_variance(
        profile,
        wavelength_m,
        zenith_rad=zenith_rad,
        h_ground=h_ground,
        h_top=h_top,
        wave=wave,
        method=method,
        n_nodes=n_nodes,
        warn_strong=warn_strong,
    )


def coherence_length_to_seeing(r0_m: float, wavelength_m: float) -> float:
    r"""Seeing-disc FWHM (rad) from r0: ``epsilon = 0.98 lambda / r0``.

    The 0.98 factor is the standard long-exposure Kolmogorov result quoted by
    Roddier (1981) and Hardy (1998) (some authors write 0.976); it assumes an
    infinite outer scale and a long exposure.  Included because r0 is almost
    always reported as seeing in the astronomical literature, which is how the
    validation ranges are quoted.

    Units: metres in, radians out (multiply by 206264.8 for arcsec).
    """
    r0 = float(r0_m)
    if r0 <= 0.0:
        raise ValueError(f"r0_m must be > 0, got {r0!r}")
    lam = check_wavelength(wavelength_m)
    return 0.98 * lam / r0


@dataclass(frozen=True)
class TurbulenceSummary:
    """All metrics for one (profile, wavelength, zenith) case.

    Units: ``wavelength_m`` m, ``zenith_deg`` deg, ``r0_m`` m,
    ``theta0_urad`` microradians, ``f_greenwood_hz`` Hz, Rytov variances and
    the scintillation index dimensionless, ``seeing_arcsec`` arcseconds.
    """

    profile: str
    wavelength_m: float
    zenith_deg: float
    r0_m: float
    r0_spherical_m: float
    theta0_urad: float
    f_greenwood_hz: float | None
    rytov_plane: float
    rytov_spherical: float
    scintillation_index_plane: float
    effective_height_m: float
    seeing_arcsec: float
    weak_fluctuation_valid: bool

    def as_dict(self) -> dict[str, object]:
        """Return the summary as a plain dictionary (for JSON output)."""
        return asdict(self)


def summarize(
    profile: Cn2Profile,
    wavelength_m: float,
    *,
    zenith_rad: float = 0.0,
    wind: WindProfile | None = None,
    h_ground: float = 0.0,
    h_top: float | None = None,
    method: str = "quad",
    n_nodes: int = 2001,
) -> TurbulenceSummary:
    """Evaluate every metric for one case and return a :class:`TurbulenceSummary`.

    ``f_greenwood_hz`` is ``None`` when no wind profile is supplied, because the
    Greenwood frequency is not computable from Cn^2 alone.
    """
    common = {
        "zenith_rad": zenith_rad,
        "h_ground": h_ground,
        "h_top": h_top,
        "method": method,
        "n_nodes": n_nodes,
    }
    r0 = fried_parameter(profile, wavelength_m, wave="plane", **common)
    r0_sph = fried_parameter(profile, wavelength_m, wave="spherical", **common)
    theta0 = isoplanatic_angle(profile, wavelength_m, **common)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        rytov_p = rytov_variance(profile, wavelength_m, wave="plane", **common)
        rytov_s = rytov_variance(profile, wavelength_m, wave="spherical", **common)
    fg = (
        None
        if wind is None
        else greenwood_frequency(profile, wind, wavelength_m, **common)
    )
    hbar = effective_turbulence_height(
        profile, h_ground=h_ground, h_top=h_top, method=method, n_nodes=n_nodes
    )
    return TurbulenceSummary(
        profile=profile.name,
        wavelength_m=float(wavelength_m),
        zenith_deg=math.degrees(float(zenith_rad)),
        r0_m=r0,
        r0_spherical_m=r0_sph,
        theta0_urad=theta0 * 1e6,
        f_greenwood_hz=fg,
        rytov_plane=rytov_p,
        rytov_spherical=rytov_s,
        scintillation_index_plane=rytov_p,
        effective_height_m=hbar,
        seeing_arcsec=math.degrees(coherence_length_to_seeing(r0, wavelength_m)) * 3600.0,
        weak_fluctuation_valid=bool(rytov_p < WEAK_FLUCTUATION_LIMIT),
    )
