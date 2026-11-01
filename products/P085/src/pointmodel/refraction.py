"""Atmospheric refraction: an empirical formula and an independent ray trace.

Two calculations of the same quantity live in this module, and they are
deliberately built from different mathematics so that one can check the other.

**The empirical formula.** :func:`bennett_refraction` is Bennett's (1982)
closed-form fit, as published by Meeus (1998) "Astronomical Algorithms", 2nd
edition, chapter 16 "Atmospheric Refraction", equation (16.4):

    R = 1 / tan( h + 7.31 / (h + 4.4) )                                    (1)

with ``h`` the **true** (geometric, unrefracted) altitude in degrees and ``R``
the refraction in **minutes of arc**. Meeus states the accuracy of this fit as
about 0.07 arcminutes (4.2 arcsec). Equation (1) is stated for standard
conditions of 1010 millibar and 10 degrees Celsius; Meeus gives the correction
for other conditions in the same chapter as the multiplicative factor

    f = (P / 1010) * (283 / (273 + T))                                     (2)

with ``P`` in millibar (equivalently hectopascal) and ``T`` in degrees Celsius.
Equation (2) is a density ratio and nothing more: it carries no humidity term
and no wavelength term.

Validity range. Equation (1) is an altitude-domain fit and is used here only
for ``h >= 2 degrees``; :func:`bennett_refraction` raises below that. Near the
horizon the real refraction depends on the actual temperature profile far more
strongly than any two-parameter fit can represent, and the formula's residual
there is not bounded by the 0.07 arcminute figure above.

**The independent calculation.** :func:`raytrace_refraction` integrates the
refraction integral through a spherically symmetric atmosphere. It shares no
mathematics with equation (1): no fitted constant, no arctangent form, no
altitude-domain polynomial. For a spherically symmetric refractive index
``n(r)`` Snell's invariant holds along a ray,

    n(r) * r * sin(zeta(r)) = constant,                                    (3)

with ``zeta`` the angle between the ray and the local vertical, and the total
bending between the observer and space is

    R = - integral_{r0}^{r_top} tan(zeta(r)) * (1/n) * (dn/dr) dr.         (4)

Equation (4) is the standard refraction integral; see Smart (1977) "Textbook
on Spherical Astronomy", 6th edition, chapter 3, and Auer & Standish (2000),
Astronomical Journal 119, 2472, "Astronomical refraction: computational
method for all zenith angles", which derives the same integral and discusses
the substitutions needed at large zenith angles. The flat-atmosphere limit of
(4) is a check on the sign and the factor: holding ``zeta`` constant at the
zenith distance ``z`` gives ``R = (n0 - 1) * tan(z)``, the classical
plane-parallel result, and :func:`raytrace_refraction` is verified against
that limit in the test suite.

The constant in (3) is evaluated at the observer, so the argument of
:func:`raytrace_refraction` is the **apparent** (refracted) altitude at the
observer, whereas equation (1) takes the **true** altitude. The two differ by
the refraction itself, which is why :func:`raytrace_refraction` accepts
``altitude_is_true=True`` and solves for the apparent altitude by fixed-point
iteration before integrating. Comparisons in ``validation/`` use that mode so
that both calculations take the same input.

**Refractivity.** The refractive index comes from the refractivity

    N = (n - 1) * 1e6 = K * P / T                                          (5)

with ``P`` in hectopascal, ``T`` in kelvin and ``K`` in K/hPa. Two values of
``K`` are provided and the choice is an argument, never a hidden default:

- ``K_RADIO_DRY = 77.6`` K/hPa, the dry-air term of Smith & Weintraub (1953),
  "The constants in the equation for atmospheric refractive index at radio
  frequencies", Proceedings of the IRE 41, 1035. This is the right constant
  for a radio antenna and it omits the water-vapour term, which at radio
  frequencies is large and is not modelled here.
- ``K_OPTICAL_550NM = 78.95`` K/hPa. This is **not** quoted from a source as a
  constant: it is back-computed in this module from the standard-air optical
  refractivity at 550 nm, ``(n - 1) = 2.7765e-4`` at 1013.25 hPa and 288.15 K,
  which is the value Edlen's (1966) dispersion formula gives for dry air at
  that wavelength and which is reproduced in the Ciddor (1996) revision,
  Applied Optics 35, 1566. The arithmetic is
  ``2.7765e-4 * 288.15 / 1013.25 * 1e6``; it is done in the source below so a
  reader can check it. It is 1.74 per cent above the radio dry-air constant,
  and that difference is a measured part of the disagreement reported in
  ``validation/validate_refraction_output.txt``.

The atmosphere between the observer and space is the ICAO/US Standard
Atmosphere 1976 troposphere and lower stratosphere: a constant lapse rate of
6.5 K/km to 11 km geopotential height above the observer, isothermal above,
with pressure from the corresponding hydrostatic solution. The ray trace is
therefore *not* isothermal-exponential, and no part of it is fitted to
equation (1).

**What the pointing model actually uses.** The refraction column of the
pointing-model design matrix is the *normalised shape*

    rho_hat(E) = R(E) / R(45 degrees),                                     (6)

because the fitted coefficient ``RF`` carries the scale. Equation (6) is
dimensionless and ``RF`` is in arcsec, so ``RF`` is the refraction at 45
degrees elevation that the fit attributes to the atmosphere. This matters for
the cross-check: the comparison that bears on the product is the comparison of
*shapes*, which is independent of ``K`` entirely, and it is reported separately
from the absolute comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad

__all__ = [
    "BENNETT_MAX_ALTITUDE_DEG",
    "BENNETT_MIN_ALTITUDE_DEG",
    "K_OPTICAL_550NM",
    "K_RADIO_DRY",
    "RefractionComparison",
    "STANDARD_PRESSURE_HPA",
    "STANDARD_TEMPERATURE_C",
    "bennett_refraction",
    "compare_refraction",
    "plane_parallel_refraction",
    "raytrace_refraction",
    "refraction_shape",
    "refractivity",
]

#: Arcseconds per radian.
ARCSEC_PER_RAD = 180.0 * 3600.0 / np.pi

#: Reference conditions of equation (1), from Meeus (1998) chapter 16.
STANDARD_PRESSURE_HPA = 1010.0
STANDARD_TEMPERATURE_C = 10.0

#: Elevation at which the refraction shape of equation (6) is normalised, deg.
SHAPE_REFERENCE_EL_DEG = 45.0

#: Lowest true altitude at which equation (1) is evaluated, degrees.
BENNETT_MIN_ALTITUDE_DEG = 2.0

#: Highest true altitude at which equation (1) is evaluated, degrees.
#:
#: Equation (1) returns a NEGATIVE refraction near the zenith. The argument of
#: the tangent is ``h + 7.31 / (h + 4.4)``, which reaches 90 degrees when
#: ``h^2 - 85.6 h - 388.69 = 0``, that is at
#: ``h = (85.6 + sqrt(85.6^2 + 4 * 388.69)) / 2 = 89.92250...`` degrees, above
#: which the tangent is negative and so is the fitted refraction. That is an
#: artefact of a two-parameter fit and not a physical statement, so the
#: formula is refused above 89.9 degrees rather than being allowed to put a
#: sign error into a design matrix. The quadratic above is solved in the test
#: suite and the crossing is checked.
BENNETT_MAX_ALTITUDE_DEG = 89.9

#: Dry-air radio refractivity constant, K/hPa. Smith & Weintraub (1953).
K_RADIO_DRY = 77.6

#: Optical refractivity constant at 550 nm, K/hPa, back-computed here from the
#: standard-air refractivity 2.7765e-4 at 1013.25 hPa and 288.15 K (Edlen 1966;
#: Ciddor 1996). The arithmetic is written out so that it can be checked.
K_OPTICAL_550NM = 2.7765e-4 * 288.15 / 1013.25 * 1e6

#: Mean Earth radius, metres. IUGG mean radius R1 of the GRS80 ellipsoid.
EARTH_RADIUS_M = 6_371_000.0

#: US Standard Atmosphere 1976 troposphere: lapse rate K/m, gravity m/s^2,
#: specific gas constant of dry air J/(kg K), tropopause height m.
LAPSE_RATE_K_PER_M = 0.0065
GRAVITY_M_PER_S2 = 9.80665
DRY_AIR_R_J_PER_KG_K = 287.0528
TROPOPAUSE_M = 11_000.0

#: Upper limit of the refraction integral, metres above the observer. At 120 km
#: the refractivity is below 1e-11 of its surface value, so the truncated tail
#: is far below the quadrature tolerance.
INTEGRATION_TOP_M = 120_000.0


def _atmosphere(height_m: np.ndarray | float, p0_hpa: float, t0_k: float) -> tuple:
    """Pressure (hPa) and temperature (K) at ``height_m`` above the observer.

    US Standard Atmosphere 1976 form: constant lapse rate to the tropopause,
    isothermal above, hydrostatic pressure throughout. ``height_m`` is a
    geometric height above the observer and is treated as a geopotential
    height, an approximation worth about 0.3 per cent of the pressure at 11 km
    and far below the disagreement this module reports.
    """
    height = np.asarray(height_m, dtype=float)
    exponent = GRAVITY_M_PER_S2 / (DRY_AIR_R_J_PER_KG_K * LAPSE_RATE_K_PER_M)
    capped = np.minimum(height, TROPOPAUSE_M)
    t_trop = t0_k - LAPSE_RATE_K_PER_M * capped
    pressure = p0_hpa * (t_trop / t0_k) ** exponent
    above = height > TROPOPAUSE_M
    if np.any(above):
        t_11 = t0_k - LAPSE_RATE_K_PER_M * TROPOPAUSE_M
        p_11 = p0_hpa * (t_11 / t0_k) ** exponent
        isothermal = p_11 * np.exp(
            -GRAVITY_M_PER_S2 * (height - TROPOPAUSE_M) / (DRY_AIR_R_J_PER_KG_K * t_11)
        )
        pressure = np.where(above, isothermal, pressure)
    temperature = np.where(above, t0_k - LAPSE_RATE_K_PER_M * TROPOPAUSE_M, t_trop)
    return pressure, temperature


def refractivity(
    height_m: np.ndarray | float,
    pressure_hpa: float = STANDARD_PRESSURE_HPA,
    temperature_c: float = STANDARD_TEMPERATURE_C,
    refractivity_constant: float = K_RADIO_DRY,
) -> np.ndarray:
    """``n - 1`` at ``height_m`` above the observer, dimensionless.

    Equation (5) with the standard-atmosphere profile. ``height_m`` is metres
    above the observer, ``pressure_hpa`` and ``temperature_c`` are the
    conditions AT the observer, and ``refractivity_constant`` is in K/hPa.
    """
    if pressure_hpa <= 0.0:
        raise ValueError(f"pressure must be positive, got {pressure_hpa} hPa")
    t0_k = 273.15 + temperature_c
    if t0_k <= 0.0:
        raise ValueError(f"temperature must be above absolute zero, got {temperature_c} C")
    pressure, temperature = _atmosphere(height_m, pressure_hpa, t0_k)
    return refractivity_constant * 1e-6 * pressure / temperature


def bennett_refraction(
    elevation_deg: np.ndarray | float,
    pressure_hpa: float = STANDARD_PRESSURE_HPA,
    temperature_c: float = STANDARD_TEMPERATURE_C,
) -> np.ndarray:
    """Refraction in **arcseconds** from equations (1) and (2).

    Parameters
    ----------
    elevation_deg
        True (unrefracted) altitude in degrees. Must be at least
        ``BENNETT_MIN_ALTITUDE_DEG`` = 2 degrees; equation (1) is a fit and its
        quoted 0.07 arcminute accuracy does not extend to the horizon.
    pressure_hpa
        Pressure at the observer, hectopascal.
    temperature_c
        Temperature at the observer, degrees Celsius.

    Returns
    -------
    numpy.ndarray
        Refraction in arcseconds, positive, decreasing with elevation.
    """
    elevation = np.asarray(elevation_deg, dtype=float)
    if np.any(~np.isfinite(elevation)):
        raise ValueError("elevation_deg contains a non-finite value")
    if np.any(elevation < BENNETT_MIN_ALTITUDE_DEG):
        raise ValueError(
            f"bennett_refraction is used only above {BENNETT_MIN_ALTITUDE_DEG} deg "
            f"elevation; got a minimum of {float(np.min(elevation)):.4f} deg"
        )
    if np.any(elevation > BENNETT_MAX_ALTITUDE_DEG):
        raise ValueError(
            f"bennett_refraction is refused above {BENNETT_MAX_ALTITUDE_DEG} deg "
            f"elevation, where equation (1) returns a negative refraction; got a "
            f"maximum of {float(np.max(elevation)):.4f} deg"
        )
    arcmin = 1.0 / np.tan(np.deg2rad(elevation + 7.31 / (elevation + 4.4)))
    factor = (pressure_hpa / STANDARD_PRESSURE_HPA) * (283.0 / (273.0 + temperature_c))
    return 60.0 * arcmin * factor


def plane_parallel_refraction(
    elevation_deg: np.ndarray | float,
    pressure_hpa: float = STANDARD_PRESSURE_HPA,
    temperature_c: float = STANDARD_TEMPERATURE_C,
    refractivity_constant: float = K_RADIO_DRY,
) -> np.ndarray:
    """``(n0 - 1) * tan(z)`` in arcseconds: the flat-atmosphere limit of (4).

    This is not a product-quality refraction model. It exists because it is the
    analytic limit of equation (4) as the Earth radius goes to infinity, and
    the test suite uses it to pin the sign and the leading factor of the ray
    trace near the zenith, where curvature is negligible.
    """
    elevation = np.asarray(elevation_deg, dtype=float)
    n0_minus_1 = refractivity(0.0, pressure_hpa, temperature_c, refractivity_constant)
    return n0_minus_1 * np.tan(np.deg2rad(90.0 - elevation)) * ARCSEC_PER_RAD


def _raytrace_apparent(
    apparent_el_deg: float,
    pressure_hpa: float,
    temperature_c: float,
    refractivity_constant: float,
    top_m: float,
) -> tuple[float, float]:
    """Evaluate equation (4) for one apparent altitude. Returns (arcsec, quad error)."""
    n0 = 1.0 + float(
        refractivity(0.0, pressure_hpa, temperature_c, refractivity_constant)
    )
    zenith = np.deg2rad(90.0 - apparent_el_deg)
    invariant = n0 * EARTH_RADIUS_M * np.sin(zenith)
    step = 1.0  # metres, central-difference step for dn/dr

    def integrand(height: float) -> float:
        index = 1.0 + float(
            refractivity(height, pressure_hpa, temperature_c, refractivity_constant)
        )
        radius = EARTH_RADIUS_M + height
        sine = min(invariant / (index * radius), 1.0)
        zeta = np.arcsin(sine)
        lower = float(
            refractivity(
                max(height - step, 0.0), pressure_hpa, temperature_c, refractivity_constant
            )
        )
        upper = float(
            refractivity(height + step, pressure_hpa, temperature_c, refractivity_constant)
        )
        span = (height + step) - max(height - step, 0.0)
        gradient = (upper - lower) / span
        return float(np.tan(zeta) * gradient / index)

    value, error = quad(integrand, 0.0, top_m, limit=400, epsabs=1e-14, epsrel=1e-11)
    return -value * ARCSEC_PER_RAD, error * ARCSEC_PER_RAD


def raytrace_refraction(
    elevation_deg: np.ndarray | float,
    pressure_hpa: float = STANDARD_PRESSURE_HPA,
    temperature_c: float = STANDARD_TEMPERATURE_C,
    refractivity_constant: float = K_RADIO_DRY,
    altitude_is_true: bool = True,
    top_m: float = INTEGRATION_TOP_M,
) -> np.ndarray:
    """Refraction in **arcseconds** by numerical integration of equation (4).

    This is the independent calculation. It is written in this repository and
    is not a wrapper around any external refraction routine; no refraction
    package is installed in the measured environment.

    Parameters
    ----------
    elevation_deg
        Altitude in degrees. Interpreted as the **true** altitude when
        ``altitude_is_true`` (the default), in which case the apparent altitude
        is found by fixed-point iteration ``h_app = h_true - R(h_app)`` before
        the integral is evaluated; interpreted as the apparent altitude
        otherwise. The iteration is capped at 40 steps and converges to
        1e-9 arcsec in fewer than 10 above 5 degrees.
    pressure_hpa, temperature_c
        Conditions at the observer.
    refractivity_constant
        ``K`` of equation (5), K/hPa. :data:`K_RADIO_DRY` or
        :data:`K_OPTICAL_550NM`, or any other declared value.
    altitude_is_true
        See ``elevation_deg``.
    top_m
        Upper limit of the integral, metres above the observer.

    Returns
    -------
    numpy.ndarray
        Refraction in arcseconds, same shape as ``elevation_deg``.

    Notes
    -----
    Validity. The integrand is well behaved while ``n(r) * r`` increases
    monotonically, which it does for the standard atmosphere, so there is no
    turning point and no singularity for any altitude above 0. Accuracy is
    limited by the atmospheric profile, not by the quadrature: the reported
    absolute quadrature error is below 1e-8 arcsec everywhere measured. Below
    about 5 degrees the real atmosphere's temperature profile dominates and
    neither this calculation nor equation (1) should be trusted.
    """
    elevation = np.atleast_1d(np.asarray(elevation_deg, dtype=float))
    if np.any(~np.isfinite(elevation)):
        raise ValueError("elevation_deg contains a non-finite value")
    if np.any(elevation <= 0.0):
        raise ValueError(
            f"raytrace_refraction needs a positive altitude; got a minimum of "
            f"{float(np.min(elevation)):.4f} deg"
        )
    if np.any(elevation > 90.0):
        raise ValueError(f"elevation above 90 deg: {float(np.max(elevation)):.4f}")
    out = np.empty(elevation.shape, dtype=float)
    for index, value in enumerate(elevation.ravel()):
        if altitude_is_true:
            apparent = float(value)
            for _ in range(40):
                refraction, _err = _raytrace_apparent(
                    apparent, pressure_hpa, temperature_c, refractivity_constant, top_m
                )
                updated = float(value) - refraction / 3600.0
                if abs(updated - apparent) * 3600.0 < 1e-9:
                    apparent = updated
                    break
                apparent = updated
            refraction, _err = _raytrace_apparent(
                apparent, pressure_hpa, temperature_c, refractivity_constant, top_m
            )
        else:
            refraction, _err = _raytrace_apparent(
                float(value), pressure_hpa, temperature_c, refractivity_constant, top_m
            )
        out.ravel()[index] = refraction
    return out.reshape(np.shape(elevation_deg)) if np.ndim(elevation_deg) else out[0]


def refraction_shape(
    elevation_rad: np.ndarray | float,
    pressure_hpa: float = STANDARD_PRESSURE_HPA,
    temperature_c: float = STANDARD_TEMPERATURE_C,
) -> np.ndarray:
    """Equation (6): dimensionless refraction shape, 1.0 at 45 degrees elevation.

    This is the refraction column of the pointing-model design matrix. Its
    argument is in **radians** because every other angle in
    :mod:`pointmodel.terms` is, and it is normalised at 45 degrees so that the
    fitted coefficient ``RF`` is the refraction at 45 degrees in arcseconds.

    The multiplicative conditions factor of equation (2) cancels in the ratio,
    so the shape is independent of pressure and temperature; the arguments are
    accepted so that a caller cannot be surprised by them being ignored
    silently, and they are asserted to be in range.
    """
    elevation = np.asarray(elevation_rad, dtype=float)
    degrees = np.rad2deg(elevation)
    reference = bennett_refraction(
        SHAPE_REFERENCE_EL_DEG, pressure_hpa, temperature_c
    )
    return bennett_refraction(degrees, pressure_hpa, temperature_c) / reference


@dataclass(frozen=True)
class RefractionComparison:
    """Result of :func:`compare_refraction`. All angles in arcseconds.

    Attributes
    ----------
    elevation_deg
        The true altitudes compared.
    bennett_arcsec
        Equation (1) at those altitudes.
    raytrace_arcsec
        Equation (4) at those altitudes, with ``refractivity_constant``.
    shape_bennett, shape_raytrace
        Equation (6) from each calculation, normalised at 45 degrees.
    max_absolute_arcsec
        Largest ``|bennett - raytrace|`` over the grid.
    max_shape_relative
        Largest ``|shape_bennett - shape_raytrace| / shape_raytrace``.
    mean_ratio
        Mean of ``bennett / raytrace``, which isolates the scale difference
        that the fitted coefficient ``RF`` would absorb.
    refractivity_constant
        The ``K`` of equation (5) used for the ray trace, K/hPa.
    """

    elevation_deg: np.ndarray
    bennett_arcsec: np.ndarray
    raytrace_arcsec: np.ndarray
    shape_bennett: np.ndarray
    shape_raytrace: np.ndarray
    max_absolute_arcsec: float
    max_shape_relative: float
    mean_ratio: float
    refractivity_constant: float


def compare_refraction(
    elevation_deg: np.ndarray,
    pressure_hpa: float = STANDARD_PRESSURE_HPA,
    temperature_c: float = STANDARD_TEMPERATURE_C,
    refractivity_constant: float = K_RADIO_DRY,
) -> RefractionComparison:
    """Compare equations (1) and (4) on ``elevation_deg`` (true altitudes, deg).

    The two numbers that matter are different. ``max_shape_relative`` is the
    disagreement in the quantity the pointing model actually uses, equation
    (6), and it is independent of ``refractivity_constant``.
    ``max_absolute_arcsec`` and ``mean_ratio`` describe the scale, which the
    fitted ``RF`` coefficient absorbs and which depends on
    ``refractivity_constant`` directly.
    """
    elevation = np.atleast_1d(np.asarray(elevation_deg, dtype=float))
    bennett = bennett_refraction(elevation, pressure_hpa, temperature_c)
    ray = raytrace_refraction(
        elevation,
        pressure_hpa,
        temperature_c,
        refractivity_constant,
        altitude_is_true=True,
    )
    ray = np.atleast_1d(ray)
    bennett_ref = float(
        bennett_refraction(SHAPE_REFERENCE_EL_DEG, pressure_hpa, temperature_c)
    )
    ray_ref = float(
        np.atleast_1d(
            raytrace_refraction(
                SHAPE_REFERENCE_EL_DEG,
                pressure_hpa,
                temperature_c,
                refractivity_constant,
                altitude_is_true=True,
            )
        )[0]
    )
    shape_bennett = bennett / bennett_ref
    shape_ray = ray / ray_ref
    return RefractionComparison(
        elevation_deg=elevation,
        bennett_arcsec=bennett,
        raytrace_arcsec=ray,
        shape_bennett=shape_bennett,
        shape_raytrace=shape_ray,
        max_absolute_arcsec=float(np.max(np.abs(bennett - ray))),
        max_shape_relative=float(np.max(np.abs(shape_bennett - shape_ray) / shape_ray)),
        mean_ratio=float(np.mean(bennett / ray)),
        refractivity_constant=float(refractivity_constant),
    )
