"""The pointing-model term set and its design matrix.

The term set
------------
The nine terms below are the classical alt-azimuth mount model. The *list* is
standard and citable:

- Ulich, B. L. (1981), "Millimeter-wave radio telescopes: gain and pointing
  characteristics", International Journal of Infrared and Millimeter Waves
  2(2), 293-310. Section on pointing gives the index, collimation, axis-tilt
  and non-perpendicularity terms for a large alt-azimuth antenna.
- Wallace, P. T. (2002), "A rigorous algorithm for telescope pointing",
  Proceedings of SPIE 4848, 125-136. This is the published description of the
  geometric model that the TPOINT term names ``IA``, ``IE``, ``CA``, ``AN``,
  ``AW``, ``NPAE``, ``TF`` come from, and those names are used here because
  they are what a pointing engineer will recognise.
- Mangum, J. G. & Wallace, P. (2015), "Atmospheric refractive electromagnetic
  wave bending and propagation delay", Publications of the Astronomical
  Society of the Pacific 127, 74 — for the refraction term's place in a
  pointing model and for the statement that the refraction coefficient is
  fitted rather than assumed.

**The functional forms and the signs below are derived in this module, not
copied.** Sign conventions differ between implementations and a sign taken on
trust is a defect waiting to happen, so each column is derived from the
rotation geometry here and is verified against an exact composition of
rotation matrices in :mod:`pointmodel.geometry`, which shares no algebra with
this module. The verification is a first-order-truncation check whose
remainder is bounded in :func:`pointmodel.geometry.linearisation_bound`, and
it is run over the whole term set in ``validation/validate_linearisation.py``.
A reader comparing against another implementation should expect to find signs
flipped on individual coefficients; that is a convention difference, not a
different model.

Coordinates, units, and what a residual is
------------------------------------------
Azimuth ``A`` is measured from north toward east, elevation ``E`` from the
horizon, both in radians in every function here. The unit sky vector in a local
north-east-up frame is

    v(A, E) = [cos(E) cos(A), cos(E) sin(A), sin(E)].                      (7)

An observation contributes two residuals. ``dE`` is the elevation residual in
arcseconds. ``dX`` is the **cross-elevation** residual in arcseconds, which is
the azimuth residual projected on the sky,

    dX = dA * cos(E).                                                      (8)

Cross-elevation is used rather than raw azimuth because it is the quantity with
a constant metric on the sky: ``(dX, dE)`` is an orthonormal tangent pair at
every point, which is what makes an unweighted least squares over both
components meaningful and what makes the condition number of the design matrix
a statement about geometry rather than about a coordinate singularity at the
zenith.

The nine terms
--------------
Every coefficient is in **arcseconds**. The design matrix is dimensionless.

=======  =======================================  =======================  =====================
name     physical meaning                         contribution to ``dX``   contribution to ``dE``
=======  =======================================  =======================  =====================
``IA``   azimuth encoder index (zero point)       ``cos(E)``               0
``IE``   elevation encoder index (zero point)     0                        1
``CA``   collimation: beam-to-elevation-axis      1                        0
         angle, in the cross-elevation sense
``AN``   azimuth-axis tilt toward north           ``sin(A) sin(E)``        ``cos(A)``
``AW``   azimuth-axis tilt toward east            ``-cos(A) sin(E)``       ``sin(A)``
``NPAE`` non-perpendicularity of the azimuth      ``sin(E)``               0
         and elevation axes
``TF``   gravitational flexure, cosine component   0                        ``cos(E)``
``TS``   gravitational flexure, sine component     0                        ``sin(E)``
``RF``   refraction at 45 deg elevation           0                        ``rho_hat(E)``
=======  =======================================  =======================  =====================

Derivations, one line each.

``IA``: the azimuth encoder reads ``A`` while the mount has turned
``A + IA``, so ``dA = IA`` and ``dX = IA cos(E)`` by (8).

``CA``: a fixed angular offset of the beam from the plane perpendicular to the
elevation axis appears on the sky as a constant cross-elevation offset, so
``dX = CA`` directly and ``dA = CA / cos(E)``.

``NPAE``: if the elevation axis is off perpendicular to the azimuth axis by
``NPAE``, the beam traces a great circle inclined by that angle, giving
``dA = NPAE tan(E)`` and hence ``dX = NPAE sin(E)``.

``AN`` and ``AW``: let the azimuth axis lean by ``theta`` toward azimuth
``phi``, and write ``AN = theta cos(phi)``, ``AW = theta sin(phi)``. The
mount's horizon is then tilted, and a first-order rotation of the sky vector
about the axis ``theta * [sin(phi), -cos(phi), 0]`` gives

    dE = theta cos(A - phi) = AN cos(A) + AW sin(A),                       (9)
    dA = theta tan(E) sin(A - phi) = (AN sin(A) - AW cos(A)) tan(E),      (10)

so ``dX = (AN sin(A) - AW cos(A)) sin(E)`` by (8). The pair (9)-(10) is
**not** independent: the same two numbers set both components, which is why a
tilt is identifiable from elevation residuals alone over a full azimuth turn
and why its ``dX`` signature vanishes at the horizon. The relative sign between
(9) and (10) is the part most often wrong when a convention is copied; it is
fixed here by the derivation and checked numerically.

``TF`` and ``TS``: gravitational sag of the structure is a function of
elevation only, and the first two terms of its Fourier expansion on the
elevation interval are ``cos(E)`` and ``sin(E)``. ``TF`` is the classical tube
flexure term, largest at the horizon. ``TS`` is included because the mission
specification names a sine component, and because a one-term flexure model and
a two-term one are not distinguishable from each other on a narrow elevation
band — which is itself one of the degeneracies this product exists to report.
Neither is a physical model of the structure: they are the two leading terms of
an expansion, valid only over the elevation range actually fitted, and they must
not be extrapolated outside it.

``RF``: the atmosphere raises the apparent elevation of a source, so a model
that omits refraction leaves a residual with the refraction's elevation
dependence. The column is the normalised shape of equation (6) in
:mod:`pointmodel.refraction`, so ``RF`` in arcseconds is the refraction at 45
degrees elevation that the fit attributes to the atmosphere. At standard
conditions the expected value is about 59.7 arcsec.

Validity range
--------------
Every form above is first order in the coefficients. The truncation error is
bounded by :func:`pointmodel.geometry.linearisation_bound` and measured in
``validation/``: at a total misalignment of 100 arcsec it is below 1 arcsec on
the sky, and at 1000 arcsec it is tens of arcseconds, so a mount with
degree-level misalignments is outside the model's range and must be
mechanically corrected before this model is fitted. The elevation range is
limited by two things rather than by the algebra: ``tan(E)`` in (10) diverges
at the zenith, so ``dX`` carries no tilt information there, and the refraction
column rests on a fit that is not trustworthy below about 5 degrees.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .refraction import (
    BENNETT_MAX_ALTITUDE_DEG,
    BENNETT_MIN_ALTITUDE_DEG,
    STANDARD_PRESSURE_HPA,
    STANDARD_TEMPERATURE_C,
    refraction_shape,
)

__all__ = [
    "N_TERMS",
    "TERMS",
    "TERM_INDEX",
    "TermDefinition",
    "coefficients_from_mapping",
    "design_matrix",
    "mapping_from_coefficients",
    "stack_residuals",
    "term_table",
]


@dataclass(frozen=True)
class TermDefinition:
    """One pointing-model term. ``unit`` is the unit of its coefficient."""

    name: str
    unit: str
    meaning: str
    cross_elevation: str
    elevation: str


TERMS: tuple[TermDefinition, ...] = (
    TermDefinition("IA", "arcsec", "azimuth encoder index error", "cos(E)", "0"),
    TermDefinition("IE", "arcsec", "elevation encoder index error", "0", "1"),
    TermDefinition("CA", "arcsec", "collimation error (cross-elevation)", "1", "0"),
    TermDefinition("AN", "arcsec", "azimuth-axis tilt toward north", "sin(A)sin(E)", "cos(A)"),
    TermDefinition("AW", "arcsec", "azimuth-axis tilt toward east", "-cos(A)sin(E)", "sin(A)"),
    TermDefinition("NPAE", "arcsec", "azimuth/elevation non-perpendicularity", "sin(E)", "0"),
    TermDefinition("TF", "arcsec", "gravitational flexure, cosine term", "0", "cos(E)"),
    TermDefinition("TS", "arcsec", "gravitational flexure, sine term", "0", "sin(E)"),
    TermDefinition("RF", "arcsec", "refraction at 45 deg elevation", "0", "rho_hat(E)"),
)

#: Term names in design-matrix column order.
TERM_NAMES: tuple[str, ...] = tuple(term.name for term in TERMS)

#: Number of terms in the declared model.
N_TERMS = len(TERMS)

#: Name to column index.
TERM_INDEX: dict[str, int] = {term.name: index for index, term in enumerate(TERMS)}


def design_matrix(
    azimuth_rad: np.ndarray,
    elevation_rad: np.ndarray,
    pressure_hpa: float = STANDARD_PRESSURE_HPA,
    temperature_c: float = STANDARD_TEMPERATURE_C,
) -> tuple[np.ndarray, np.ndarray]:
    """Cross-elevation and elevation blocks of the design matrix.

    Parameters
    ----------
    azimuth_rad, elevation_rad
        One-dimensional arrays of equal length, radians. Azimuth is from north
        toward east; elevation is from the horizon.
    pressure_hpa, temperature_c
        Site conditions, passed to :func:`pointmodel.refraction.refraction_shape`.
        They cancel in the normalised shape and are accepted so that the
        signature of this function states where they would enter.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        Two ``(n, 9)`` dimensionless matrices: the cross-elevation block and the
        elevation block, in the column order of :data:`TERM_NAMES`. Multiplying
        either by a coefficient vector in arcseconds gives residuals in
        arcseconds.

    Raises
    ------
    ValueError
        If the inputs are not one-dimensional, differ in length, are empty,
        contain a non-finite value, or carry an elevation outside
        ``(-pi/2, pi/2)`` or outside the refraction model's declared range.
    """
    azimuth = np.asarray(azimuth_rad, dtype=float)
    elevation = np.asarray(elevation_rad, dtype=float)
    if azimuth.ndim != 1 or elevation.ndim != 1:
        raise ValueError(
            f"azimuth_rad and elevation_rad must be 1-D, got shapes "
            f"{azimuth.shape} and {elevation.shape}"
        )
    if azimuth.size != elevation.size:
        raise ValueError(
            f"azimuth_rad has {azimuth.size} entries but elevation_rad has {elevation.size}"
        )
    if azimuth.size == 0:
        raise ValueError("design_matrix needs at least one observation, got zero")
    if not np.all(np.isfinite(azimuth)) or not np.all(np.isfinite(elevation)):
        raise ValueError("azimuth_rad or elevation_rad contains a non-finite value")
    low = np.deg2rad(BENNETT_MIN_ALTITUDE_DEG)
    high = np.deg2rad(BENNETT_MAX_ALTITUDE_DEG)
    if np.any(elevation < low) or np.any(elevation > high):
        raise ValueError(
            f"elevation_rad must lie within [{BENNETT_MIN_ALTITUDE_DEG}, "
            f"{BENNETT_MAX_ALTITUDE_DEG}] degrees, the declared validity range of the "
            "refraction column; got a range of "
            f"[{np.rad2deg(float(np.min(elevation))):.6f}, "
            f"{np.rad2deg(float(np.max(elevation))):.6f}] deg"
        )

    sin_a = np.sin(azimuth)
    cos_a = np.cos(azimuth)
    sin_e = np.sin(elevation)
    cos_e = np.cos(elevation)
    one = np.ones_like(azimuth)
    zero = np.zeros_like(azimuth)
    rho = refraction_shape(elevation, pressure_hpa, temperature_c)

    cross = np.stack(
        [cos_e, zero, one, sin_a * sin_e, -cos_a * sin_e, sin_e, zero, zero, zero],
        axis=1,
    )
    elev = np.stack(
        [zero, one, zero, cos_a, sin_a, zero, cos_e, sin_e, rho],
        axis=1,
    )
    return cross, elev


def stack_residuals(cross: np.ndarray, elevation: np.ndarray) -> np.ndarray:
    """Stack a cross-elevation block above an elevation block.

    Every fit in this package works on the stacked ``2n`` system: the first
    ``n`` rows are cross-elevation, the last ``n`` are elevation. The order is
    blocks, not interleaved, and it is fixed everywhere so that a row index can
    be mapped back to an observation by ``index % n``.
    """
    cross = np.asarray(cross, dtype=float)
    elevation = np.asarray(elevation, dtype=float)
    if cross.shape[0] != elevation.shape[0]:
        raise ValueError(
            f"blocks have different row counts: {cross.shape[0]} and {elevation.shape[0]}"
        )
    return np.concatenate([cross, elevation], axis=0)


def coefficients_from_mapping(mapping: dict[str, float]) -> np.ndarray:
    """Coefficient vector in :data:`TERM_NAMES` order from a name-to-value dict.

    Missing terms are zero. An unknown name raises, because a silently ignored
    coefficient name is how a typo becomes a wrong fit.
    """
    unknown = sorted(set(mapping) - set(TERM_NAMES))
    if unknown:
        raise ValueError(
            f"unknown pointing-model term(s) {unknown}; the declared terms are "
            f"{list(TERM_NAMES)}"
        )
    values = np.zeros(N_TERMS, dtype=float)
    for name, value in mapping.items():
        if not np.isfinite(value):
            raise ValueError(f"coefficient {name} is not finite: {value}")
        values[TERM_INDEX[name]] = float(value)
    return values


def mapping_from_coefficients(coefficients: np.ndarray) -> dict[str, float]:
    """Name-to-value dict from a coefficient vector in :data:`TERM_NAMES` order."""
    values = np.asarray(coefficients, dtype=float)
    if values.shape != (N_TERMS,):
        raise ValueError(f"expected a length-{N_TERMS} coefficient vector, got {values.shape}")
    return {name: float(values[index]) for index, name in enumerate(TERM_NAMES)}


def term_table() -> list[list[str]]:
    """The term table as rows, for :mod:`pointmodel.report`."""
    return [
        [term.name, term.unit, term.meaning, term.cross_elevation, term.elevation]
        for term in TERMS
    ]
