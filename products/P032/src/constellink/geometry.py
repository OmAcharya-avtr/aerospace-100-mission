"""Line-of-sight geometry for inter-satellite and satellite-ground links.

Earth-limb blockage
-------------------
An inter-satellite link (ISL) between positions ``r1`` and ``r2`` (both from
the Earth's centre, km) is geometrically clear when the straight segment
joining them stays outside a sphere of radius ``r_block = R_e + h_grazing``,
where ``h_grazing`` is a grazing-altitude margin that keeps the ray out of the
dense atmosphere.  The test is the exact point-to-segment distance:

    d_min = min_{s in [0, 1]} | r1 + s (r2 - r1) |
    clear <=> d_min >= r_block

Closed form for the equal-radius case
-------------------------------------
For two satellites at the same geocentric radius ``r`` separated by Earth
central angle ``gamma``, the chord's closest approach to the centre is
``r cos(gamma / 2)``, so the link is clear iff

    gamma <= gamma_max = 2 arccos(r_block / r)                            (1)

This is the standard maximum-central-angle result for a circular
constellation; see Wertz & Larson (eds.) 1999, "Space Mission Analysis and
Design", 3rd ed., Ch. 5 (Earth geometry viewed from space), and Vallado 2013,
"Fundamentals of Astrodynamics and Applications", 4th ed., Ch. 11.
Units: ``r`` and ``r_block`` in km, ``gamma`` in rad.  Validity: spherical
Earth; equal radii; no refraction.  Equation (1) is used only as an
independent check on the general segment test (see
``validation/validate_contact_windows.py``).

Ground-station access
---------------------
For a circular orbit of radius ``r`` and a minimum elevation ``eps``, the
maximum Earth central angle between the sub-satellite point and the station is

    lambda_max = arccos( (R_e / r) cos(eps) ) - eps                       (2)

(Wertz & Larson 1999, Ch. 5, the standard Earth-coverage relation; equivalent
forms appear in Vallado 2013, Ch. 11).  Units: angles in rad.  Validity:
spherical Earth of radius ``R_e``, no refraction, station at zero altitude.
Equation (2) is used for the analytic cross-check of the numerically found
ground contact windows.

Slant range at a given elevation
--------------------------------
From the same spherical geometry, the slant range to a satellite at radius
``r`` seen at elevation ``eps`` from a station at radius ``R_e`` is

    rho = -R_e sin(eps) + sqrt(r^2 - R_e^2 cos^2(eps))                    (3)

(law of cosines in the station-centre-satellite triangle; Wertz & Larson 1999,
Ch. 5).  Units: km.
"""

from __future__ import annotations

import numpy as np

from .frames import WGS84_A_KM

__all__ = [
    "DEFAULT_GRAZING_ALTITUDE_KM",
    "segment_min_radius",
    "isl_clear",
    "max_isl_central_angle",
    "central_angle",
    "ground_max_central_angle",
    "slant_range_at_elevation",
]

# Grazing-altitude margin [km].  100 km is the conventional round figure for
# the top of the dense atmosphere (the "von Karman line"); it is a modelling
# choice, not a propagation result, and is exposed as a parameter everywhere.
DEFAULT_GRAZING_ALTITUDE_KM = 100.0


def segment_min_radius(r1_km: np.ndarray, r2_km: np.ndarray) -> np.ndarray:
    """Minimum distance from the Earth's centre to the segment ``r1 -> r2`` [km].

    Exact point-to-segment distance.  Accepts ``(3,)`` or ``(n, 3)`` arrays;
    returns a scalar array or shape ``(n,)``.
    """
    a = np.atleast_2d(np.asarray(r1_km, dtype=float))
    b = np.atleast_2d(np.asarray(r2_km, dtype=float))
    if a.shape[-1] != 3 or b.shape[-1] != 3:
        raise ValueError("positions must have trailing dimension 3")
    d = b - a
    dd = np.einsum("ij,ij->i", d, d)
    # Parameter of the foot of the perpendicular from the origin, clamped to
    # the segment.  dd == 0 means coincident endpoints.
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(dd > 0.0, -np.einsum("ij,ij->i", a, d) / np.where(dd > 0.0, dd, 1.0), 0.0)
    s = np.clip(s, 0.0, 1.0)
    closest = a + s[:, None] * d
    return np.linalg.norm(closest, axis=1)


def isl_clear(r1_km: np.ndarray, r2_km: np.ndarray,
              grazing_altitude_km: float = DEFAULT_GRAZING_ALTITUDE_KM,
              earth_radius_km: float = WGS84_A_KM) -> np.ndarray:
    """Boolean Earth-limb clearance of the ISL ray (see module docstring).

    Parameters
    ----------
    r1_km, r2_km : geocentric positions [km], shape ``(3,)`` or ``(n, 3)``.
    grazing_altitude_km : altitude margin above the Earth's surface the ray
        must clear [km], >= 0.
    earth_radius_km : blocking sphere radius [km], > 0 (spherical Earth; the
        WGS-84 equatorial radius is the conservative choice).

    Returns a bool array; scalar inputs give a 1-element array.
    """
    if grazing_altitude_km < 0.0:
        raise ValueError(f"grazing_altitude_km must be >= 0, got {grazing_altitude_km}")
    if earth_radius_km <= 0.0:
        raise ValueError(f"earth_radius_km must be > 0, got {earth_radius_km}")
    r_block = earth_radius_km + grazing_altitude_km
    return segment_min_radius(r1_km, r2_km) >= r_block


def max_isl_central_angle(orbit_radius_km: float,
                          grazing_altitude_km: float = DEFAULT_GRAZING_ALTITUDE_KM,
                          earth_radius_km: float = WGS84_A_KM) -> float:
    """Closed-form maximum ISL central angle [rad] for equal radii, Eq. (1).

    ``gamma_max = 2 arccos(r_block / r)``.  Raises ``ValueError`` if the orbit
    is inside the blocking sphere.
    """
    r_block = earth_radius_km + grazing_altitude_km
    if orbit_radius_km <= r_block:
        raise ValueError(
            f"orbit radius {orbit_radius_km} km is not above the blocking sphere {r_block} km")
    return float(2.0 * np.arccos(r_block / orbit_radius_km))


def central_angle(r1_km: np.ndarray, r2_km: np.ndarray) -> np.ndarray:
    """Earth central angle between two geocentric position vectors [rad].

    ``gamma = atan2(|r1 x r2|, r1 . r2)``, which is numerically better
    conditioned than ``arccos`` of the normalised dot product near 0 and pi.
    """
    a = np.atleast_2d(np.asarray(r1_km, dtype=float))
    b = np.atleast_2d(np.asarray(r2_km, dtype=float))
    dot = np.einsum("ij,ij->i", a, b)
    cross = np.linalg.norm(np.cross(a, b), axis=1)
    return np.arctan2(cross, dot)


def ground_max_central_angle(orbit_radius_km: float, min_elevation_deg: float,
                             earth_radius_km: float = WGS84_A_KM) -> float:
    """Closed-form maximum station central angle [rad], Eq. (2).

    ``lambda_max = arccos((R_e / r) cos eps) - eps``.  Spherical Earth,
    station at zero altitude, no refraction.
    """
    if orbit_radius_km <= earth_radius_km:
        raise ValueError(
            f"orbit radius {orbit_radius_km} km must exceed Earth radius {earth_radius_km} km")
    if not 0.0 <= min_elevation_deg < 90.0:
        raise ValueError(f"min_elevation_deg must be in [0, 90), got {min_elevation_deg}")
    eps = np.deg2rad(min_elevation_deg)
    arg = (earth_radius_km / orbit_radius_km) * np.cos(eps)
    return float(np.arccos(np.clip(arg, -1.0, 1.0)) - eps)


def slant_range_at_elevation(orbit_radius_km: float, elevation_deg: float,
                             earth_radius_km: float = WGS84_A_KM) -> float:
    """Slant range [km] to a satellite at ``orbit_radius_km`` seen at elevation, Eq. (3)."""
    if orbit_radius_km <= earth_radius_km:
        raise ValueError("orbit radius must exceed Earth radius")
    if not -90.0 <= elevation_deg <= 90.0:
        raise ValueError(f"elevation_deg must be in [-90, 90], got {elevation_deg}")
    eps = np.deg2rad(elevation_deg)
    disc = orbit_radius_km ** 2 - (earth_radius_km * np.cos(eps)) ** 2
    if disc < 0.0:
        raise ValueError("no solution: satellite below the local horizon geometry")
    return float(-earth_radius_km * np.sin(eps) + np.sqrt(disc))
