"""Element plausibility bounds derived from orbital mechanics, not from taste.

Every bound in this module is either a physical consequence of the two-body
problem, a limit of what the two-line element text format can express, or a
declared policy band that is labelled as such wherever it is reported. The
three are kept apart because they carry different authority: a physical bound
that is violated means the element set describes nothing, while a policy band
that is violated means the element set describes something this screen was not
configured to expect.

Constants
---------
The gravitational parameter and equatorial radius are the WGS-72 values, which
are the values the SGP4 propagator itself uses, so that a semi-major axis
computed here and a semi-major axis computed inside ``sgp4`` are comparable:

- ``MU_KM3_S2 = 398600.8`` km^3 s^-2
- ``EARTH_RADIUS_KM = 6378.135`` km

Source: Hoots, F. R. and Roehrich, R. L. (1980), *Models for Propagation of
NORAD Element Sets*, Spacetrack Report No. 3, and Vallado, D. A., Crawford, P.,
Hujsak, R. and Kelso, T. S. (2006), *Revisiting Spacetrack Report #3*,
AIAA 2006-6753, section 3, which states the WGS-72 constants the reference
implementation carries and which this package therefore adopts.

Equations
---------
(1) Kepler's third law, inverted for the semi-major axis of a mean motion:

        a = (mu / omega^2)^(1/3),   omega = 2 pi n / 86400

    with ``n`` in revolutions per day, ``omega`` in radians per second and
    ``a`` in kilometres. Source: Vallado (2013), *Fundamentals of Astrodynamics
    and Applications*, 4th edition, equation (1-29).

    **Assumption and validity range.** The mean motion written in a two-line
    element set is a Kozai mean motion, not the Brouwer mean motion that the
    Kepler relation above is stated for, so equation (1) returns a semi-major
    axis that is biased. The bias is measured rather than assumed: see
    ``validation/validate_bounds.py``, which compares equation (1) against the
    semi-major axis ``sgp4`` computes from the same element set after its own
    Kozai-to-Brouwer conversion, over a swept grid of orbits. The screen's
    bands are set orders of magnitude looser than the measured bias, and the
    bias is reported as a determinability limit.

(2) Perigee and apogee radius:

        r_p = a (1 - e),   r_a = a (1 + e)

    Altitudes subtract ``EARTH_RADIUS_KM``. Source: Vallado (2013), equations
    (1-16) and (1-17).

(3) The mean-motion ceiling for an orbit whose perigee is at the equatorial
    radius, which is the fastest a circular Earth orbit can be:

        n_max = (86400 / 2 pi) sqrt(mu / R_e^3)

    This evaluates to 17.0436 rev/day with the constants above and is a
    physical bound, not a policy one.

(4) The apogee ceiling is the radius of the Earth's Hill sphere,

        r_H = a_E (m_E / (3 m_sun))^(1/3),

    about 1.5e6 km. Beyond it the two-body Earth model the element set is
    expressed in does not hold at all. Source: Hamilton, D. P. and Burns,
    J. A. (1992), *Orbital stability zones about asteroids*, Icarus 92, 118,
    for the form of the stability radius; the numerical value is the standard
    Earth figure and is used here only as an order-of-magnitude ceiling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .elements import ElementSet

MU_KM3_S2 = 398600.8
"""WGS-72 Earth gravitational parameter, km^3 s^-2."""

EARTH_RADIUS_KM = 6378.135
"""WGS-72 Earth equatorial radius, km."""

SECONDS_PER_DAY = 86400.0

HILL_RADIUS_KM = 1.5e6
"""Order-of-magnitude radius of the Earth's Hill sphere, km."""


def semi_major_axis_km(mean_motion_rev_per_day: float) -> float:
    """Semi-major axis from mean motion by equation (1), in kilometres.

    Parameters
    ----------
    mean_motion_rev_per_day:
        Mean motion in revolutions per day. Must be strictly positive.

    Returns
    -------
    float
        Semi-major axis in kilometres, or ``nan`` if the input is not a
        positive finite number.

    Notes
    -----
    This is the Kepler inversion of a Kozai mean motion and is therefore
    biased; the bias is measured in ``validation/validate_bounds.py``. Do not
    use this function as an orbit determination result.
    """
    n = float(mean_motion_rev_per_day)
    if not math.isfinite(n) or n <= 0.0:
        return float("nan")
    omega = 2.0 * math.pi * n / SECONDS_PER_DAY
    return (MU_KM3_S2 / (omega * omega)) ** (1.0 / 3.0)


def mean_motion_ceiling_rev_per_day() -> float:
    """Equation (3): fastest circular Earth orbit, revolutions per day."""
    omega = math.sqrt(MU_KM3_S2 / EARTH_RADIUS_KM**3)
    return SECONDS_PER_DAY * omega / (2.0 * math.pi)


@dataclass(frozen=True)
class BoundsPolicy:
    """The declared policy bands, separate from the physical bounds.

    Every field here is a configuration choice, not a law, and every report
    that uses one says so. Defaults are stated in the README.
    """

    min_perigee_altitude_km: float = 60.0
    """Below this the object has re-entered within hours. Policy."""

    max_apogee_altitude_km: float = HILL_RADIUS_KM - EARTH_RADIUS_KM
    """Equation (4). Physical in origin, held here for configurability."""

    min_mean_motion_rev_per_day: float = 0.5
    """Policy floor. 0.5 rev/day is an apogee well beyond geostationary."""

    max_eccentricity: float = 0.95
    """Policy ceiling. The text format itself can express up to 0.9999999."""

    max_abs_bstar: float = 0.1
    """Policy ceiling on |B*| in inverse Earth radii."""

    expected_ephemeris_type: int = 0
    """Distributed element sets carry 0, meaning SGP4/SDP4."""


@dataclass(frozen=True)
class BoundFinding:
    """One bound violation, with the kind of authority it carries.

    ``kind`` is ``"physical"``, ``"format"`` or ``"policy"``. A reader who
    disagrees with a policy band can change it; a physical violation means the
    element set does not describe an orbit.
    """

    name: str
    kind: str
    detail: str
    value: float


@dataclass(frozen=True)
class DerivedOrbit:
    """Quantities derived from the element set by equations (1) and (2)."""

    semi_major_axis_km: float
    perigee_altitude_km: float
    apogee_altitude_km: float
    period_minutes: float


def derive(element_set: ElementSet) -> DerivedOrbit:
    """Derived orbit geometry by equations (1) and (2). Units in the fields."""
    a = semi_major_axis_km(element_set.mean_motion_rev_per_day)
    e = element_set.eccentricity
    if not math.isfinite(a) or not math.isfinite(e):
        nan = float("nan")
        return DerivedOrbit(a, nan, nan, nan)
    return DerivedOrbit(
        semi_major_axis_km=a,
        perigee_altitude_km=a * (1.0 - e) - EARTH_RADIUS_KM,
        apogee_altitude_km=a * (1.0 + e) - EARTH_RADIUS_KM,
        period_minutes=1440.0 / element_set.mean_motion_rev_per_day,
    )


def check(element_set: ElementSet, policy: BoundsPolicy | None = None) -> list[BoundFinding]:
    """Return every bound violation of ``element_set``, with no verdict attached.

    Parameters
    ----------
    element_set:
        Parsed element set. Non-finite fields produce ``not_a_number``
        findings rather than being skipped, because a silently skipped field
        is how a screen reports a pass it did not earn.
    policy:
        Declared policy bands. ``None`` uses :class:`BoundsPolicy` defaults.

    Returns
    -------
    list[BoundFinding]
        Empty when every bound holds.
    """
    policy = policy or BoundsPolicy()
    findings: list[BoundFinding] = []

    numeric = {
        "inclination_deg": element_set.inclination_deg,
        "raan_deg": element_set.raan_deg,
        "eccentricity": element_set.eccentricity,
        "arg_perigee_deg": element_set.arg_perigee_deg,
        "mean_anomaly_deg": element_set.mean_anomaly_deg,
        "mean_motion_rev_per_day": element_set.mean_motion_rev_per_day,
        "bstar": element_set.bstar,
        "epoch_day": element_set.epoch_day,
    }
    for name, value in numeric.items():
        if not math.isfinite(value):
            findings.append(
                BoundFinding(
                    name=f"{name}_not_a_number",
                    kind="format",
                    detail=f"{name} did not parse to a finite number",
                    value=float("nan"),
                )
            )
    if any(f.kind == "format" for f in findings):
        return findings

    e = element_set.eccentricity
    if not 0.0 <= e < 1.0:
        findings.append(
            BoundFinding(
                "eccentricity_not_in_unit_interval",
                "physical",
                f"eccentricity {e:.7f} is outside [0, 1); the orbit is not closed",
                e,
            )
        )
    elif e > policy.max_eccentricity:
        findings.append(
            BoundFinding(
                "eccentricity_above_policy",
                "policy",
                f"eccentricity {e:.7f} exceeds the declared band "
                f"{policy.max_eccentricity:.7f}",
                e,
            )
        )

    inclination = element_set.inclination_deg
    if not 0.0 <= inclination <= 180.0:
        findings.append(
            BoundFinding(
                "inclination_out_of_range",
                "physical",
                f"inclination {inclination:.4f} deg is outside [0, 180]",
                inclination,
            )
        )
    for name, value in (
        ("raan_deg", element_set.raan_deg),
        ("arg_perigee_deg", element_set.arg_perigee_deg),
        ("mean_anomaly_deg", element_set.mean_anomaly_deg),
    ):
        if not 0.0 <= value < 360.0:
            findings.append(
                BoundFinding(
                    f"{name}_out_of_range",
                    "physical",
                    f"{name} {value:.4f} deg is outside [0, 360)",
                    value,
                )
            )

    n = element_set.mean_motion_rev_per_day
    ceiling = mean_motion_ceiling_rev_per_day()
    if n <= 0.0:
        findings.append(
            BoundFinding(
                "mean_motion_not_positive",
                "physical",
                f"mean motion {n:.8f} rev/day is not positive",
                n,
            )
        )
    elif n > ceiling:
        findings.append(
            BoundFinding(
                "mean_motion_above_physical_ceiling",
                "physical",
                f"mean motion {n:.8f} rev/day exceeds equation (3) ceiling "
                f"{ceiling:.4f} rev/day for a circular orbit at the equatorial radius",
                n,
            )
        )
    elif n < policy.min_mean_motion_rev_per_day:
        findings.append(
            BoundFinding(
                "mean_motion_below_policy",
                "policy",
                f"mean motion {n:.8f} rev/day is below the declared band "
                f"{policy.min_mean_motion_rev_per_day:.4f} rev/day",
                n,
            )
        )

    orbit = derive(element_set)
    if math.isfinite(orbit.perigee_altitude_km):
        if orbit.perigee_altitude_km < 0.0:
            findings.append(
                BoundFinding(
                    "perigee_below_surface",
                    "physical",
                    f"perigee altitude {orbit.perigee_altitude_km:.1f} km is below the "
                    "equatorial radius; equation (2) places the orbit underground",
                    orbit.perigee_altitude_km,
                )
            )
        elif orbit.perigee_altitude_km < policy.min_perigee_altitude_km:
            findings.append(
                BoundFinding(
                    "perigee_below_policy",
                    "policy",
                    f"perigee altitude {orbit.perigee_altitude_km:.1f} km is below the "
                    f"declared band {policy.min_perigee_altitude_km:.1f} km",
                    orbit.perigee_altitude_km,
                )
            )
        if orbit.apogee_altitude_km > policy.max_apogee_altitude_km:
            findings.append(
                BoundFinding(
                    "apogee_above_hill_sphere",
                    "physical",
                    f"apogee altitude {orbit.apogee_altitude_km:.1f} km exceeds the "
                    "Hill-sphere ceiling of equation (4); a geocentric two-body "
                    "element set does not describe this",
                    orbit.apogee_altitude_km,
                )
            )

    bstar = element_set.bstar
    if abs(bstar) > policy.max_abs_bstar:
        findings.append(
            BoundFinding(
                "bstar_magnitude_above_policy",
                "policy",
                f"|bstar| {abs(bstar):.6e} exceeds the declared band "
                f"{policy.max_abs_bstar:.6e} inverse Earth radii",
                bstar,
            )
        )

    if element_set.ephemeris_type != policy.expected_ephemeris_type:
        findings.append(
            BoundFinding(
                "ephemeris_type_unexpected",
                "policy",
                f"ephemeris type {element_set.ephemeris_type} is not the "
                f"{policy.expected_ephemeris_type} that distributed element sets carry",
                float(element_set.ephemeris_type),
            )
        )

    if not 1.0 <= element_set.epoch_day < 367.0:
        findings.append(
            BoundFinding(
                "epoch_day_out_of_range",
                "format",
                f"epoch day {element_set.epoch_day:.8f} is outside [1, 367)",
                element_set.epoch_day,
            )
        )

    return findings
