"""Spacecraft propagation through ``sgp4``, and the scenario time base.

The propagator is ``sgp4`` (Rhodes' Python wrapper around the Vallado C++
implementation of SGP4/SDP4). This package does not reimplement SGP4 and makes
no claim to. What it adds is the frame rotation into an Earth-fixed frame, a
uniform time base, and the batch interface the visibility search needs.

Time base
---------
A :class:`Scenario` fixes an epoch as a Julian day number plus a fraction, and
every other time in this package is a float count of **seconds after that
epoch**. Propagation splits the Julian date as ``jd`` plus ``fr`` and keeps the
fraction small, which is what ``sgp4``'s own interface asks for: adding a
quarter of a day to a number near 2.46e6 loses about 0.2 microseconds of
resolution, and keeping the day number separate avoids it.

UT1 against UTC is not modelled. ``sgp4`` wants UTC and the sidereal rotation
wants UT1; the difference is below 0.9 s, which is up to 0.0037 deg of Earth
rotation and up to roughly 0.4 km of station displacement. That is a declared
limitation, not a bug, and it is why window boundaries in this package are
reported to one second and not finer.

References
----------
Hoots, F. R. and Roehrich, R. L. (1980). *Models for Propagation of NORAD
Element Sets*, Spacetrack Report No. 3.

Vallado, D. A., Crawford, P., Hujsak, R. and Kelso, T. S. (2006). *Revisiting
Spacetrack Report #3*, AIAA 2006-6753. The reference implementation ``sgp4``
wraps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sgp4.api import SGP4_ERRORS, Satrec, jday

from .geometry import teme_to_ecef_km
from .tle import OrbitElements


@dataclass(frozen=True)
class Scenario:
    """A time base: an epoch and a duration, both declared.

    Attributes
    ----------
    jd : float
        Julian day number of the epoch, integer part carried here.
    fr : float
        Fraction of a day added to :attr:`jd`, in ``[0, 1)``.
    duration_s : float
        Length of the scheduling horizon, seconds after the epoch. Must be
        positive.
    """

    jd: float
    fr: float
    duration_s: float

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.fr) < 1.0:
            raise ValueError(f"fr must be within [0, 1), got {self.fr}")
        if float(self.duration_s) <= 0.0:
            raise ValueError(f"duration_s must be positive, got {self.duration_s}")

    @classmethod
    def from_calendar(
        cls,
        year: int,
        month: int,
        day: int,
        hour: int = 0,
        minute: int = 0,
        second: float = 0.0,
        duration_s: float = 86400.0,
    ) -> Scenario:
        """Build a scenario from a UTC calendar date and a horizon length."""
        jd, fr = jday(year, month, day, hour, minute, second)
        return cls(jd=jd, fr=fr, duration_s=float(duration_s))

    def julian_dates(self, times_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Split ``times_s`` after the epoch into ``sgp4``'s ``(jd, fr)`` pair."""
        t = np.asarray(times_s, dtype=float)
        return np.full(t.shape, float(self.jd)), float(self.fr) + t / 86400.0

    def full_julian_dates(self, times_s: np.ndarray) -> np.ndarray:
        """Single-float Julian dates, for sidereal time only.

        Sidereal time is a smooth function of the date with no sensitivity at
        the microsecond level, so collapsing the split representation here costs
        nothing measurable, unlike inside the propagator.
        """
        jd, fr = self.julian_dates(times_s)
        return jd + fr


class Spacecraft:
    """One propagated object: elements, a ``Satrec``, and Earth-fixed positions.

    Parameters
    ----------
    elements : OrbitElements
        Synthetic elements; see :mod:`contactsched.tle` for why they are
        synthetic.
    mission : str
        Owning mission identifier. Several spacecraft may belong to one mission,
        and fairness is accounted per mission, not per spacecraft.

    Notes
    -----
    Propagation failures are raised rather than returned. ``sgp4`` signals
    decayed or diverged states through an error code; a schedule built over a
    window where the propagator failed would be silently meaningless, so the
    error code is turned into a :class:`RuntimeError` naming the code and the
    time.
    """

    def __init__(self, elements: OrbitElements, mission: str = "UNASSIGNED") -> None:
        self.elements = elements
        self.mission = str(mission)
        self.line1, self.line2 = elements.to_lines()
        self.satrec = Satrec.twoline2rv(self.line1, self.line2)

    @property
    def name(self) -> str:
        """Spacecraft label, from the elements."""
        return self.elements.name

    def __repr__(self) -> str:
        return f"Spacecraft(name={self.name!r}, mission={self.mission!r})"

    def teme_km(self, scenario: Scenario, times_s: np.ndarray) -> np.ndarray:
        """TEME positions, km, shape ``(n, 3)``.

        Raises
        ------
        RuntimeError
            If ``sgp4`` reports a propagation error at any requested time.
        """
        times = np.atleast_1d(np.asarray(times_s, dtype=float))
        jd, fr = scenario.julian_dates(times)
        codes, positions, _velocities = self.satrec.sgp4_array(jd, fr)
        codes = np.asarray(codes)
        if np.any(codes != 0):
            index = int(np.flatnonzero(codes != 0)[0])
            code = int(codes[index])
            message = SGP4_ERRORS.get(code, f"unknown sgp4 error code {code}")
            raise RuntimeError(
                f"sgp4 failed for {self.name} at t={times[index]:.3f} s after the "
                f"scenario epoch: code {code}, {message}"
            )
        return np.asarray(positions, dtype=float)

    def ecef_km(self, scenario: Scenario, times_s: np.ndarray) -> np.ndarray:
        """Earth-fixed positions, km, shape ``(n, 3)``."""
        times = np.atleast_1d(np.asarray(times_s, dtype=float))
        teme = self.teme_km(scenario, times)
        return teme_to_ecef_km(teme, scenario.full_julian_dates(times))

    def radius_km(self, scenario: Scenario, times_s: np.ndarray) -> np.ndarray:
        """Geocentric radius, km, shape ``(n,)``. Frame-independent."""
        return np.linalg.norm(self.teme_km(scenario, times_s), axis=1)


def default_fleet() -> tuple[Spacecraft, ...]:
    """Six synthetic spacecraft across four missions, used by the examples.

    The elements are chosen so that contention actually occurs: four of the six
    are in near-polar orbits at similar altitudes, which puts their passes over
    a high-latitude station close together in time. Mission ``POLARIS`` owns
    three spacecraft, which is what makes per-mission fairness differ from
    per-spacecraft fairness.
    """
    base = dict(epoch_year=26, epoch_day=120.0, eccentricity=0.0012)
    return (
        Spacecraft(
            OrbitElements(
                satnum=90001, inclination_deg=97.8, raan_deg=12.0, argp_deg=40.0,
                mean_anomaly_deg=0.0, mean_motion_rev_day=15.21, bstar=8.0e-5,
                name="POLARIS-A", **base,
            ),
            mission="POLARIS",
        ),
        Spacecraft(
            OrbitElements(
                satnum=90002, inclination_deg=97.8, raan_deg=12.4, argp_deg=40.0,
                mean_anomaly_deg=35.0, mean_motion_rev_day=15.21, bstar=8.0e-5,
                name="POLARIS-B", **base,
            ),
            mission="POLARIS",
        ),
        Spacecraft(
            OrbitElements(
                satnum=90003, inclination_deg=97.9, raan_deg=13.1, argp_deg=40.0,
                mean_anomaly_deg=72.0, mean_motion_rev_day=15.19, bstar=8.0e-5,
                name="POLARIS-C", **base,
            ),
            mission="POLARIS",
        ),
        Spacecraft(
            OrbitElements(
                satnum=90011, inclination_deg=98.4, raan_deg=200.0, argp_deg=120.0,
                mean_anomaly_deg=10.0, mean_motion_rev_day=14.78, bstar=4.0e-5,
                name="TERRASCAN-1", **base,
            ),
            mission="TERRASCAN",
        ),
        Spacecraft(
            OrbitElements(
                satnum=90021, inclination_deg=51.6, raan_deg=80.0, argp_deg=300.0,
                mean_anomaly_deg=200.0, mean_motion_rev_day=15.49, bstar=2.0e-4,
                name="EQUINOX-1", **base,
            ),
            mission="EQUINOX",
        ),
        Spacecraft(
            OrbitElements(
                satnum=90031, inclination_deg=63.4, raan_deg=310.0, argp_deg=270.0,
                mean_anomaly_deg=90.0, mean_motion_rev_day=13.42, bstar=1.0e-5,
                name="MERIDIAN-1", **base,
            ),
            mission="MERIDIAN",
        ),
    )
