"""Crosslink scenarios built from orbital elements, and seeded families of them.

Everything here is a geometry generator. No flight data is used anywhere in
this repository, and every number any validation script reports is a statement
about these generators and nothing else.

A circular orbit is specified by altitude, inclination, right ascension of the
ascending node and argument of latitude, and converted to an inertial state
vector by the standard rotation sequence

    r = R3(-raan) R1(-inclination) R3(-argument_of_latitude) [a, 0, 0]^T

with the velocity the in-track derivative at circular speed ``sqrt(mu / a)``.
Source for the sequence: Vallado, *Fundamentals of Astrodynamics and
Applications*, 4th ed., Microcosm Press 2013, section 2.5 (coordinate
transformations). Structure only; no numerical value is taken from it.

The frame is an unspecified Earth-centred inertial frame. It is deliberately
unspecified: the point-ahead total is invariant under a change of inertial
frame to first order and the attribution is not, and a scenario that claimed to
be in J2000 without a time system behind it would be claiming more than it can
support. The frame label is carried through every result so a reader is never
left guessing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import rotation_matrix
from .kepler import MU_EARTH, R_EARTH
from .terminal import BeamContext, Crosslink, TerminalState

FRAME_LABEL = "unspecified Earth-centred inertial"


@dataclass(frozen=True)
class CircularOrbit:
    """A circular orbit element set, all angles in radians, altitude in metres."""

    altitude_m: float
    inclination_rad: float
    raan_rad: float
    argument_of_latitude_rad: float

    def __post_init__(self) -> None:
        altitude = float(self.altitude_m)
        if not np.isfinite(altitude) or altitude <= 0.0:
            raise ValueError(f"altitude_m must be positive and finite, got {self.altitude_m!r}")
        for name in ("inclination_rad", "raan_rad", "argument_of_latitude_rad"):
            if not np.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite, got {getattr(self, name)!r}")

    @property
    def semi_major_axis_m(self) -> float:
        """Orbital radius, metres, measured from the WGS-84 equatorial radius."""
        return R_EARTH + float(self.altitude_m)

    def speed_mps(self, mu: float = MU_EARTH) -> float:
        """Circular orbital speed, m/s."""
        return float(np.sqrt(float(mu) / self.semi_major_axis_m))

    def period_s(self, mu: float = MU_EARTH) -> float:
        """Orbital period, seconds."""
        a = self.semi_major_axis_m
        return float(2.0 * np.pi * np.sqrt(a**3 / float(mu)))

    def state(self, name: str = "terminal", mu: float = MU_EARTH) -> TerminalState:
        """Inertial state at the declared argument of latitude, epoch zero."""
        a = self.semi_major_axis_m
        speed = self.speed_mps(mu)
        rotation = (
            rotation_matrix([0.0, 0.0, 1.0], float(self.raan_rad))
            @ rotation_matrix([1.0, 0.0, 0.0], float(self.inclination_rad))
            @ rotation_matrix([0.0, 0.0, 1.0], float(self.argument_of_latitude_rad))
        )
        position = rotation @ np.array([a, 0.0, 0.0])
        velocity = rotation @ np.array([0.0, speed, 0.0])
        return TerminalState(position, velocity, 0.0, name)


def crosslink_from_orbits(
    local: CircularOrbit,
    remote: CircularOrbit,
    label: str = "crosslink",
    trajectory_model: str = "kepler",
    mu: float = MU_EARTH,
) -> Crosslink:
    """Build a :class:`~pointahead.terminal.Crosslink` from two circular orbits."""
    return Crosslink(
        local=local.state("local", mu),
        remote=remote.state("remote", mu),
        trajectory_model=trajectory_model,
        mu=mu,
        label=label,
        frame=FRAME_LABEL,
    )


def leo_crossing_pair(
    altitude_m: float = 600.0e3,
    inclination_deg: float = 60.0,
    separation_rad: float = 0.3,
    label: str = "leo_crossing",
) -> Crosslink:
    """A same-altitude pair on crossing planes.

    The local terminal is equatorial and the remote terminal is inclined by
    ``inclination_deg``, with the remote terminal ``separation_rad`` of arc
    along its own orbit. This is the geometry with the largest point-ahead
    angle in the families below and it is also the geometry in which the
    relative acceleration is exactly radial, so it is the one that exposed the
    missing jerk term in the ephemeris staleness line.
    """
    local = CircularOrbit(altitude_m, 0.0, 0.0, 0.0)
    remote = CircularOrbit(altitude_m, float(np.deg2rad(inclination_deg)), 0.0, separation_rad)
    return crosslink_from_orbits(local, remote, label)


def leo_trailing_pair(
    altitude_m: float = 600.0e3,
    separation_rad: float = 0.3,
    label: str = "leo_trailing",
) -> Crosslink:
    """A same-plane, same-altitude trailing pair.

    The two velocities differ only by the rotation of the in-track direction
    over the arc, so the transverse relative velocity -- and with it the
    point-ahead angle -- is small. Included because it is the geometry a reader
    will assume is representative, and it is the one where the answer is
    smallest.
    """
    local = CircularOrbit(altitude_m, 0.0, 0.0, 0.0)
    remote = CircularOrbit(altitude_m, 0.0, 0.0, separation_rad)
    return crosslink_from_orbits(local, remote, label)


def geo_to_leo_pair(
    leo_altitude_m: float = 600.0e3,
    geo_altitude_m: float = 35786.0e3,
    leo_inclination_deg: float = 51.6,
    leo_argument_rad: float = 0.4,
    label: str = "geo_relay_to_leo",
) -> Crosslink:
    """A geostationary-altitude relay looking down at an inclined low orbiter.

    Range is about sixty times the crossing-pair case, so the light time is
    sixty times longer. Included to test whether the closed form degrades with
    light time, which is the thing a reader would reasonably expect it to do.
    """
    local = CircularOrbit(geo_altitude_m, 0.0, 0.0, 0.0)
    remote = CircularOrbit(
        leo_altitude_m, float(np.deg2rad(leo_inclination_deg)), 0.0, leo_argument_rad
    )
    return crosslink_from_orbits(local, remote, label)


def random_crosslink(rng: np.random.Generator, label: str = "random") -> Crosslink:
    """A seeded random same-regime pair, for sweeps.

    Altitudes are drawn in 400 to 1400 km, inclinations in 0 to 90 degrees,
    nodes and arguments uniformly, and the draw is rejected and retried if the
    two terminals land closer than 50 km or if the chord passes within the
    WGS-84 equatorial radius of the Earth centre, because an occulted link is
    not a link. The rejection is geometric only; no atmosphere or limb margin
    is modelled, and that is a limitation, not a model.
    """
    for _ in range(200):
        orbits = [
            CircularOrbit(
                float(rng.uniform(400.0e3, 1400.0e3)),
                float(rng.uniform(0.0, 0.5 * np.pi)),
                float(rng.uniform(0.0, 2.0 * np.pi)),
                float(rng.uniform(0.0, 2.0 * np.pi)),
            )
            for _ in range(2)
        ]
        link = crosslink_from_orbits(orbits[0], orbits[1], label)
        if link.range_m < 50.0e3:
            continue
        if _chord_perigee_m(link) > R_EARTH:
            return link
    raise RuntimeError(
        "random_crosslink could not find an unoccluded pair in 200 draws; "
        "this should not happen and indicates a change in the sampling ranges"
    )


def _chord_perigee_m(link: Crosslink) -> float:
    """Closest approach of the line-of-sight segment to the frame origin, metres."""
    r0 = link.local.position_m
    d = link.separation_m
    length_squared = float(np.dot(d, d))
    t = -float(np.dot(r0, d)) / length_squared
    t = min(max(t, 0.0), 1.0)
    return float(np.linalg.norm(r0 + t * d))


def scenario_family(seed: int, count: int) -> tuple[Crosslink, ...]:
    """``count`` seeded random crosslinks. Deterministic in ``seed``."""
    if int(count) <= 0:
        raise ValueError(f"count must be positive, got {count!r}")
    rng = np.random.default_rng(int(seed))
    return tuple(random_crosslink(rng, f"random_{index:03d}") for index in range(int(count)))


NAMED_SCENARIOS = {
    "leo_crossing": leo_crossing_pair,
    "leo_trailing": leo_trailing_pair,
    "geo_relay_to_leo": geo_to_leo_pair,
}
"""The three hand-built geometries used by the examples and validation scripts."""


DEFAULT_BEAM = BeamContext(1550.0e-9, 0.025, "1550 nm, 25 mm waist")
"""A 1550 nm beam with a 25 mm waist radius. A declared input, not a product
specification: the divergence half-angle that follows is about 19.7 microradians
and it is reported by ``validation/validate_budget.py``."""

NARROW_BEAM = BeamContext(1064.0e-9, 0.075, "1064 nm, 75 mm waist")
"""A narrower beam, for the context comparison in the worked example."""
