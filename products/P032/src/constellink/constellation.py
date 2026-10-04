"""Constellation definition, TLE handling and ephemeris generation.

Propagation
-----------
Orbit propagation is delegated entirely to the ``sgp4`` package (Vallado's
SGP4/SDP4 implementation; Vallado, Crawford, Hujsak & Kelso 2006, "Revisiting
Spacetrack Report #3", AIAA 2006-6753).  This module does not reimplement
SGP4; it wraps it with constellation bookkeeping, epoch-validity checking and
a vectorised ephemeris interface.  ``validation/validate_sgp4_vector.py``
reproduces the verification output shipped with ``sgp4`` so the wrapper's use
of the library is checked rather than assumed.

Walker constellations
---------------------
``walker_delta`` builds a Walker delta pattern ``i: T/P/F`` (Walker 1984,
"Satellite constellations", J. British Interplanetary Society 37, 559-572;
also described in Wertz & Larson (eds.) 1999, "Space Mission Analysis and
Design", 3rd ed., Ch. 7):

* ``T`` satellites in ``P`` equally spaced orbital planes, ``S = T / P`` per
  plane, all at inclination ``i`` and the same circular radius;
* plane ``p`` has right ascension of the ascending node
  ``Omega_p = p * 360 / P`` deg;
* satellite ``s`` in plane ``p`` has mean anomaly
  ``M = s * 360 / S + p * F * 360 / T`` deg, where ``F`` in ``0..P-1`` is the
  inter-plane phasing parameter.

Satellites are created as ``sgp4`` ``Satrec`` objects through ``sgp4init``, so
every satellite -- generated or TLE-loaded -- is propagated by the same code
path.  Mean motion is converted from the circular-orbit radius by the
two-body relation ``n = sqrt(mu / a^3)``, which is the osculating/Kepler mean
motion; SGP4 consumes the Brouwer-Lyddane "Kozai" mean motion.  The two differ
by the J2 secular correction, which for the LEO altitudes used here is of
order 1e-3 relative.  The consequence is that a generated Walker satellite's
realised altitude differs from the requested one by a few km.  The error is
measured and reported in ``validation/validate_walker.py`` and in
``docs/REQUIREMENTS.md`` (REQ-04); generated constellations are design aids,
not replacements for a published element set.

Epoch validity
--------------
SGP4 is an analytic theory fitted to a specific epoch.  Propagating a long
way from epoch degrades silently -- the library returns a position with no
error code.  ``Satellite.propagate`` therefore raises ``TleEpochError`` when
the requested time is further from the element epoch than
``max_epoch_age_days`` (default 7 days).  The default is a modelling choice,
documented as such: TLE accuracy degradation is dominated by drag-model error
and is mission-specific, so there is no single correct threshold.  Published
discussions of TLE accuracy growth away from epoch include Vallado & Cefola
2012, "Two-line element sets - practice and use", IAC-12.C1.6.7, and Kelso
2007, "Validation of SGP4 and IS-GPS-200D against GPS precision ephemerides",
AAS 07-127.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import numpy as np
from sgp4.api import WGS72, Satrec, jday

from .frames import MU_EARTH_KM3_S2, WGS84_A_KM, to_utc

__all__ = [
    "TleEpochError",
    "PropagationError",
    "TLE",
    "Satellite",
    "GroundStation",
    "Constellation",
    "walker_delta",
    "Ephemeris",
    "CircularOrbit",
]

# Days from 1949-12-31 00:00 UT to the J2000 epoch used by ``sgp4init``.
_SGP4_EPOCH = datetime(1949, 12, 31, 0, 0, 0, tzinfo=UTC)

DEFAULT_MAX_EPOCH_AGE_DAYS = 7.0


class TleEpochError(ValueError):
    """Raised when a propagation time is too far from the element-set epoch."""


class PropagationError(RuntimeError):
    """Raised when the SGP4 propagator returns a non-zero error code."""


def _tle_checksum(line: str) -> int:
    """Modulo-10 TLE checksum: digits count as their value, '-' counts as 1."""
    total = 0
    for ch in line[:68]:
        if ch.isdigit():
            total += int(ch)
        elif ch == "-":
            total += 1
    return total % 10


@dataclass(frozen=True)
class TLE:
    """A two-line element set with a display name.

    ``line1`` and ``line2`` must be standard 69-character TLE lines with valid
    modulo-10 checksums.
    """

    name: str
    line1: str
    line2: str

    def __post_init__(self) -> None:
        for i, line in enumerate((self.line1, self.line2), start=1):
            if len(line) != 69:
                raise ValueError(
                    f"TLE line {i} for '{self.name}' must be 69 characters, got {len(line)}")
            if line[0] != str(i):
                raise ValueError(f"TLE line {i} for '{self.name}' must start with '{i}'")
            if _tle_checksum(line) != int(line[68]):
                raise ValueError(f"TLE line {i} for '{self.name}' fails its modulo-10 checksum")

    def to_satrec(self) -> Satrec:
        """Parse into an ``sgp4`` ``Satrec``."""
        return Satrec.twoline2rv(self.line1, self.line2)


@dataclass
class Satellite:
    """One propagatable satellite: a name plus an ``sgp4`` ``Satrec``.

    Attributes
    ----------
    name : display name, must be unique within a :class:`Constellation`.
    satrec : the ``sgp4`` propagator object.
    max_epoch_age_days : propagation times further than this from the element
        epoch raise :class:`TleEpochError` [days], > 0.
    """

    name: str
    satrec: Satrec
    max_epoch_age_days: float = DEFAULT_MAX_EPOCH_AGE_DAYS

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("satellite name must be non-empty")
        if self.max_epoch_age_days <= 0.0:
            raise ValueError(
                f"max_epoch_age_days must be > 0, got {self.max_epoch_age_days}")

    @classmethod
    def from_tle(cls, tle: TLE, max_epoch_age_days: float = DEFAULT_MAX_EPOCH_AGE_DAYS,
                 ) -> Satellite:
        """Build from a :class:`TLE`."""
        return cls(name=tle.name, satrec=tle.to_satrec(),
                   max_epoch_age_days=max_epoch_age_days)

    @property
    def epoch(self) -> datetime:
        """Element-set epoch as a timezone-aware UTC datetime."""
        return _SGP4_EPOCH + timedelta(days=self.satrec.jdsatepoch
                                       + self.satrec.jdsatepochF - _jd_of(_SGP4_EPOCH))

    def epoch_age_days(self, t: datetime) -> float:
        """Signed age of ``t`` relative to the element epoch [days]."""
        jd, fr = _jd_pair(t)
        return float((jd - self.satrec.jdsatepoch) + (fr - self.satrec.jdsatepochF))

    def check_epoch(self, times: list[datetime] | np.ndarray) -> None:
        """Raise :class:`TleEpochError` if any time is outside the epoch window.

        The check is applied to the extreme requested times only; the window is
        symmetric about the epoch.
        """
        ages = [abs(self.epoch_age_days(t)) for t in times]
        worst = max(ages) if ages else 0.0
        if worst > self.max_epoch_age_days:
            raise TleEpochError(
                f"satellite '{self.name}': requested window reaches {worst:.3f} days from the "
                f"element epoch {self.epoch.isoformat()}, exceeding max_epoch_age_days="
                f"{self.max_epoch_age_days}. SGP4 degrades silently far from epoch; supply a "
                f"fresher element set, or raise max_epoch_age_days deliberately and record "
                f"the resulting accuracy loss.")

    def propagate(self, times: list[datetime], check_epoch: bool = True,
                  ) -> tuple[np.ndarray, np.ndarray]:
        """Propagate to ``times``; return TEME ``(positions_km, velocities_km_s)``.

        Both arrays have shape ``(len(times), 3)``.  Raises
        :class:`TleEpochError` if ``check_epoch`` and any time is outside the
        epoch window, and :class:`PropagationError` on an SGP4 error code.
        """
        if len(times) == 0:
            return np.zeros((0, 3)), np.zeros((0, 3))
        if check_epoch:
            self.check_epoch(times)
        jds = np.empty(len(times))
        frs = np.empty(len(times))
        for k, t in enumerate(times):
            jds[k], frs[k] = _jd_pair(t)
        err, r, v = self.satrec.sgp4_array(jds, frs)
        bad = np.nonzero(err)[0]
        if bad.size:
            code = int(err[bad[0]])
            raise PropagationError(
                f"satellite '{self.name}': SGP4 error code {code} at sample index "
                f"{int(bad[0])} ({times[int(bad[0])].isoformat()}); "
                f"see sgp4 documentation for code meanings")
        return np.asarray(r, dtype=float), np.asarray(v, dtype=float)


def _jd_pair(t: datetime) -> tuple[float, float]:
    t = to_utc(t)
    return jday(t.year, t.month, t.day, t.hour, t.minute,
                t.second + t.microsecond * 1e-6)


def _jd_of(t: datetime) -> float:
    jd, fr = _jd_pair(t)
    return jd + fr


@dataclass(frozen=True)
class GroundStation:
    """A ground terminal.

    Attributes
    ----------
    name : unique identifier.
    lat_deg : geodetic latitude [deg], [-90, 90].
    lon_deg : longitude east [deg].
    alt_km : height above the WGS-84 ellipsoid [km].
    min_elevation_deg : elevation mask [deg], [0, 90).
    """

    name: str
    lat_deg: float
    lon_deg: float
    alt_km: float = 0.0
    min_elevation_deg: float = 10.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ground-station name must be non-empty")
        if not -90.0 <= self.lat_deg <= 90.0:
            raise ValueError(f"lat_deg must be in [-90, 90], got {self.lat_deg}")
        if not -540.0 <= self.lon_deg <= 540.0:
            raise ValueError(f"lon_deg must be in [-540, 540], got {self.lon_deg}")
        if self.alt_km < -0.5:
            raise ValueError(f"alt_km must be >= -0.5, got {self.alt_km}")
        if not 0.0 <= self.min_elevation_deg < 90.0:
            raise ValueError(
                f"min_elevation_deg must be in [0, 90), got {self.min_elevation_deg}")


@dataclass
class Constellation:
    """A named set of satellites plus optional ground stations."""

    satellites: list[Satellite]
    stations: list[GroundStation] = field(default_factory=list)
    name: str = "constellation"

    def __post_init__(self) -> None:
        if not self.satellites:
            raise ValueError("a constellation needs at least one satellite")
        names = [s.name for s in self.satellites]
        if len(set(names)) != len(names):
            raise ValueError("satellite names must be unique")
        st = [g.name for g in self.stations]
        if len(set(st)) != len(st):
            raise ValueError("ground-station names must be unique")
        if set(names) & set(st):
            raise ValueError("satellite and ground-station names must not collide")

    @property
    def n_sat(self) -> int:
        """Number of satellites."""
        return len(self.satellites)

    def satellite(self, name: str) -> Satellite:
        """Look up a satellite by name."""
        for s in self.satellites:
            if s.name == name:
                return s
        raise KeyError(f"no satellite named '{name}'")

    def ephemeris(self, t0: datetime, t1: datetime, step_s: float,
                  check_epoch: bool = True) -> Ephemeris:
        """Propagate every satellite on a uniform grid over ``[t0, t1]``."""
        times = time_grid(t0, t1, step_s)
        pos = np.empty((self.n_sat, len(times), 3))
        vel = np.empty((self.n_sat, len(times), 3))
        for i, sat in enumerate(self.satellites):
            r, v = sat.propagate(times, check_epoch=check_epoch)
            pos[i] = r
            vel[i] = v
        return Ephemeris(times=times, sat_names=[s.name for s in self.satellites],
                         r_teme_km=pos, v_teme_km_s=vel)


def time_grid(t0: datetime, t1: datetime, step_s: float) -> list[datetime]:
    """Uniform UTC time grid over ``[t0, t1]`` inclusive of ``t0``.

    The last sample is at or before ``t1``; ``t1`` itself is appended when the
    step does not divide the span exactly.
    """
    t0 = to_utc(t0)
    t1 = to_utc(t1)
    if t1 <= t0:
        raise ValueError(f"t1 ({t1}) must be after t0 ({t0})")
    if step_s <= 0.0:
        raise ValueError(f"step_s must be > 0, got {step_s}")
    span = (t1 - t0).total_seconds()
    n = int(np.floor(span / step_s))
    times = [t0 + timedelta(seconds=step_s * k) for k in range(n + 1)]
    if times[-1] < t1:
        times.append(t1)
    return times


@dataclass(frozen=True)
class Ephemeris:
    """Propagated states on a shared time grid.

    Attributes
    ----------
    times : ``n_t`` UTC datetimes.
    sat_names : ``n_sat`` names, index-aligned with the state arrays.
    r_teme_km : TEME positions, shape ``(n_sat, n_t, 3)`` [km].
    v_teme_km_s : TEME velocities, shape ``(n_sat, n_t, 3)`` [km/s].
    """

    times: list[datetime]
    sat_names: list[str]
    r_teme_km: np.ndarray
    v_teme_km_s: np.ndarray

    def __post_init__(self) -> None:
        n_sat, n_t = len(self.sat_names), len(self.times)
        if self.r_teme_km.shape != (n_sat, n_t, 3):
            raise ValueError(
                f"r_teme_km must have shape {(n_sat, n_t, 3)}, got {self.r_teme_km.shape}")
        if self.v_teme_km_s.shape != (n_sat, n_t, 3):
            raise ValueError(
                f"v_teme_km_s must have shape {(n_sat, n_t, 3)}, got {self.v_teme_km_s.shape}")

    @property
    def step_s(self) -> float:
        """Grid step [s] (from the first two samples)."""
        if len(self.times) < 2:
            raise ValueError("ephemeris has fewer than two time samples")
        return (self.times[1] - self.times[0]).total_seconds()

    def index(self, sat_name: str) -> int:
        """Row index of ``sat_name``."""
        try:
            return self.sat_names.index(sat_name)
        except ValueError as exc:
            raise KeyError(f"no satellite named '{sat_name}' in this ephemeris") from exc

    def drop(self, sat_names: list[str]) -> Ephemeris:
        """Return a copy without the named satellites (a satellite-loss case)."""
        keep = [i for i, n in enumerate(self.sat_names) if n not in set(sat_names)]
        if not keep:
            raise ValueError("cannot drop every satellite from an ephemeris")
        return Ephemeris(times=list(self.times),
                         sat_names=[self.sat_names[i] for i in keep],
                         r_teme_km=self.r_teme_km[keep],
                         v_teme_km_s=self.v_teme_km_s[keep])


def walker_delta(n_total: int, n_planes: int, phasing_f: int,
                 inclination_deg: float, altitude_km: float,
                 epoch: datetime, name_prefix: str = "W",
                 max_epoch_age_days: float = DEFAULT_MAX_EPOCH_AGE_DAYS,
                 ) -> Constellation:
    """Build a Walker delta constellation ``i: T/P/F`` (see module docstring).

    Parameters
    ----------
    n_total : total satellites ``T``, must be a positive multiple of ``n_planes``.
    n_planes : number of planes ``P``, >= 1.
    phasing_f : inter-plane phasing ``F`` in ``0 .. n_planes - 1``.
    inclination_deg : common inclination [deg], [0, 180].
    altitude_km : circular altitude above the WGS-84 equatorial radius [km], > 0.
    epoch : element epoch (UTC) for every satellite.
    name_prefix : satellite names are ``f"{prefix}{plane:02d}-{slot:02d}"``.
    max_epoch_age_days : propagated to each :class:`Satellite`.

    Notes
    -----
    Mean motion is set from the two-body relation ``n = sqrt(mu / a^3)``; see
    the module docstring for the Kozai/Kepler mean-motion caveat and the
    measured altitude offset it produces.
    """
    if n_planes < 1:
        raise ValueError(f"n_planes must be >= 1, got {n_planes}")
    if n_total < 1:
        raise ValueError(f"n_total must be >= 1, got {n_total}")
    if n_total % n_planes != 0:
        raise ValueError(
            f"n_total ({n_total}) must be a multiple of n_planes ({n_planes})")
    if not 0 <= phasing_f <= n_planes - 1:
        raise ValueError(
            f"phasing_f must be in [0, {n_planes - 1}], got {phasing_f}")
    if not 0.0 <= inclination_deg <= 180.0:
        raise ValueError(f"inclination_deg must be in [0, 180], got {inclination_deg}")
    if altitude_km <= 0.0:
        raise ValueError(f"altitude_km must be > 0, got {altitude_km}")

    per_plane = n_total // n_planes
    a_km = WGS84_A_KM + altitude_km
    # Two-body mean motion [rad/min] -- the unit sgp4init expects.
    n_rad_min = np.sqrt(MU_EARTH_KM3_S2 / a_km ** 3) * 60.0
    epoch = to_utc(epoch)
    epoch_days = _jd_of(epoch) - _jd_of(_SGP4_EPOCH)

    sats: list[Satellite] = []
    satnum = 90000
    for p in range(n_planes):
        raan_deg = 360.0 * p / n_planes
        for s in range(per_plane):
            ma_deg = (360.0 * s / per_plane + 360.0 * phasing_f * p / n_total) % 360.0
            satrec = Satrec()
            satrec.sgp4init(
                WGS72,                       # gravity model
                "i",                         # improved mode
                satnum,                      # satellite number
                epoch_days,                  # epoch: days since 1949-12-31 00:00 UT
                0.0,                         # bstar [1/earth radii]
                0.0,                         # ndot [rev/day^2] (unused in 'i' mode)
                0.0,                         # nddot [rev/day^3] (unused in 'i' mode)
                0.0,                         # eccentricity (circular)
                0.0,                         # argument of perigee [rad]
                float(np.deg2rad(inclination_deg)),  # inclination [rad]
                float(np.deg2rad(ma_deg)),   # mean anomaly [rad]
                float(n_rad_min),            # mean motion [rad/min]
                float(np.deg2rad(raan_deg)), # RAAN [rad]
            )
            sats.append(Satellite(name=f"{name_prefix}{p:02d}-{s:02d}", satrec=satrec,
                                  max_epoch_age_days=max_epoch_age_days))
            satnum += 1
    return Constellation(satellites=sats,
                         name=f"Walker-{inclination_deg:g}:{n_total}/{n_planes}/{phasing_f}")


@dataclass(frozen=True)
class CircularOrbit:
    """Analytic two-body circular orbit, used only as an independent reference.

    Kepler's third law for a circular orbit, ``n = sqrt(mu / a^3)``
    (Vallado 2013, Ch. 1).  The orbit is specified by radius, inclination,
    RAAN and the argument of latitude at ``epoch``; the position is

        r(t) = R3(-Omega) R1(-i) R3(-u(t)) [a, 0, 0]^T,  u(t) = u0 + n (t - t0)

    in an inertial frame (here treated as TEME, consistent with SGP4 output;
    the two-body model has no frame-dependent perturbations so this is exact
    within the model).  Units: km, rad, s.  No J2, no drag -- this exists so
    that contact windows can be compared against a closed form, not to
    propagate real satellites.
    """

    radius_km: float
    inclination_deg: float
    raan_deg: float
    arg_lat0_deg: float
    epoch: datetime
    name: str = "circular"
    max_epoch_age_days: float = float("inf")

    def __post_init__(self) -> None:
        if self.radius_km <= 0.0:
            raise ValueError(f"radius_km must be > 0, got {self.radius_km}")
        if not 0.0 <= self.inclination_deg <= 180.0:
            raise ValueError(f"inclination_deg must be in [0, 180], got {self.inclination_deg}")
        if not self.name:
            raise ValueError("name must be non-empty")

    @property
    def mean_motion_rad_s(self) -> float:
        """Two-body mean motion [rad/s]."""
        return float(np.sqrt(MU_EARTH_KM3_S2 / self.radius_km ** 3))

    @property
    def period_s(self) -> float:
        """Orbital period [s]."""
        return float(2.0 * np.pi / self.mean_motion_rad_s)

    def _rotate(self, x_p: np.ndarray, y_p: np.ndarray) -> np.ndarray:
        inc = np.deg2rad(self.inclination_deg)
        raan = np.deg2rad(self.raan_deg)
        x_i = x_p * np.cos(raan) - y_p * np.cos(inc) * np.sin(raan)
        y_i = x_p * np.sin(raan) + y_p * np.cos(inc) * np.cos(raan)
        z_i = y_p * np.sin(inc)
        return np.stack([x_i, y_i, z_i], axis=1)

    def position(self, times: list[datetime]) -> np.ndarray:
        """Inertial positions [km], shape ``(len(times), 3)``."""
        t0 = to_utc(self.epoch)
        dt = np.array([(to_utc(t) - t0).total_seconds() for t in times], dtype=float)
        u = np.deg2rad(self.arg_lat0_deg) + self.mean_motion_rad_s * dt
        return self._rotate(self.radius_km * np.cos(u), self.radius_km * np.sin(u))

    def propagate(self, times: list[datetime], check_epoch: bool = False,
                  ) -> tuple[np.ndarray, np.ndarray]:
        """Positions [km] and velocities [km/s], matching :meth:`Satellite.propagate`.

        ``check_epoch`` is accepted and ignored: a two-body circular orbit has
        no epoch-validity window.  Having the same signature as
        :class:`Satellite` lets the contact finders in
        :mod:`constellink.contacts` run against this analytic orbit without a
        separate code path, which is what makes the closed-form validation in
        ``validation/validate_contact_windows.py`` a check of the shipped
        window finder rather than of a parallel implementation.
        """
        del check_epoch
        if len(times) == 0:
            return np.zeros((0, 3)), np.zeros((0, 3))
        t0 = to_utc(self.epoch)
        dt = np.array([(to_utc(t) - t0).total_seconds() for t in times], dtype=float)
        n = self.mean_motion_rad_s
        u = np.deg2rad(self.arg_lat0_deg) + n * dt
        r = self._rotate(self.radius_km * np.cos(u), self.radius_km * np.sin(u))
        v = self._rotate(-self.radius_km * n * np.sin(u),
                         self.radius_km * n * np.cos(u))
        return r, v

    def ephemeris(self, t0: datetime, t1: datetime, step_s: float) -> Ephemeris:
        """Uniform-grid :class:`Ephemeris` for this single analytic orbit."""
        times = time_grid(t0, t1, step_s)
        r, v = self.propagate(times)
        return Ephemeris(times=times, sat_names=[self.name],
                         r_teme_km=r[None, :, :], v_teme_km_s=v[None, :, :])
