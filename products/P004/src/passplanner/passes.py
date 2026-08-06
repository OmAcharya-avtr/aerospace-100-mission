"""Pass finding: coarse elevation scan + bisection refinement of rise/set.

Method
------
The station-relative elevation e(t) is sampled on a coarse grid (default 30 s).
Sign changes of e(t) - mask bracket rise/set events, which are refined by
bisection to ``refine_tol_s`` (default 0.05 s).  The culmination is located by
a bounded scalar minimisation of -e(t) inside the pass (elevation is unimodal
over a single pass for near-circular LEO orbits; for safety the interval is
pre-scanned and the search bracketed around the best coarse sample).

Limitations: passes whose total duration above the mask is shorter than the
coarse step can be missed; choose ``coarse_step_s`` at most half the shortest
expected pass duration (LEO passes above a 5-30 deg mask last several
minutes, so the 30 s default is conservative).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import minimize_scalar
from sgp4.api import Satrec

from .frames import datetime_to_jd, ecef_to_azel, teme_to_ecef, to_utc
from .stations import Station


@dataclass(frozen=True)
class TLE:
    """A two-line element set with a display name.

    ``line1``/``line2`` must be standard 69-character TLE lines (checksummed).
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
                raise ValueError(f"TLE line {i} for '{self.name}' fails its checksum")

    def to_satrec(self) -> Satrec:
        """Parse into an ``sgp4`` Satrec propagator object."""
        return Satrec.twoline2rv(self.line1, self.line2)


def _tle_checksum(line: str) -> int:
    """Modulo-10 TLE checksum: digits count as value, '-' counts as 1."""
    total = 0
    for ch in line[:68]:
        if ch.isdigit():
            total += int(ch)
        elif ch == "-":
            total += 1
    return total % 10


@dataclass(frozen=True)
class Pass:
    """A single station contact opportunity.

    Times are timezone-aware UTC datetimes.  ``max_elevation_deg`` is the
    culmination elevation [deg]; ``duration_s`` is (t_set - t_rise) [s].
    """

    satellite: str
    station: Station
    t_rise: datetime
    t_set: datetime
    t_culminate: datetime
    max_elevation_deg: float

    @property
    def duration_s(self) -> float:
        """Pass duration above the elevation mask [s]."""
        return (self.t_set - self.t_rise).total_seconds()

    def overlaps(self, other: "Pass", setup_time_s: float = 0.0) -> bool:
        """True if the two time intervals (padded by setup_time_s) intersect."""
        pad = timedelta(seconds=setup_time_s)
        return self.t_rise < other.t_set + pad and other.t_rise < self.t_set + pad


ElevationFn = Callable[[float], float]
PositionFn = Callable[[datetime], np.ndarray]


def find_passes(tle: TLE | Sequence[str] | Satrec,
                station: Station,
                t0: datetime,
                t1: datetime,
                min_elev_deg: float | None = None,
                coarse_step_s: float = 30.0,
                refine_tol_s: float = 0.05,
                satellite_name: str | None = None) -> list[Pass]:
    """Find all passes of a TLE-defined satellite over ``station`` in [t0, t1].

    Parameters
    ----------
    tle : :class:`TLE`, (line1, line2) pair, or a pre-built ``Satrec``.
    station : ground station (elevation mask taken from
        ``station.min_elevation_deg`` unless ``min_elev_deg`` is given).
    t0, t1 : UTC window start/end (naive datetimes are treated as UTC).
    min_elev_deg : optional mask override [deg], in [0, 90).
    coarse_step_s : coarse scan step [s], > 0.
    refine_tol_s : bisection tolerance on rise/set times [s], > 0.

    Returns passes sorted by rise time.  SGP4 propagation errors raise
    ``ValueError`` (e.g. decayed orbits far from epoch).
    """
    if isinstance(tle, TLE):
        sat = tle.to_satrec()
        name = satellite_name or tle.name
    elif isinstance(tle, Satrec):
        sat = tle
        name = satellite_name or f"NORAD-{tle.satnum}"
    else:
        lines = list(tle)
        if len(lines) != 2:
            raise ValueError("tle sequence must contain exactly (line1, line2)")
        parsed = TLE(satellite_name or "SAT", lines[0], lines[1])
        sat = parsed.to_satrec()
        name = parsed.name

    def position_ecef(t: datetime) -> np.ndarray:
        jd, fr = datetime_to_jd(t)
        err, r_teme, _v = sat.sgp4(jd, fr)
        if err != 0:
            raise ValueError(f"SGP4 propagation failed for '{name}' at {t.isoformat()} "
                             f"(sgp4 error code {err})")
        return teme_to_ecef(np.array(r_teme), jd, fr)

    return find_passes_from_position_fn(
        position_ecef, station, t0, t1,
        min_elev_deg=min_elev_deg, coarse_step_s=coarse_step_s,
        refine_tol_s=refine_tol_s, satellite_name=name)


def find_passes_from_position_fn(position_ecef_fn: PositionFn,
                                 station: Station,
                                 t0: datetime,
                                 t1: datetime,
                                 min_elev_deg: float | None = None,
                                 coarse_step_s: float = 30.0,
                                 refine_tol_s: float = 0.05,
                                 satellite_name: str = "SAT") -> list[Pass]:
    """Pass finder for an arbitrary ECEF ephemeris ``t -> r_ecef [km]``.

    This is the generic core used by :func:`find_passes`; it also allows
    analytic (e.g. Keplerian circular-orbit) test ephemerides without a TLE.
    See module docstring for the coarse-scan + bisection method.
    """
    t0 = to_utc(t0)
    t1 = to_utc(t1)
    if t1 <= t0:
        raise ValueError(f"t1 ({t1.isoformat()}) must be after t0 ({t0.isoformat()})")
    if coarse_step_s <= 0:
        raise ValueError(f"coarse_step_s must be > 0, got {coarse_step_s}")
    if refine_tol_s <= 0:
        raise ValueError(f"refine_tol_s must be > 0, got {refine_tol_s}")
    mask = station.min_elevation_deg if min_elev_deg is None else float(min_elev_deg)
    if not 0.0 <= mask < 90.0:
        raise ValueError(f"min_elev_deg must be in [0, 90) deg, got {mask}")

    def elev(offset_s: float) -> float:
        t = t0 + timedelta(seconds=offset_s)
        _az, el, _rng = ecef_to_azel(position_ecef_fn(t),
                                     station.lat_deg, station.lon_deg, station.alt_km)
        return el - mask

    window_s = (t1 - t0).total_seconds()
    n_steps = int(np.ceil(window_s / coarse_step_s)) + 1
    grid = np.minimum(np.arange(n_steps) * coarse_step_s, window_s)
    vals = np.array([elev(s) for s in grid])

    # Bracket rise/set events at sign changes of (elevation - mask).
    events: list[tuple[float, str]] = []
    for i in range(len(grid) - 1):
        if vals[i] < 0.0 <= vals[i + 1]:
            events.append((_bisect(elev, grid[i], grid[i + 1], refine_tol_s), "rise"))
        elif vals[i] >= 0.0 > vals[i + 1]:
            events.append((_bisect(elev, grid[i], grid[i + 1], refine_tol_s), "set"))

    # Assemble [rise, set] intervals, handling passes clipped by the window.
    intervals: list[tuple[float, float]] = []
    pending_rise: float | None = 0.0 if vals[0] >= 0.0 else None
    for time_s, kind in events:
        if kind == "rise":
            pending_rise = time_s
        elif pending_rise is not None:
            intervals.append((pending_rise, time_s))
            pending_rise = None
    if pending_rise is not None:
        intervals.append((pending_rise, window_s))

    passes = []
    for rise_s, set_s in intervals:
        if set_s <= rise_s:
            continue
        culm_s, max_el = _refine_culmination(elev, rise_s, set_s)
        passes.append(Pass(
            satellite=satellite_name,
            station=station,
            t_rise=t0 + timedelta(seconds=rise_s),
            t_set=t0 + timedelta(seconds=set_s),
            t_culminate=t0 + timedelta(seconds=culm_s),
            max_elevation_deg=max_el + mask,
        ))
    return sorted(passes, key=lambda p: p.t_rise)


def _bisect(f: ElevationFn, lo: float, hi: float, tol_s: float) -> float:
    """Bisection root of f on [lo, hi] (f(lo), f(hi) of opposite sign) to tol_s."""
    f_lo = f(lo)
    while hi - lo > tol_s:
        mid = 0.5 * (lo + hi)
        f_mid = f(mid)
        if (f_lo < 0.0) == (f_mid < 0.0):
            lo, f_lo = mid, f_mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _refine_culmination(f: ElevationFn, rise_s: float, set_s: float,
                        n_scan: int = 64) -> tuple[float, float]:
    """Locate max of f on [rise_s, set_s]: coarse scan then bounded refinement."""
    ts = np.linspace(rise_s, set_s, n_scan)
    es = np.array([f(s) for s in ts])
    i_best = int(np.argmax(es))
    lo = ts[max(i_best - 1, 0)]
    hi = ts[min(i_best + 1, n_scan - 1)]
    if hi <= lo:
        return float(ts[i_best]), float(es[i_best])
    res = minimize_scalar(lambda s: -f(s), bounds=(lo, hi), method="bounded",
                          options={"xatol": 0.01})
    if -res.fun >= es[i_best]:
        return float(res.x), float(-res.fun)
    return float(ts[i_best]), float(es[i_best])
