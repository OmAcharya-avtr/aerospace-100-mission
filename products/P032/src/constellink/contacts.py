"""Contact-window computation for inter-satellite and satellite-ground links.

Method
------
A link-specific scalar *access function* ``f(t)`` is defined so that the link
is open exactly when ``f(t) >= 0``:

* inter-satellite: ``f(t) = min(d_limb(t) - r_block, R_max - rho(t))`` where
  ``d_limb`` is the segment's closest approach to the Earth's centre, ``rho``
  the inter-satellite range, and ``R_max`` the maximum supported range.  Both
  terms are in km, so the minimum is a well-defined margin;
* satellite-ground: ``f(t) = min(el(t) - el_mask, R_max - rho(t))`` with the
  elevation term in degrees and the range term in km.  The two are not
  commensurate, so the combined function is used only for sign changes, never
  as a metric.

Windows are found by sampling ``f`` on a uniform grid (the ephemeris grid),
bracketing sign changes, and refining each crossing by bisection to
``refine_tol_s``.  A window already open at ``t0`` or still open at the end of
the horizon is clipped to the horizon and flagged ``clipped_start`` /
``clipped_end``.

Sampling limitation (stated, not hidden): a window shorter than the grid step
can be missed entirely, and a brief closure inside a window can be missed the
same way.  The grid step must be at most half the shortest window of interest.
For a 24-satellite Walker shell at 550 km, intra-plane ISLs are permanently
closed-form clear and inter-plane windows last minutes, so a 30-60 s step is
adequate; ground passes above a 10 deg mask last several minutes.
``contact_windows_isl`` reports the grid step it used in every window so this
is traceable in the output.

Bisection refines the time of the zero crossing of the *interpolated* access
function, evaluated by re-propagating -- not by interpolating the sampled
values -- so the refined edge is accurate to ``refine_tol_s`` against the
propagator, independently of the grid step.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from .constellation import Ephemeris, GroundStation, Satellite
from .frames import (
    SPEED_OF_LIGHT_KM_S,
    WGS84_A_KM,
    datetime_to_jd,
    ecef_to_azel,
    teme_to_ecef,
    to_utc,
)
from .geometry import DEFAULT_GRAZING_ALTITUDE_KM, segment_min_radius

__all__ = [
    "ContactWindow",
    "contact_windows_isl",
    "contact_windows_ground",
    "all_contact_windows",
]

DEFAULT_REFINE_TOL_S = 0.05


@dataclass(frozen=True)
class ContactWindow:
    """One interval during which a link is geometrically available.

    Attributes
    ----------
    node_a, node_b : endpoint names (satellite or ground-station names).
    kind : ``"isl"`` or ``"ground"``.
    t_open, t_close : UTC interval edges.
    min_range_km, max_range_km : range extremes over the window [km].
    max_elevation_deg : culmination elevation for a ground link [deg], or
        ``None`` for an ISL.
    grid_step_s : the sampling step used to detect the window [s].
    clipped_start, clipped_end : the window was already open at the start of
        the horizon / still open at its end.
    """

    node_a: str
    node_b: str
    kind: str
    t_open: datetime
    t_close: datetime
    min_range_km: float
    max_range_km: float
    max_elevation_deg: float | None
    grid_step_s: float
    clipped_start: bool = False
    clipped_end: bool = False

    @property
    def duration_s(self) -> float:
        """Window duration [s]."""
        return (self.t_close - self.t_open).total_seconds()

    @property
    def mid_time(self) -> datetime:
        """Mid-window UTC time."""
        return self.t_open + (self.t_close - self.t_open) / 2

    @property
    def min_one_way_delay_s(self) -> float:
        """Shortest one-way propagation delay over the window [s]."""
        return self.min_range_km / SPEED_OF_LIGHT_KM_S

    def __post_init__(self) -> None:
        if self.kind not in ("isl", "ground"):
            raise ValueError(f"kind must be 'isl' or 'ground', got {self.kind!r}")
        if self.t_close <= self.t_open:
            raise ValueError(
                f"window {self.node_a}-{self.node_b}: t_close must be after t_open")


def _bisect_crossing(f: Callable[[datetime], float],
                     t_a: datetime, t_b: datetime, tol_s: float) -> datetime:
    """Bisect the sign change of ``f`` between ``t_a`` and ``t_b`` to ``tol_s`` seconds.

    The two bounds may be given in either chronological order; they are sorted
    internally.  ``f`` must have opposite signs at the two bounds.
    """
    t_lo, t_hi = (t_a, t_b) if t_b > t_a else (t_b, t_a)
    f_lo = f(t_lo)
    for _ in range(200):
        span = (t_hi - t_lo).total_seconds()
        if span <= tol_s:
            break
        t_mid = t_lo + timedelta(seconds=span / 2.0)
        f_mid = f(t_mid)
        if (f_mid >= 0.0) == (f_lo >= 0.0):
            t_lo, f_lo = t_mid, f_mid
        else:
            t_hi = t_mid
    return t_lo + (t_hi - t_lo) / 2


def _windows_from_signs(times: list[datetime], values: np.ndarray,
                        f: Callable[[datetime], float], tol_s: float,
                        ) -> list[tuple[datetime, datetime, bool, bool, int, int]]:
    """Convert sampled access values into refined open intervals.

    Returns tuples ``(t_open, t_close, clipped_start, clipped_end, i0, i1)``
    where ``i0:i1+1`` are the sample indices inside the window (used for the
    range/elevation extremes).
    """
    open_mask = values >= 0.0
    out: list[tuple[datetime, datetime, bool, bool, int, int]] = []
    n = len(times)
    k = 0
    while k < n:
        if not open_mask[k]:
            k += 1
            continue
        i0 = k
        while k + 1 < n and open_mask[k + 1]:
            k += 1
        i1 = k
        clipped_start = i0 == 0
        clipped_end = i1 == n - 1
        t_open = times[i0] if clipped_start else _bisect_crossing(f, times[i0 - 1], times[i0],
                                                                  tol_s)
        t_close = times[i1] if clipped_end else _bisect_crossing(f, times[i1 + 1], times[i1],
                                                                 tol_s)
        if t_close > t_open:
            out.append((t_open, t_close, clipped_start, clipped_end, i0, i1))
        k += 1
    return out


def contact_windows_isl(eph: Ephemeris, sat_a: Satellite, sat_b: Satellite,
                        max_range_km: float = 5000.0,
                        grazing_altitude_km: float = DEFAULT_GRAZING_ALTITUDE_KM,
                        earth_radius_km: float = WGS84_A_KM,
                        refine_tol_s: float = DEFAULT_REFINE_TOL_S,
                        ) -> list[ContactWindow]:
    """Inter-satellite contact windows between two satellites over an ephemeris.

    Parameters
    ----------
    eph : ephemeris containing both satellites (provides the sampling grid).
    sat_a, sat_b : the :class:`Satellite` objects, needed for bisection
        refinement off the grid.
    max_range_km : maximum supported ISL range [km], > 0.  A terminal-design
        limit, not a physical one.
    grazing_altitude_km : Earth-limb clearance margin [km], >= 0.
    earth_radius_km : blocking sphere radius [km], > 0.
    refine_tol_s : bisection tolerance on window edges [s], > 0.
    """
    if max_range_km <= 0.0:
        raise ValueError(f"max_range_km must be > 0, got {max_range_km}")
    if refine_tol_s <= 0.0:
        raise ValueError(f"refine_tol_s must be > 0, got {refine_tol_s}")
    if sat_a.name == sat_b.name:
        raise ValueError("an ISL needs two distinct satellites")
    ia, ib = eph.index(sat_a.name), eph.index(sat_b.name)
    r_a, r_b = eph.r_teme_km[ia], eph.r_teme_km[ib]
    r_block = earth_radius_km + grazing_altitude_km
    rng = np.linalg.norm(r_b - r_a, axis=1)
    limb = segment_min_radius(r_a, r_b)
    values = np.minimum(limb - r_block, max_range_km - rng)

    def access(t: datetime) -> float:
        pa, _ = sat_a.propagate([t], check_epoch=False)
        pb, _ = sat_b.propagate([t], check_epoch=False)
        d = float(np.linalg.norm(pb[0] - pa[0]))
        lb = float(segment_min_radius(pa[0], pb[0])[0])
        return min(lb - r_block, max_range_km - d)

    step = eph.step_s
    out = []
    for t_open, t_close, cs, ce, i0, i1 in _windows_from_signs(
            eph.times, values, access, refine_tol_s):
        seg = rng[i0:i1 + 1]
        out.append(ContactWindow(
            node_a=sat_a.name, node_b=sat_b.name, kind="isl",
            t_open=t_open, t_close=t_close,
            min_range_km=float(seg.min()), max_range_km=float(seg.max()),
            max_elevation_deg=None, grid_step_s=step,
            clipped_start=cs, clipped_end=ce))
    return out


def contact_windows_ground(eph: Ephemeris, sat: Satellite, station: GroundStation,
                           max_range_km: float = 3000.0,
                           min_elevation_deg: float | None = None,
                           refine_tol_s: float = DEFAULT_REFINE_TOL_S,
                           ) -> list[ContactWindow]:
    """Satellite-to-ground contact windows for one satellite and one station.

    ``min_elevation_deg`` overrides ``station.min_elevation_deg`` when given.
    ``max_range_km`` is the terminal's maximum supported slant range [km].
    """
    if max_range_km <= 0.0:
        raise ValueError(f"max_range_km must be > 0, got {max_range_km}")
    mask = station.min_elevation_deg if min_elevation_deg is None else float(min_elevation_deg)
    if not 0.0 <= mask < 90.0:
        raise ValueError(f"elevation mask must be in [0, 90), got {mask}")
    i = eph.index(sat.name)
    n_t = len(eph.times)
    el = np.empty(n_t)
    rng = np.empty(n_t)
    for k, t in enumerate(eph.times):
        jd, fr = datetime_to_jd(t)
        r_ecef = teme_to_ecef(eph.r_teme_km[i, k], jd, fr)
        _, el[k], rng[k] = ecef_to_azel(r_ecef, station.lat_deg, station.lon_deg,
                                        station.alt_km)
    values = np.minimum(el - mask, max_range_km - rng)

    def access(t: datetime) -> float:
        p, _ = sat.propagate([t], check_epoch=False)
        jd, fr = datetime_to_jd(t)
        r_ecef = teme_to_ecef(p[0], jd, fr)
        _, e, d = ecef_to_azel(r_ecef, station.lat_deg, station.lon_deg, station.alt_km)
        return min(e - mask, max_range_km - d)

    step = eph.step_s
    out = []
    for t_open, t_close, cs, ce, i0, i1 in _windows_from_signs(
            eph.times, values, access, refine_tol_s):
        out.append(ContactWindow(
            node_a=sat.name, node_b=station.name, kind="ground",
            t_open=t_open, t_close=t_close,
            min_range_km=float(rng[i0:i1 + 1].min()),
            max_range_km=float(rng[i0:i1 + 1].max()),
            max_elevation_deg=float(el[i0:i1 + 1].max()), grid_step_s=step,
            clipped_start=cs, clipped_end=ce))
    return out


def all_contact_windows(eph: Ephemeris, satellites: list[Satellite],
                        stations: list[GroundStation] | None = None,
                        isl_max_range_km: float = 5000.0,
                        ground_max_range_km: float = 3000.0,
                        grazing_altitude_km: float = DEFAULT_GRAZING_ALTITUDE_KM,
                        refine_tol_s: float = DEFAULT_REFINE_TOL_S,
                        ) -> list[ContactWindow]:
    """Every ISL and ground contact window over the ephemeris horizon.

    Cost is ``O(n_sat^2 n_t)`` for the ISL scan plus ``O(n_sat n_gs n_t)`` for
    the ground scan, so it is sized for tens of satellites over hours, not
    hundreds over weeks.  See the README compute budget.
    """
    by_name = {s.name: s for s in satellites}
    missing = [n for n in eph.sat_names if n not in by_name]
    if missing:
        raise KeyError(f"ephemeris contains satellites not in the list: {missing}")
    out: list[ContactWindow] = []
    names = list(eph.sat_names)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            out.extend(contact_windows_isl(
                eph, by_name[names[i]], by_name[names[j]],
                max_range_km=isl_max_range_km,
                grazing_altitude_km=grazing_altitude_km,
                refine_tol_s=refine_tol_s))
    for station in (stations or []):
        for n in names:
            out.extend(contact_windows_ground(
                eph, by_name[n], station, max_range_km=ground_max_range_km,
                refine_tol_s=refine_tol_s))
    out.sort(key=lambda w: (to_utc(w.t_open), w.node_a, w.node_b))
    return out
