"""Visibility windows: when a spacecraft is above a station's mask.

Algorithm
---------
Define the *mask margin*

    ``m(t) = elevation(t) - mask(azimuth(t))``

in degrees, with elevation measured against the local ellipsoid normal and the
mask evaluated at the instantaneous azimuth. A contact window is a maximal
interval on which ``m(t) >= 0``.

The search is a coarse uniform scan followed by bisection on each sign change.
The scan step is a declared parameter and the bisection tolerance is a declared
parameter, and neither is a guess: ``validation/validate_convergence.py``
measures how far window boundaries move as the scan step is refined from 120 s
down to 2 s and reports the step at which the boundary stops moving by more than
the reported resolution.

The coarse scan can miss a pass shorter than its step, and it can miss a pass
that rises and sets between two samples even when it is longer, if the mask is
steep in azimuth. This is a property of any sampled root search and is not
hidden: :func:`compute_windows` returns the scan step it used and
``validation/validate_convergence.py`` reports the number of windows found as a
function of step, which is the measurement that tells a reader whether their
chosen step is fine enough for their orbits.

Bisection is used rather than a secant or Newton step because ``m`` is not
smooth: the mask is only piecewise linear in azimuth, so ``m`` has corners, and
a derivative-based method can step across one. Bisection converges on the
bracket regardless, in ``ceil(log2(step / tolerance))`` evaluations.

Units: seconds after the scenario epoch, degrees, km.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .ephemeris import Scenario, Spacecraft
from .geometry import look_angles_deg
from .station import GroundStation


@dataclass(frozen=True)
class ContactWindow:
    """One maximal interval of visibility between one spacecraft and one antenna.

    Attributes
    ----------
    station : str
        Antenna name.
    spacecraft : str
        Spacecraft label.
    mission : str
        Owning mission of the spacecraft.
    rise_s : float
        Start of visibility, seconds after the scenario epoch.
    set_s : float
        End of visibility, seconds after the scenario epoch.
    max_elevation_deg : float
        Highest elevation reached inside the window, degrees, located to the
        sampling resolution named in :attr:`peak_resolution_s`.
    max_elevation_s : float
        Time of :attr:`max_elevation_deg`, seconds after the epoch.
    peak_resolution_s : float
        Spacing of the samples the peak was located on. The peak time is not
        refined beyond this and should not be read as finer.
    clipped_start : bool
        True when the spacecraft was already above the mask at the start of the
        horizon, so :attr:`rise_s` is the horizon start and not a real rise.
    clipped_end : bool
        True when the spacecraft was still above the mask at the end of the
        horizon.
    mask_limited : bool
        True when the mask at the rise or set azimuth exceeds the station's
        floor, so this boundary was set by a mask breakpoint rather than by the
        flat operational limit.
    """

    station: str
    spacecraft: str
    mission: str
    rise_s: float
    set_s: float
    max_elevation_deg: float
    max_elevation_s: float
    peak_resolution_s: float
    clipped_start: bool = False
    clipped_end: bool = False
    mask_limited: bool = False

    def __post_init__(self) -> None:
        if float(self.set_s) <= float(self.rise_s):
            raise ValueError(
                f"window for {self.spacecraft} at {self.station} has set_s "
                f"{self.set_s} not after rise_s {self.rise_s}"
            )

    @property
    def duration_s(self) -> float:
        """Length of the window, seconds."""
        return float(self.set_s) - float(self.rise_s)

    def overlaps(self, other: ContactWindow) -> bool:
        """True when the two windows share more than a single instant."""
        return self.rise_s < other.set_s and other.rise_s < self.set_s


def mask_margin_deg(
    scenario: Scenario,
    station: GroundStation,
    spacecraft: Spacecraft,
    times_s: np.ndarray,
) -> np.ndarray:
    """Elevation above the station mask at each time, degrees.

    Positive means visible. This is the function whose zeros are the window
    boundaries, and it is the single place elevation and mask are combined, so
    that the scan, the bisection and every test agree by construction about what
    visibility means.
    """
    times = np.atleast_1d(np.asarray(times_s, dtype=float))
    positions = spacecraft.ecef_km(scenario, times)
    azimuth, elevation, _rng = look_angles_deg(positions, station.ecef_km, station.sez)
    return elevation - station.mask.min_elevation_deg(azimuth)


def _bisect_roots(
    scenario: Scenario,
    station: GroundStation,
    spacecraft: Spacecraft,
    lo: np.ndarray,
    hi: np.ndarray,
    tolerance_s: float,
) -> np.ndarray:
    """Refine bracketed sign changes of the mask margin to ``tolerance_s``.

    ``lo`` and ``hi`` bracket one root each, with the margin of opposite sign at
    the two ends. All brackets are refined simultaneously so that one batched
    propagation serves every root.
    """
    lo = np.asarray(lo, dtype=float).copy()
    hi = np.asarray(hi, dtype=float).copy()
    if lo.size == 0:
        return lo
    width = float(np.max(hi - lo))
    iterations = max(1, int(math.ceil(math.log2(width / float(tolerance_s)))))
    margin_lo = mask_margin_deg(scenario, station, spacecraft, lo)
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        margin_mid = mask_margin_deg(scenario, station, spacecraft, mid)
        same_side = np.sign(margin_mid) == np.sign(margin_lo)
        lo = np.where(same_side, mid, lo)
        hi = np.where(same_side, hi, mid)
        margin_lo = np.where(same_side, margin_mid, margin_lo)
    return 0.5 * (lo + hi)


def compute_windows(
    scenario: Scenario,
    station: GroundStation,
    spacecraft: Spacecraft,
    step_s: float = 30.0,
    tolerance_s: float = 0.5,
    peak_samples: int = 65,
) -> list[ContactWindow]:
    """Visibility windows for one spacecraft at one antenna over the horizon.

    Parameters
    ----------
    scenario : Scenario
        Epoch and horizon length.
    station : GroundStation
        The antenna.
    spacecraft : Spacecraft
        The object.
    step_s : float
        Coarse scan step, seconds. Must be positive. A step larger than the
        shortest pass will miss passes; see the module docstring.
    tolerance_s : float
        Bisection tolerance on each boundary, seconds. Must be positive and
        below ``step_s``.
    peak_samples : int
        Number of samples used inside each window to locate the elevation peak.
        The returned :attr:`ContactWindow.peak_resolution_s` records the
        resulting spacing so the peak time is not over-read.

    Returns
    -------
    list of ContactWindow
        Ordered by rise time.

    Raises
    ------
    ValueError
        On a non-positive step, a tolerance not below the step, or fewer than
        three peak samples.
    """
    step_s = float(step_s)
    tolerance_s = float(tolerance_s)
    if step_s <= 0.0:
        raise ValueError(f"step_s must be positive, got {step_s}")
    if not 0.0 < tolerance_s < step_s:
        raise ValueError(f"tolerance_s must be within (0, {step_s}), got {tolerance_s}")
    if int(peak_samples) < 3:
        raise ValueError(f"peak_samples must be at least 3, got {peak_samples}")

    count = int(math.floor(scenario.duration_s / step_s)) + 1
    grid = np.minimum(np.arange(count) * step_s, scenario.duration_s)
    if grid[-1] < scenario.duration_s:
        grid = np.append(grid, scenario.duration_s)
    margin = mask_margin_deg(scenario, station, spacecraft, grid)
    above = margin >= 0.0

    rise_brackets_lo: list[float] = []
    rise_brackets_hi: list[float] = []
    set_brackets_lo: list[float] = []
    set_brackets_hi: list[float] = []
    transitions = np.flatnonzero(above[:-1] != above[1:])
    for index in transitions:
        if above[index + 1]:
            rise_brackets_lo.append(float(grid[index]))
            rise_brackets_hi.append(float(grid[index + 1]))
        else:
            set_brackets_lo.append(float(grid[index]))
            set_brackets_hi.append(float(grid[index + 1]))

    rises = list(
        _bisect_roots(
            scenario,
            station,
            spacecraft,
            np.array(rise_brackets_lo),
            np.array(rise_brackets_hi),
            tolerance_s,
        )
    )
    sets = list(
        _bisect_roots(
            scenario,
            station,
            spacecraft,
            np.array(set_brackets_lo),
            np.array(set_brackets_hi),
            tolerance_s,
        )
    )

    clipped_start = bool(above[0])
    clipped_end = bool(above[-1])
    if clipped_start:
        rises.insert(0, 0.0)
    if clipped_end:
        sets.append(float(scenario.duration_s))
    if len(rises) != len(sets):
        raise RuntimeError(
            f"visibility scan for {spacecraft.name} at {station.name} produced "
            f"{len(rises)} rises and {len(sets)} sets; the scan step {step_s} s "
            "is too coarse for this orbit"
        )

    windows: list[ContactWindow] = []
    for index, (rise, close) in enumerate(zip(rises, sets, strict=True)):
        if close - rise <= 0.0:
            continue
        samples = np.linspace(rise, close, int(peak_samples))
        positions = spacecraft.ecef_km(scenario, samples)
        azimuth, elevation, _rng = look_angles_deg(
            positions, station.ecef_km, station.sez
        )
        peak = int(np.argmax(elevation))
        boundary_azimuth = np.array([azimuth[0], azimuth[-1]])
        boundary_mask = station.mask.min_elevation_deg(boundary_azimuth)
        windows.append(
            ContactWindow(
                station=station.name,
                spacecraft=spacecraft.name,
                mission=spacecraft.mission,
                rise_s=float(rise),
                set_s=float(close),
                max_elevation_deg=float(elevation[peak]),
                max_elevation_s=float(samples[peak]),
                peak_resolution_s=float(samples[1] - samples[0]),
                clipped_start=bool(clipped_start and index == 0),
                clipped_end=bool(clipped_end and index == len(rises) - 1),
                mask_limited=bool(
                    np.any(boundary_mask > station.mask.floor_deg + 1e-9)
                ),
            )
        )
    return windows


def all_windows(
    scenario: Scenario,
    stations: tuple[GroundStation, ...],
    fleet: tuple[Spacecraft, ...],
    step_s: float = 30.0,
    tolerance_s: float = 0.5,
) -> list[ContactWindow]:
    """Every window for every spacecraft at every antenna, ordered by rise time."""
    windows: list[ContactWindow] = []
    for station in stations:
        for spacecraft in fleet:
            windows.extend(
                compute_windows(scenario, station, spacecraft, step_s, tolerance_s)
            )
    windows.sort(key=lambda w: (w.rise_s, w.station, w.spacecraft))
    return windows


def visible_seconds(windows: list[ContactWindow]) -> float:
    """Total window duration, seconds. Not de-overlapped across stations."""
    return float(sum(w.duration_s for w in windows))


def supply_seconds_by_station(windows: list[ContactWindow]) -> dict[str, float]:
    """Union of window coverage per antenna, seconds.

    This is the *supply* side of the contention ratio and must be a union, not
    a sum: an antenna cannot serve two spacecraft at once, so two overlapping
    windows supply only the time they jointly cover, counted once.
    """
    by_station: dict[str, list[tuple[float, float]]] = {}
    for window in windows:
        by_station.setdefault(window.station, []).append((window.rise_s, window.set_s))
    supply: dict[str, float] = {}
    for station, intervals in by_station.items():
        merged: list[list[float]] = []
        for start, end in sorted(intervals):
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        supply[station] = float(sum(end - start for start, end in merged))
    return supply
