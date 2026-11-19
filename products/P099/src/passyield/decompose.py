"""Per-pass yield accounting: nominal, capacity, realised, and the shortfall.

THE IDENTITY THIS MODULE IMPLEMENTS
-----------------------------------
Let :math:`W=[0,T]` be the geometric contact window, mask crossing to mask
crossing, and :math:`r(t)` the achievable rate from :mod:`passyield.link`
evaluated on the geometry from :mod:`passyield.geometry`. Define

.. math::
   Y_{nom}  &= T_{nom} R_{nom}                      &&\\text{what the schedule books}\\\\
   Y_{cap}  &= \\int_W r(t)\\,dt                      &&\\text{what a perfect link delivers}\\\\
   Y_{real} &= \\int_A r(t)\\,dt,\\quad A \\subseteq W  &&\\text{what the record accounts for}

with :math:`A` the **active** set: the window less the acquisition interval at
its start, less every re-synchronisation interval, less every idle interval.
Then the headline shortfall splits as

.. math::
   S = Y_{nom} - Y_{real}
     = \\underbrace{(Y_{nom}-Y_{cap})}_{\\text{geometry}}
     + \\underbrace{\\int_{acq} r}_{\\text{acquisition}}
     + \\underbrace{\\int_{resync} r}_{\\text{re-sync}}
     + \\underbrace{\\int_{idle} r}_{\\text{idle}}
     + \\rho

and :math:`\\rho` is **zero in exact arithmetic**. It is not zero here, and
this module reports it rather than assuming it away, for a reason that is
structural and not a bug: each term on the right is integrated **on its own
quadrature grid over its own interval**, and :math:`Y_{real}` is integrated on
grids over the active intervals. Four composite-Simpson sums over different
partitions of the same window do not telescope. :math:`\\rho` is therefore a
measurement of the quadrature, it is reported on every
:class:`Decomposition`, and ``validation/validate_decomposition.py`` publishes
its distribution over the whole committed campaign against the Simpson error
bound below.

That is deliberate. Computing the losses and then *defining* the realised
yield as the capacity minus them would force :math:`\\rho=0` by construction
and the identity would be untestable. Here it is testable, Hypothesis tests
it, and the residual has a number.

THE QUADRATURE AND ITS ERROR
----------------------------
Composite Simpson's rule on each interval with
:data:`DEFAULT_NODES_PER_INTERVAL` nodes. The classical error bound is

.. math:: |E| \\le \\frac{(b-a) h^4}{180} \\max_{[a,b]} |r^{(4)}(t)|

with :math:`h=(b-a)/(n-1)` (see any numerical-analysis text; for instance
Burden & Faires, "Numerical Analysis", composite Simpson error theorem).
:func:`simpson_error_scale` returns the :math:`(b-a)h^4/180` factor so that a
caller or a validation script can state the bound in Mbit once it has a
bound on :math:`|r^{(4)}|`, which is estimated numerically rather than
asserted: ``validation/validate_decomposition.py`` differentiates the rate
profile four times by finite differences over the campaign and reports the
maximum it found.

WHAT HAPPENS AT THE EDGES, EXPLICITLY
-------------------------------------
Each of these returns or raises something documented. None of them produces a
silent negative yield, which is the failure mode a yield accountant is for.

- **A pass that never rises above the mask.** There is no window, so
  :func:`passyield.geometry.find_passes` returns nothing and there is nothing
  to decompose. :func:`decompose_pass` raises :class:`ValueError` if handed a
  window whose peak elevation does not clear the mask.
- **A zero-duration pass.** :func:`decompose_pass` raises
  :class:`ValueError`. A zero-length window has no booked volume and no
  realised volume, so every ratio the product reports would be 0/0.
- **Acquisition longer than the whole pass.** The acquisition interval is
  clipped to the window, ``acquisition_truncated`` is set True on the result,
  the realised yield is exactly 0.0 Mbit and the acquisition term carries the
  entire capacity. The shortfall is then the whole booked volume, which is
  the correct accounting answer, and the flag tells the reader the input was
  out of range rather than the link being bad.
- **Overlapping loss intervals.** Loss intervals are clipped to the window
  and then resolved by a declared priority -- acquisition, then
  re-synchronisation, then idle -- so that every instant of the window is
  charged to at most one term. Without that, a re-sync inside an idle period
  would be charged twice and the decomposition would overstate the shortfall.
  :func:`normalise_intervals` is the implementation and it is tested directly.

UNITS
-----
Times seconds, rates Mbit/s, volumes **Mbit**.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sgp4.api import Satrec

from .geometry import GroundStation, PassWindow, elevation_profile
from .link import DEFAULT_BUDGET, LinkBudget, rate_mbit_s

#: Simpson nodes per interval. Odd, so the number of sub-intervals is even as
#: the composite rule requires.
DEFAULT_NODES_PER_INTERVAL = 33

#: Nodes used for the whole-window capacity integral. Larger, because the
#: window is the longest interval and its fourth derivative is the one the
#: error bound is dominated by.
DEFAULT_NODES_PER_WINDOW = 513

#: Largest elevation shortfall below the mask, degrees, tolerated at a window
#: endpoint before :func:`rate_at_offsets` raises.
#:
#: The window endpoints are bisection roots of ``elevation - mask``, so the
#: evaluated elevation there lands a rounding either side of the mask, and the
#: ITU-R P.618 model in :mod:`passyield.link` refuses anything below 10
#: degrees. The value is therefore clamped up to the mask, and the clamp is
#: bounded by arithmetic rather than by taste:
#:
#: SGP4 is called with a Julian date near 2.454e6, whose double-precision
#: resolution is ``2**-52 * 2**22 = 4.8e-10`` days, which is ``4.1e-5`` s. The
#: elevation rate of this element set at the mask is below 0.2 deg/s, so the
#: representable resolution of the rise instant is worth under ``1e-5``
#: degrees of elevation. 1e-4 degrees leaves an order of magnitude of margin
#: and still fails a window that is wrong by as little as a millisecond.
#:
#: What the clamp costs: the rate gradient at a 10 degree mask is about
#: 3 Mbit/s per degree for this budget, so 1e-4 degrees is 3e-4 Mbit/s, which
#: is 1e-5 per cent of the rate ceiling. ``validation/validate_geometry.py``
#: measures the gradient and ``validation/validate_decomposition.py`` reports
#: the largest clamp actually applied over the committed campaign.
MASK_CLAMP_TOLERANCE_DEG = 1.0e-4


@dataclass(frozen=True)
class PassEvents:
    """The operator-side timeline of one pass, in seconds from mask rise.

    Attributes
    ----------
    acquisition_s:
        Time from mask rise until the link carries user data: antenna slew
        and pointing, carrier and symbol acquisition, frame synchronisation.
        Always at the start of the window, which is why it is a scalar and not
        an interval -- the geometry makes it the lowest-rate part of the pass
        and that asymmetry is one of the things the product measures.
    resync_intervals_s:
        ``((start, duration), ...)``, each a loss of lock and the
        re-acquisition that follows it.
    idle_intervals_s:
        ``((start, duration), ...)``: link up, no user data moving. Protocol
        turnaround, retransmission stalls, buffer underrun, teardown.
    """

    acquisition_s: float
    resync_intervals_s: tuple[tuple[float, float], ...] = ()
    idle_intervals_s: tuple[tuple[float, float], ...] = ()

    def __post_init__(self) -> None:
        if not np.isfinite(self.acquisition_s) or self.acquisition_s < 0.0:
            raise ValueError(
                f"acquisition_s must be a finite non-negative number of seconds, "
                f"got {self.acquisition_s!r}"
            )
        for label, intervals in (
            ("resync_intervals_s", self.resync_intervals_s),
            ("idle_intervals_s", self.idle_intervals_s),
        ):
            for start, duration in intervals:
                if not np.isfinite(start) or start < 0.0:
                    raise ValueError(
                        f"{label} start offsets must be finite and non-negative, got {start!r}"
                    )
                if not np.isfinite(duration) or duration < 0.0:
                    raise ValueError(
                        f"{label} durations must be finite and non-negative, got {duration!r}"
                    )

    @property
    def resync_total_s(self) -> float:
        """Sum of the declared re-sync durations, seconds, before clipping."""
        return float(sum(duration for _start, duration in self.resync_intervals_s))

    @property
    def idle_total_s(self) -> float:
        """Sum of the declared idle durations, seconds, before clipping."""
        return float(sum(duration for _start, duration in self.idle_intervals_s))


@dataclass(frozen=True)
class Decomposition:
    """The shortfall of one pass and its additive parts, all in Mbit.

    Every field is a measured number from the quadrature described in the
    module docstring. ``residual_bits`` is the identity's closing error and is
    reported, not assumed.
    """

    station: str
    nominal_bits: float
    capacity_bits: float
    realised_bits: float
    shortfall_bits: float
    geometry_bits: float
    acquisition_bits: float
    resync_bits: float
    idle_bits: float
    residual_bits: float
    window_duration_s: float
    active_duration_s: float
    acquisition_truncated: bool
    max_mask_clamp_deg: float
    nodes_per_interval: int = DEFAULT_NODES_PER_INTERVAL
    nodes_per_window: int = DEFAULT_NODES_PER_WINDOW

    @property
    def decomposed_bits(self) -> float:
        """Sum of the four named terms, Mbit. Equals the shortfall up to rho."""
        return self.geometry_bits + self.acquisition_bits + self.resync_bits + self.idle_bits

    @property
    def realised_fraction(self) -> float:
        """``Y_real / Y_cap``, dimensionless, in ``[0, 1]``.

        The learned component's target. Normalising by capacity rather than by
        the booked volume makes the target a property of the operator's
        conduct of the pass with the geometry divided out, which is what makes
        it transferable to a station the model never saw.
        """
        if self.capacity_bits <= 0.0:
            raise ValueError(
                "realised_fraction is undefined for a window of zero capacity; "
                f"capacity_bits={self.capacity_bits!r}"
            )
        return self.realised_bits / self.capacity_bits


def simpson_error_scale(interval_s: float, nodes: int) -> float:
    """The :math:`(b-a)h^4/180` factor of the composite Simpson error bound.

    Multiply by a bound on :math:`\\max|r^{(4)}|` in Mbit/s per s^4 to get a
    bound on the quadrature error in Mbit. Returned separately from any
    derivative estimate so that a validation script supplies the derivative it
    measured instead of this module asserting one.
    """
    if interval_s < 0.0:
        raise ValueError(f"interval_s must be non-negative, got {interval_s!r}")
    if nodes < 3 or nodes % 2 == 0:
        raise ValueError(f"nodes must be an odd integer of at least 3, got {nodes!r}")
    step = interval_s / (nodes - 1)
    return interval_s * step**4 / 180.0


def normalise_intervals(
    window_duration_s: float,
    acquisition_s: float,
    resync_intervals_s: tuple[tuple[float, float], ...],
    idle_intervals_s: tuple[tuple[float, float], ...],
) -> tuple[list[tuple[float, float]], list[tuple[float, float]], list[tuple[float, float]], bool]:
    """Clip and disjointify the loss intervals under the declared priority.

    Acquisition first, then re-synchronisation, then idle: an instant already
    charged to a higher-priority term is removed from the lower-priority one,
    so the three returned lists are pairwise disjoint and each is internally
    disjoint and sorted. Without this a re-sync nested inside an idle period
    would be charged twice and the decomposition would overstate the
    shortfall -- a failure that makes the product look better at measuring
    shortfall than it is.

    Returns
    -------
    tuple
        ``(acquisition, resync, idle, acquisition_truncated)`` where the first
        three are lists of ``(start, end)`` in seconds from mask rise and the
        flag says whether the acquisition interval was clipped by the end of
        the window.
    """
    if window_duration_s <= 0.0:
        raise ValueError(f"window_duration_s must be positive, got {window_duration_s!r}")
    truncated = bool(acquisition_s > window_duration_s)
    acquisition = _clip([(0.0, min(acquisition_s, window_duration_s))], window_duration_s)
    resync = _subtract(
        _merge(_clip([(s, s + d) for s, d in resync_intervals_s], window_duration_s)), acquisition
    )
    idle = _subtract(
        _subtract(
            _merge(_clip([(s, s + d) for s, d in idle_intervals_s], window_duration_s)),
            acquisition,
        ),
        resync,
    )
    return acquisition, resync, idle, truncated


def _clip(intervals: list[tuple[float, float]], end: float) -> list[tuple[float, float]]:
    """Clip to ``[0, end]`` and drop anything of non-positive length."""
    clipped = []
    for start, stop in intervals:
        low, high = max(0.0, start), min(end, stop)
        if high > low:
            clipped.append((low, high))
    return sorted(clipped)


def _merge(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Union of a list of intervals, sorted and disjoint."""
    merged: list[tuple[float, float]] = []
    for start, stop in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], stop))
        else:
            merged.append((start, stop))
    return merged


def _subtract(
    intervals: list[tuple[float, float]], removed: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """Set difference of two sorted disjoint interval lists."""
    result = list(intervals)
    for cut_start, cut_stop in removed:
        nxt: list[tuple[float, float]] = []
        for start, stop in result:
            if cut_stop <= start or cut_start >= stop:
                nxt.append((start, stop))
                continue
            if start < cut_start:
                nxt.append((start, cut_start))
            if cut_stop < stop:
                nxt.append((cut_stop, stop))
        result = nxt
    return result


def complement(
    window_duration_s: float, removed: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """The active set: ``[0, T]`` less the union of ``removed``."""
    return _subtract([(0.0, window_duration_s)], _merge(sorted(removed)))


def rate_at_offsets(
    satrec: Satrec,
    station: GroundStation,
    window: PassWindow,
    offsets_s: np.ndarray,
    budget: LinkBudget = DEFAULT_BUDGET,
) -> tuple[np.ndarray, float]:
    """Achievable rate at offsets from mask rise, Mbit/s, and the mask clamp.

    The window endpoints are bisection roots of ``elevation - mask``, so the
    elevation evaluated there can land a rounding below the mask, which the
    ITU-R P.618 model refuses. The elevation is clamped up to the mask and the
    **largest clamp applied** is returned alongside the rate so that a caller
    can report it. A clamp larger than
    :data:`MASK_CLAMP_TOLERANCE_DEG` is a real geometry error and raises.
    """
    elevation, slant = elevation_profile(satrec, station, window, offsets_s)
    deficit = float(np.max(np.maximum(station.mask_deg - elevation, 0.0), initial=0.0))
    if deficit > MASK_CLAMP_TOLERANCE_DEG:
        raise ValueError(
            f"elevation fell {deficit!r} degrees below the {station.mask_deg!r} degree mask "
            f"inside the window, which exceeds the clamp tolerance "
            f"{MASK_CLAMP_TOLERANCE_DEG!r}; the window is not a valid mask-to-mask interval"
        )
    clamped = np.clip(elevation, station.mask_deg, 90.0)
    rates = rate_mbit_s(
        clamped, slant, station.gain_over_temperature_db, station.zenith_attenuation_db, budget
    )
    return np.asarray(rates, dtype=float), deficit


def integrate_rate_mbit(
    satrec: Satrec,
    station: GroundStation,
    window: PassWindow,
    start_s: float,
    stop_s: float,
    nodes: int = DEFAULT_NODES_PER_INTERVAL,
    budget: LinkBudget = DEFAULT_BUDGET,
) -> tuple[float, float]:
    """Data volume over ``[start_s, stop_s]``, Mbit, by composite Simpson.

    Returns ``(volume_mbit, max_mask_clamp_deg)``. A zero-length interval
    returns ``(0.0, 0.0)`` without propagating anything.
    """
    if nodes < 3 or nodes % 2 == 0:
        raise ValueError(f"nodes must be an odd integer of at least 3, got {nodes!r}")
    if stop_s < start_s:
        raise ValueError(f"stop_s must not precede start_s, got {start_s!r} and {stop_s!r}")
    if stop_s == start_s:
        return 0.0, 0.0
    offsets = np.linspace(start_s, stop_s, nodes)
    rates, clamp = rate_at_offsets(satrec, station, window, offsets, budget)
    step = (stop_s - start_s) / (nodes - 1)
    weights = np.ones(nodes)
    weights[1:-1:2] = 4.0
    weights[2:-1:2] = 2.0
    return float(step / 3.0 * np.dot(weights, rates)), clamp


def nominal_rate_mbit_s(
    satrec: Satrec,
    station: GroundStation,
    window: PassWindow,
    reference_elevation_deg: float,
    budget: LinkBudget = DEFAULT_BUDGET,
) -> float:
    """The rate a schedule books for this station, Mbit/s.

    Evaluated at a declared reference elevation using the station's own slant
    range at that elevation taken from the SGP4 geometry of this window --
    specifically, the rate at the first offset at which the elevation reaches
    ``reference_elevation_deg``, or at the peak if the pass never reaches it.
    That is the strongest honest form of "the nominal rate": a scheduler that
    knows the station and the mask can compute exactly this before the pass.

    Raises
    ------
    ValueError
        If ``reference_elevation_deg`` is below the station mask, because a
        schedule cannot book a rate at an elevation it will not use.
    """
    if reference_elevation_deg < station.mask_deg:
        raise ValueError(
            f"reference_elevation_deg {reference_elevation_deg!r} is below the station mask "
            f"{station.mask_deg!r}; a contact schedule cannot book a rate it never sees"
        )
    offsets = np.linspace(0.0, window.duration_s, DEFAULT_NODES_PER_INTERVAL)
    elevation, slant = elevation_profile(satrec, station, window, offsets)
    reached = np.flatnonzero(elevation >= reference_elevation_deg)
    index = int(reached[0]) if reached.size else int(np.argmax(elevation))
    clamped = float(np.clip(elevation[index], station.mask_deg, 90.0))
    return float(
        rate_mbit_s(
            clamped,
            float(slant[index]),
            station.gain_over_temperature_db,
            station.zenith_attenuation_db,
            budget,
        )
    )


@dataclass(frozen=True)
class DecompositionOptions:
    """Quadrature and booking settings, so they appear in one place."""

    reference_elevation_deg: float = 30.0
    nodes_per_interval: int = DEFAULT_NODES_PER_INTERVAL
    nodes_per_window: int = DEFAULT_NODES_PER_WINDOW
    budget: LinkBudget = field(default=DEFAULT_BUDGET)


def decompose_pass(
    satrec: Satrec,
    station: GroundStation,
    window: PassWindow,
    events: PassEvents,
    options: DecompositionOptions | None = None,
) -> Decomposition:
    """Decompose one pass's nominal-versus-realised shortfall into four terms.

    See the module docstring for the identity, the quadrature, the residual
    and every documented edge case.

    Raises
    ------
    ValueError
        On a zero- or negative-duration window; on a window whose peak
        elevation does not clear the station mask; on quadrature node counts
        that are not odd integers of at least 3.
    """
    settings = options if options is not None else DecompositionOptions()
    if window.duration_s <= 0.0:
        raise ValueError(
            "a pass of zero or negative duration has no booked and no realised volume, so "
            f"every ratio this product reports is undefined; duration_s={window.duration_s!r}"
        )
    if window.max_elevation_deg < station.mask_deg:
        raise ValueError(
            f"the window's peak elevation {window.max_elevation_deg!r} degrees does not clear "
            f"the station mask {station.mask_deg!r} degrees, so there is no contact to decompose"
        )

    acquisition, resync, idle, truncated = normalise_intervals(
        window.duration_s,
        events.acquisition_s,
        events.resync_intervals_s,
        events.idle_intervals_s,
    )
    active = complement(window.duration_s, acquisition + resync + idle)
    clamps = [0.0]

    def volume(intervals: list[tuple[float, float]]) -> float:
        total = 0.0
        for start, stop in intervals:
            piece, clamp = integrate_rate_mbit(
                satrec, station, window, start, stop,
                settings.nodes_per_interval, settings.budget,
            )
            total += piece
            clamps.append(clamp)
        return total

    capacity, window_clamp = integrate_rate_mbit(
        satrec, station, window, 0.0, window.duration_s,
        settings.nodes_per_window, settings.budget,
    )
    clamps.append(window_clamp)

    realised = volume(active)
    acquisition_bits = volume(acquisition)
    resync_bits = volume(resync)
    idle_bits = volume(idle)

    nominal_rate = nominal_rate_mbit_s(
        satrec, station, window, settings.reference_elevation_deg, settings.budget
    )
    nominal = float(window.duration_s * nominal_rate)
    shortfall = float(nominal - realised)
    geometry = float(nominal - capacity)
    residual = float(shortfall - (geometry + acquisition_bits + resync_bits + idle_bits))

    return Decomposition(
        station=station.name,
        nominal_bits=nominal,
        capacity_bits=capacity,
        realised_bits=realised,
        shortfall_bits=shortfall,
        geometry_bits=geometry,
        acquisition_bits=acquisition_bits,
        resync_bits=resync_bits,
        idle_bits=idle_bits,
        residual_bits=residual,
        window_duration_s=window.duration_s,
        active_duration_s=float(sum(stop - start for start, stop in active)),
        acquisition_truncated=truncated,
        max_mask_clamp_deg=max(clamps),
        nodes_per_interval=settings.nodes_per_interval,
        nodes_per_window=settings.nodes_per_window,
    )
