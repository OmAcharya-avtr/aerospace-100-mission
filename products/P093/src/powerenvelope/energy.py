"""Energy integration over a rail trace, whole-run and restricted to active windows.

The integral
------------

Energy over a window is the **trapezoidal integral of ``vdd_in_mw`` over
``t_mono``**, which is the measurement model stated in
:mod:`powerenvelope.trace`. In mW and seconds the result is mJ.

The trapezoid rule is used rather than a rectangle rule because the sampler's
period is not exactly uniform -- it targets 20 Hz with a sleep correction and
drifts by whatever the scheduler does to it -- and the trapezoid is the integral
of the piecewise-linear interpolant through the samples it actually took. What
is guaranteed is agreement with that interpolant's integral to within floating
rounding, not agreement with the true continuous integral: the INA3221 averages
512 conversions in hardware, so the samples are already a low-pass view of the
rail and no quadrature rule recovers what the averaging removed.

The run, and why it is not the sampler's span
---------------------------------------------

**A run is ``[t0_mono, t1_mono]``, the interval the orchestrator records in
``manifest.json``: 38 cycles of 2.0 s beginning at ``t0_mono``.** The sampler's
span is deliberately WIDER. The campaign starts the sampler about half a second
before ``t0_mono`` and gives it ``duration + 1.0`` s, so that it brackets the
run and no edge sample is missing. On the measured reference cell the sampler
spans 76.950 s against a run of 76.001 s.

Integrating the sampler's full span and calling the result the run's energy
charges about 0.95 s of lead-in and tail-out to decisions that were not taken in
it -- 1.1 per cent on this campaign. :func:`whole_run_energy` integrates the
SAMPLER SPAN and is kept only as a diagnostic; :func:`run_energy` integrates the
run and is what every reported figure uses.

The declared phase partition
----------------------------

``E_active`` and ``E_sleep`` partition the run's cycles exactly. For cycle
``c`` the **declared** active phase is
``[t0 + c*CYCLE, t0 + c*CYCLE + duty*CYCLE]`` and the sleep phase is the rest of
that cycle, with ``P`` linearly interpolated at every boundary. The two sum to
the energy over ``[t0, t0 + cycles*CYCLE]``; the run may extend a few
milliseconds past the last cycle, and that tail is the whole of the closure
residual, which ``validate_measured_baselines.py`` reports per cell with its
worst value named.

Using the DECLARED phases rather than the merged inference intervals is what
makes the decomposition exact: ``F = (N/A) * (E_total/E_active)`` with no
residual term. How well the pipeline's inferences actually filled the declared
phase is then a separate, reported diagnostic -- see :func:`phase_occupancy` --
and not a factor quietly absorbing the difference.

Active windows from the decisions
---------------------------------

An active window derived from the decisions is used for the diagnostic and for
the interstitial count, never for a reported energy. The inference intervals
``[t_start, t_end]`` from ``decisions_<cell>.csv`` are merged and intersected
with the sampler's timestamps. The two processes share ``CLOCK_MONOTONIC``,
which is a system-wide clock, so comparing a timestamp taken in the sampler with
one taken in the pipeline is exact and not an approximation.

**The edge case: a sampler timestamp that falls between two inferences inside an
active window.** During an active phase the pipeline runs inferences
back-to-back, but ``t_end`` of one inference and ``t_start`` of the next are not
the same instant: between them the loop reads the clock, appends a row and tests
the phase boundary. A sampler timestamp can land in such a gap. Three choices
exist and only one of them is defensible:

1. Treat the sample as inactive. This is wrong: the gaps are tens of
   microseconds and the INA3221's 512-conversion hardware averaging means the
   reading at that instant is dominated by the active power on either side. It
   would also make the active duration smaller than the active phase, inflating
   the mean active power and therefore the Baseline-1 figure this product exists
   to compare against.
2. Treat every sample between the first and last inference of a cycle as
   active. This is wrong in the other direction: it would swallow a genuine
   sleep window if a run's phase accounting were ever broken, and the point of
   deriving the windows is that such a break should be visible.
3. Merge two consecutive inference intervals when the gap between them is
   shorter than :data:`GAP_TOLERANCE_S`, which is **half the sampler period**:
   the 20 Hz sampler cannot resolve a gap shorter than that, so it carries no
   information about one. A gap longer than the tolerance splits the window, and
   any sample landing in such a split is counted and reported as
   **interstitial** -- it is excluded from the active integral and from the
   active duration, and its count is part of every report rather than being
   silently absorbed.

Option 3 is what this module does, and the interstitial count is reported so a
reader can see it was small.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .trace import CYCLE_S, Cell, DecisionTrace, PowerTrace

#: Nominal rail sampler period, seconds, as the campaign declares it.
SAMPLE_PERIOD_S = 0.05

#: Two consecutive inference intervals separated by less than this are merged
#: into one active window. Half the sampler period: a 20 Hz sampler carries no
#: information about a shorter gap. See this module's docstring.
GAP_TOLERANCE_S = SAMPLE_PERIOD_S / 2.0


def trapezoid_energy_mj(t_s: np.ndarray, power_mw: np.ndarray) -> float:
    """Trapezoidal integral of ``power_mw`` over ``t_s``, in mJ.

    Parameters
    ----------
    t_s
        Timestamps, seconds, non-decreasing, at least two of them.
    power_mw
        Instantaneous power at those timestamps, mW.

    Returns
    -------
    float
        Energy in mJ, equal to the integral of the piecewise-linear interpolant
        through the samples, to within floating rounding.

    Raises
    ------
    ValueError
        On mismatched lengths, fewer than two samples, non-finite values or
        decreasing timestamps.
    """
    t = np.asarray(t_s, dtype=np.float64)
    p = np.asarray(power_mw, dtype=np.float64)
    if t.ndim != 1 or p.ndim != 1:
        raise ValueError(f"t_s and power_mw must be one-dimensional, got {t.shape} and {p.shape}")
    if t.size != p.size:
        raise ValueError(f"t_s has {t.size} samples and power_mw has {p.size}")
    if t.size < 2:
        raise ValueError(
            f"a trapezoidal integral needs at least two samples, got {t.size}. A window "
            "with one sample has no measured duration and no energy is reported for it."
        )
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(p)):
        raise ValueError("t_s and power_mw must be finite")
    if np.any(np.diff(t) < 0.0):
        raise ValueError("t_s must be non-decreasing; CLOCK_MONOTONIC does not go backwards")
    return float(np.trapezoid(p, t))


def merge_intervals(
    starts: np.ndarray, ends: np.ndarray, gap_tolerance_s: float = GAP_TOLERANCE_S
) -> tuple[np.ndarray, np.ndarray]:
    """Merge ``[start, end]`` intervals separated by at most ``gap_tolerance_s``.

    Parameters
    ----------
    starts, ends
        Interval bounds, seconds, same length. Not required to be sorted.
    gap_tolerance_s
        Gaps of at most this length are absorbed. Must be non-negative.

    Returns
    -------
    tuple of ndarray
        Merged ``(starts, ends)``, sorted and disjoint by more than the
        tolerance.

    Raises
    ------
    ValueError
        On mismatched lengths, an interval with ``end < start``, or a negative
        tolerance.
    """
    lo = np.asarray(starts, dtype=np.float64)
    hi = np.asarray(ends, dtype=np.float64)
    if lo.size != hi.size:
        raise ValueError(f"starts has {lo.size} entries and ends has {hi.size}")
    if gap_tolerance_s < 0.0:
        raise ValueError(f"gap_tolerance_s must be non-negative, got {gap_tolerance_s}")
    if lo.size == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    if np.any(hi < lo):
        raise ValueError("every interval must have end >= start")
    order = np.argsort(lo, kind="stable")
    lo, hi = lo[order], hi[order]
    out_lo = [float(lo[0])]
    out_hi = [float(hi[0])]
    for index in range(1, lo.size):
        if lo[index] - out_hi[-1] <= gap_tolerance_s:
            out_hi[-1] = max(out_hi[-1], float(hi[index]))
        else:
            out_lo.append(float(lo[index]))
            out_hi.append(float(hi[index]))
    return np.asarray(out_lo, dtype=np.float64), np.asarray(out_hi, dtype=np.float64)


def active_windows(
    decisions: DecisionTrace, gap_tolerance_s: float = GAP_TOLERANCE_S
) -> tuple[np.ndarray, np.ndarray]:
    """Merged active windows of a run, from its inference intervals.

    Returns
    -------
    tuple of ndarray
        ``(starts, ends)`` in ``CLOCK_MONOTONIC`` seconds.
    """
    return merge_intervals(decisions.t_start, decisions.t_end, gap_tolerance_s)


def clip_series(
    t_s: np.ndarray, power_mw: np.ndarray, lo: float, hi: float
) -> tuple[np.ndarray, np.ndarray]:
    """Restrict a sampled series to ``[lo, hi]``, interpolating at both ends.

    The endpoints are inserted by linear interpolation of the power between the
    bracketing samples, which is the same interpolant the trapezoid rule
    integrates, so clipping and integrating is consistent with integrating the
    whole series: the energies over a partition of the span sum to the energy
    over the span, to within floating rounding.

    ``[lo, hi]`` is first intersected with the span of ``t_s``: a window that
    starts before the first sample or ends after the last is truncated to what
    was measured, because nothing is extrapolated.

    Returns
    -------
    tuple of ndarray
        ``(t, p)`` with at least two entries, or two empty arrays when the
        intersection is empty or degenerate.
    """
    t = np.asarray(t_s, dtype=np.float64)
    p = np.asarray(power_mw, dtype=np.float64)
    if t.size != p.size:
        raise ValueError(f"t_s has {t.size} samples and power_mw has {p.size}")
    empty = (np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64))
    if t.size < 2:
        return empty
    left = max(float(lo), float(t[0]))
    right = min(float(hi), float(t[-1]))
    if not right > left:
        return empty
    inner = (t > left) & (t < right)
    out_t = np.concatenate(([left], t[inner], [right]))
    out_p = np.concatenate((np.interp([left], t, p), p[inner], np.interp([right], t, p)))
    return out_t, out_p


@dataclass(frozen=True)
class WindowEnergy:
    """Energy restricted to a set of windows, with the accounting that goes with it.

    Attributes
    ----------
    energy_mj
        Summed trapezoidal energy over the windows, mJ.
    duration_s
        Summed measured duration of the windows after intersection with the
        sampler's span, seconds.
    n_windows
        Number of windows that contributed.
    n_samples
        Number of sampler timestamps strictly inside the windows.
    n_between_windows
        Number of sampler timestamps that lie inside the convex span of the
        window set but in none of its windows. For the active-window set of a
        duty-cycled run these are the SLEEP-window samples and the count is
        large by construction; it is here so that the three accountings --
        active, between, and whole-run -- can be seen to add up. The genuinely
        interesting case, a sample inside one active PHASE but between two
        inferences, has its own counter: :func:`interstitial_samples`.
    mean_power_mw
        ``energy_mj / duration_s``: the TIME-WEIGHTED mean power over the
        windows, mW. This is the mean that an energy integral implies, and it is
        the one the Baseline-1 figure uses.
    sample_mean_power_mw
        The arithmetic mean of the samples inside the windows, mW, reported
        beside the time-weighted mean because a reader will want to know whether
        they differ. They differ when the sampler period is not uniform.
    """

    energy_mj: float
    duration_s: float
    n_windows: int
    n_samples: int
    n_between_windows: int
    mean_power_mw: float
    sample_mean_power_mw: float


def energy_over_windows(
    power: PowerTrace,
    starts: np.ndarray,
    ends: np.ndarray,
    column: str = "vdd_in_mw",
) -> WindowEnergy:
    """Integrate one power column of ``power`` over the given windows.

    Parameters
    ----------
    power
        The run's rail trace.
    starts, ends
        Window bounds, seconds. Assumed already merged and disjoint; pass the
        output of :func:`active_windows`.
    column
        ``"vdd_in_mw"`` for the measured rail, ``"nominal_mw"`` for the same
        current against the nominal 5.0 V rail, or any other property of
        :class:`~powerenvelope.trace.PowerTrace` that returns an array of the
        same length.

    Raises
    ------
    ValueError
        If ``column`` is not a power-like array of the right length.
    """
    values = getattr(power, column, None)
    if not isinstance(values, np.ndarray) or values.size != power.t.size:
        raise ValueError(
            f"column {column!r} is not a power array of length {power.t.size} on a "
            "PowerTrace"
        )
    total_energy = 0.0
    total_duration = 0.0
    inside = np.zeros(power.t.size, dtype=bool)
    contributing = 0
    for lo, hi in zip(starts, ends, strict=True):
        t, p = clip_series(power.t, values, float(lo), float(hi))
        if t.size < 2:
            continue
        contributing += 1
        total_energy += trapezoid_energy_mj(t, p)
        total_duration += float(t[-1] - t[0])
        inside |= (power.t >= t[0]) & (power.t <= t[-1])
    n_samples = int(inside.sum())
    if starts.size:
        span = (power.t >= float(np.min(starts))) & (power.t <= float(np.max(ends)))
        n_between = int(np.sum(span & ~inside))
    else:
        n_between = 0
    mean = total_energy / total_duration if total_duration > 0.0 else float("nan")
    sample_mean = float(values[inside].mean()) if n_samples else float("nan")
    return WindowEnergy(
        energy_mj=total_energy,
        duration_s=total_duration,
        n_windows=contributing,
        n_samples=n_samples,
        n_between_windows=n_between,
        mean_power_mw=mean,
        sample_mean_power_mw=sample_mean,
    )


@dataclass(frozen=True)
class Interstitial:
    """The edge case of this module's docstring, counted rather than described.

    An inference's ``t_end`` and the next inference's ``t_start`` are not the
    same instant: between them the pipeline's loop reads the clock, appends a
    row and tests the phase boundary. A sampler timestamp can land in such a
    gap. Gaps of at most :data:`GAP_TOLERANCE_S` are merged, so a sample in one
    of those is inside an active window and is counted as active. A longer gap
    splits the window, and a sample landing in the split belongs to neither the
    active windows nor the declared sleep windows.

    Attributes
    ----------
    n_samples
        How many sampler timestamps landed in such a split, summed over cycles.
    n_gaps
        How many splits there were.
    max_gap_s
        The longest gap between consecutive inferences of one cycle, seconds.
        Reported whether or not it exceeded the tolerance, because the number a
        reader wants is how close the pipeline's loop overhead came to being
        visible at 20 Hz.
    """

    n_samples: int
    n_gaps: int
    max_gap_s: float


def interstitial_samples(
    cell: Cell, gap_tolerance_s: float = GAP_TOLERANCE_S
) -> Interstitial:
    """Count sampler timestamps that fell inside an active phase but in no window.

    The active PHASE of a cycle is taken from the decisions' own ``cycle``
    column: it spans the first ``t_start`` to the last ``t_end`` of that cycle.
    Any sample inside that span but outside every merged active window is
    interstitial.
    """
    decisions = cell.require_decisions()
    starts, ends = merge_intervals(decisions.t_start, decisions.t_end, gap_tolerance_s)
    inside = np.zeros(cell.power.t.size, dtype=bool)
    for lo, hi in zip(starts, ends, strict=True):
        inside |= (cell.power.t >= lo) & (cell.power.t <= hi)
    total = 0
    gaps = 0
    widest = 0.0
    for cycle in np.unique(decisions.cycle):
        rows = decisions.cycle == cycle
        phase_lo = float(decisions.t_start[rows].min())
        phase_hi = float(decisions.t_end[rows].max())
        order = np.argsort(decisions.t_start[rows], kind="stable")
        gap = decisions.t_start[rows][order][1:] - decisions.t_end[rows][order][:-1]
        if gap.size:
            widest = max(widest, float(gap.max()))
            gaps += int(np.sum(gap > gap_tolerance_s))
        in_phase = (cell.power.t >= phase_lo) & (cell.power.t <= phase_hi)
        total += int(np.sum(in_phase & ~inside))
    return Interstitial(n_samples=total, n_gaps=gaps, max_gap_s=widest)


def run_bounds(cell: Cell) -> tuple[float, float]:
    """``(t0_mono, t1_mono)`` of the run, from the manifest's own record."""
    return cell.t0_mono, cell.t1_mono


def declared_phases(
    cell: Cell, cycle_s: float = CYCLE_S
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """The declared active and sleep phases of every cycle of the run.

    Returns ``((active_lo, active_hi), (sleep_lo, sleep_hi))``. The phases are
    taken from the DECLARED cycle grid beginning at ``t0_mono``, not from the
    decisions, so together they tile ``[t0, t0 + cycles*cycle_s]`` exactly with
    no gap and no overlap. A duty of 0 gives no active phase and a duty of 1
    gives no sleep phase.

    Raises
    ------
    ValueError
        On a non-positive cycle length or a duty outside ``[0, 1]``.
    """
    if cycle_s <= 0.0:
        raise ValueError(f"cycle_s must be positive, got {cycle_s}")
    if not 0.0 <= cell.duty <= 1.0:
        raise ValueError(f"duty must be in [0, 1], got {cell.duty}")
    start = cell.t0_mono
    index = np.arange(cell.cycles, dtype=np.float64)
    cycle_start = start + index * cycle_s
    active_end = cycle_start + cell.duty * cycle_s
    cycle_end = cycle_start + cycle_s
    active = (cycle_start[active_end > cycle_start], active_end[active_end > cycle_start])
    sleep = (active_end[cycle_end > active_end], cycle_end[cycle_end > active_end])
    return active, sleep


def run_energy(cell: Cell, column: str = "vdd_in_mw") -> WindowEnergy:
    """Integrate one power column over the RUN, ``[t0_mono, t1_mono]``.

    This is the numerator of the energy per useful decision. The sleep windows
    are in it on purpose: see the measurement model in
    :mod:`powerenvelope.trace`. The sampler's wider bracketing span is NOT in
    it: see this module's docstring.
    """
    low, high = run_bounds(cell)
    return energy_over_windows(
        cell.power, np.asarray([low]), np.asarray([high]), column=column
    )


def declared_active_energy(
    cell: Cell, cycle_s: float = CYCLE_S, column: str = "vdd_in_mw"
) -> WindowEnergy:
    """Integrate one power column over the run's declared ACTIVE phases."""
    (low, high), _sleep = declared_phases(cell, cycle_s)
    return energy_over_windows(cell.power, low, high, column=column)


def declared_sleep_energy(
    cell: Cell, cycle_s: float = CYCLE_S, column: str = "vdd_in_mw"
) -> WindowEnergy:
    """Integrate one power column over the run's declared SLEEP phases."""
    _active, (low, high) = declared_phases(cell, cycle_s)
    return energy_over_windows(cell.power, low, high, column=column)


@dataclass(frozen=True)
class PhaseOccupancy:
    """How well the pipeline's inferences filled the DECLARED active phase.

    This used to be a factor of the decomposition, absorbing the difference
    between the declared phase and whatever the loop actually did. Under the
    declared-phase partition the decomposition is exact in its named factors and
    this is what it always was: a diagnostic that the pipeline ran when it said
    it would.

    Attributes
    ----------
    declared_s
        ``cycles * duty * cycle_s``: the active time the configuration declares.
    occupied_s
        ``sum(t_end - t_start)``: the time inferences actually occupied. It is
        SHORTER than declared, because the loop checks the clock before starting
        an inference and does not start one that would run past the boundary,
        and because a few microseconds pass between consecutive inferences.
    spanned_s
        Summed span of the merged inference intervals, first ``t_start`` to last
        ``t_end`` per cycle. It is LONGER than occupied by the inter-inference
        gaps, and can exceed the declared time because the last inference of a
        phase starts before the boundary and finishes after it.
    occupancy
        ``occupied_s / declared_s``.
    overshoot_s
        ``spanned_s - declared_s``: positive when the last inference straddles
        the declared boundary.
    shortfall_s
        ``declared_s - occupied_s``: positive when the loop leaves the end of
        the phase unfilled.
    """

    declared_s: float
    occupied_s: float
    spanned_s: float
    occupancy: float
    overshoot_s: float
    shortfall_s: float


def phase_occupancy(cell: Cell, cycle_s: float = CYCLE_S) -> PhaseOccupancy:
    """Measure how the pipeline's inferences filled the declared active phase."""
    decisions = cell.require_decisions()
    declared = cell.cycles * cell.duty * cycle_s
    occupied = float(np.sum(decisions.latency_s))
    low, high = active_windows(decisions)
    spanned = float(np.sum(high - low))
    return PhaseOccupancy(
        declared_s=declared,
        occupied_s=occupied,
        spanned_s=spanned,
        occupancy=occupied / declared if declared else float("nan"),
        overshoot_s=spanned - declared,
        shortfall_s=declared - occupied,
    )


def whole_run_energy(power: PowerTrace, column: str = "vdd_in_mw") -> WindowEnergy:
    """Integrate one power column over the SAMPLER'S SPAN. A diagnostic only.

    **This is not the run.** The campaign starts the sampler about half a second
    before ``t0_mono`` and gives it ``duration + 1.0`` s so that it brackets the
    run; integrating all of it charges the lead-in and tail-out to decisions
    that were not taken in them, which on the measured campaign is 1.1 per cent
    of the energy. Use :func:`run_energy`, which takes ``[t0_mono, t1_mono]``
    from the manifest.

    It is kept because the difference between the two is worth reporting, and
    because the rail-accounting ratio is a ratio of two integrals over the same
    span and is insensitive to which span that is.
    """
    if power.t.size < 2:
        raise ValueError("a run with fewer than two rail samples has no measured energy")
    return energy_over_windows(
        power, np.asarray([power.t[0]]), np.asarray([power.t[-1]]), column=column
    )


def cell_energy(cell: Cell, column: str = "vdd_in_mw") -> tuple[WindowEnergy, WindowEnergy]:
    """``(run, declared active phases)`` energy for one cell.

    Both integrals are over the run and its declared cycle grid, which is
    definition C: see this module's docstring. The floor cell is allowed here --
    it has a run and a zero-duty phase grid, so its active energy is zero over
    no windows.
    """
    return run_energy(cell, column=column), declared_active_energy(cell, column=column)


def sleep_energy(cell: Cell, column: str = "vdd_in_mw") -> WindowEnergy:
    """Energy in the run's declared SLEEP phases. The partner of :func:`cell_energy`.

    Because :func:`clip_series` interpolates every endpoint on the same
    piecewise-linear interpolant the trapezoid rule integrates, and because the
    declared active and sleep phases share those endpoints and tile the cycles
    exactly, the two energies sum to the energy over
    ``[t0, t0 + cycles*cycle_s]``. The run may run a few milliseconds past the
    last cycle, and that tail is the whole of the residual against
    :func:`run_energy`. ``validate_measured_baselines.py`` reports it per cell
    with its worst value named, and ``tests/test_energy.py`` bounds it.
    """
    return declared_sleep_energy(cell, column=column)
