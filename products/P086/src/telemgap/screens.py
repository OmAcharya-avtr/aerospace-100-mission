"""The rule screens. This is the analytic baseline and the operational default.

Five independent screens are run over a record. Each is an explicit, stated
rule with no fitted parameter, and each returns *evidence* rather than a
verdict; the verdict is assembled from the evidence in
:mod:`telemgap.classify`.

The screens
-----------
1. **Counter continuity.** With a counter modulus ``M`` and a reorder horizon
   ``H`` (default ``M // 2``), the forward delta between consecutive received
   frames is

       d = (c_i - c_{i-1}) mod M                                        (2)

   and is read as: ``d == 0`` a repeated counter, ``d == 1`` nominal,
   ``2 <= d <= M - H`` a forward gap of ``d - 1`` frames, and ``d > M - H``
   backward motion of ``M - d`` frames, that is, a reorder. The split at
   ``M - H`` is a *choice*, and the ambiguity it resolves by fiat is
   irreducible: a gap of ``M - k`` frames and a reorder by ``k`` frames produce
   the identical delta. :mod:`telemgap.determinability` reports the largest gap
   this record could be hiding as a reorder.

2. **Timestamp monotonicity.** A decrease beyond the declared jitter is a
   backward time jump. If the counter moved *forward* across the same pair, the
   stream did not go backward and the clock did, which is a clock reset. If the
   counter moved backward too, the two agree and it is a reorder.

3. **Cadence.** For an inter-frame interval ``dt`` and nominal cadence ``T``,
   the frame deficit is ``k = round(dt / T)`` and the residual is
   ``r = dt - k T``. ``|r| > jitter`` is a cadence violation; ``k >= 2`` is a
   time-domain gap of ``k - 1`` frames; ``k == 0`` with ``dt >= 0`` means two
   frames arrived inside one cadence slot.

4. **Declared range.** A channel value outside its inclusive ``[low, high]``,
   or non-finite, is a range violation. It is evidence that something is wrong
   and says nothing on its own about what.

5. **Frame error control.** A frame whose error-control field failed carries
   damaged bits, and that is the strongest evidence of corruption available --
   stronger than any inference from the values, because it is a statement about
   the bits. What it does **not** say is *which* word in the frame is damaged:
   a frame-level checksum is one number over the whole frame. The frame is
   therefore reported corrupt, and every channel of it whose value is inside
   its declared range and consistent with its neighbours is reported
   **undecided at the value level** rather than clean. That is the honest
   reading and it is where most of this package's undecided set comes from.

6. **Local residual (the strengthened screen).** A robust z-score against a
   centred median window,

       z_i = |v_i - median(window)| / (1.4826 * MAD(window) + eps)       (5)

   with the 1.4826 factor the standard consistency constant making the scaled
   MAD an unbiased estimator of the standard deviation for Gaussian data
   (Rousseeuw & Croux 1993, *JASA* 88(424), 1273-1283). This screen is **not**
   in the product specification's named baseline, which lists only monotonicity,
   cadence, counter continuity and declared ranges. It is implemented and
   reported beside the specified baseline because the specified baseline is
   structurally blind to a bit error that leaves the value inside its declared
   range, and the build guide requires the strongest honest baseline to be the
   one the learned model is scored against. Both are reported.

Screens 1 to 3 are evaluated in received order, because reordering and
duplication are properties of the order. Gap consolidation is evaluated on a
time-sorted, counter-deduplicated view, because a gap is a property of the
timeline and not of the arrival order.

Units
-----
Durations and timestamps in seconds; counters dimensionless; channel values in
each channel's declared unit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .record import TelemetryRecord

#: Consistency constant making the scaled median absolute deviation an unbiased
#: estimator of the standard deviation for normally distributed data
#: (Rousseeuw & Croux 1993). Dimensionless.
MAD_TO_SIGMA = 1.4826

#: Floor added to the scaled MAD in equation (5) as a fraction of the channel's
#: declared range span, so a window of identical values does not divide by zero
#: and does not report an infinite z-score for a one-count wobble.
MAD_FLOOR_FRACTION = 1e-9


@dataclass(frozen=True)
class CounterFinding:
    """One counter-continuity observation between received frames ``i-1`` and ``i``."""

    index: int
    previous_counter: int
    counter: int
    delta: int
    kind: str
    frames_implied: int

    KINDS = ("nominal", "repeated_counter", "forward_gap", "reorder")


@dataclass(frozen=True)
class ClockFinding:
    """One timestamp-monotonicity observation between received frames ``i-1`` and ``i``."""

    index: int
    previous_timestamp: float
    timestamp: float
    delta_s: float
    counter_moved_forward: bool
    kind: str

    KINDS = ("clock_reset", "reorder_consistent")


@dataclass(frozen=True)
class CadenceFinding:
    """One cadence observation between received frames ``i-1`` and ``i``."""

    index: int
    delta_s: float
    slots: int
    residual_s: float
    kind: str

    KINDS = ("nominal", "jitter_exceeded", "same_slot", "time_gap")


@dataclass(frozen=True)
class RangeFinding:
    """One declared-range violation, at one frame and one channel."""

    index: int
    channel: int
    channel_name: str
    value: float
    low: float
    high: float
    excess: float
    non_finite: bool


@dataclass(frozen=True)
class ValidityFinding:
    """One frame whose error-control field failed."""

    index: int
    counter: int


@dataclass(frozen=True)
class ResidualFinding:
    """One local-residual violation from the strengthened screen, equation (5)."""

    index: int
    channel: int
    channel_name: str
    value: float
    local_median: float
    scaled_mad: float
    z: float


@dataclass(frozen=True)
class Gap:
    """A consolidated interval of the timeline with no received frame in it.

    Attributes
    ----------
    after_index, before_index
        Received-order indices of the frames bounding the gap in time.
    start_time, end_time
        Timestamps of those two frames, seconds.
    duration_s
        ``end_time - start_time``, seconds. This is the *frame-to-frame*
        interval, which exceeds the unobserved interval by one cadence.
    unobserved_s
        ``duration_s - cadence_s``, floored at zero: the length of the interval
        in which no frame could have been received, seconds. The policy is
        written against this, not against ``duration_s``.
    missing_frames_from_time
        ``round(duration_s / cadence_s) - 1``.
    missing_frames_from_counter
        Counter-implied deficit, or ``-1`` when the bounding pair is not
        adjacent in received order so no counter delta applies.
    agrees
        Whether the two deficits agree. A disagreement is evidence about the
        *clock*, not about the data, and is surfaced as such.
    """

    after_index: int
    before_index: int
    start_time: float
    end_time: float
    duration_s: float
    unobserved_s: float
    missing_frames_from_time: int
    missing_frames_from_counter: int
    agrees: bool


@dataclass
class ScreenFindings:
    """Everything the five screens observed, with no verdict attached."""

    n_frames: int
    counter: tuple[CounterFinding, ...] = ()
    clock: tuple[ClockFinding, ...] = ()
    cadence: tuple[CadenceFinding, ...] = ()
    ranges: tuple[RangeFinding, ...] = ()
    residuals: tuple[ResidualFinding, ...] = ()
    validity_failures: tuple[ValidityFinding, ...] = ()
    gaps: tuple[Gap, ...] = ()
    duplicate_indices: tuple[int, ...] = ()
    reorder_indices: tuple[int, ...] = ()
    clock_reset_indices: tuple[int, ...] = ()
    notes: list[str] = field(default_factory=list)

    @property
    def total_missing_frames(self) -> int:
        """Sum of the time-domain frame deficits over all consolidated gaps."""
        return int(sum(gap.missing_frames_from_time for gap in self.gaps))

    @property
    def range_violation_indices(self) -> tuple[int, ...]:
        """Sorted unique frame indices carrying at least one range violation."""
        return tuple(sorted({finding.index for finding in self.ranges}))

    @property
    def validity_failed_indices(self) -> tuple[int, ...]:
        """Frame indices whose error-control field failed."""
        return tuple(finding.index for finding in self.validity_failures)

    @property
    def residual_violation_indices(self) -> tuple[int, ...]:
        """Sorted unique frame indices carrying at least one residual violation."""
        return tuple(sorted({finding.index for finding in self.residuals}))

    def summary(self) -> dict[str, int]:
        """Counts per screen, for reports and regression tests."""
        return {
            "frames": self.n_frames,
            "gaps": len(self.gaps),
            "missing_frames": self.total_missing_frames,
            "duplicates": len(self.duplicate_indices),
            "reorders": len(self.reorder_indices),
            "clock_resets": len(self.clock_reset_indices),
            "cadence_violations": sum(
                1 for finding in self.cadence if finding.kind == "jitter_exceeded"
            ),
            "range_violations": len(self.ranges),
            "residual_violations": len(self.residuals),
            "validity_failures": len(self.validity_failures),
        }


def counter_delta(previous: int, counter: int, modulus: int) -> int:
    """Forward counter delta of equation (2), in ``0 .. modulus-1``."""
    if modulus < 2:
        raise ValueError(f"modulus must be at least 2, got {modulus!r}")
    return int((int(counter) - int(previous)) % int(modulus))


def screen_counters(record: TelemetryRecord) -> tuple[CounterFinding, ...]:
    """Counter-continuity screen, equation (2), in received order."""
    modulus = int(record.schema.counter_modulus)
    horizon = record.schema.effective_reorder_horizon
    backward_threshold = modulus - horizon
    findings: list[CounterFinding] = []
    counters = record.counters
    for index in range(1, counters.size):
        previous = int(counters[index - 1])
        current = int(counters[index])
        delta = counter_delta(previous, current, modulus)
        if delta == 0:
            kind, implied = "repeated_counter", 0
        elif delta == 1:
            kind, implied = "nominal", 0
        elif delta <= backward_threshold:
            kind, implied = "forward_gap", delta - 1
        else:
            kind, implied = "reorder", modulus - delta
        findings.append(
            CounterFinding(
                index=index,
                previous_counter=previous,
                counter=current,
                delta=delta,
                kind=kind,
                frames_implied=implied,
            )
        )
    return tuple(findings)


def screen_clock(
    record: TelemetryRecord, counter_findings: tuple[CounterFinding, ...]
) -> tuple[ClockFinding, ...]:
    """Timestamp-monotonicity screen. Only backward motion is reported."""
    jitter = float(record.schema.jitter_s)
    by_index = {finding.index: finding for finding in counter_findings}
    findings: list[ClockFinding] = []
    timestamps = record.timestamps
    for index in range(1, timestamps.size):
        delta = float(timestamps[index] - timestamps[index - 1])
        if delta >= -jitter:
            continue
        counter_finding = by_index.get(index)
        forward = counter_finding is None or counter_finding.kind in ("nominal", "forward_gap")
        findings.append(
            ClockFinding(
                index=index,
                previous_timestamp=float(timestamps[index - 1]),
                timestamp=float(timestamps[index]),
                delta_s=delta,
                counter_moved_forward=bool(forward),
                kind="clock_reset" if forward else "reorder_consistent",
            )
        )
    return tuple(findings)


def screen_cadence(record: TelemetryRecord) -> tuple[CadenceFinding, ...]:
    """Cadence screen. Reports every interval that is not a nominal single slot."""
    cadence = float(record.schema.cadence_s)
    jitter = float(record.schema.jitter_s)
    findings: list[CadenceFinding] = []
    timestamps = record.timestamps
    for index in range(1, timestamps.size):
        delta = float(timestamps[index] - timestamps[index - 1])
        if delta < 0.0:
            continue
        slots = int(np.rint(delta / cadence))
        residual = delta - slots * cadence
        if abs(residual) > jitter:
            kind = "jitter_exceeded"
        elif slots == 0:
            kind = "same_slot"
        elif slots == 1:
            kind = "nominal"
        else:
            kind = "time_gap"
        findings.append(
            CadenceFinding(
                index=index, delta_s=delta, slots=slots, residual_s=residual, kind=kind
            )
        )
    return tuple(findings)


def screen_ranges(record: TelemetryRecord) -> tuple[RangeFinding, ...]:
    """Declared-range screen over every frame and channel."""
    findings: list[RangeFinding] = []
    for column, channel in enumerate(record.schema.channels):
        values = record.values[:, column]
        finite = np.isfinite(values)
        outside = np.empty(values.shape, dtype=bool)
        outside[:] = ~finite
        outside[finite] = (values[finite] < channel.low) | (values[finite] > channel.high)
        for index in np.flatnonzero(outside):
            value = float(values[index])
            if np.isfinite(value):
                excess = float(max(channel.low - value, value - channel.high))
            else:
                excess = float("inf")
            findings.append(
                RangeFinding(
                    index=int(index),
                    channel=column,
                    channel_name=channel.name,
                    value=value,
                    low=channel.low,
                    high=channel.high,
                    excess=excess,
                    non_finite=not np.isfinite(value),
                )
            )
    return tuple(sorted(findings, key=lambda finding: (finding.index, finding.channel)))


def _robust_local_statistics_loop(
    values: np.ndarray, window: int, span: float
) -> tuple[np.ndarray, np.ndarray]:
    """Reference implementation of equation (5), written as an explicit loop.

    Kept because it is obviously correct and the vectorised implementation is
    not. ``tests/test_screens.py`` asserts the two agree to within rounding on
    random inputs, including inputs containing non-finite values, which is the
    only reason to trust the fast path.
    """
    n = values.size
    medians = np.empty(n, dtype=np.float64)
    scaled = np.empty(n, dtype=np.float64)
    floor = MAD_FLOOR_FRACTION * abs(float(span))
    for index in range(n):
        low = max(0, index - window)
        high = min(n, index + window + 1)
        neighbourhood = np.concatenate((values[low:index], values[index + 1 : high]))
        neighbourhood = neighbourhood[np.isfinite(neighbourhood)]
        if neighbourhood.size == 0:
            medians[index] = values[index] if np.isfinite(values[index]) else 0.0
            scaled[index] = floor
            continue
        median = float(np.median(neighbourhood))
        mad = float(np.median(np.abs(neighbourhood - median)))
        medians[index] = median
        scaled[index] = MAD_TO_SIGMA * mad + floor
    return medians, scaled


def screen_validity(record: TelemetryRecord) -> tuple[ValidityFinding, ...]:
    """Frame error-control screen. Empty when the record carries no validity field."""
    if record.validity is None:
        return ()
    return tuple(
        ValidityFinding(index=int(index), counter=int(record.counters[index]))
        for index in np.flatnonzero(~record.validity)
    )


def robust_local_statistics(
    values: np.ndarray, window: int, span: float
) -> tuple[np.ndarray, np.ndarray]:
    """Centred median and scaled MAD of ``values``, equation (5), excluding self.

    Parameters
    ----------
    values
        ``(n,)`` channel values.
    window
        Half-width of the centred window in samples. Must be at least 2.
    span
        Declared range span of the channel, used only for the MAD floor.

    Returns
    -------
    (median, scaled_mad)
        Both ``(n,)``. The sample at the centre is excluded from its own
        window, so a single wild value does not pull the statistic it is being
        compared against. Non-finite neighbours are excluded. Where no finite
        neighbour exists the median falls back to the sample itself (or zero if
        that is not finite) and the scaled MAD to the floor.

    Implementation note
    -------------------
    A sliding window over a NaN-padded copy, with the centre column blanked, so
    that ``nanmedian`` performs the exclusion. Agreement with
    :func:`_robust_local_statistics_loop` is asserted by test on random inputs;
    it is not assumed.
    """
    if window < 2:
        raise ValueError(f"window must be at least 2 samples, got {window!r}")
    values = np.asarray(values, dtype=np.float64)
    n = values.size
    floor = MAD_FLOOR_FRACTION * abs(float(span))
    clean = np.where(np.isfinite(values), values, np.nan)
    padded = np.concatenate((np.full(window, np.nan), clean, np.full(window, np.nan)))
    view = np.lib.stride_tricks.sliding_window_view(padded, 2 * window + 1).copy()
    view[:, window] = np.nan
    empty = np.count_nonzero(~np.isnan(view), axis=1) == 0
    view[empty, :] = 0.0
    medians = np.nanmedian(view, axis=1)
    mad = np.nanmedian(np.abs(view - medians[:, None]), axis=1)
    if np.any(empty):
        medians[empty] = np.where(np.isfinite(values[empty]), values[empty], 0.0)
        mad[empty] = 0.0
    scaled = MAD_TO_SIGMA * mad + floor
    if n != medians.size:  # pragma: no cover - defensive shape guard
        raise AssertionError(f"window view produced {medians.size} rows for {n} samples")
    return medians, scaled


def screen_local_residual(
    record: TelemetryRecord, window: int = 8, z_threshold: float = 8.0
) -> tuple[ResidualFinding, ...]:
    """Strengthened local-residual screen, equation (5).

    Parameters
    ----------
    window
        Half-width of the centred median window, samples.
    z_threshold
        Robust z above which a sample is reported. The default of 8 is chosen
        so that, for normally distributed data, the expected number of false
        reports in a 10,000-sample record is below 1e-11; the realised false
        report rate on clean records is measured in
        ``validation/validate_rule_screens.py`` rather than assumed.
    """
    if z_threshold <= 0.0:
        raise ValueError(f"z_threshold must be positive, got {z_threshold!r}")
    findings: list[ResidualFinding] = []
    for column, channel in enumerate(record.schema.channels):
        values = record.values[:, column]
        medians, scaled = robust_local_statistics(values, window, channel.span)
        with np.errstate(invalid="ignore"):
            deviation = np.abs(values - medians)
            z = np.where(np.isfinite(deviation), deviation / scaled, np.inf)
        for index in np.flatnonzero(z > z_threshold):
            findings.append(
                ResidualFinding(
                    index=int(index),
                    channel=column,
                    channel_name=channel.name,
                    value=float(values[index]),
                    local_median=float(medians[index]),
                    scaled_mad=float(scaled[index]),
                    z=float(z[index]),
                )
            )
    return tuple(sorted(findings, key=lambda finding: (finding.index, finding.channel)))


def timeline_order(record: TelemetryRecord) -> tuple[int, ...]:
    """Received-order indices sorted by timestamp, with repeated counters dropped.

    A gap and a reconstruction are properties of the timeline, not of the
    arrival order, so both are computed over this view. The first frame
    carrying a given counter wins, which is the conventional de-duplication and
    the one that does not depend on which copy arrived second.
    """
    order = np.argsort(record.timestamps, kind="stable")
    seen: set[int] = set()
    kept: list[int] = []
    for index in order:
        counter = int(record.counters[index])
        if counter in seen:
            continue
        seen.add(counter)
        kept.append(int(index))
    return tuple(kept)


def consolidate_gaps(
    record: TelemetryRecord, counter_findings: tuple[CounterFinding, ...]
) -> tuple[Gap, ...]:
    """Build the gap list from the time-sorted, counter-deduplicated timeline.

    A gap is a property of the timeline, so duplicates are removed and frames
    are sorted by timestamp before consolidation. Both frame deficits are
    reported: the time-domain one from the cadence, and the counter-domain one
    where the bounding frames happen to be adjacent in received order.
    """
    cadence = float(record.schema.cadence_s)
    jitter = float(record.schema.jitter_s)
    counter_by_index = {finding.index: finding for finding in counter_findings}
    kept = list(timeline_order(record))

    gaps: list[Gap] = []
    for position in range(1, len(kept)):
        left, right = kept[position - 1], kept[position]
        duration = float(record.timestamps[right] - record.timestamps[left])
        slots = int(np.rint(duration / cadence))
        if slots < 2 and duration - cadence <= jitter:
            continue
        from_time = max(0, slots - 1)
        finding = counter_by_index.get(right)
        if finding is not None and finding.index - 1 == left and finding.kind == "forward_gap":
            from_counter = finding.frames_implied
        elif finding is not None and finding.index - 1 == left and finding.kind == "nominal":
            from_counter = 0
        else:
            from_counter = -1
        if from_time == 0 and from_counter <= 0:
            continue
        gaps.append(
            Gap(
                after_index=left,
                before_index=right,
                start_time=float(record.timestamps[left]),
                end_time=float(record.timestamps[right]),
                duration_s=duration,
                unobserved_s=max(0.0, duration - cadence),
                missing_frames_from_time=from_time,
                missing_frames_from_counter=from_counter,
                agrees=bool(from_counter < 0 or from_counter == from_time),
            )
        )
    return tuple(gaps)


def screen(
    record: TelemetryRecord, residual_window: int = 8, residual_z: float = 8.0
) -> ScreenFindings:
    """Run all five screens and return their evidence.

    Parameters
    ----------
    record
        The record to screen.
    residual_window, residual_z
        Parameters of the strengthened local-residual screen, equation (5).

    Returns
    -------
    ScreenFindings
        Evidence only. No anomaly class is assigned here; that is
        :func:`telemgap.classify.classify`.
    """
    if not isinstance(record, TelemetryRecord):
        raise TypeError(f"screen expects a TelemetryRecord, got {type(record).__name__}")
    counters = screen_counters(record)
    clock = screen_clock(record, counters)
    cadence = screen_cadence(record)
    ranges = screen_ranges(record)
    validity = screen_validity(record)
    residuals = screen_local_residual(record, window=residual_window, z_threshold=residual_z)
    gaps = consolidate_gaps(record, counters)
    findings = ScreenFindings(
        n_frames=len(record),
        counter=counters,
        clock=clock,
        cadence=cadence,
        ranges=ranges,
        residuals=residuals,
        validity_failures=validity,
        gaps=gaps,
        duplicate_indices=tuple(
            finding.index for finding in counters if finding.kind == "repeated_counter"
        ),
        reorder_indices=tuple(finding.index for finding in counters if finding.kind == "reorder"),
        clock_reset_indices=tuple(
            finding.index for finding in clock if finding.kind == "clock_reset"
        ),
    )
    disagreeing = [gap for gap in findings.gaps if not gap.agrees]
    if disagreeing:
        findings.notes.append(
            f"{len(disagreeing)} of {len(findings.gaps)} gaps have a time-domain frame deficit "
            "that disagrees with the counter-domain deficit; that is evidence about the clock, "
            "not about the data"
        )
    return findings
