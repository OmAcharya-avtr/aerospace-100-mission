"""Declared link conditions: rates, asymmetry, light time, loss and outages.

Every quantity here is *declared by the operator*, not estimated from
telemetry. That is the whole premise of this package: the operator knows,
approximately, what the link does, and wants to know what a file transfer over
it will cost. Nothing in this module measures a real link.

Units are SI throughout and are stated on every field. Bit rates are bits per
second of *channel* capacity available to the file transfer, so framing and
coding overhead below the transfer layer is assumed already removed.

The outage process is the two-state burst model of Gilbert (1960),
"Capacity of a burst-noise channel", *Bell System Technical Journal* 39(5),
1253--1265, in its continuous-time form: alternating exponentially distributed
up and down intervals. It is the simplest process that produces *correlated*
loss, which is the regime where the closed-form geometric-loss result in
:mod:`relfiledelivery.analytic` stops applying. Validity range: it reproduces
mean outage duration and duty cycle by construction and nothing else. A real
link loss process has structure this model does not have, and a prediction from
it is a statement about the model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

# Two windows whose boundaries agree to within this relative tolerance are
# treated as adjacent rather than overlapping. The strict comparison that this
# replaces rejected a window list produced by ``uniform_windows`` itself:
# ``uniform_windows(11, 114.06854261351316, 114.06854261351316)`` builds
# back-to-back windows whose tenth boundary lands two parts in 1e16 apart from
# the eleventh start, because ``first + 10 * period`` and ``first + 9 * period +
# duration`` are not the same floating-point number. The condition being tested
# -- do these windows overlap -- is a question about an exact quantity being
# approximated, which is the one thing a strict floating-point comparison cannot
# answer. Hypothesis found this; it is pinned in tests/test_regressions.py.
WINDOW_ADJACENCY_TOLERANCE = 1e-9


@dataclass(frozen=True)
class ContactWindow:
    """One contact window, in seconds on a common mission timeline.

    Parameters
    ----------
    start_s, end_s:
        Window bounds in seconds. ``end_s`` must be strictly greater than
        ``start_s``.
    """

    start_s: float
    end_s: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.start_s) or not math.isfinite(self.end_s):
            raise ValueError("contact window bounds must be finite seconds")
        if self.end_s <= self.start_s:
            raise ValueError(
                f"contact window must have positive duration; got start_s={self.start_s}, "
                f"end_s={self.end_s}"
            )

    @property
    def duration_s(self) -> float:
        """Window length in seconds."""
        return self.end_s - self.start_s

    def contains(self, t_s: float) -> bool:
        """True when ``t_s`` lies in ``[start_s, end_s)``."""
        return self.start_s <= t_s < self.end_s


@dataclass(frozen=True)
class OutageModel:
    """Continuous-time two-state burst model (Gilbert 1960).

    The link alternates between up and down intervals drawn from exponential
    distributions with the given means. It starts up at the beginning of the
    first contact. A protocol data unit whose transmission interval intersects
    any down interval is lost in full; partial corruption is not modelled,
    because a unit that fails its frame check is discarded whole.

    Parameters
    ----------
    mean_up_s:
        Mean duration of an up interval, seconds. ``math.inf`` means the link
        never drops, which is the no-outage limit.
    mean_down_s:
        Mean duration of a down interval, seconds. Zero also means the
        no-outage limit.
    """

    mean_up_s: float
    mean_down_s: float

    def __post_init__(self) -> None:
        if self.mean_up_s <= 0.0:
            raise ValueError(f"mean_up_s must be positive seconds; got {self.mean_up_s}")
        if self.mean_down_s < 0.0:
            raise ValueError(f"mean_down_s must be non-negative seconds; got {self.mean_down_s}")
        if math.isinf(self.mean_down_s):
            raise ValueError("mean_down_s must be finite; an infinite outage never completes")

    @classmethod
    def none(cls) -> OutageModel:
        """The no-outage limit: the link is up for all time."""
        return cls(mean_up_s=math.inf, mean_down_s=0.0)

    @property
    def is_outage_free(self) -> bool:
        """True when the model produces no down intervals at all."""
        return self.mean_down_s == 0.0 or math.isinf(self.mean_up_s)

    @property
    def duty_cycle(self) -> float:
        """Long-run fraction of time the link is up, dimensionless in ``(0, 1]``.

        ``mean_up_s / (mean_up_s + mean_down_s)``, the stationary probability of
        the up state of the two-state process. Exactly 1.0 in the no-outage
        limit.
        """
        if self.is_outage_free:
            return 1.0
        return self.mean_up_s / (self.mean_up_s + self.mean_down_s)


@dataclass(frozen=True)
class LinkConditions:
    """The declared link a transfer has to cross.

    Parameters
    ----------
    forward_rate_bps:
        Spacecraft-bound (or ground-bound) rate carrying the file, bits per
        second.
    return_rate_bps:
        Rate carrying acknowledgements, bits per second. Asymmetry is the
        ratio ``forward_rate_bps / return_rate_bps`` and is usually large.
    one_way_light_time_s:
        One-way propagation delay, seconds.
    pdu_loss_probability:
        Probability that a protocol data unit is lost *independently* of the
        outage process, dimensionless in ``[0, 1)``. This is the residual loss
        after coding, and it is the parameter the closed form in
        :mod:`relfiledelivery.analytic` is exact in. A value of 1.0 is rejected
        because nothing is ever delivered.
    outage:
        The burst-loss process. Defaults to the no-outage limit.
    windows:
        Contact windows, in increasing time order and non-overlapping. An empty
        tuple means continuous contact from time zero.
    nak_bytes:
        Size of one negative-acknowledgement unit on the return link, bytes.
        Declared constant. A real CFDP NAK grows with the number of missing
        segments; see ``nak_bytes_per_segment``.
    nak_bytes_per_segment:
        Additional return-link bytes per missing segment named in a NAK.
        Defaults to zero, which is the condition under which the closed form is
        exact. Setting it non-zero makes the turnaround cost depend on how much
        was lost, so :func:`relfiledelivery.analytic.expected_delivery_time`
        refuses to produce a closed form and only the simulation applies.
    eof_bytes:
        Size of the end-of-file unit on the forward link, bytes.
    """

    forward_rate_bps: float
    return_rate_bps: float
    one_way_light_time_s: float = 0.0
    pdu_loss_probability: float = 0.0
    outage: OutageModel = field(default_factory=OutageModel.none)
    windows: tuple[ContactWindow, ...] = ()
    nak_bytes: int = 64
    nak_bytes_per_segment: float = 0.0
    eof_bytes: int = 16

    def __post_init__(self) -> None:
        if self.forward_rate_bps <= 0.0 or not math.isfinite(self.forward_rate_bps):
            raise ValueError(
                f"forward_rate_bps must be a positive finite bit rate; got {self.forward_rate_bps}"
            )
        if self.return_rate_bps <= 0.0 or not math.isfinite(self.return_rate_bps):
            raise ValueError(
                f"return_rate_bps must be a positive finite bit rate; got {self.return_rate_bps}"
            )
        if self.one_way_light_time_s < 0.0 or not math.isfinite(self.one_way_light_time_s):
            raise ValueError(
                "one_way_light_time_s must be a non-negative finite number of seconds; got "
                f"{self.one_way_light_time_s}"
            )
        p = self.pdu_loss_probability
        if not (0.0 <= p < 1.0):
            raise ValueError(
                f"pdu_loss_probability must lie in [0, 1); got {p}. A loss probability of 1 "
                "delivers nothing and has no expected delivery time."
            )
        if self.nak_bytes <= 0:
            raise ValueError(f"nak_bytes must be positive; got {self.nak_bytes}")
        if self.nak_bytes_per_segment < 0.0:
            raise ValueError(
                f"nak_bytes_per_segment must be non-negative; got {self.nak_bytes_per_segment}"
            )
        if self.eof_bytes <= 0:
            raise ValueError(f"eof_bytes must be positive; got {self.eof_bytes}")
        if not isinstance(self.outage, OutageModel):
            raise TypeError(f"outage must be an OutageModel; got {type(self.outage).__name__}")
        windows = tuple(self.windows)
        object.__setattr__(self, "windows", windows)
        for earlier, later in zip(windows, windows[1:], strict=False):
            scale = max(1.0, abs(earlier.end_s), abs(later.start_s))
            if later.start_s < earlier.end_s - WINDOW_ADJACENCY_TOLERANCE * scale:
                raise ValueError(
                    "contact windows must be non-overlapping and in increasing time order; "
                    f"window starting {later.start_s} overlaps one ending {earlier.end_s}"
                )

    @property
    def asymmetry(self) -> float:
        """``forward_rate_bps / return_rate_bps``, dimensionless."""
        return self.forward_rate_bps / self.return_rate_bps

    @property
    def is_continuous(self) -> bool:
        """True when no contact windows are declared, so contact never breaks."""
        return len(self.windows) == 0

    @property
    def contact_duty_cycle(self) -> float:
        """Fraction of the declared contact span actually inside a window.

        1.0 for continuous contact. For declared windows it is the summed
        window duration over the span from the first window start to the last
        window end, so it does not include dead time before the first contact.

        The result is clamped to 1.0. Non-overlapping windows inside the span
        cannot sum to more than the span in exact arithmetic, but they can in
        floating point: contiguous windows from ``uniform_windows(11, d, d)``
        sum to ``1.0000000000000002`` of their span, because the summed
        durations and the differenced endpoints are two different roundings of
        the same exact quantity. Hypothesis found this; it is pinned in
        tests/test_regressions.py. The clamp removes a rounding artefact and
        nothing else -- it is the reason this property can be documented as a
        probability, and :attr:`availability` is used as a divisor downstream.
        """
        if self.is_continuous:
            return 1.0
        span = self.windows[-1].end_s - self.windows[0].start_s
        if span <= 0.0:
            return 1.0
        return min(1.0, sum(w.duration_s for w in self.windows) / span)

    @property
    def availability(self) -> float:
        """Product of contact duty cycle and outage duty cycle, dimensionless.

        The fraction of wall-clock time, over the contact span, during which a
        forward unit can actually get through. This is the inflation factor the
        analytic predictor uses to extend the no-outage closed form, and
        measuring how badly that extension does is a deliverable of this
        package.
        """
        return self.contact_duty_cycle * self.outage.duty_cycle

    def without_outages(self) -> LinkConditions:
        """A copy with the outage process removed and contact made continuous.

        This is the regime the closed form is exact in.
        """
        return replace(self, outage=OutageModel.none(), windows=())

    def nak_seconds(self, missing_segments: int) -> float:
        """Return-link time for one NAK naming ``missing_segments`` segments, seconds."""
        if missing_segments < 0:
            raise ValueError(f"missing_segments must be non-negative; got {missing_segments}")
        wire_bytes = self.nak_bytes + self.nak_bytes_per_segment * missing_segments
        return 8.0 * wire_bytes / self.return_rate_bps

    @property
    def eof_seconds(self) -> float:
        """Forward-link time for the end-of-file unit, seconds."""
        return 8.0 * self.eof_bytes / self.forward_rate_bps

    def describe(self) -> list[str]:
        """Human-readable lines for a report. No units are left implicit."""
        lines = [
            f"forward rate            : {self.forward_rate_bps:.6g} bit/s",
            f"return rate             : {self.return_rate_bps:.6g} bit/s",
            f"asymmetry (fwd/ret)     : {self.asymmetry:.6g}",
            f"one-way light time      : {self.one_way_light_time_s:.6g} s",
            f"independent PDU loss    : {self.pdu_loss_probability:.6g}",
        ]
        if self.outage.is_outage_free:
            lines.append("outage process          : none (no-outage limit)")
        else:
            lines.append(
                f"outage process          : mean up {self.outage.mean_up_s:.6g} s, mean down "
                f"{self.outage.mean_down_s:.6g} s, duty {self.outage.duty_cycle:.6f}"
            )
        if self.is_continuous:
            lines.append("contact                 : continuous")
        else:
            lines.append(
                f"contact                 : {len(self.windows)} windows, duty "
                f"{self.contact_duty_cycle:.6f}"
            )
        lines.append(f"availability            : {self.availability:.6f}")
        return lines


class OutageTrace:
    """One seeded realisation of an :class:`OutageModel`, extended lazily.

    The realisation is an alternating sequence of up and down intervals
    starting up at ``t0``. Boundaries are generated on demand, so a transfer
    that finishes early never pays for the rest of the timeline, and a transfer
    that runs long is never truncated.

    The trace is a function of the seeded generator alone, so two traces built
    from equal seeds are identical. That is asserted by test.
    """

    def __init__(self, model: OutageModel, rng: np.random.Generator, t0_s: float = 0.0) -> None:
        self.model = model
        self._rng = rng
        self._t0 = float(t0_s)
        # ``_edges[i]`` is the end of the i-th interval. Interval 0 is up,
        # interval 1 down, and so on; parity of the index gives the state.
        self._edges: list[float] = []
        self._horizon = self._t0

    def _extend_to(self, t_s: float) -> None:
        if self.model.is_outage_free:
            return
        while self._horizon <= t_s:
            index = len(self._edges)
            mean = self.model.mean_up_s if index % 2 == 0 else self.model.mean_down_s
            duration = float(self._rng.exponential(mean))
            self._horizon += duration
            self._edges.append(self._horizon)

    def is_down(self, t_s: float) -> bool:
        """True when the link is in a down interval at ``t_s``."""
        if self.model.is_outage_free:
            return False
        self._extend_to(t_s)
        # Index of the interval containing t_s.
        index = int(np.searchsorted(self._edges, t_s, side="right"))
        return index % 2 == 1

    def intersects_outage(self, t0_s: float, t1_s: float) -> bool:
        """True when ``[t0_s, t1_s)`` intersects any down interval.

        A protocol data unit is lost if any part of its transmission falls in
        an outage, because the frame check covers the whole unit.
        """
        if self.model.is_outage_free:
            return False
        if t1_s <= t0_s:
            return self.is_down(t0_s)
        self._extend_to(t1_s)
        start = int(np.searchsorted(self._edges, t0_s, side="right"))
        end = int(np.searchsorted(self._edges, t1_s, side="left"))
        # Any interval index in [start, end] with odd parity is a down interval
        # overlapping the transmission.
        for index in range(start, end + 1):
            if index % 2 == 1:
                return True
        return False

    def outage_seconds_in(self, t0_s: float, t1_s: float) -> float:
        """Total down time inside ``[t0_s, t1_s)``, seconds."""
        if self.model.is_outage_free or t1_s <= t0_s:
            return 0.0
        self._extend_to(t1_s)
        total = 0.0
        previous = self._t0
        for index, edge in enumerate(self._edges):
            lo, hi = previous, edge
            previous = edge
            if index % 2 == 0:
                continue
            total += max(0.0, min(hi, t1_s) - max(lo, t0_s))
            if lo > t1_s:
                break
        return total


class ContactSchedule:
    """Maps requested transmission durations onto the declared contact windows.

    A protocol data unit is never split across a contact boundary: if the
    remaining time in the current window cannot carry it, the transmission
    starts at the beginning of the next window. That is a declared modelling
    choice and not a property of any protocol.
    """

    def __init__(self, windows: tuple[ContactWindow, ...]) -> None:
        self.windows = tuple(windows)

    @property
    def is_continuous(self) -> bool:
        """True when no windows are declared."""
        return len(self.windows) == 0

    @property
    def horizon_s(self) -> float:
        """End of the last declared window, or ``math.inf`` for continuous contact."""
        return math.inf if self.is_continuous else self.windows[-1].end_s

    def start_time_s(self) -> float:
        """Earliest time any transmission can begin, seconds."""
        return 0.0 if self.is_continuous else self.windows[0].start_s

    def place(self, after_s: float, duration_s: float) -> tuple[float, float, int] | None:
        """Place a transmission of ``duration_s`` at or after ``after_s``.

        Returns ``(start_s, end_s, window_index)``, or ``None`` when no
        declared window can carry it. ``window_index`` is ``-1`` for continuous
        contact.
        """
        if duration_s < 0.0:
            raise ValueError(f"duration_s must be non-negative; got {duration_s}")
        if self.is_continuous:
            start = max(after_s, 0.0)
            return start, start + duration_s, -1
        for index, window in enumerate(self.windows):
            start = max(after_s, window.start_s)
            if start >= window.end_s:
                continue
            if start + duration_s <= window.end_s:
                return start, start + duration_s, index
        return None

    def window_index_at(self, t_s: float) -> int:
        """Index of the window containing ``t_s``, ``-1`` for continuous contact.

        Raises :class:`ValueError` when ``t_s`` is outside every declared window.
        """
        if self.is_continuous:
            return -1
        for index, window in enumerate(self.windows):
            if window.contains(t_s):
                return index
        raise ValueError(f"t_s={t_s} lies outside every declared contact window")


def uniform_windows(
    count: int, duration_s: float, period_s: float, first_start_s: float = 0.0
) -> tuple[ContactWindow, ...]:
    """``count`` windows of ``duration_s``, one every ``period_s`` seconds.

    A convenience for declaring a repeating ground-station pass pattern. It is
    not an orbit propagation and makes no claim about when a real pass occurs.
    """
    if count <= 0:
        raise ValueError(f"count must be positive; got {count}")
    if duration_s <= 0.0:
        raise ValueError(f"duration_s must be positive seconds; got {duration_s}")
    if period_s < duration_s:
        raise ValueError(
            f"period_s={period_s} is shorter than duration_s={duration_s}; windows would overlap"
        )
    starts = [first_start_s + index * period_s for index in range(count)]
    return tuple(ContactWindow(start, start + duration_s) for start in starts)
