"""Contact plan: windows, rate profiles, and the capacity integral.

A contact window is not a rectangle. The achievable bit rate over a pass varies
with elevation because slant range and atmospheric loss do, and an adaptive
modem steps its modulation and coding accordingly. The deliverable capacity of
a window is therefore the integral of the rate over the window,

    C = integral from t_start to t_end of R(t) dt,                          (1)

in bits, with ``R`` in bits per second and ``t`` in seconds. This module
represents ``R`` as a piecewise-linear function of time through a sequence of
breakpoints, for which (1) is the trapezoidal sum over consecutive breakpoints
*exactly*, not approximately -- the trapezoidal rule is exact for a linear
integrand on each sub-interval. ``validation/validate_capacity.py`` checks the
closed-form sum against an independent adaptive quadrature of the same callable
through :func:`scipy.integrate.quad`, which shares no code with it.

The piecewise-linear rate profile is a modelling choice, not physics. A real
link budget gives rate as a step function of elevation through a modcod table
(Mod-Cod switching is discrete), and a step profile is representable here by
placing two breakpoints at the same time; a genuinely continuous rate law such
as the Shannon capacity of the instantaneous carrier-to-noise ratio is only
approximated. The error of that approximation is a property of the breakpoint
spacing the caller supplies and is not estimated by this package.

Reliability is a per-window scalar probability in ``[0, 1]`` that the window is
achieved at all: weather at a Ka-band station, antenna availability, a shared
ground segment. It multiplies value, not capacity, which encodes the assumption
that a window either happens as planned or does not happen -- there is no
partial pass in this model. That assumption is restrictive and is stated in
README.md under Limitations.

References
----------
Shannon, C. E. (1948), A mathematical theory of communication, Bell System
Technical Journal 27(3), 379-423 and 27(4), 623-656. Source coding and channel
capacity; used in :mod:`downlinkpick.compress` for the entropy floor, and the
reason a rate profile rather than a constant rate is the honest representation
of an adaptive link.

CCSDS 131.0-B-5 (2023), TM Synchronization and Channel Coding. The existence of
a discrete modcod table, hence the step-profile remark above. No numerical value
in this repository is taken from it.
"""

from __future__ import annotations

import bisect
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field

#: Absolute tolerance in bits, used where a bit count is compared against a
#: capacity in a user-facing query -- :meth:`ContactWindow.time_for_bits` --
#: rather than inside a feasibility test. Bits are integers in reality, so one
#: bit of slop on "can this window carry this much" is harmless and avoids
#: rejecting a request that is short by a rounding error.
CAPACITY_TOLERANCE_BITS = 1.0

#: Relative slack allowed when *this package* tests committed bits against a
#: window capacity, dimensionless. The quantity being tested is a sum of
#: products of doubles, so its rounding is proportional to the capacity and the
#: tolerance must be too. An absolute one-bit tolerance was used here first and
#: a Hypothesis property test falsified it: on a window of 8 bits carrying a
#: product of 9 bits, the one-bit slack made the exhaustive enumeration call the
#: selection feasible while the MILP, whose own feasibility test is relative,
#: called it infeasible, and the two routes disagreed about the optimum. The
#: counterexample is pinned in ``tests/test_regressions.py``.
ROUNDING_TOLERANCE_REL = 1.0e-12

#: Relative slack allowed when checking a selection that came from *somewhere
#: else*, typically a branch-and-bound solver. HiGHS has a primal feasibility
#: tolerance of its own, around 1e-7 relative, so a returned solution may sit
#: that far outside a constraint; an invariant checker that did not allow for it
#: would report the solver's tolerance as a defect in this package.
CHECK_TOLERANCE_REL = 1.0e-6


def feasibility_slack_bits(capacity_bits: float) -> float:
    """Slack in bits allowed by this package's own feasibility tests.

    ``ROUNDING_TOLERANCE_REL * capacity``, floored at zero. Relative because
    the rounding of the committed-bits sum is proportional to its magnitude.
    """
    return ROUNDING_TOLERANCE_REL * max(capacity_bits, 0.0)


def check_slack_bits(capacity_bits: float) -> float:
    """Slack in bits allowed when checking someone else's selection.

    ``max(CAPACITY_TOLERANCE_BITS, CHECK_TOLERANCE_REL * capacity)``, which
    accommodates a solver's own primal feasibility tolerance without accepting a
    violation that would cost a real product.
    """
    return max(CAPACITY_TOLERANCE_BITS, CHECK_TOLERANCE_REL * max(capacity_bits, 0.0))


@dataclass(frozen=True)
class ContactWindow:
    """One ground contact with a piecewise-linear achievable-rate profile.

    Parameters
    ----------
    name:
        Unique identifier within a :class:`ContactPlan`.
    start_s, end_s:
        Window bounds in seconds after the plan epoch, ``start_s < end_s``.
    rate_breakpoints:
        Ascending ``(time_s, rate_bits_per_second)`` pairs. The first time must
        equal ``start_s`` and the last must equal ``end_s``; rates must be
        non-negative. Between breakpoints the rate is linear in time. Two
        breakpoints at the same time encode a rate step.
    reliability:
        Probability in ``[0, 1]`` that the window is achieved. Multiplies value.
    station:
        Ground-station label, carried for reporting only.
    """

    name: str
    start_s: float
    end_s: float
    rate_breakpoints: tuple[tuple[float, float], ...]
    reliability: float = 1.0
    station: str = "unknown"
    _times: tuple[float, ...] = field(init=False, repr=False, compare=False)
    _rates: tuple[float, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("ContactWindow.name must be a non-empty string")
        if not (self.end_s > self.start_s):
            raise ValueError(
                f"{self.name}: end_s must exceed start_s, got start_s={self.start_s!r} "
                f"end_s={self.end_s!r}"
            )
        if self.start_s < 0.0:
            raise ValueError(
                f"{self.name}: start_s must be non-negative seconds after the plan epoch, "
                f"got {self.start_s!r}"
            )
        if not (0.0 <= self.reliability <= 1.0):
            raise ValueError(
                f"{self.name}: reliability must lie in [0, 1], got {self.reliability!r}"
            )
        points = tuple(self.rate_breakpoints)
        if len(points) < 2:
            raise ValueError(f"{self.name}: rate_breakpoints needs at least two points")
        times = tuple(float(t) for t, _ in points)
        rates = tuple(float(r) for _, r in points)
        if any(later < earlier for earlier, later in zip(times, times[1:], strict=False)):
            raise ValueError(f"{self.name}: rate_breakpoint times must be non-decreasing")
        if times[0] != self.start_s or times[-1] != self.end_s:
            raise ValueError(
                f"{self.name}: rate_breakpoints must span exactly [start_s, end_s] = "
                f"[{self.start_s!r}, {self.end_s!r}], got [{times[0]!r}, {times[-1]!r}]"
            )
        if any(rate < 0.0 for rate in rates):
            raise ValueError(f"{self.name}: rate_breakpoints must have non-negative rates")
        object.__setattr__(self, "_times", times)
        object.__setattr__(self, "_rates", rates)

    @property
    def duration_s(self) -> float:
        """Window length in seconds."""
        return self.end_s - self.start_s

    @property
    def peak_rate_bps(self) -> float:
        """Largest breakpoint rate in bits per second."""
        return max(self._rates)

    @property
    def final_rate_bps(self) -> float:
        """Rate at ``end_s`` in bits per second."""
        return self._rates[-1]

    def rate_bps(self, time_s: float) -> float:
        """Achievable rate at ``time_s`` in bits per second.

        Outside ``[start_s, end_s]`` the rate is zero: no contact, no bits. At a
        rate step (two breakpoints sharing a time) the later segment's value is
        returned, which makes the function right-continuous there.
        """
        if time_s < self.start_s or time_s > self.end_s:
            return 0.0
        times = self._times
        rates = self._rates
        if time_s == times[-1]:
            return rates[-1]
        index = bisect.bisect_right(times, time_s) - 1
        index = min(max(index, 0), len(times) - 2)
        while index + 2 < len(times) and times[index + 1] == time_s:
            index += 1
        t0, t1 = times[index], times[index + 1]
        r0, r1 = rates[index], rates[index + 1]
        if t1 == t0:
            return r1
        return r0 + (r1 - r0) * (time_s - t0) / (t1 - t0)

    def delivered_bits(self, from_s: float, to_s: float) -> float:
        """Bits deliverable over ``[from_s, to_s]`` intersected with the window.

        Closed-form evaluation of equation (1) restricted to the interval: the
        trapezoidal sum over whole breakpoint segments plus the exact trapezoid
        of the two partial end segments. Exact for the piecewise-linear profile
        in exact arithmetic, and agreeing with an independent adaptive
        quadrature to within rounding; see ``validation/validate_capacity.py``.

        The rate at a segment endpoint is interpolated *within that segment*
        rather than read back through :meth:`rate_bps`. The difference matters
        only where two breakpoints share a time -- a rate step -- and there it
        matters by six per cent: :meth:`rate_bps` is right-continuous at a step,
        so reading the left segment's upper endpoint through it returns the
        step's higher value and over-integrates the left segment. The first run
        of ``validation/validate_capacity.py`` found exactly this, which is why
        the irregular profile in that script has a deliberate step in it.
        """
        lower = max(from_s, self.start_s)
        upper = min(to_s, self.end_s)
        if not (upper > lower):
            return 0.0
        total = 0.0
        times = self._times
        rates = self._rates
        for index in range(len(times) - 1):
            t0, t1 = times[index], times[index + 1]
            if t1 <= lower or t0 >= upper or t1 == t0:
                continue
            a = max(t0, lower)
            b = min(t1, upper)
            if not (b > a):
                continue
            r0, r1 = rates[index], rates[index + 1]
            slope = (r1 - r0) / (t1 - t0)
            total += 0.5 * ((r0 + slope * (a - t0)) + (r0 + slope * (b - t0))) * (b - a)
        return total

    def capacity_bits(self) -> float:
        """Deliverable capacity of the whole window in bits, equation (1)."""
        return self.delivered_bits(self.start_s, self.end_s)

    def time_for_bits(self, bits: float, *, from_s: float | None = None) -> float:
        """Seconds from ``from_s`` needed to deliver ``bits``, in seconds.

        Inverts the monotone non-decreasing integral (1) by bisection to a
        relative tolerance of ``1e-12`` on the elapsed time. Raises
        :class:`ValueError` if the remaining window cannot deliver ``bits``.
        """
        if bits < 0.0:
            raise ValueError(f"{self.name}: bits must be non-negative, got {bits!r}")
        origin = self.start_s if from_s is None else max(from_s, self.start_s)
        available = self.delivered_bits(origin, self.end_s)
        if bits > available + CAPACITY_TOLERANCE_BITS:
            raise ValueError(
                f"{self.name}: cannot deliver {bits:.6e} bits from t={origin:.6f} s; "
                f"only {available:.6e} bits remain in the window"
            )
        if bits <= 0.0:
            return 0.0
        low, high = 0.0, self.end_s - origin
        for _ in range(200):
            middle = 0.5 * (low + high)
            if self.delivered_bits(origin, origin + middle) < bits:
                low = middle
            else:
                high = middle
            if high - low <= 1.0e-12 * max(1.0, high):
                break
        return 0.5 * (low + high)

    def extended(self, extra_s: float) -> ContactWindow:
        """A copy whose end is ``extra_s`` later, holding the final rate.

        Used by :mod:`downlinkpick.marginal` to price the next unit of contact
        time. Holding the final rate is the optimistic reading of a profile that
        is simply not defined past ``end_s``; a real extension would follow the
        pass geometry down and deliver less, measured at a factor of 2.0 for a
        300-second extension of the shipped sine profile in
        ``validation/validate_marginal.py``.

        Note what else the extension does: it moves the window's *end*, which is
        also the epoch at which everything in the window is delivered. A product
        whose expiry falls inside the extension therefore loses this window
        entirely. Extending a pass can lower the achievable value, and
        :attr:`downlinkpick.marginal.WindowMarginal.expiry_crossings` reports
        when that is happening.
        """
        if extra_s < 0.0:
            raise ValueError(f"{self.name}: extra_s must be non-negative, got {extra_s!r}")
        if extra_s == 0.0:
            return self
        points = (*self.rate_breakpoints, (self.end_s + extra_s, self.final_rate_bps))
        return ContactWindow(
            name=self.name,
            start_s=self.start_s,
            end_s=self.end_s + extra_s,
            rate_breakpoints=points,
            reliability=self.reliability,
            station=self.station,
        )


@dataclass(frozen=True)
class ContactPlan:
    """Non-overlapping contact windows in ascending time order.

    Overlap is rejected. The spacecraft modelled here has one transmit chain, so
    two overlapping windows would double-count capacity that the vehicle cannot
    deliver. A multi-antenna vehicle is out of scope and is stated as such in
    README.md under Limitations.
    """

    windows: tuple[ContactWindow, ...]
    _index: dict[str, int] = field(init=False, repr=False, compare=False)

    def __init__(self, windows: Iterable[ContactWindow]) -> None:
        items = tuple(windows)
        if not items:
            raise ValueError("ContactPlan must contain at least one ContactWindow")
        names: dict[str, int] = {}
        for position, window in enumerate(items):
            if not isinstance(window, ContactWindow):
                raise TypeError(
                    f"ContactPlan element {position} is {type(window).__name__}, "
                    "not ContactWindow"
                )
            if window.name in names:
                raise ValueError(f"duplicate window name in contact plan: {window.name!r}")
            names[window.name] = position
        ordered = sorted(items, key=lambda w: (w.start_s, w.name))
        for earlier, later in zip(ordered, ordered[1:], strict=False):
            if later.start_s < earlier.end_s:
                raise ValueError(
                    f"windows {earlier.name!r} and {later.name!r} overlap in time "
                    f"([{earlier.start_s}, {earlier.end_s}] and "
                    f"[{later.start_s}, {later.end_s}]); one transmit chain is assumed"
                )
        object.__setattr__(self, "windows", tuple(ordered))
        object.__setattr__(self, "_index", {w.name: i for i, w in enumerate(ordered)})

    def __len__(self) -> int:
        return len(self.windows)

    def __iter__(self) -> Iterator[ContactWindow]:
        return iter(self.windows)

    def __getitem__(self, name: str) -> ContactWindow:
        try:
            return self.windows[self._index[name]]
        except KeyError as error:
            raise KeyError(f"no window named {name!r} in contact plan") from error

    def __contains__(self, name: object) -> bool:
        return name in self._index

    @property
    def names(self) -> tuple[str, ...]:
        """Window names in ascending start time."""
        return tuple(window.name for window in self.windows)

    def capacity_bits(self) -> dict[str, float]:
        """Per-window capacity in bits, equation (1)."""
        return {window.name: window.capacity_bits() for window in self.windows}

    def total_capacity_bits(self) -> float:
        """Sum of per-window capacities in bits."""
        return sum(window.capacity_bits() for window in self.windows)

    def total_contact_s(self) -> float:
        """Sum of window durations in seconds."""
        return sum(window.duration_s for window in self.windows)

    def windows_ending_by(self, deadline_s: float) -> tuple[str, ...]:
        """Names of windows whose end is at or before ``deadline_s``."""
        return tuple(w.name for w in self.windows if w.end_s <= deadline_s)

    def replace_window(self, window: ContactWindow) -> ContactPlan:
        """A copy with the same-named window replaced by ``window``."""
        if window.name not in self._index:
            raise KeyError(f"no window named {window.name!r} in contact plan")
        return ContactPlan(
            window if existing.name == window.name else existing for existing in self.windows
        )

    def extended_window(self, name: str, extra_s: float) -> ContactPlan:
        """A copy with window ``name`` extended by ``extra_s`` seconds.

        Raises :class:`KeyError` on an unknown window name -- a typo must not
        silently return an unchanged plan, which is what a plain map over the
        windows would do -- and :class:`ValueError` if the extension would
        overlap the next window, because an overlapping plan is not
        representable.
        """
        if name not in self._index:
            raise KeyError(f"no window named {name!r} in contact plan")
        return ContactPlan(
            w.extended(extra_s) if w.name == name else w for w in self.windows
        )


def rate_profile_from_elevation(
    start_s: float,
    end_s: float,
    peak_rate_bps: float,
    *,
    floor_fraction: float = 0.25,
    points: int = 9,
) -> tuple[tuple[float, float], ...]:
    """A symmetric rise-and-fall rate profile for a single pass.

    The profile is ``peak_rate_bps * (floor_fraction + (1 - floor_fraction) *
    sin(pi * u))`` sampled at ``points`` equally spaced ``u`` in ``[0, 1]``, with
    the endpoints forced to ``floor_fraction * peak_rate_bps``. The sine shape is
    a convenient stand-in for the elevation profile of a near-overhead pass, not
    a link budget: it has no antenna pattern, no atmospheric model and no modcod
    table in it. Use measured breakpoints from a real link budget where you have
    them, and treat this helper as a way to make an example reproducible.

    Returns
    -------
    tuple of (float, float)
        Breakpoints in ``(seconds, bits per second)``, suitable for
        :class:`ContactWindow`.
    """
    import math

    if points < 3:
        raise ValueError(f"points must be >= 3, got {points!r}")
    if not (0.0 <= floor_fraction <= 1.0):
        raise ValueError(f"floor_fraction must lie in [0, 1], got {floor_fraction!r}")
    if not (peak_rate_bps > 0.0):
        raise ValueError(f"peak_rate_bps must be positive, got {peak_rate_bps!r}")
    if not (end_s > start_s):
        raise ValueError(f"end_s must exceed start_s, got {start_s!r} and {end_s!r}")
    span = end_s - start_s
    breakpoints: list[tuple[float, float]] = []
    for index in range(points):
        u = index / (points - 1)
        shape = floor_fraction + (1.0 - floor_fraction) * math.sin(math.pi * u)
        if index in (0, points - 1):
            shape = floor_fraction
        breakpoints.append((start_s + u * span, peak_rate_bps * shape))
    return tuple(breakpoints)


def constant_rate_profile(start_s: float, end_s: float, rate_bps: float) -> tuple[
    tuple[float, float], ...
]:
    """A flat rate profile, for which capacity is ``rate_bps * (end_s - start_s)``."""
    if not (rate_bps >= 0.0):
        raise ValueError(f"rate_bps must be non-negative, got {rate_bps!r}")
    if not (end_s > start_s):
        raise ValueError(f"end_s must exceed start_s, got {start_s!r} and {end_s!r}")
    return ((start_s, rate_bps), (end_s, rate_bps))


def rate_callable(window: ContactWindow) -> Sequence[float]:
    """Deprecated placeholder kept out of the public API."""
    raise NotImplementedError
