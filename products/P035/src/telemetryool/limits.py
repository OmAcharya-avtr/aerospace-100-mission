"""Out-of-limit checking with full operational semantics.

This is the part of the package that has no direct equivalent in the mature
anomaly-detection libraries: not the detection statistic, but the bookkeeping a
real housekeeping-telemetry monitor has to get right.

Semantics implemented, each a deliberate choice and each pinned by a test
=========================================================================

**Soft and hard limits.** Four thresholds per channel per mode, ordered
``hard_low <= soft_low <= soft_high <= hard_high``.  Any bound may be ``None``
(unbounded on that side).  A sample's *breach level* is
:class:`BreachLevel`: ``IN_LIMIT``, ``SOFT`` or ``HARD``, with ``HARD`` taking
precedence.

**Per-channel validity mask.** Each sample carries a validity flag -- the frame
was marked stale, the sensor reported a fault, the value failed a
reasonableness screen upstream.  An invalid sample is *not* evidence either way.
The default :class:`InvalidPolicy` is ``HOLD``: counters and latched state are
left untouched and the sample is reported with ``level = None``.  ``RESET``
zeroes the raise counters (treat the gap as breaking the run) and ``BREACH``
counts the sample as a hard breach (fail-safe).  No policy silently treats an
invalid sample as in-limit, because that is the failure mode that hides a dead
sensor.

**Persistence / debounce.** Separate consecutive-sample counts to raise a soft
alarm and to raise a hard alarm, plus a separate count of consecutive samples
*below* the latched state required to step the latch down one level.  The latch
therefore clears HARD -> SOFT -> NOMINAL, one level per satisfied clear count,
never in one jump.  Raising is immediate once the count is met; clearing is
never immediate unless ``clear_persistence = 1``.

**Mode dependence.** The limit table is selected by mode name, with ``"*"`` as
the fallback entry.  On a mode change the raise and clear counters are zeroed,
because the evidence they accumulated was measured against a limit table that
no longer applies.  The latched alarm state survives the mode change by default
(``latch_across_mode_change = True``); an operator clears it under the new
table, rather than a mode change clearing it for them.

The exact per-sample update order is given in :meth:`OolChecker.update` and
hand-traced in ``tests/test_persistence.py``.

No literature equation appears in this module.  These are operational
conventions, of the kind described in ECSS-E-ST-70-41C (*Telemetry and telecommand
packet utilization*, ESA/ECSS, 2016) service 12 "on-board monitoring", which
defines per-parameter limit checking with a mode-dependent limit set, a validity
parameter and a repetition (persistence) count.  The semantics here are
consistent with that structure; this package is **not** an implementation of
that standard and makes no conformance claim.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "BreachLevel",
    "AlarmState",
    "InvalidPolicy",
    "LimitSet",
    "ChannelSpec",
    "OolSample",
    "OolChecker",
    "TelemetryMonitor",
    "scale_limits",
    "DEFAULT_MODE",
]

#: Key of the fallback limit set used when a mode has no entry of its own.
DEFAULT_MODE = "*"


class BreachLevel(IntEnum):
    """How far outside its limits one sample is."""

    IN_LIMIT = 0
    SOFT = 1
    HARD = 2


class AlarmState(IntEnum):
    """Latched alarm state of a channel.  Numerically equal to BreachLevel."""

    NOMINAL = 0
    SOFT_ALARM = 1
    HARD_ALARM = 2


class InvalidPolicy(IntEnum):
    """What to do with a sample whose validity flag is false.

    HOLD
        Counters and latched state unchanged; the sample contributes nothing.
    RESET
        Raise and clear counters zeroed; the latched state is kept.
    BREACH
        The sample is counted as a HARD breach (fail-safe).
    """

    HOLD = 0
    RESET = 1
    BREACH = 2


@dataclass(frozen=True)
class LimitSet:
    """Four ordered limits for one channel in one mode.

    Parameters
    ----------
    soft_low, soft_high
        Soft (warning) bounds in the channel's engineering units.  ``None``
        means unbounded on that side.
    hard_low, hard_high
        Hard (action) bounds in the channel's engineering units.  ``None``
        means unbounded on that side.

    Raises
    ------
    ValueError
        If the supplied bounds are not ordered
        ``hard_low <= soft_low <= soft_high <= hard_high`` among those present.
    """

    soft_low: float | None = None
    soft_high: float | None = None
    hard_low: float | None = None
    hard_high: float | None = None

    def __post_init__(self) -> None:
        chain = [
            ("hard_low", self.hard_low),
            ("soft_low", self.soft_low),
            ("soft_high", self.soft_high),
            ("hard_high", self.hard_high),
        ]
        present = [(n, float(v)) for n, v in chain if v is not None]
        for (n_a, a), (n_b, b) in zip(present, present[1:], strict=False):
            if a > b:
                raise ValueError(
                    f"limits must satisfy hard_low <= soft_low <= soft_high <= hard_high; "
                    f"got {n_a}={a} > {n_b}={b}"
                )

    def level(self, value: float) -> BreachLevel:
        """Breach level of ``value`` (engineering units) against this limit set."""
        v = float(value)
        if self.hard_high is not None and v > self.hard_high:
            return BreachLevel.HARD
        if self.hard_low is not None and v < self.hard_low:
            return BreachLevel.HARD
        if self.soft_high is not None and v > self.soft_high:
            return BreachLevel.SOFT
        if self.soft_low is not None and v < self.soft_low:
            return BreachLevel.SOFT
        return BreachLevel.IN_LIMIT

    @classmethod
    def symmetric(cls, centre: float, soft: float, hard: float) -> LimitSet:
        """Limit set at ``centre +/- soft`` and ``centre +/- hard`` (same units)."""
        if hard < soft:
            raise ValueError(f"hard half-width {hard} must be >= soft half-width {soft}")
        if soft < 0:
            raise ValueError(f"soft half-width must be >= 0, got {soft}")
        c = float(centre)
        return cls(c - soft, c + soft, c - hard, c + hard)


@dataclass(frozen=True)
class ChannelSpec:
    """Monitoring configuration for one telemetry channel.

    Parameters
    ----------
    name
        Channel identifier.
    units
        Engineering units of the raw value, for reporting only.
    limits
        Mapping from mode name to :class:`LimitSet`.  The key
        :data:`DEFAULT_MODE` (``"*"``) is the fallback.  A mode with no entry
        and no fallback is an error at update time.
    persistence_soft, persistence_hard
        Consecutive breaching samples required to raise a soft / hard alarm.
        Integers >= 1.  ``1`` means raise on the first breaching sample.
    clear_persistence
        Consecutive samples *strictly below* the latched state required to step
        the latch down one level.  Integer >= 1.
    invalid_policy
        See :class:`InvalidPolicy`.
    latch_across_mode_change
        If true (default) a mode change zeroes the counters but keeps the
        latched alarm state.  If false the latch is also cleared.
    """

    name: str
    units: str
    limits: Mapping[str, LimitSet]
    persistence_soft: int = 1
    persistence_hard: int = 1
    clear_persistence: int = 1
    invalid_policy: InvalidPolicy = InvalidPolicy.HOLD
    latch_across_mode_change: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be a non-empty string")
        if not self.limits:
            raise ValueError(f"channel {self.name!r}: limits mapping is empty")
        for field_name in ("persistence_soft", "persistence_hard", "clear_persistence"):
            v = getattr(self, field_name)
            if not isinstance(v, (int, np.integer)) or v < 1:
                raise ValueError(
                    f"channel {self.name!r}: {field_name} must be an integer >= 1, got {v!r}"
                )
        for mode, ls in self.limits.items():
            if not isinstance(ls, LimitSet):
                raise TypeError(
                    f"channel {self.name!r}: limits[{mode!r}] must be a LimitSet, "
                    f"got {type(ls).__name__}"
                )

    def limits_for(self, mode: str) -> LimitSet:
        """Limit set applying in ``mode``, falling back to :data:`DEFAULT_MODE`."""
        if mode in self.limits:
            return self.limits[mode]
        if DEFAULT_MODE in self.limits:
            return self.limits[DEFAULT_MODE]
        raise KeyError(
            f"channel {self.name!r}: no limit set for mode {mode!r} and no "
            f"{DEFAULT_MODE!r} fallback; defined modes: {sorted(self.limits)}"
        )


@dataclass(frozen=True)
class OolSample:
    """Result of one :meth:`OolChecker.update` call.

    Attributes
    ----------
    index
        Zero-based sample counter for this checker.
    value
        The raw value presented, engineering units.
    valid
        Validity flag as presented.
    mode
        Mode in force for this sample.
    level
        Breach level of this sample, or ``None`` when the sample was invalid and
        the policy was HOLD or RESET (no level is attributed).
    state
        Latched alarm state *after* this sample.
    soft_count, hard_count
        Raise counters after this sample.
    below_count
        Consecutive-samples-below-latch counter after this sample.
    raised
        True if ``state`` is higher than before this sample.
    cleared
        True if ``state`` is lower than before this sample.
    """

    index: int
    value: float
    valid: bool
    mode: str
    level: BreachLevel | None
    state: AlarmState
    soft_count: int
    hard_count: int
    below_count: int
    raised: bool
    cleared: bool


class OolChecker:
    """Stateful out-of-limit checker for one channel.

    Examples
    --------
    >>> spec = ChannelSpec("BATT_V", "V", {"*": LimitSet(27.0, 29.0, 26.0, 30.0)},
    ...                    persistence_soft=2, persistence_hard=1, clear_persistence=2)
    >>> ck = OolChecker(spec)
    >>> [ck.update(v).state.name for v in (28.0, 29.5, 29.5, 28.0, 28.0)]
    ['NOMINAL', 'NOMINAL', 'SOFT_ALARM', 'SOFT_ALARM', 'NOMINAL']
    """

    def __init__(self, spec: ChannelSpec, mode: str = DEFAULT_MODE) -> None:
        self.spec = spec
        self._mode = mode
        self._state = AlarmState.NOMINAL
        self._soft = 0
        self._hard = 0
        self._below = 0
        self._index = -1

    # -- introspection ----------------------------------------------------- #

    @property
    def state(self) -> AlarmState:
        """Current latched alarm state."""
        return self._state

    @property
    def mode(self) -> str:
        """Mode currently in force."""
        return self._mode

    @property
    def counters(self) -> tuple[int, int, int]:
        """``(soft_count, hard_count, below_count)``."""
        return (self._soft, self._hard, self._below)

    def reset(self, mode: str | None = None) -> None:
        """Clear the latch and all counters; optionally set a new mode."""
        self._state = AlarmState.NOMINAL
        self._soft = self._hard = self._below = 0
        self._index = -1
        if mode is not None:
            self._mode = mode

    # -- the update rule --------------------------------------------------- #

    def update(self, value: float, valid: bool = True, mode: str | None = None) -> OolSample:
        """Process one sample and return the resulting :class:`OolSample`.

        The update order is fixed and is the specification of this package's
        persistence semantics:

        1. If ``mode`` is given and differs from the mode in force, zero
           ``soft_count``, ``hard_count`` and ``below_count``; clear the latch
           only if ``latch_across_mode_change`` is false.  Then adopt the mode.
        2. If the sample is invalid, apply :class:`InvalidPolicy`.  Under HOLD,
           return immediately with every counter and the latch unchanged and
           ``level = None``.  Under RESET, zero the three counters and return
           with ``level = None``.  Under BREACH, continue to step 3 with
           ``level = BreachLevel.HARD``.
        3. Compute the breach level of the value against the limit set for the
           mode in force (skipped if BREACH already fixed it).
        4. ``soft_count += 1`` if ``level >= SOFT`` else ``soft_count = 0``;
           ``hard_count += 1`` if ``level >= HARD`` else ``hard_count = 0``.
           Both counters saturate at their persistence thresholds.
        5. Escalate: the candidate state is the highest of the current state,
           ``HARD_ALARM`` if ``hard_count >= persistence_hard``, and
           ``SOFT_ALARM`` if ``soft_count >= persistence_soft``.
        6. If the candidate is above the current state, adopt it and zero
           ``below_count``.  Otherwise, if ``level`` is strictly below the
           current state, increment ``below_count`` and, when it reaches
           ``clear_persistence``, step the state down exactly one level and
           zero ``below_count``.  Otherwise zero ``below_count``.

        Parameters
        ----------
        value
            Raw channel value, engineering units.  Must be finite unless the
            sample is marked invalid.
        valid
            Validity flag for this sample.
        mode
            Mode for this sample; ``None`` keeps the mode in force.

        Returns
        -------
        OolSample
        """
        self._index += 1
        if mode is not None and mode != self._mode:
            self._soft = self._hard = self._below = 0
            if not self.spec.latch_across_mode_change:
                self._state = AlarmState.NOMINAL
            self._mode = mode

        before = self._state
        level: BreachLevel | None

        if not valid:
            policy = self.spec.invalid_policy
            if policy is InvalidPolicy.HOLD:
                return self._emit(value, valid, None, before)
            if policy is InvalidPolicy.RESET:
                self._soft = self._hard = self._below = 0
                return self._emit(value, valid, None, before)
            level = BreachLevel.HARD
        else:
            v = float(value)
            if not np.isfinite(v):
                raise ValueError(
                    f"channel {self.spec.name!r}: value {value!r} is not finite; mark the "
                    f"sample invalid instead of passing a non-finite value"
                )
            level = self.spec.limits_for(self._mode).level(v)

        self._soft = min(self._soft + 1, self.spec.persistence_soft) if level >= 1 else 0
        self._hard = min(self._hard + 1, self.spec.persistence_hard) if level >= 2 else 0

        candidate = int(before)
        if self._hard >= self.spec.persistence_hard:
            candidate = max(candidate, int(AlarmState.HARD_ALARM))
        if self._soft >= self.spec.persistence_soft:
            candidate = max(candidate, int(AlarmState.SOFT_ALARM))

        if candidate > int(before):
            self._state = AlarmState(candidate)
            self._below = 0
        elif int(level) < int(before):
            self._below += 1
            if self._below >= self.spec.clear_persistence:
                self._state = AlarmState(int(before) - 1)
                self._below = 0
        else:
            self._below = 0

        return self._emit(value, valid, level, before)

    def _emit(
        self, value: float, valid: bool, level: BreachLevel | None, before: AlarmState
    ) -> OolSample:
        return OolSample(
            index=self._index,
            value=float(value),
            valid=bool(valid),
            mode=self._mode,
            level=level,
            state=self._state,
            soft_count=self._soft,
            hard_count=self._hard,
            below_count=self._below,
            raised=int(self._state) > int(before),
            cleared=int(self._state) < int(before),
        )

    def update_series(
        self,
        values: Sequence[float] | NDArray[np.float64],
        valid: Sequence[bool] | NDArray[np.bool_] | None = None,
        modes: Sequence[str] | None = None,
    ) -> list[OolSample]:
        """Process a whole series, returning one :class:`OolSample` per sample.

        ``valid`` defaults to all true; ``modes`` defaults to the mode in force.
        All three sequences must have the same length.
        """
        vals = np.asarray(values, dtype=float)
        if vals.ndim != 1:
            raise ValueError(f"values must be 1-D, got shape {vals.shape}")
        n = vals.size
        if valid is None:
            valid_arr = np.ones(n, dtype=bool)
        else:
            valid_arr = np.asarray(valid, dtype=bool)
            if valid_arr.shape != (n,):
                raise ValueError(f"valid must have shape ({n},), got {valid_arr.shape}")
        if modes is not None and len(modes) != n:
            raise ValueError(f"modes must have length {n}, got {len(modes)}")
        out = []
        for i in range(n):
            out.append(
                self.update(
                    float(vals[i]),
                    bool(valid_arr[i]),
                    None if modes is None else modes[i],
                )
            )
        return out


class TelemetryMonitor:
    """A bank of :class:`OolChecker` instances sharing a spacecraft mode.

    Parameters
    ----------
    specs
        Iterable of :class:`ChannelSpec`.  Channel names must be unique.
    mode
        Initial mode, applied to every channel.

    Examples
    --------
    >>> specs = [ChannelSpec("A", "K", {"*": LimitSet(-1, 1, -2, 2)}),
    ...          ChannelSpec("B", "K", {"*": LimitSet(-1, 1, -2, 2)})]
    >>> mon = TelemetryMonitor(specs)
    >>> res = mon.update({"A": 0.0, "B": 3.0})
    >>> res["B"].state.name
    'HARD_ALARM'
    >>> mon.active_alarms()
    {'B': <AlarmState.HARD_ALARM: 2>}
    """

    def __init__(self, specs: Iterable[ChannelSpec], mode: str = DEFAULT_MODE) -> None:
        self.checkers: dict[str, OolChecker] = {}
        for spec in specs:
            if spec.name in self.checkers:
                raise ValueError(f"duplicate channel name {spec.name!r}")
            self.checkers[spec.name] = OolChecker(spec, mode)
        if not self.checkers:
            raise ValueError("at least one ChannelSpec is required")
        self._mode = mode

    @property
    def mode(self) -> str:
        """Mode in force for the whole bank."""
        return self._mode

    def set_mode(self, mode: str) -> None:
        """Change the mode for every channel on the next update."""
        self._mode = mode

    def update(
        self,
        frame: Mapping[str, float],
        valid: Mapping[str, bool] | None = None,
        mode: str | None = None,
    ) -> dict[str, OolSample]:
        """Process one telemetry frame.

        Parameters
        ----------
        frame
            Mapping channel name -> raw value.  Every configured channel must be
            present; an unknown channel name is an error.
        valid
            Mapping channel name -> validity flag; missing entries default True.
        mode
            Mode for this frame; ``None`` keeps the current mode.

        Returns
        -------
        dict
            Channel name -> :class:`OolSample`.
        """
        if mode is not None:
            self._mode = mode
        unknown = set(frame) - set(self.checkers)
        if unknown:
            raise KeyError(f"unknown channel(s) in frame: {sorted(unknown)}")
        missing = set(self.checkers) - set(frame)
        if missing:
            raise KeyError(f"frame is missing channel(s): {sorted(missing)}")
        valid = {} if valid is None else valid
        return {
            name: ck.update(frame[name], bool(valid.get(name, True)), self._mode)
            for name, ck in self.checkers.items()
        }

    def active_alarms(self) -> dict[str, AlarmState]:
        """Channels whose latched state is not NOMINAL, with that state."""
        return {n: c.state for n, c in self.checkers.items() if c.state != AlarmState.NOMINAL}

    def reset(self) -> None:
        """Clear every channel's latch and counters."""
        for ck in self.checkers.values():
            ck.reset(self._mode)


def scale_limits(limit_set: LimitSet, factor: float) -> LimitSet:
    """Multiply every present bound of ``limit_set`` by ``factor`` (same units)."""
    if factor <= 0:
        raise ValueError(f"factor must be > 0, got {factor!r}")
    return replace(
        limit_set,
        soft_low=None if limit_set.soft_low is None else limit_set.soft_low * factor,
        soft_high=None if limit_set.soft_high is None else limit_set.soft_high * factor,
        hard_low=None if limit_set.hard_low is None else limit_set.hard_low * factor,
        hard_high=None if limit_set.hard_high is None else limit_set.hard_high * factor,
    )
