"""Injection wrapper: faults applied around an unmodified target.

The wrapper implements the *same* call interface as the target it wraps --
``step(step_index, measurement_mapping) -> command_mapping`` -- so a target is
instrumented by substitution, not by edit:

    stepper = InjectionWrapper(GncController(), injections, rng)

With an empty injection list the wrapper is required to be bit-transparent:
the commands it returns are the same binary64 values the bare target returns.
That is asserted in ``tests/test_wrapper.py`` and is the property the whole
"no target modification" claim rests on.

Restrictions, stated rather than hidden: at most one TRANSPORT-stage and at
most one TARGET-stage injection may be active in a single wrapper, because two
competing frame-reordering models have no well-defined composition.  Multiple
SENSOR- and ACTUATOR-stage injections compose in taxonomy order.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .faults import Handler, Injection, Stage, make_handler

SMALLEST_NORMAL = 2.2250738585072014e-308
"""Smallest positive IEEE 754 binary64 normal number (2**-1022)."""

LARGE_MAGNITUDE = 1.3407807929942596e154
"""Threshold above which a single multiplication overflows: sqrt(DBL_MAX).

Not an IEEE 754 class.  It is a declared surveillance threshold, chosen so that
any flagged value squares to infinity, which makes it the smallest magnitude
that is certain to produce an overflow one operation downstream.  Magnitudes
below it are *not* flagged by the monitor even though they may still be
physically absurd -- that is a stated limitation, checked in
``validation/validate_nan_detection.py``."""


class Target(Protocol):
    """The call interface a target must expose. Nothing else is required."""

    def reset(self) -> None: ...

    def step(self, k: int, meas: Mapping[str, float]) -> Mapping[str, float]: ...


@dataclass(frozen=True)
class Event:
    """One thing the wrapper observed. Stored in order of occurrence."""

    step: int
    stage: str
    signal: str
    event: str
    detail: float = 0.0


def classify_value(value: float) -> str | None:
    """Numerical anomaly class of ``value``, or ``None`` if ordinary.

    Three classes follow IEEE 754-2019: ``nan`` (Sec. 6.2), ``inf`` (Sec. 7.4)
    and ``subnormal`` (Sec. 3.4).  The fourth, ``large``, is the declared
    surveillance threshold :data:`LARGE_MAGNITUDE` rather than an IEEE class.
    Exact zero is ordinary, not subnormal.
    """
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf"
    av = abs(value)
    if av > LARGE_MAGNITUDE:
        return "large"
    if value != 0.0 and av < SMALLEST_NORMAL:
        return "subnormal"
    return None


@dataclass
class NumericalMonitor:
    """Watches every value crossing the wrapper boundary.

    The monitor is an observer: it never changes a value.  It records, per
    anomaly class, the first step and signal at which the class was seen on the
    way *in* to the target and on the way *out* of it.  A class seen on the way
    in but never on the way out is reported as ``absorbed`` -- the target
    swallowed it -- which is a finding, not a pass.
    """

    first_in: dict[str, tuple[int, str]] = field(default_factory=dict)
    first_out: dict[str, tuple[int, str]] = field(default_factory=dict)
    counts_in: dict[str, int] = field(default_factory=dict)
    counts_out: dict[str, int] = field(default_factory=dict)

    def reset(self) -> None:
        self.first_in.clear()
        self.first_out.clear()
        self.counts_in.clear()
        self.counts_out.clear()

    def _record(
        self,
        first: dict[str, tuple[int, str]],
        counts: dict[str, int],
        step: int,
        signal: str,
        value: float,
    ) -> str | None:
        cls = classify_value(value)
        if cls is None:
            return None
        counts[cls] = counts.get(cls, 0) + 1
        first.setdefault(cls, (step, signal))
        return cls

    def check_in(self, step: int, frame: Mapping[str, float]) -> list[str]:
        """Classify every channel of an inbound frame. Returns classes seen."""
        seen = []
        for name in ("pos", "vel"):
            cls = self._record(self.first_in, self.counts_in, step, name, float(frame[name]))
            if cls is not None:
                seen.append(cls)
        return seen

    def check_out(self, step: int, cmd: Mapping[str, float]) -> list[str]:
        """Classify every channel of an outbound command. Returns classes seen."""
        seen = []
        for name, value in cmd.items():
            cls = self._record(self.first_out, self.counts_out, step, name, float(value))
            if cls is not None:
                seen.append(cls)
        return seen

    @property
    def detected(self) -> bool:
        """True if any anomaly class was seen anywhere."""
        return bool(self.first_in or self.first_out)

    def verdict(self, cls: str) -> str:
        """``propagated``, ``absorbed``, ``emitted`` or ``absent`` for ``cls``.

        ``propagated``  seen entering the target and leaving it.
        ``absorbed``    seen entering, never leaving -- the target hid it.
        ``emitted``     never entered, but left: the target created it.
        ``absent``      not seen at all.
        """
        inn = cls in self.first_in
        out = cls in self.first_out
        if inn and out:
            return "propagated"
        if inn:
            return "absorbed"
        if out:
            return "emitted"
        return "absent"

    def report(self) -> dict[str, object]:
        """Flat serialisable summary."""
        classes = sorted(set(self.first_in) | set(self.first_out))
        return {
            "detected": self.detected,
            "classes": classes,
            "verdicts": {c: self.verdict(c) for c in classes},
            "first_in": {c: list(v) for c, v in sorted(self.first_in.items())},
            "first_out": {c: list(v) for c, v in sorted(self.first_out.items())},
            "counts_in": dict(sorted(self.counts_in.items())),
            "counts_out": dict(sorted(self.counts_out.items())),
        }


class InjectionWrapper:
    """Wrap a target so that faults are injected around its call interface."""

    def __init__(
        self,
        target: Target,
        injections: Sequence[Injection],
        rng: np.random.Generator,
    ) -> None:
        self.target = target
        self.injections = tuple(injections)
        self.rng = rng
        self.handlers: list[tuple[Injection, Handler]] = [
            (inj, make_handler(inj)) for inj in self.injections
        ]
        self.sensor = [(i, h) for i, h in self.handlers if i.stage is Stage.SENSOR]
        self.actuator = [(i, h) for i, h in self.handlers if i.stage is Stage.ACTUATOR]
        transport = [(i, h) for i, h in self.handlers if i.stage is Stage.TRANSPORT]
        target_stage = [(i, h) for i, h in self.handlers if i.stage is Stage.TARGET]
        if len(transport) > 1:
            raise ValueError(
                "at most one TRANSPORT-stage injection per wrapper; got "
                f"{[i.kind.value for i, _ in transport]}"
            )
        if len(target_stage) > 1:
            raise ValueError(
                "at most one TARGET-stage injection per wrapper; got "
                f"{[i.kind.value for i, _ in target_stage]}"
            )
        self.transport = transport[0] if transport else None
        self.target_stage = target_stage[0] if target_stage else None
        self.monitor = NumericalMonitor()
        self.events: list[Event] = []
        self.history: list[Mapping[str, float]] = []
        self.last_cmd: dict[str, float] = {"u": 0.0}
        self.delivered_source: list[int | None] = []

    def reset(self) -> None:
        """Reset the wrapper, every handler and the wrapped target."""
        self.target.reset()
        for _, h in self.handlers:
            h.reset()
        self.monitor.reset()
        self.events = []
        self.history = []
        self.last_cmd = {"u": 0.0}
        self.delivered_source = []

    def step(self, k: int, meas: Mapping[str, float]) -> dict[str, float]:
        """One wrapped control step. Same signature as the target's ``step``."""
        frame = {"pos": float(meas["pos"]), "vel": float(meas["vel"])}
        frame["valid"] = float(meas.get("valid", 1.0))

        for inj, h in self.sensor:
            if inj.active(k):
                frame[inj.channel] = h.apply_signal(k, frame[inj.channel], self.rng)
                if getattr(h, "dropped_this_step", False):
                    frame["valid"] = 0.0
                    self.events.append(Event(k, "sensor", inj.channel, "dropout"))
            else:
                h.observe_signal(frame[inj.channel])

        for cls in self.monitor.check_in(k, frame):
            self.events.append(Event(k, "sensor", "frame", f"numeric_in:{cls}"))
        self.history.append(dict(frame))

        if self.transport is not None:
            inj, h = self.transport
            delivered, src = h.deliver(k, frame, self.history, self.rng, inj.active(k))
            delivered = dict(delivered)
            if src is None:
                self.events.append(Event(k, "transport", "bus", f"no_frame:{inj.kind.value}"))
            elif src != k:
                self.events.append(
                    Event(k, "transport", "bus", f"stale:{inj.kind.value}", float(k - src))
                )
        else:
            delivered, src = dict(frame), k
        self.delivered_source.append(src)

        if self.target_stage is not None:
            inj, h = self.target_stage
            if inj.active(k) and h.suppress(k, self.rng):
                self.events.append(Event(k, "target", "task", "overrun"))
                return dict(self.last_cmd)

        cmd = dict(self.target.step(k, delivered))

        for inj, h in self.actuator:
            if inj.active(k):
                cmd[inj.channel] = h.apply_signal(k, cmd[inj.channel], self.rng)
            else:
                h.observe_signal(cmd[inj.channel])

        for cls in self.monitor.check_out(k, cmd):
            self.events.append(Event(k, "actuator", "u", f"numeric_out:{cls}"))

        self.last_cmd = dict(cmd)
        return cmd

    def event_counts(self) -> dict[str, int]:
        """Number of events of each kind."""
        out: dict[str, int] = {}
        for e in self.events:
            out[e.event] = out.get(e.event, 0) + 1
        return dict(sorted(out.items()))
