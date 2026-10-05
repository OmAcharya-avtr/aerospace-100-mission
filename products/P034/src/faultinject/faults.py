"""Fault models: the executable half of the taxonomy.

Each taxonomy kind maps to one handler class here.  A handler is a small state
machine with a single ``apply`` method; all randomness comes from an injected
``numpy.random.Generator`` and every draw happens unconditionally on every
active step, so the draw sequence depends only on the seed and the active
window.  That is what makes seeded replay bit-identical (see
``validation/validate_replay.py``).

Stages
------
Injection points are ordered the way the signals physically flow:

    plant -> SENSOR -> TRANSPORT -> TARGET -> ACTUATOR -> plant

``SENSOR`` faults perturb a named channel of a freshly sampled measurement
frame.  ``TRANSPORT`` faults act on whole frames (delay, reorder, loss, late
read).  ``TARGET`` faults suppress the target's execution (overrun).
``ACTUATOR`` faults perturb the command the target produced.  Numerical faults
act at whichever stage their channel belongs to.

Units are the channel units declared in :mod:`faultinject.taxonomy`.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from .taxonomy import FaultClass, FaultKind, spec

SUBNORMAL_MIN = 5e-324
"""Smallest positive IEEE 754 binary64 subnormal (2**-1074)."""

INVALID_FRAME = {"pos": 0.0, "vel": 0.0, "valid": 0.0}
"""Frame delivered when transport produced nothing this step."""


class Stage(StrEnum):
    """Where in the loop a fault acts."""

    SENSOR = "sensor"
    TRANSPORT = "transport"
    TARGET = "target"
    ACTUATOR = "actuator"


_TRANSPORT_KINDS = frozenset(
    {
        FaultKind.BUS_DELAY,
        FaultKind.BUS_REORDER,
        FaultKind.BUS_LOSS,
        FaultKind.TIMING_LATE_SAMPLE,
    }
)


def stage_of(kind: FaultKind, channel: str) -> Stage:
    """Stage at which ``kind`` on ``channel`` is applied."""
    sp = spec(kind)
    if kind in _TRANSPORT_KINDS:
        return Stage.TRANSPORT
    if kind is FaultKind.TIMING_OVERRUN:
        return Stage.TARGET
    if sp.fault_class is FaultClass.ACTUATOR:
        return Stage.ACTUATOR
    if sp.fault_class is FaultClass.NUMERICAL:
        return Stage.ACTUATOR if channel == "u" else Stage.SENSOR
    return Stage.SENSOR


@dataclass(frozen=True)
class Injection:
    """One fault instance: kind, channel, parameters and an active window.

    ``params`` is stored as a sorted tuple of pairs so that the object is
    hashable and serialises to a canonical form.  Use :meth:`create` rather
    than the constructor.
    """

    kind: FaultKind
    channel: str
    param_items: tuple[tuple[str, float], ...]
    start_step: int
    duration_steps: int

    @classmethod
    def create(
        cls,
        kind: FaultKind | str,
        channel: str,
        params: Mapping[str, float],
        start_step: int,
        duration_steps: int,
    ) -> Injection:
        """Validated constructor. Raises ``ValueError`` on an invalid combination."""
        sp = spec(kind)
        sp.validate(channel, params)
        if start_step < 0:
            raise ValueError(f"start_step must be >= 0, got {start_step}")
        if duration_steps < 1:
            raise ValueError(f"duration_steps must be >= 1, got {duration_steps}")
        items = tuple(sorted((str(k), float(v)) for k, v in params.items()))
        return cls(sp.kind, channel, items, int(start_step), int(duration_steps))

    @property
    def params(self) -> dict[str, float]:
        """Parameters as a plain mapping."""
        return dict(self.param_items)

    @property
    def stage(self) -> Stage:
        """Stage at which this injection acts."""
        return stage_of(self.kind, self.channel)

    @property
    def end_step(self) -> int:
        """First step at which the injection is no longer active (exclusive)."""
        return self.start_step + self.duration_steps

    def active(self, step: int) -> bool:
        """True if the injection is active at ``step``."""
        return self.start_step <= step < self.end_step

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable form."""
        return {
            "kind": self.kind.value,
            "channel": self.channel,
            "params": self.params,
            "start_step": self.start_step,
            "duration_steps": self.duration_steps,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Injection:
        """Inverse of :meth:`to_dict`."""
        return cls.create(
            str(d["kind"]),
            str(d["channel"]),
            dict(d["params"]),  # type: ignore[arg-type]
            int(d["start_step"]),  # type: ignore[arg-type]
            int(d["duration_steps"]),  # type: ignore[arg-type]
        )


class Handler:
    """Base class for an executable fault model."""

    def __init__(self, injection: Injection) -> None:
        self.injection = injection
        self.params = injection.params
        self.channel = injection.channel
        self.applications = 0

    def reset(self) -> None:
        """Clear internal state. Called once before a run."""
        self.applications = 0

    # Signal-stage interface -------------------------------------------------
    def apply_signal(self, step: int, value: float, rng: np.random.Generator) -> float:
        """Transform one channel value. Signal-stage handlers override this."""
        raise NotImplementedError

    def observe_signal(self, value: float) -> None:
        """Record the nominal value on inactive steps (for stuck/drift memory)."""

    # Transport-stage interface ---------------------------------------------
    def deliver(
        self,
        step: int,
        frame: Mapping[str, float],
        history: list[Mapping[str, float]],
        rng: np.random.Generator,
        active: bool,
    ) -> tuple[Mapping[str, float], int | None]:
        """Return ``(frame_to_deliver, source_step)``. ``source_step`` is ``None``
        when nothing was delivered."""
        raise NotImplementedError

    # Target-stage interface -------------------------------------------------
    def suppress(self, step: int, rng: np.random.Generator) -> bool:
        """True if the target must not run this step."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Sensor and numerical signal handlers
# --------------------------------------------------------------------------- #


class BiasHandler(Handler):
    """Additive constant offset. ``value + offset`` in channel units."""

    def apply_signal(self, step: int, value: float, rng: np.random.Generator) -> float:
        self.applications += 1
        return value + self.params["offset"]


class DriftHandler(Handler):
    """Additive ramp: ``value + rate * (step - start_step + 1)``."""

    def apply_signal(self, step: int, value: float, rng: np.random.Generator) -> float:
        self.applications += 1
        n = step - self.injection.start_step + 1
        return value + self.params["rate"] * n


class StuckHandler(Handler):
    """Freeze the channel at the last value seen before injection began."""

    def __init__(self, injection: Injection) -> None:
        super().__init__(injection)
        self.held: float | None = None

    def reset(self) -> None:
        super().reset()
        self.held = None

    def observe_signal(self, value: float) -> None:
        self.held = value

    def apply_signal(self, step: int, value: float, rng: np.random.Generator) -> float:
        self.applications += 1
        if self.held is None:
            self.held = value
        return self.held


class DropoutHandler(Handler):
    """Mark the frame invalid with probability ``dropout_prob``.

    Signalled by returning NaN sentinel-free: the wrapper reads
    :attr:`dropped_this_step` after calling ``apply_signal``.
    """

    def __init__(self, injection: Injection) -> None:
        super().__init__(injection)
        self.dropped_this_step = False
        self.drops = 0

    def reset(self) -> None:
        super().reset()
        self.dropped_this_step = False
        self.drops = 0

    def apply_signal(self, step: int, value: float, rng: np.random.Generator) -> float:
        self.applications += 1
        draw = float(rng.random())
        self.dropped_this_step = draw < self.params["dropout_prob"]
        if self.dropped_this_step:
            self.drops += 1
        return value


class QuantCollapseHandler(Handler):
    """Quantise to a coarse LSB: ``lsb * round(value / lsb)``.

    Uses round-half-away-from-zero, so the result is exactly representable as a
    multiple of ``lsb`` up to binary64 rounding of the product.
    """

    def apply_signal(self, step: int, value: float, rng: np.random.Generator) -> float:
        self.applications += 1
        lsb = self.params["lsb"]
        return lsb * math.floor(value / lsb + 0.5)


class NanHandler(Handler):
    """Write a quiet NaN to the channel."""

    def apply_signal(self, step: int, value: float, rng: np.random.Generator) -> float:
        self.applications += 1
        return float("nan")


class DenormalHandler(Handler):
    """Write the smallest positive subnormal to the channel."""

    def apply_signal(self, step: int, value: float, rng: np.random.Generator) -> float:
        self.applications += 1
        return SUBNORMAL_MIN


class OverflowHandler(Handler):
    """Write a large magnitude, preserving the sign of the nominal value."""

    def apply_signal(self, step: int, value: float, rng: np.random.Generator) -> float:
        self.applications += 1
        mag = self.params["magnitude"]
        return -mag if value < 0.0 else mag


# --------------------------------------------------------------------------- #
# Actuator handlers
# --------------------------------------------------------------------------- #


class LossEffectivenessHandler(Handler):
    """Scale the command by the retained effectiveness fraction."""

    def apply_signal(self, step: int, value: float, rng: np.random.Generator) -> float:
        self.applications += 1
        return value * self.params["retained"]


class RunawayHandler(Handler):
    """Ramp the command away from the request at a fixed slew rate.

    Sign is taken from the command at the step injection began, so a runaway
    always moves the actuator further in the direction it was already going;
    if that command was zero the runaway is positive.
    """

    def __init__(self, injection: Injection) -> None:
        super().__init__(injection)
        self.sign = 0.0
        self.base: float | None = None

    def reset(self) -> None:
        super().reset()
        self.sign = 0.0
        self.base = None

    def observe_signal(self, value: float) -> None:
        self.base = value

    def apply_signal(self, step: int, value: float, rng: np.random.Generator) -> float:
        self.applications += 1
        if self.base is None:
            self.base = value
        if self.sign == 0.0:
            self.sign = -1.0 if self.base < 0.0 else 1.0
        n = step - self.injection.start_step + 1
        return self.base + self.sign * self.params["slew"] * n


# --------------------------------------------------------------------------- #
# Transport handlers
# --------------------------------------------------------------------------- #


class DelayHandler(Handler):
    """FIFO transport delay of ``delay_steps`` frames.

    A FIFO delay is a *persistent* change of transport latency, not a transient
    one: once ``delay_steps`` frames are in flight they stay in flight, because
    the producer and the consumer both run at one frame per step and nothing
    flushes the queue.  Delivery therefore remains ``delay_steps`` steps stale
    for the rest of the run, even after the active window closes.  That is the
    behaviour of a real queue, and it is what distinguishes this kind from
    :class:`LateSampleHandler`, which returns to fresh delivery on the first
    step after its window.  Both behaviours are checked in
    ``validation/validate_transparency.py``.
    """

    def __init__(self, injection: Injection) -> None:
        super().__init__(injection)
        self.queue: deque[tuple[int, Mapping[str, float]]] = deque()

    def reset(self) -> None:
        super().reset()
        self.queue = deque()

    def deliver(
        self,
        step: int,
        frame: Mapping[str, float],
        history: list[Mapping[str, float]],
        rng: np.random.Generator,
        active: bool,
    ) -> tuple[Mapping[str, float], int | None]:
        d = int(self.params["delay_steps"])
        self.queue.append((step, frame))
        if active:
            self.applications += 1
            if len(self.queue) > d:
                src, f = self.queue.popleft()
                return f, src
            return INVALID_FRAME, None
        src, f = self.queue.popleft()
        return f, src


class LossHandler(Handler):
    """Drop the whole frame with probability ``loss_prob``."""

    def __init__(self, injection: Injection) -> None:
        super().__init__(injection)
        self.losses = 0

    def reset(self) -> None:
        super().reset()
        self.losses = 0

    def deliver(
        self,
        step: int,
        frame: Mapping[str, float],
        history: list[Mapping[str, float]],
        rng: np.random.Generator,
        active: bool,
    ) -> tuple[Mapping[str, float], int | None]:
        if not active:
            return frame, step
        self.applications += 1
        if float(rng.random()) < self.params["loss_prob"]:
            self.losses += 1
            return INVALID_FRAME, None
        return frame, step


class ReorderHandler(Handler):
    """Exchange adjacent frames with probability ``swap_prob``.

    Causal reordering needs somewhere to hold a frame, so while active the
    transport runs with a one-slot buffer and therefore one step of latency.
    With no swap the delivered sequence is exactly a one-step delay; a swap
    delivers the newer frame first and leaves the older one held, which arrives
    out of order on the next non-swap step.

    The out-of-order *rate* is therefore non-monotonic in ``swap_prob``: an
    exchange needs a swap followed by a non-swap, so the rate is zero at
    ``swap_prob = 0``, peaks near 0.5, and is zero again at ``swap_prob = 1``,
    where every step swaps and the held frame is never released.  That is a
    property of the mechanism, not a defect, and it is measured in
    ``validation/validate_transparency.py`` rather than left for a user to
    discover.
    """

    def __init__(self, injection: Injection) -> None:
        super().__init__(injection)
        self.buffer: deque[tuple[int, Mapping[str, float]]] = deque()
        self.swaps = 0
        self.out_of_order = 0
        self._max_src = -1

    def reset(self) -> None:
        super().reset()
        self.buffer = deque()
        self.swaps = 0
        self.out_of_order = 0
        self._max_src = -1

    def deliver(
        self,
        step: int,
        frame: Mapping[str, float],
        history: list[Mapping[str, float]],
        rng: np.random.Generator,
        active: bool,
    ) -> tuple[Mapping[str, float], int | None]:
        self.buffer.append((step, frame))
        if not active:
            src, f = self.buffer.popleft()
            if src < self._max_src:
                self.out_of_order += 1
            self._max_src = max(self._max_src, src)
            return f, src
        self.applications += 1
        draw = float(rng.random())
        if len(self.buffer) < 2:
            # Building up the one-slot latency an exchange needs. Nothing is
            # delivered on this step; the frame is held.
            return INVALID_FRAME, None
        if draw < self.params["swap_prob"]:
            src, f = self.buffer.pop()
            self.swaps += 1
        else:
            src, f = self.buffer.popleft()
        if src < self._max_src:
            self.out_of_order += 1
        self._max_src = max(self._max_src, src)
        return f, src


class LateSampleHandler(Handler):
    """Read the port ``late_steps`` steps stale, with no queueing.

    Unlike :class:`DelayHandler` nothing is buffered, so delivery returns to
    the fresh frame on the first step after the active window.
    """

    def deliver(
        self,
        step: int,
        frame: Mapping[str, float],
        history: list[Mapping[str, float]],
        rng: np.random.Generator,
        active: bool,
    ) -> tuple[Mapping[str, float], int | None]:
        if not active:
            return frame, step
        self.applications += 1
        n = int(self.params["late_steps"])
        idx = step - n
        if idx < 0:
            return INVALID_FRAME, None
        return history[idx], idx


class OverrunHandler(Handler):
    """Suppress the target's execution with probability ``overrun_prob``."""

    def __init__(self, injection: Injection) -> None:
        super().__init__(injection)
        self.overruns = 0

    def reset(self) -> None:
        super().reset()
        self.overruns = 0

    def suppress(self, step: int, rng: np.random.Generator) -> bool:
        self.applications += 1
        if float(rng.random()) < self.params["overrun_prob"]:
            self.overruns += 1
            return True
        return False


_HANDLERS: dict[FaultKind, type[Handler]] = {
    FaultKind.SENSOR_BIAS: BiasHandler,
    FaultKind.SENSOR_DRIFT: DriftHandler,
    FaultKind.SENSOR_STUCK: StuckHandler,
    FaultKind.SENSOR_DROPOUT: DropoutHandler,
    FaultKind.SENSOR_QUANT_COLLAPSE: QuantCollapseHandler,
    FaultKind.ACTUATOR_LOSS_EFFECTIVENESS: LossEffectivenessHandler,
    FaultKind.ACTUATOR_STUCK: StuckHandler,
    FaultKind.ACTUATOR_RUNAWAY: RunawayHandler,
    FaultKind.BUS_DELAY: DelayHandler,
    FaultKind.BUS_REORDER: ReorderHandler,
    FaultKind.BUS_LOSS: LossHandler,
    FaultKind.TIMING_LATE_SAMPLE: LateSampleHandler,
    FaultKind.TIMING_OVERRUN: OverrunHandler,
    FaultKind.NUMERICAL_NAN: NanHandler,
    FaultKind.NUMERICAL_DENORMAL: DenormalHandler,
    FaultKind.NUMERICAL_OVERFLOW: OverflowHandler,
}


def make_handler(injection: Injection) -> Handler:
    """Instantiate the handler for ``injection``."""
    try:
        cls = _HANDLERS[injection.kind]
    except KeyError as exc:  # pragma: no cover - taxonomy and table kept in sync
        raise KeyError(f"no handler registered for {injection.kind!r}") from exc
    return cls(injection)


def handler_coverage() -> tuple[FaultKind, ...]:
    """Kinds with an executable handler. Equals the taxonomy by construction."""
    return tuple(_HANDLERS)
