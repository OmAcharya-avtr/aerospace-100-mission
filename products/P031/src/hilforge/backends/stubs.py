"""Fault-injection stubs and the paired-backend helper.

Each stub injects exactly one of the six failure modes the Level 3
specification requires, at the driver level, so the loop under test takes its
normal path and the fault arrives where a real fault would.

===============================  ==========================================
Stub                             Failure mode
===============================  ==========================================
:class:`DisconnectingDriver`     device disconnects mid-run
:class:`DroppingDriver`          sample dropped
:class:`RejectingDriver`         actuator write rejected
:class:`SteppingBackDriver`      timebase steps backwards
:class:`CountingActuator`        (not a fault) counts or forbids writes
===============================  ==========================================

"device absent" is :class:`hilforge.backends.device.AbsentDriver` and the
overrun cascade is injected through
:attr:`hilforge.loop.LoopConfig.injected_durations_s`, both of which live with
the code they exercise.
"""

from __future__ import annotations

import numpy as np

from ..errors import (
    DeviceDisconnectedError,
    DryRunViolationError,
    SampleDroppedError,
    WriteRejectedError,
)
from ..hal import ChannelSpec, WriteAck
from ..plant import PlantConfig
from .device import DeviceBackend, LoopbackDriver
from .simulated import SimulatedBackend

__all__ = [
    "CountingActuator",
    "DisconnectingDriver",
    "DroppingDriver",
    "RejectingDriver",
    "SteppingBackDriver",
    "make_backend_pair",
]


class CountingActuator:
    """Actuator that counts writes, and optionally forbids them.

    Parameters
    ----------
    spec:
        Channel spec to present.
    forbid_writes:
        If ``True``, any write raises
        :class:`~hilforge.errors.DryRunViolationError`. This is how the
        dry-run guarantee is proved by execution rather than by inspection.
    """

    def __init__(self, spec: ChannelSpec, *, forbid_writes: bool = False) -> None:
        self._spec = spec
        self._forbid = bool(forbid_writes)
        self._count = 0
        self._last: np.ndarray | None = None
        self.attempts = 0

    @property
    def spec(self) -> ChannelSpec:
        return self._spec

    @property
    def write_count(self) -> int:
        """Writes accepted."""
        return self._count

    def last_command(self) -> np.ndarray | None:
        return None if self._last is None else self._last.copy()

    def write(self, command: np.ndarray) -> WriteAck:
        """Count the write; raise if this stub forbids writes."""
        self.attempts += 1
        if self._forbid:
            raise DryRunViolationError(
                f"write reached actuator {self._spec.name!r} during a dry run "
                f"(attempt {self.attempts})"
            )
        arr = np.asarray(command, dtype=np.float64).ravel()
        self._count += 1
        self._last = arr.copy()
        return WriteAck(
            channel=self._spec.name,
            applied=arr.copy(),
            saturated=False,
            sequence=self._count,
        )


class DisconnectingDriver(LoopbackDriver):
    """Loopback driver that loses the link at a chosen read.

    Parameters
    ----------
    fail_at_read:
        1-based index of the read at which
        :class:`~hilforge.errors.DeviceDisconnectedError` is raised and the
        link is marked down. Must be >= 1.
    """

    def __init__(self, *args, fail_at_read: int = 5, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if fail_at_read < 1:
            raise ValueError(f"fail_at_read must be >= 1, got {fail_at_read!r}")
        self.fail_at_read = int(fail_at_read)
        self.reads = 0

    @property
    def name(self) -> str:
        return "loopback-disconnecting"

    def read_channel(self, name: str, dt_s: float) -> np.ndarray:
        self.reads += 1
        if self.reads == self.fail_at_read:
            self._connected = False
            raise DeviceDisconnectedError(
                f"link lost at read {self.reads} (injected by DisconnectingDriver)"
            )
        return super().read_channel(name, dt_s)


class DroppingDriver(LoopbackDriver):
    """Loopback driver that drops the samples at chosen read indices.

    Parameters
    ----------
    drop_at_reads:
        1-based read indices at which
        :class:`~hilforge.errors.SampleDroppedError` is raised. The link stays
        up: a dropped sample is not a disconnect.
    """

    def __init__(self, *args, drop_at_reads=(3, 4), **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.drop_at_reads = {int(i) for i in drop_at_reads}
        if any(i < 1 for i in self.drop_at_reads):
            raise ValueError(f"drop_at_reads must all be >= 1, got {sorted(self.drop_at_reads)}")
        self.reads = 0
        self.dropped = 0

    @property
    def name(self) -> str:
        return "loopback-dropping"

    def read_channel(self, name: str, dt_s: float) -> np.ndarray:
        self.reads += 1
        if self.reads in self.drop_at_reads:
            self.dropped += 1
            raise SampleDroppedError(
                f"no sample at read {self.reads} (injected by DroppingDriver)"
            )
        return super().read_channel(name, dt_s)


class RejectingDriver(LoopbackDriver):
    """Loopback driver that refuses the writes at chosen write indices.

    Parameters
    ----------
    reject_at_writes:
        1-based write indices to refuse with
        :class:`~hilforge.errors.WriteRejectedError`.
    reason:
        Value placed on the exception's ``reason`` attribute.
    """

    def __init__(self, *args, reject_at_writes=(4,), reason: str = "bus_nak", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.reject_at_writes = {int(i) for i in reject_at_writes}
        if any(i < 1 for i in self.reject_at_writes):
            raise ValueError(
                f"reject_at_writes must all be >= 1, got {sorted(self.reject_at_writes)}"
            )
        self.reason = reason
        self.writes = 0
        self.rejected = 0

    @property
    def name(self) -> str:
        return "loopback-rejecting"

    def write_channel(self, name: str, command: np.ndarray) -> tuple[np.ndarray, bool]:
        self.writes += 1
        if self.writes in self.reject_at_writes:
            self.rejected += 1
            raise WriteRejectedError(
                f"device refused write {self.writes} (injected by RejectingDriver)",
                reason=self.reason,
            )
        return super().write_channel(name, command)


class SteppingBackDriver(LoopbackDriver):
    """Loopback driver whose clock jumps backwards once.

    A monotonic clock cannot do this; a settable or badly virtualised one can,
    and a loop that accepts it computes negative durations. The regression is
    caught by :class:`hilforge.timing.MonotonicGuard`, not here.

    Parameters
    ----------
    step_back_at_read:
        1-based read index after which the clock is rewound.
    step_back_s:
        How far back to jump [s], > 0.
    """

    def __init__(
        self, *args, step_back_at_read: int = 4, step_back_s: float = 0.05, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        if step_back_at_read < 1:
            raise ValueError(f"step_back_at_read must be >= 1, got {step_back_at_read!r}")
        if not (step_back_s > 0.0):
            raise ValueError(f"step_back_s must be > 0, got {step_back_s!r}")
        self.step_back_at_read = int(step_back_at_read)
        self.step_back_s = float(step_back_s)
        self.reads = 0
        self.stepped = False

    @property
    def name(self) -> str:
        return "loopback-stepping-back"

    def read_channel(self, name: str, dt_s: float) -> np.ndarray:
        self.reads += 1
        out = super().read_channel(name, dt_s)
        if self.reads == self.step_back_at_read and not self.stepped:
            self.stepped = True
            ticks = round(self.step_back_s / self._clock.resolution_s)
            self._clock.set_ticks(
                max(0, round(self._clock.now() / self._clock.resolution_s) - ticks)
            )
        return out


def make_backend_pair(
    *,
    seed: int = 20261004,
    config: PlantConfig | None = None,
    sample_dt_s: float = 0.01,
) -> tuple[SimulatedBackend, DeviceBackend]:
    """Build a simulated backend and a loopback device backend with one seed.

    Both drive the same :class:`~hilforge.plant.AttitudePlant` class with the
    same seed and the same configuration, by different code paths — the
    simulated backend directly, the device backend through
    :class:`~hilforge.backends.device.LoopbackDriver`. A parity failure
    therefore localises to the HAL path, which is what the parity check is
    for. It says nothing about hardware timing.

    Returns
    -------
    tuple
        ``(simulated_backend, device_backend)``, both unopened.
    """
    cfg = config or PlantConfig()
    sim = SimulatedBackend(cfg, seed=seed, sample_dt_s=sample_dt_s)
    dev = DeviceBackend(LoopbackDriver(cfg, seed=seed), sample_dt_s=sample_dt_s)
    return sim, dev
