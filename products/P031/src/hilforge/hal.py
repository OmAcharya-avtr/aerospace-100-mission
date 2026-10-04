"""The hardware abstraction layer: sensors, actuators, timebase, backend.

One contract, two backends
--------------------------
Everything a loop touches goes through four protocols —
:class:`SensorChannel`, :class:`ActuatorChannel`,
:class:`~hilforge.timebase.Timebase` and :class:`Backend`. A simulated
backend and a device backend implement the same four, so the same loop code
and the same test suite run against both. That symmetry is the only reason
simulation/device parity is testable at all; see
``validation/parity_simdev.py``.

Conventions
-----------
* A sensor sample is a 1-D ``numpy`` array of ``float64`` with a fixed length
  and stated units.
* An actuator command is a 1-D ``numpy`` array of ``float64`` with a fixed
  length and stated units.
* A write returns a :class:`WriteAck`; refusal is an exception
  (:class:`~hilforge.errors.WriteRejectedError`), not a return code, because a
  silently ignored refusal is the failure this harness exists to catch.
* Channel names are unique within a backend and are the key used in records,
  so a replay can be matched to the run that produced it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from .errors import ConfigurationError
from .timebase import Timebase

__all__ = [
    "ActuatorChannel",
    "Backend",
    "BackendInfo",
    "ChannelSpec",
    "SensorChannel",
    "WriteAck",
    "validate_command",
    "validate_sample",
]


@dataclass(frozen=True)
class ChannelSpec:
    """Static description of one HAL channel.

    Attributes
    ----------
    name:
        Unique identifier within a backend.
    length:
        Number of ``float64`` elements in every sample or command.
    units:
        Physical units of the elements, e.g. ``"rad/s"``. Stated, not
        enforced: the HAL cannot check units, but every record carries them so
        a reader can.
    lower, upper:
        Inclusive validity range applied element-wise. A command outside it is
        refused by :func:`validate_command`; a sample outside it is a sensor
        fault and raises.
    """

    name: str
    length: int
    units: str
    lower: float = -np.inf
    upper: float = np.inf

    def __post_init__(self) -> None:
        if not self.name:
            raise ConfigurationError("channel name must be non-empty")
        if self.length < 1:
            raise ConfigurationError(
                f"channel {self.name!r}: length must be >= 1, got {self.length!r}"
            )
        if not (self.lower <= self.upper):
            raise ConfigurationError(
                f"channel {self.name!r}: lower ({self.lower}) must be <= upper ({self.upper})"
            )


@dataclass(frozen=True)
class WriteAck:
    """Acknowledgement of an accepted actuator write.

    Attributes
    ----------
    channel:
        Name of the actuator written.
    applied:
        The command as the actuator applied it, after any saturation
        the actuator itself performs. Equal to the requested command when no
        saturation occurred.
    saturated:
        ``True`` if ``applied`` differs from the request.
    sequence:
        Monotonically increasing write counter for this channel, starting at 1.
    """

    channel: str
    applied: np.ndarray
    saturated: bool
    sequence: int


@runtime_checkable
class SensorChannel(Protocol):
    """A readable channel.

    ``read`` returns a fresh sample or raises
    :class:`~hilforge.errors.SampleDroppedError` if none is available within
    the channel's budget. It must never return a stale sample silently; the
    hold-last-value policy lives in the loop, where it is counted.
    """

    @property
    def spec(self) -> ChannelSpec:
        """Static channel description."""

    def read(self) -> np.ndarray:
        """One sample, shape ``(spec.length,)``, dtype ``float64``."""


@runtime_checkable
class ActuatorChannel(Protocol):
    """A writable channel."""

    @property
    def spec(self) -> ChannelSpec:
        """Static channel description."""

    @property
    def write_count(self) -> int:
        """Number of writes this channel has accepted since construction."""

    def write(self, command: np.ndarray) -> WriteAck:
        """Apply ``command``; raise ``WriteRejectedError`` if refused."""

    def last_command(self) -> np.ndarray | None:
        """Most recently applied command, or ``None`` before the first write."""


@dataclass(frozen=True)
class BackendInfo:
    """Identity of a backend, recorded with every run.

    ``kind`` is ``"simulated"`` or ``"device"``. ``driver`` names the driver
    shim a device backend is talking to — ``"loopback"`` for the stub used in
    parity validation, which is **not** hardware and is labelled as such in
    every record it appears in.
    """

    kind: str
    name: str
    driver: str = ""
    detail: dict[str, str] = field(default_factory=dict)

    @property
    def is_hardware(self) -> bool:
        """``True`` only for a device backend not driven by a stub.

        A number produced with ``is_hardware is False`` may never be presented
        as a hardware measurement.
        """
        return self.kind == "device" and self.driver not in ("", "loopback", "absent")


@runtime_checkable
class Backend(Protocol):
    """A complete HAL instance: timebase plus named channels."""

    @property
    def info(self) -> BackendInfo:
        """Backend identity, recorded with every run."""

    @property
    def timebase(self) -> Timebase:
        """The backend's timebase."""

    @property
    def sensors(self) -> dict[str, SensorChannel]:
        """Sensor channels by name."""

    @property
    def actuators(self) -> dict[str, ActuatorChannel]:
        """Actuator channels by name."""

    @property
    def is_open(self) -> bool:
        """``True`` between a successful :meth:`open` and :meth:`close`."""

    def open(self) -> None:
        """Acquire the device; raise ``DeviceAbsentError`` if none answers."""

    def close(self) -> None:
        """Release the device. Idempotent."""


def validate_sample(spec: ChannelSpec, sample: np.ndarray) -> np.ndarray:
    """Check a sensor sample against its spec and return it as ``float64``.

    Raises
    ------
    ValueError
        Wrong length, non-finite element, or an element outside
        ``[spec.lower, spec.upper]``.
    TypeError
        Not convertible to a float array.
    """
    try:
        arr = np.asarray(sample, dtype=np.float64)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise TypeError(f"sensor {spec.name!r}: sample is not a float array: {exc}") from exc
    arr = np.atleast_1d(arr)
    if arr.shape != (spec.length,):
        raise ValueError(
            f"sensor {spec.name!r}: expected shape ({spec.length},) in {spec.units}, "
            f"got {arr.shape}"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"sensor {spec.name!r}: sample contains non-finite values: {arr}")
    if np.any(arr < spec.lower) or np.any(arr > spec.upper):
        raise ValueError(
            f"sensor {spec.name!r}: sample outside [{spec.lower}, {spec.upper}] {spec.units}: "
            f"{arr}"
        )
    return arr


def validate_command(spec: ChannelSpec, command: np.ndarray) -> np.ndarray:
    """Check an actuator command against its spec and return it as ``float64``.

    Range violation raises :class:`ValueError`; it is the caller's job to
    saturate before writing if saturation is what it wants. An actuator that
    saturates internally reports it through :attr:`WriteAck.saturated`.
    """
    try:
        arr = np.asarray(command, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"actuator {spec.name!r}: command is not a float array: {exc}") from exc
    arr = np.atleast_1d(arr)
    if arr.shape != (spec.length,):
        raise ValueError(
            f"actuator {spec.name!r}: expected shape ({spec.length},) in {spec.units}, "
            f"got {arr.shape}"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"actuator {spec.name!r}: command contains non-finite values: {arr}")
    return arr
