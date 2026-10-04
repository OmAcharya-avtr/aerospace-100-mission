"""The device backend and its driver shim.

What a driver shim is
---------------------
A :class:`DriverShim` is the thin, replaceable layer between the HAL and an
actual board: connect, read a channel, write a channel, read the device clock,
disconnect. Everything above it — the loop, the deadline accounting, the
latency histograms, the dry-run guard, the recovery procedure — is backend
agnostic and is exercised by the same test suite against both backends.

Two shims ship here:

* :class:`LoopbackDriver` wraps a :class:`~hilforge.plant.AttitudePlant` and a
  :class:`~hilforge.timebase.VirtualTimebase`. It is **not hardware**. It
  exists so that ``DeviceBackend`` can be run end to end deterministically and
  compared bit for bit against ``SimulatedBackend``; its
  :attr:`~hilforge.hal.BackendInfo.is_hardware` is ``False`` and every record
  it produces says ``driver=loopback``.
* :class:`AbsentDriver` refuses to connect, which is the "device absent"
  failure mode.

A real shim for a real board is not in this repository and cannot be: it needs
the board. The contract it would have to satisfy is the five methods of
:class:`DriverShim` and the failure semantics in :mod:`hilforge.errors`.

Honest scope note
-----------------
Running ``DeviceBackend`` against ``LoopbackDriver`` proves the HAL path, the
loop and the accounting. It proves nothing about timing on hardware. No number
obtained this way may be presented as a hardware measurement; see
``README.md`` and ``validation/VALIDATION.md``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from ..errors import DeviceAbsentError, DeviceDisconnectedError, WriteRejectedError
from ..hal import BackendInfo, ChannelSpec, WriteAck, validate_command, validate_sample
from ..plant import AttitudePlant, PlantConfig
from ..timebase import VirtualTimebase
from .simulated import ahrs_spec, torque_spec

__all__ = ["AbsentDriver", "DeviceBackend", "DriverShim", "LoopbackDriver"]


@runtime_checkable
class DriverShim(Protocol):
    """The contract a board driver must satisfy to sit under the HAL.

    Failure semantics, which are part of the contract:

    * :meth:`connect` raises :class:`~hilforge.errors.DeviceAbsentError` when
      no device answers.
    * :meth:`read_channel` raises
      :class:`~hilforge.errors.SampleDroppedError` when no sample arrives in
      time and :class:`~hilforge.errors.DeviceDisconnectedError` when the link
      is gone.
    * :meth:`write_channel` raises
      :class:`~hilforge.errors.WriteRejectedError` when the device refuses,
      and returns the applied command otherwise.
    """

    @property
    def name(self) -> str:
        """Driver identifier, recorded in every run record."""

    @property
    def is_hardware(self) -> bool:
        """``True`` only if this shim talks to a physical device."""

    def connect(self) -> None:
        """Acquire the device."""

    def disconnect(self) -> None:
        """Release the device. Idempotent."""

    def channel_specs(self) -> dict[str, ChannelSpec]:
        """All channel specs the driver offers, keyed by name."""

    def read_channel(self, name: str, dt_s: float) -> np.ndarray:
        """One sample from ``name``; ``dt_s`` [s] is the sampling interval."""

    def write_channel(self, name: str, command: np.ndarray) -> tuple[np.ndarray, bool]:
        """Apply ``command`` to ``name``; return ``(applied, saturated)``."""

    def read_clock(self) -> float:
        """Device timestamp [s] on a monotonic epoch."""

    def advance(self, dt_s: float) -> None:
        """Advance a simulated device by ``dt_s`` [s]; no-op on hardware."""

    def snapshot(self) -> dict[str, object]:
        """Restorable device-side state."""

    def restore(self, snap: dict[str, object]) -> None:
        """Restore a :meth:`snapshot`."""


class LoopbackDriver:
    """A driver shim backed by the plant model instead of a board.

    Not hardware. See the module docstring.

    Parameters
    ----------
    config:
        Plant configuration.
    seed:
        Seed of the plant noise stream. Equal to the simulated backend's seed
        is what makes bit-identical parity possible.
    clock_resolution_s:
        Tick of the virtual device clock [s].
    """

    def __init__(
        self,
        config: PlantConfig | None = None,
        *,
        seed: int = 0,
        clock_resolution_s: float = 1.0e-9,
    ) -> None:
        self._config = config or PlantConfig()
        self._plant = AttitudePlant(self._config, seed=seed)
        self._clock = VirtualTimebase(resolution_s=clock_resolution_s)
        self._connected = False
        self._specs = {
            "ahrs": ahrs_spec(),
            "torque": torque_spec(self._config.torque_limit_nm),
        }
        self.connect_calls = 0

    @property
    def name(self) -> str:
        return "loopback"

    @property
    def is_hardware(self) -> bool:
        return False

    @property
    def plant(self) -> AttitudePlant:
        """The plant standing in for the device."""
        return self._plant

    def connect(self) -> None:
        self.connect_calls += 1
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def channel_specs(self) -> dict[str, ChannelSpec]:
        return dict(self._specs)

    def _require_link(self) -> None:
        if not self._connected:
            raise DeviceDisconnectedError("loopback driver is not connected")

    def read_channel(self, name: str, dt_s: float) -> np.ndarray:
        self._require_link()
        if name != "ahrs":
            raise KeyError(f"loopback driver has no sensor channel {name!r}")
        return self._plant.measure(dt_s)

    def write_channel(self, name: str, command: np.ndarray) -> tuple[np.ndarray, bool]:
        self._require_link()
        if name != "torque":
            raise KeyError(f"loopback driver has no actuator channel {name!r}")
        applied, saturated = self._plant.apply_torque(float(np.asarray(command)[0]))
        return np.array([applied], dtype=np.float64), saturated

    def read_clock(self) -> float:
        return self._clock.now()

    def advance(self, dt_s: float) -> None:
        self._plant.step(dt_s)
        self._clock.advance(dt_s)

    def snapshot(self) -> dict[str, object]:
        return {
            "plant": self._plant.snapshot(),
            "time_s": self._clock.now(),
            "connected": self._connected,
        }

    def restore(self, snap: dict[str, object]) -> None:
        self._plant.restore(snap["plant"])  # type: ignore[arg-type]
        self._clock.set_ticks(
            round(float(snap["time_s"]) / self._clock.resolution_s)  # type: ignore[arg-type]
        )
        self._connected = bool(snap["connected"])


class AbsentDriver:
    """A driver shim for a device that is not there.

    :meth:`connect` always raises :class:`~hilforge.errors.DeviceAbsentError`.
    This is the "device absent" failure mode, and it is a driver rather than a
    monkeypatch so the test exercises the real ``open`` path.
    """

    def __init__(self, *, detail: str = "no device on the configured bus") -> None:
        self._detail = detail

    @property
    def name(self) -> str:
        return "absent"

    @property
    def is_hardware(self) -> bool:
        return False

    def connect(self) -> None:
        raise DeviceAbsentError(f"device absent: {self._detail}")

    def disconnect(self) -> None:
        return None

    def channel_specs(self) -> dict[str, ChannelSpec]:
        return {}

    def read_channel(self, name: str, dt_s: float) -> np.ndarray:
        raise DeviceAbsentError("device absent: no channels")

    def write_channel(self, name: str, command: np.ndarray) -> tuple[np.ndarray, bool]:
        raise DeviceAbsentError("device absent: no channels")

    def read_clock(self) -> float:
        raise DeviceAbsentError("device absent: no clock")

    def advance(self, dt_s: float) -> None:
        return None

    def snapshot(self) -> dict[str, object]:
        return {}

    def restore(self, snap: dict[str, object]) -> None:
        return None


class _DriverSensor:
    """Sensor channel that reads through a :class:`DriverShim`."""

    def __init__(self, driver: DriverShim, spec: ChannelSpec, *, sample_dt_s: float) -> None:
        self._driver = driver
        self._spec = spec
        self._dt = float(sample_dt_s)
        self.read_count = 0

    @property
    def spec(self) -> ChannelSpec:
        return self._spec

    def read(self) -> np.ndarray:
        self.read_count += 1
        raw = self._driver.read_channel(self._spec.name, self._dt)
        return validate_sample(self._spec, raw)


class _DriverActuator:
    """Actuator channel that writes through a :class:`DriverShim`."""

    def __init__(self, driver: DriverShim, spec: ChannelSpec) -> None:
        self._driver = driver
        self._spec = spec
        self._count = 0
        self._last: np.ndarray | None = None

    @property
    def spec(self) -> ChannelSpec:
        return self._spec

    @property
    def write_count(self) -> int:
        return self._count

    def last_command(self) -> np.ndarray | None:
        return None if self._last is None else self._last.copy()

    def write(self, command: np.ndarray) -> WriteAck:
        cmd = validate_command(self._spec, command)
        if cmd[0] < self._spec.lower or cmd[0] > self._spec.upper:
            raise WriteRejectedError(
                f"torque {cmd[0]:.6g} N*m outside "
                f"[{self._spec.lower:.6g}, {self._spec.upper:.6g}] N*m",
                reason="out_of_range",
            )
        applied, saturated = self._driver.write_channel(self._spec.name, cmd)
        self._count += 1
        self._last = np.asarray(applied, dtype=np.float64).copy()
        return WriteAck(
            channel=self._spec.name,
            applied=self._last.copy(),
            saturated=bool(saturated),
            sequence=self._count,
        )


class DeviceBackend:
    """HAL instance over a :class:`DriverShim`.

    Parameters
    ----------
    driver:
        The shim to talk to. :class:`LoopbackDriver` for deterministic
        validation, :class:`AbsentDriver` for the absent-device test, a real
        board shim on hardware.
    sample_dt_s:
        Sampling interval [s] passed down to the driver on every read.
    """

    def __init__(self, driver: DriverShim, *, sample_dt_s: float = 0.01) -> None:
        self._driver = driver
        self._sample_dt = float(sample_dt_s)
        self._open = False
        self._sensors: dict[str, _DriverSensor] = {}
        self._actuators: dict[str, _DriverActuator] = {}

    @property
    def driver(self) -> DriverShim:
        """The driver shim underneath."""
        return self._driver

    @property
    def info(self) -> BackendInfo:
        return BackendInfo(
            kind="device",
            name=f"device:{self._driver.name}",
            driver=self._driver.name,
            detail={
                "sample_dt_s": f"{self._sample_dt:.9g}",
                "is_hardware": str(bool(getattr(self._driver, "is_hardware", False))),
            },
        )

    @property
    def timebase(self) -> _DriverTimebase:
        return _DriverTimebase(self._driver)

    @property
    def sensors(self) -> dict[str, _DriverSensor]:
        return dict(self._sensors)

    @property
    def actuators(self) -> dict[str, _DriverActuator]:
        return dict(self._actuators)

    @property
    def is_open(self) -> bool:
        return self._open

    def open(self) -> None:
        """Connect the driver and build the channels it advertises.

        Raises
        ------
        DeviceAbsentError
            Propagated from the driver when nothing answers.
        """
        self._driver.connect()
        specs = self._driver.channel_specs()
        self._sensors = {
            name: _DriverSensor(self._driver, spec, sample_dt_s=self._sample_dt)
            for name, spec in specs.items()
            if name == "ahrs"
        }
        self._actuators = {
            name: _DriverActuator(self._driver, spec)
            for name, spec in specs.items()
            if name == "torque"
        }
        self._open = True

    def close(self) -> None:
        """Disconnect the driver. Idempotent."""
        self._driver.disconnect()
        self._open = False

    def advance(self, dt_s: float) -> None:
        """Advance a simulated device by ``dt_s`` [s]; no-op on hardware."""
        self._driver.advance(dt_s)

    def snapshot(self) -> dict[str, object]:
        """Restorable state: the driver's, plus the HAL write counters."""
        act = self._actuators.get("torque")
        sen = self._sensors.get("ahrs")
        return {
            "driver": self._driver.snapshot(),
            "write_count": act.write_count if act else 0,
            "read_count": sen.read_count if sen else 0,
            "last_command": (
                None
                if act is None or act.last_command() is None
                else act.last_command().tolist()
            ),
        }

    def restore(self, snap: dict[str, object]) -> None:
        """Restore a :meth:`snapshot`."""
        self._driver.restore(snap["driver"])  # type: ignore[arg-type]
        act = self._actuators.get("torque")
        sen = self._sensors.get("ahrs")
        if act is not None:
            act._count = int(snap["write_count"])
            last = snap["last_command"]
            act._last = (
                None if last is None else np.asarray(last, dtype=np.float64)
            )
        if sen is not None:
            sen.read_count = int(snap["read_count"])


class _DriverTimebase:
    """Timebase view over a driver's clock."""

    def __init__(self, driver: DriverShim) -> None:
        self._driver = driver

    @property
    def resolution_s(self) -> float:
        """Driver clock resolution [s]; 1 ns for the loopback virtual clock."""
        clock = getattr(self._driver, "_clock", None)
        return float(getattr(clock, "resolution_s", 1.0e-9))

    @property
    def is_virtual(self) -> bool:
        """``True`` unless the driver reports itself as hardware."""
        return not bool(getattr(self._driver, "is_hardware", False))

    def now(self) -> float:
        return self._driver.read_clock()

    def sleep_until(self, t_s: float) -> None:
        """Advance a virtual device clock to ``t_s``; no-op on hardware."""
        clock = getattr(self._driver, "_clock", None)
        if clock is not None:
            clock.sleep_until(t_s)
