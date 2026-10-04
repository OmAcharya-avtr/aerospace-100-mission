"""The simulated backend: plant model behind the HAL, virtual timebase.

Determinism
-----------
Nothing in this backend reads a wall clock. The timebase is a
:class:`~hilforge.timebase.VirtualTimebase` advanced by
:meth:`SimulatedBackend.advance`, and all noise comes from the plant's seeded
PCG64 stream. Two runs with the same seed, the same configuration and the
same loop therefore produce bit-identical sample and command arrays.

Channels
--------
=============  ======  ======================  ==============================
Channel        Length  Units                   Meaning
=============  ======  ======================  ==============================
``ahrs``       2       ``rad, rad/s``          measured attitude and rate
``torque``     1       ``N*m``                 commanded axis torque
=============  ======  ======================  ==============================
"""

from __future__ import annotations

import numpy as np

from ..errors import WriteRejectedError
from ..hal import BackendInfo, ChannelSpec, WriteAck, validate_command, validate_sample
from ..plant import AttitudePlant, PlantConfig
from ..timebase import VirtualTimebase

__all__ = ["PlantActuator", "PlantSensor", "SimulatedBackend", "ahrs_spec", "torque_spec"]


def ahrs_spec() -> ChannelSpec:
    """Spec of the ``ahrs`` sensor channel: ``[theta rad, omega rad/s]``."""
    return ChannelSpec(name="ahrs", length=2, units="rad, rad/s", lower=-1.0e4, upper=1.0e4)


def torque_spec(limit_nm: float) -> ChannelSpec:
    """Spec of the ``torque`` actuator channel [N*m], bounded by ``limit_nm``."""
    return ChannelSpec(
        name="torque", length=1, units="N*m", lower=-limit_nm, upper=limit_nm
    )


class PlantSensor:
    """Sensor channel reading :meth:`hilforge.plant.AttitudePlant.measure`."""

    def __init__(self, plant: AttitudePlant, *, sample_dt_s: float) -> None:
        self._plant = plant
        self._dt = float(sample_dt_s)
        self._spec = ahrs_spec()
        self.read_count = 0

    @property
    def spec(self) -> ChannelSpec:
        return self._spec

    def read(self) -> np.ndarray:
        """One ``[theta rad, omega rad/s]`` sample."""
        self.read_count += 1
        return validate_sample(self._spec, self._plant.measure(self._dt))


class PlantActuator:
    """Actuator channel writing a held torque into the plant.

    The channel refuses a command outside its spec range with
    :class:`~hilforge.errors.WriteRejectedError` rather than clipping
    silently; the *plant* saturates at its own torque limit, and that
    saturation is reported through :attr:`hilforge.hal.WriteAck.saturated`.
    """

    def __init__(self, plant: AttitudePlant) -> None:
        self._plant = plant
        self._spec = torque_spec(plant.config.torque_limit_nm)
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
        """Hold ``command`` [N*m] on the plant until the next write."""
        cmd = validate_command(self._spec, command)
        if cmd[0] < self._spec.lower or cmd[0] > self._spec.upper:
            raise WriteRejectedError(
                f"torque {cmd[0]:.6g} N*m outside "
                f"[{self._spec.lower:.6g}, {self._spec.upper:.6g}] N*m",
                reason="out_of_range",
            )
        applied, saturated = self._plant.apply_torque(float(cmd[0]))
        self._count += 1
        self._last = np.array([applied], dtype=np.float64)
        return WriteAck(
            channel=self._spec.name,
            applied=self._last.copy(),
            saturated=saturated,
            sequence=self._count,
        )


class SimulatedBackend:
    """Full HAL instance over the plant model and a virtual timebase.

    Parameters
    ----------
    config:
        Plant configuration; defaults to :class:`hilforge.plant.PlantConfig`.
    seed:
        Seed of the plant's noise stream.
    sample_dt_s:
        Sampling interval [s] handed to the gyro noise model. Should equal the
        loop period; it is a separate argument because a HIL rig can sample a
        sensor faster than the control loop runs.
    clock_resolution_s:
        Tick of the virtual timebase [s].
    """

    def __init__(
        self,
        config: PlantConfig | None = None,
        *,
        seed: int = 0,
        sample_dt_s: float = 0.01,
        clock_resolution_s: float = 1.0e-9,
    ) -> None:
        self._config = config or PlantConfig()
        self._plant = AttitudePlant(self._config, seed=seed)
        self._tb = VirtualTimebase(resolution_s=clock_resolution_s)
        self._sensor = PlantSensor(self._plant, sample_dt_s=sample_dt_s)
        self._actuator = PlantActuator(self._plant)
        self._open = False
        self._seed = int(seed)
        self._sample_dt = float(sample_dt_s)

    @property
    def info(self) -> BackendInfo:
        return BackendInfo(
            kind="simulated",
            name="plant-sim",
            driver="",
            detail={
                "seed": str(self._seed),
                "sample_dt_s": f"{self._sample_dt:.9g}",
                "model": "single-axis rigid body, ZOH double integrator",
            },
        )

    @property
    def timebase(self) -> VirtualTimebase:
        return self._tb

    @property
    def plant(self) -> AttitudePlant:
        """The plant model, for tests and for state snapshots."""
        return self._plant

    @property
    def sensors(self) -> dict[str, PlantSensor]:
        return {"ahrs": self._sensor}

    @property
    def actuators(self) -> dict[str, PlantActuator]:
        return {"torque": self._actuator}

    @property
    def is_open(self) -> bool:
        return self._open

    def open(self) -> None:
        """Mark the backend available. A simulation is always present."""
        self._open = True

    def close(self) -> None:
        """Mark the backend unavailable. Idempotent."""
        self._open = False

    def advance(self, dt_s: float) -> None:
        """Step the plant and the virtual clock by ``dt_s`` [s].

        On a real device backend this is a no-op: hardware advances whether or
        not it is asked to. Here it is the only thing that makes time pass,
        which is what keeps a seeded run reproducible.
        """
        self._plant.step(dt_s)
        self._tb.advance(dt_s)

    def snapshot(self) -> dict[str, object]:
        """Restorable backend state: plant state plus write counters."""
        return {
            "plant": self._plant.snapshot(),
            "write_count": self._actuator.write_count,
            "read_count": self._sensor.read_count,
            "last_command": (
                None
                if self._actuator.last_command() is None
                else self._actuator.last_command().tolist()
            ),
            "time_s": self._tb.now(),
        }

    def restore(self, snap: dict[str, object]) -> None:
        """Restore a :meth:`snapshot`, including the plant RNG state."""
        self._plant.restore(snap["plant"])  # type: ignore[arg-type]
        self._actuator._count = int(snap["write_count"])
        self._sensor.read_count = int(snap["read_count"])
        last = snap["last_command"]
        self._actuator._last = (
            None if last is None else np.asarray(last, dtype=np.float64)
        )
        self._tb.set_ticks(round(float(snap["time_s"]) / self._tb.resolution_s))  # type: ignore[arg-type]
