"""Record a run, store it, replay it bit-identically.

Why replay is a separate backend rather than a mode
---------------------------------------------------
A replay is just a third HAL backend whose sensor hands back stored samples
and whose actuator stores what it is given. Because it satisfies the same
contract as the simulated and device backends, the loop code, the deadline
accounting and the dry-run guard are all unchanged, and a replay can be
compared to the original run with the same digest function.

What replay reproduces, and what it does not
--------------------------------------------
Replay reproduces the **data path**: the samples the loop saw, the estimates
it formed, the commands it issued. It reproduces the timing only if the
original run had :attr:`hilforge.loop.LoopConfig.injected_durations_s` set, in
which case the stored durations are re-injected. A wall-clock-timed run
replays with its original durations re-injected, which makes the overrun
accounting reproducible even though the host would not time it the same way
twice; that is a deliberate choice and it is recorded in the replay's
``deterministic_timing`` flag.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .errors import ConfigurationError
from .hal import BackendInfo, ChannelSpec, WriteAck, validate_command
from .loop import STAGES, HilLoop, LoopConfig, RunRecord
from .timebase import VirtualTimebase

__all__ = [
    "RecordingActuator",
    "ReplayBackend",
    "ReplaySensor",
    "load_samples",
    "replay_run",
    "save_run",
]


class ReplaySensor:
    """Sensor channel that hands back stored samples in order.

    Parameters
    ----------
    samples:
        ``(n, length)`` array of stored samples.
    spec:
        Channel spec the samples satisfy.
    """

    def __init__(self, samples: np.ndarray, spec: ChannelSpec) -> None:
        arr = np.asarray(samples, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != spec.length:
            raise ConfigurationError(
                f"samples must have shape (n, {spec.length}), got {arr.shape}"
            )
        self._samples = arr
        self._spec = spec
        self.read_count = 0

    @property
    def spec(self) -> ChannelSpec:
        return self._spec

    @property
    def n_samples(self) -> int:
        """Number of stored samples."""
        return int(self._samples.shape[0])

    def read(self) -> np.ndarray:
        """Next stored sample.

        Raises
        ------
        IndexError
            When the stored samples run out, which means the replay was asked
            for more iterations than were recorded.
        """
        if self.read_count >= self._samples.shape[0]:
            raise IndexError(
                f"replay exhausted after {self.read_count} samples; "
                f"the recorded run was shorter than the replay request"
            )
        out = self._samples[self.read_count].copy()
        self.read_count += 1
        return out


class RecordingActuator:
    """Actuator channel that stores commands instead of applying them."""

    def __init__(self, spec: ChannelSpec) -> None:
        self._spec = spec
        self._count = 0
        self._last: np.ndarray | None = None
        self.commands: list[np.ndarray] = []

    @property
    def spec(self) -> ChannelSpec:
        return self._spec

    @property
    def write_count(self) -> int:
        return self._count

    def last_command(self) -> np.ndarray | None:
        return None if self._last is None else self._last.copy()

    def write(self, command: np.ndarray) -> WriteAck:
        """Store ``command``; always accepted, never saturated."""
        cmd = validate_command(self._spec, command)
        self._count += 1
        self._last = cmd.copy()
        self.commands.append(cmd.copy())
        return WriteAck(
            channel=self._spec.name, applied=cmd.copy(), saturated=False, sequence=self._count
        )


class ReplayBackend:
    """HAL backend over stored samples.

    Parameters
    ----------
    samples:
        ``(n, 2)`` array of ``[theta rad, omega rad/s]`` samples.
    torque_limit_nm:
        Actuator range [N*m] to present.
    clock_resolution_s:
        Tick of the virtual model clock [s].
    """

    def __init__(
        self,
        samples: np.ndarray,
        *,
        torque_limit_nm: float = 1.5,
        clock_resolution_s: float = 1.0e-9,
    ) -> None:
        sensor_spec = ChannelSpec(
            name="ahrs", length=2, units="rad, rad/s", lower=-1.0e4, upper=1.0e4
        )
        act_spec = ChannelSpec(
            name="torque", length=1, units="N*m", lower=-torque_limit_nm, upper=torque_limit_nm
        )
        self._sensor = ReplaySensor(samples, sensor_spec)
        self._actuator = RecordingActuator(act_spec)
        self._tb = VirtualTimebase(resolution_s=clock_resolution_s)
        self._open = False

    @property
    def info(self) -> BackendInfo:
        return BackendInfo(
            kind="replay",
            name="replay",
            driver="recorded",
            detail={"n_samples": str(self._sensor.n_samples)},
        )

    @property
    def timebase(self) -> VirtualTimebase:
        return self._tb

    @property
    def sensors(self) -> dict[str, ReplaySensor]:
        return {"ahrs": self._sensor}

    @property
    def actuators(self) -> dict[str, RecordingActuator]:
        return {"torque": self._actuator}

    @property
    def is_open(self) -> bool:
        return self._open

    def open(self) -> None:
        """Mark available. A recording is always present."""
        self._open = True

    def close(self) -> None:
        """Mark unavailable. Idempotent."""
        self._open = False

    def advance(self, dt_s: float) -> None:
        """Advance the virtual model clock by ``dt_s`` [s]."""
        self._tb.advance(dt_s)


def save_run(record: RunRecord, path: str | Path) -> Path:
    """Write a run to a compressed ``.npz``.

    Stores the signal matrix, the per-stage durations, the flags and the
    scalar metadata. Returns the path written.
    """
    p = Path(path)
    if record.n_completed == 0:
        raise ValueError("cannot save an empty run")
    sig = record.signal_matrix()
    stages = np.column_stack([record.stage_durations_s(s) for s in STAGES])
    np.savez_compressed(
        p,
        signals=sig,
        stage_durations_s=stages,
        totals_s=record.durations_s(),
        flags=np.array(
            [[it.dropped, it.rejected, it.saturated, it.flagged] for it in record.iterations],
            dtype=np.uint8,
        ),
        meta=np.array(
            [
                record.backend_kind,
                record.backend_name,
                record.backend_driver,
                str(record.is_hardware),
                str(record.deterministic_timing),
                f"{record.period_s:.17g}",
                f"{record.deadline_s:.17g}",
                str(record.dry_run),
                record.data_digest(),
            ],
            dtype=object,
        ),
    )
    return p


def load_samples(path: str | Path) -> np.ndarray:
    """Read the ``(n, 2)`` sensor samples back out of a saved run."""
    with np.load(Path(path), allow_pickle=True) as data:
        sig = np.asarray(data["signals"], dtype=np.float64)
    return sig[:, 0:2].copy()


def replay_run(record: RunRecord, config: LoopConfig | None = None) -> RunRecord:
    """Replay ``record`` through a :class:`ReplayBackend` and return the new run.

    The replay re-injects the original per-iteration durations, so the overrun
    accounting is reproduced exactly. The returned record's
    ``data_digest()`` equals the original's when the loop is deterministic,
    which is the property ``validation/replay_determinism.py`` checks.

    Parameters
    ----------
    record:
        A run recorded with ``record_signals=True``.
    config:
        Loop configuration to replay with. Defaults to one reconstructed from
        the record: same period, same iteration count, original durations
        injected. Pass a config to replay under different gains.
    """
    sig = record.signal_matrix()
    samples = sig[:, 0:2].copy()
    totals = tuple(float(x) for x in record.durations_s())
    if config is None:
        from .timing import PeriodSpec

        config = LoopConfig(
            period=PeriodSpec(period_s=record.period_s, deadline_s=record.deadline_s),
            n_iterations=record.n_completed,
            injected_durations_s=totals,
        )
    backend = ReplayBackend(samples)
    return HilLoop(backend, config).run()
