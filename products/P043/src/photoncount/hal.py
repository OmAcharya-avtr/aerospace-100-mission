"""Hardware abstraction layer: one contract, a simulated backend, a device stub.

The point of this module is that the code above it never knows whether it is
talking to a model or to a detector. :class:`PhotonCountingBackend` is the whole
interface; :class:`SimulatedBackend` implements it against
:mod:`photoncount.simulate`; :class:`DeviceBackend` carries the documented
contract a real photon-counting module must satisfy and raises
``NotImplementedError`` for every operation that needs silicon.

**Why a stub rather than a mock.** A mock that returns plausible numbers is
worse than nothing: its output is indistinguishable from a measurement in a log
file. :class:`DeviceBackend` therefore refuses, loudly, naming what is missing.
Every :class:`Acquisition` records ``backend`` and ``simulated``, and
:func:`Acquisition.is_measurement` is the single predicate anything downstream
must consult before treating a number as a measurement.

**Modes.**

``BackendMode.SIMULATION``
    The full chain against the models, deterministic under a seeded generator.

``BackendMode.DRY_RUN``
    The real command path and the real per-call timing, with every count
    discarded. A dry run exists to rehearse a deployment: it proves the
    sequence of calls, the validation, the timing and the journal all work,
    while guaranteeing that nothing downstream can consume a result. The
    returned :class:`Acquisition` has ``discarded=True`` and an empty
    ``counts`` array, and :func:`Acquisition.is_measurement` is False.

**What is still missing for Level 4.** Everything in this module runs in a
shared cloud container. No latency, memory or throughput number produced here
may be presented as a Jetson Orin Nano figure; closing Level 4 requires running
``benchmark/run_benchmark.py`` on the board itself and keeping its raw output.
A simulated backend, an extrapolation and a vendor datasheet are all equally
inadmissible.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .simulate import DetectorSpec, simulate_run

__all__ = [
    "REQUIRED_DESCRIPTION_KEYS",
    "Acquisition",
    "AcquisitionRequest",
    "BackendMode",
    "DeviceBackend",
    "PhotonCountingBackend",
    "SimulatedBackend",
]

#: Keys every backend's :meth:`PhotonCountingBackend.describe` must return.
#: The contract test suite enforces this for the simulated and device backends
#: alike, which is the only reason the device stub is useful before hardware
#: exists: its contract is testable now.
REQUIRED_DESCRIPTION_KEYS: tuple[str, ...] = (
    "name",
    "simulated",
    "dead_time_s",
    "dead_time_model",
    "afterpulse_probability",
    "afterpulse_mean_delay_s",
    "max_sustained_rate_hz",
    "timestamp_resolution_s",
    "hardware_required",
)


class BackendMode(Enum):
    """How a backend executes an acquisition."""

    SIMULATION = "simulation"
    DRY_RUN = "dry_run"


@dataclass(frozen=True)
class AcquisitionRequest:
    """One acquisition: ``n_windows`` consecutive windows of ``window_s``.

    Attributes
    ----------
    window_s:
        Counting window length, s, ``> 0``.
    n_windows:
        Number of windows, ``>= 1``.
    label:
        Short identifier used for the run journal filename. Must be a bare
        name: no path separators, so that nothing in this package can be made
        to write outside the directory it was given.
    """

    window_s: float
    n_windows: int
    label: str = "run"

    def __post_init__(self) -> None:
        if not np.isfinite(self.window_s) or self.window_s <= 0.0:
            raise ValueError(f"window_s must be finite and > 0, got {self.window_s!r}")
        if int(self.n_windows) < 1:
            raise ValueError(f"n_windows must be >= 1, got {self.n_windows!r}")
        if not self.label or any(c in self.label for c in "/\\" ) or self.label in (".", ".."):
            raise ValueError(
                f"label must be a bare name without path separators, got {self.label!r}"
            )

    @property
    def total_seconds(self) -> float:
        """``window_s * n_windows``, s."""
        return self.window_s * int(self.n_windows)


@dataclass
class Acquisition:
    """Result of one acquisition.

    Attributes
    ----------
    backend:
        Backend name, recorded so a number can always be traced to its source.
    mode:
        The :class:`BackendMode` used.
    simulated:
        True when the counts came from a model. Never overwrite this.
    discarded:
        True for a dry run, where counts were deliberately thrown away.
    counts:
        Per-window registered counts, int array of length ``n_windows``, or
        empty for a dry run.
    window_s, n_windows:
        Echo of the request.
    wall_seconds:
        Wall-clock time the backend spent, s. Real in every mode, including a
        dry run --- that is what a dry run is for.
    metadata:
        Backend-specific extras. For the simulated backend this carries the
        true incident rate, which a device can never supply.
    """

    backend: str
    mode: BackendMode
    simulated: bool
    discarded: bool
    counts: np.ndarray
    window_s: float
    n_windows: int
    wall_seconds: float
    metadata: dict[str, float | str] = field(default_factory=dict)

    @property
    def is_measurement(self) -> bool:
        """True only for non-simulated, non-discarded counts.

        The one predicate anything reporting a number must consult.
        """
        return (not self.simulated) and (not self.discarded)

    @property
    def observed_rate_hz(self) -> float:
        """Pooled registered rate, counts/s. ``nan`` for a dry run."""
        if self.discarded or self.counts.size == 0:
            return float("nan")
        return float(self.counts.sum()) / (self.window_s * self.counts.size)

    @property
    def fano_factor(self) -> float:
        """Sample ``Var/mean`` of the per-window counts (-). ``nan`` if undefined."""
        if self.counts.size < 2:
            return float("nan")
        mean = float(self.counts.mean())
        if mean <= 0.0:
            return float("nan")
        return float(self.counts.var(ddof=1)) / mean


class PhotonCountingBackend(ABC):
    """The whole interface to a photon-counting detector.

    Contract, enforced by ``tests/test_hal_contract.py`` for every backend:

    1. :meth:`describe` returns a dict containing every key in
       :data:`REQUIRED_DESCRIPTION_KEYS`, works **without** hardware, and does
       not open the device.
    2. :meth:`acquire` validates its request and raises ``ValueError`` on a bad
       one **before** touching hardware, so a malformed request is never a
       hardware error.
    3. :meth:`acquire` raises ``RuntimeError`` if the backend is not open.
    4. An operation that needs hardware the backend does not have raises
       ``NotImplementedError`` whose message names what is missing.
    5. :meth:`close` is idempotent and safe after a failed :meth:`open`.
    6. The object is a context manager; leaving the block closes it.
    """

    #: Short backend identifier, written into every Acquisition.
    name: str = "abstract"

    #: True when counts come from a model rather than a detector.
    simulated: bool = True

    #: True when the backend cannot function without physical hardware.
    hardware_required: bool = False

    def __init__(self) -> None:
        self._open = False

    @property
    def is_open(self) -> bool:
        """Whether :meth:`open` has succeeded and :meth:`close` has not run."""
        return self._open

    @abstractmethod
    def describe(self) -> dict[str, float | str | bool]:
        """Static capability description. Must not require hardware."""

    @abstractmethod
    def open(self) -> None:
        """Acquire resources. Idempotent."""

    @abstractmethod
    def close(self) -> None:
        """Release resources. Idempotent and safe after a failed open."""

    @abstractmethod
    def self_test(self) -> dict[str, bool | str]:
        """Cheap health check. Returns at least ``{"passed": bool}``."""

    @abstractmethod
    def acquire(
        self, request: AcquisitionRequest, mode: BackendMode = BackendMode.SIMULATION
    ) -> Acquisition:
        """Run one acquisition."""

    def _require_open(self) -> None:
        if not self._open:
            raise RuntimeError(
                f"backend {self.name!r} is not open; call open() or use it as a "
                "context manager"
            )

    def __enter__(self) -> PhotonCountingBackend:
        self.open()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class SimulatedBackend(PhotonCountingBackend):
    """Full implementation of the contract against :mod:`photoncount.simulate`.

    Parameters
    ----------
    spec:
        The detector non-idealities to simulate.
    incident_rate_hz:
        The true rate to simulate, counts/s. Recorded in
        ``Acquisition.metadata["incident_rate_hz"]`` because the whole point of
        a simulated backend is that the answer is known; a device backend has
        no such field and must never be given one.
    rng:
        Generator; the backend consumes it and nothing else random, so a run is
        reproducible from its seed.
    timestamp_resolution_s:
        Reported in :meth:`describe` for contract parity with a real module. The
        simulator works in float seconds and does not quantise to it.
    """

    name = "simulated"
    simulated = True
    hardware_required = False

    def __init__(
        self,
        spec: DetectorSpec,
        incident_rate_hz: float,
        rng: np.random.Generator,
        timestamp_resolution_s: float = 1e-10,
    ) -> None:
        super().__init__()
        if not isinstance(spec, DetectorSpec):
            raise TypeError(f"spec must be a DetectorSpec, got {type(spec).__name__}")
        n = float(incident_rate_hz)
        if not np.isfinite(n) or n < 0.0:
            raise ValueError(f"incident_rate_hz must be finite and >= 0, got {n!r}")
        self.spec = spec
        self.incident_rate_hz = n
        self._rng = rng
        self._resolution = float(timestamp_resolution_s)

    def describe(self) -> dict[str, float | str | bool]:
        tau = self.spec.dead_time_s
        if tau <= 0.0:
            max_rate = float("inf")
        elif self.spec.model == "paralyzable":
            max_rate = 1.0 / (np.e * tau)
        else:
            max_rate = 1.0 / tau
        return {
            "name": self.name,
            "simulated": True,
            "dead_time_s": tau,
            "dead_time_model": self.spec.model,
            "afterpulse_probability": self.spec.afterpulse_probability,
            "afterpulse_mean_delay_s": self.spec.afterpulse_mean_delay_s,
            "max_sustained_rate_hz": float(max_rate),
            "timestamp_resolution_s": self._resolution,
            "hardware_required": False,
        }

    def open(self) -> None:
        self._open = True

    def close(self) -> None:
        self._open = False

    def self_test(self) -> dict[str, bool | str]:
        """Checks the model is self-consistent: a zero-rate window yields zero counts."""
        probe = simulate_run(0.0, 1e-3, self.spec, self._rng)
        passed = probe.registered == 0
        return {
            "passed": bool(passed),
            "detail": f"zero-rate probe registered {probe.registered} counts (expected 0)",
        }

    def acquire(
        self, request: AcquisitionRequest, mode: BackendMode = BackendMode.SIMULATION
    ) -> Acquisition:
        if not isinstance(request, AcquisitionRequest):
            raise ValueError(
                f"request must be an AcquisitionRequest, got {type(request).__name__}"
            )
        if not isinstance(mode, BackendMode):
            raise ValueError(f"mode must be a BackendMode, got {mode!r}")
        self._require_open()
        n_windows = int(request.n_windows)
        start = time.perf_counter()
        counts = np.empty(n_windows, dtype=np.int64)
        afterpulses = 0
        for i in range(n_windows):
            run = simulate_run(self.incident_rate_hz, request.window_s, self.spec, self._rng)
            counts[i] = run.registered
            afterpulses += run.registered_afterpulse
        wall = time.perf_counter() - start
        discarded = mode is BackendMode.DRY_RUN
        total = int(counts.sum())
        return Acquisition(
            backend=self.name,
            mode=mode,
            simulated=True,
            discarded=discarded,
            counts=np.empty(0, dtype=np.int64) if discarded else counts,
            window_s=request.window_s,
            n_windows=n_windows,
            wall_seconds=wall,
            metadata={
                "incident_rate_hz": self.incident_rate_hz,
                "afterpulse_fraction": afterpulses / total if total else 0.0,
                "label": request.label,
            },
        )


class DeviceBackend(PhotonCountingBackend):
    """The contract a real photon-counting module must satisfy. Not implemented.

    This class is deliberately inert. It exists so that the contract is written
    down, version-controlled and **tested** before any hardware is connected,
    and so that no code path can silently fall back to simulated numbers.

    **What an implementation must provide.**

    ``open``
        Claim the device (USB/serial/PCIe), verify firmware and serial number,
        latch the discriminator threshold and bias, and fail if any differs
        from :meth:`describe`.
    ``self_test``
        Dark-count rate within the declared bound with the shutter closed, and
        a gate-length readback matching the commanded window to the declared
        timestamp resolution.
    ``acquire``
        Arm, gate ``n_windows`` windows of ``window_s``, read back per-window
        counts, and return them with ``simulated=False``. In
        ``BackendMode.DRY_RUN`` it must execute the identical command sequence
        with identical timing and discard the readback.
    ``close``
        Disarm, release the interface, idempotent.

    **What must be measured on hardware and cannot be inferred:** the real dead
    time and whether it extends, the afterpulse probability and its delay
    distribution, the dark-count rate, timestamp resolution and jitter, the
    maximum sustained rate, and the latency, memory and throughput numbers that
    Level 4 requires. None of these may come from this package's models.
    """

    name = "device"
    simulated = False
    hardware_required = True

    def __init__(
        self,
        dead_time_s: float,
        dead_time_model: str = "paralyzable",
        afterpulse_probability: float = 0.0,
        afterpulse_mean_delay_s: float = 1e-7,
        max_sustained_rate_hz: float = 0.0,
        timestamp_resolution_s: float = 0.0,
        serial_number: str = "unset",
    ) -> None:
        super().__init__()
        self._declared = {
            "name": self.name,
            "simulated": False,
            "dead_time_s": float(dead_time_s),
            "dead_time_model": str(dead_time_model),
            "afterpulse_probability": float(afterpulse_probability),
            "afterpulse_mean_delay_s": float(afterpulse_mean_delay_s),
            "max_sustained_rate_hz": float(max_sustained_rate_hz),
            "timestamp_resolution_s": float(timestamp_resolution_s),
            "hardware_required": True,
            "serial_number": str(serial_number),
        }

    def describe(self) -> dict[str, float | str | bool]:
        """The declared contract. These are *declared* values, not measurements."""
        return dict(self._declared)

    def open(self) -> None:
        raise NotImplementedError(
            "DeviceBackend.open requires a physical photon-counting module: a "
            "USB/serial/PCIe interface, firmware and serial-number readback, and "
            "a latched bias and discriminator threshold. None of these exist in "
            "this environment."
        )

    def close(self) -> None:
        """Idempotent no-op: nothing was ever opened, so there is nothing to release."""
        self._open = False

    def self_test(self) -> dict[str, bool | str]:
        raise NotImplementedError(
            "DeviceBackend.self_test requires a shuttered dark-count measurement "
            "and a gate-length readback from real hardware."
        )

    def acquire(
        self, request: AcquisitionRequest, mode: BackendMode = BackendMode.SIMULATION
    ) -> Acquisition:
        if not isinstance(request, AcquisitionRequest):
            raise ValueError(
                f"request must be an AcquisitionRequest, got {type(request).__name__}"
            )
        if not isinstance(mode, BackendMode):
            raise ValueError(f"mode must be a BackendMode, got {mode!r}")
        raise NotImplementedError(
            "DeviceBackend.acquire requires a physical photon-counting module. "
            "Measured counts, latency, memory and throughput must come from the "
            "device itself; this package will not synthesise them."
        )
