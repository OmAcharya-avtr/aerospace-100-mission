"""The hardware abstraction layer: one contract, a simulated and a device side.

One contract, two implementations
---------------------------------
Everything an acquisition touches goes through three protocols --
:class:`~timetransfer.timebase.Timebase`, :class:`TimestampDriver` and
:class:`Backend`. A simulated backend and a device backend implement the same
:class:`Backend` protocol, so :func:`timetransfer.acquire.acquire` and every
estimator downstream of it run unchanged against either. That symmetry is the
only reason simulation/device parity is testable at all; the test is
``tests/test_parity.py`` and the evidence is
``validation/validate_parity_output.txt``.

Why ``is_hardware`` is a property of the backend
------------------------------------------------
The most damaging thing this package could do is present a simulated timing
figure as a board measurement. :attr:`BackendInfo.is_hardware` is ``False``
for every backend in this repository, including the device backend, because
its only drivers are a loopback stub and a recording replayer. Every exchange
set, every benchmark record and every validation output carries the flag, and
:class:`~timetransfer.errors.HardwareClaimError` is raised if a number
produced with the flag ``False`` is about to be labelled as hardware.

Conventions
-----------
* Timestamps crossing the HAL are **integer nanoseconds** (see
  :mod:`timetransfer.timebase`).
* A lost exchange is reported as an :class:`ExchangeOutcome` with
  ``lost=True`` and ``exchange=None``, not as an exception from the backend.
  The driver below it may raise
  :class:`~timetransfer.errors.ExchangeLostError`; the backend converts it,
  because a loss is data and an exception is a fault.
* A backend that cannot be opened raises
  :class:`~timetransfer.errors.DeviceAbsentError` from :meth:`Backend.open`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from .errors import ConfigurationError
from .exchange import Exchange
from .timebase import Timebase

__all__ = [
    "STUB_DRIVERS",
    "Backend",
    "BackendInfo",
    "ExchangeOutcome",
    "ExchangeTruth",
    "TimestampDriver",
]


STUB_DRIVERS: frozenset[str] = frozenset({"", "loopback", "recorded", "absent"})
"""Driver names that are **not** hardware. A ``guarded-`` prefix is stripped first."""


@dataclass(frozen=True)
class BackendInfo:
    """Identity of a backend, recorded with every set of exchanges it produces.

    Attributes
    ----------
    kind:
        ``"simulated"`` or ``"device"``.
    name:
        Short human-readable label.
    driver:
        For a device backend, the driver shim it is talking to. ``"loopback"``
        and ``"recorded"`` are stubs and are **not** hardware; they are named
        here so that every record says so.
    detail:
        Free-text key/value pairs, e.g. the model parameters or the driver's
        probe result.
    """

    kind: str
    name: str
    driver: str = ""
    detail: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in ("simulated", "device"):
            raise ConfigurationError(
                f"backend kind must be 'simulated' or 'device', got {self.kind!r}"
            )
        if not self.name:
            raise ConfigurationError("backend name must be non-empty")

    @property
    def is_hardware(self) -> bool:
        """``True`` only for a device backend not driven by a stub.

        A ``"guarded-"`` prefix is stripped before the comparison, so wrapping
        a stub in :class:`~timetransfer.backends.stubs.GuardedDriver` cannot
        turn it into hardware. That is not a hypothetical: the first version of
        this property did not strip the prefix, and a guarded recording
        reported ``is_hardware=True``, which ``tests/test_dryrun.py`` caught.

        A number produced with this ``False`` may never be presented as a
        hardware measurement.
        """
        name = self.driver.removeprefix("guarded-")
        return self.kind == "device" and name not in STUB_DRIVERS


@dataclass(frozen=True)
class ExchangeTruth:
    """Simulation truth for one exchange. ``None`` on a real device.

    Attributes
    ----------
    offset_s:
        True remote clock offset at the exchange midpoint [s]; the estimand.
    asymmetry_s:
        True ``d_ab - d_ba`` [s] for this exchange.
    forward_delay_s, reverse_delay_s:
        The two true one-way delays [s].
    """

    offset_s: float
    asymmetry_s: float
    forward_delay_s: float
    reverse_delay_s: float

    @property
    def bias_s(self) -> float:
        """``asymmetry_s / 2`` [s]: the error this exchange puts on Baseline 1."""
        return 0.5 * float(self.asymmetry_s)


@dataclass(frozen=True)
class ExchangeOutcome:
    """What one attempted exchange produced.

    Attributes
    ----------
    seq:
        Sequence number of the attempt.
    exchange:
        The four timestamps, or ``None`` if the attempt was lost.
    lost:
        ``True`` if the attempt did not complete.
    truth:
        Simulation truth, or ``None``.
    """

    seq: int
    exchange: Exchange | None
    lost: bool
    truth: ExchangeTruth | None = None

    def __post_init__(self) -> None:
        if self.lost and self.exchange is not None:
            raise ConfigurationError(f"outcome {self.seq}: lost outcome carries timestamps")
        if not self.lost and self.exchange is None:
            raise ConfigurationError(f"outcome {self.seq}: completed outcome carries no timestamps")


@runtime_checkable
class TimestampDriver(Protocol):
    """The five-method shim a device backend talks to.

    A real implementation would wrap a NIC's hardware timestamping, a GPIO
    pulse-per-second capture, or a serial time code. None exists in this
    repository: the two implementations are a loopback stub over the
    simulation engine and a replayer over a recorded exchange set.
    """

    @property
    def name(self) -> str:
        """Driver name, recorded in :attr:`BackendInfo.driver`."""

    @property
    def touch_count(self) -> int:
        """Number of times the driver has been asked to perform an exchange.

        Counted rather than inferred, because the dry-run claim is that this
        stays at zero and a claim about a device not being touched has to be a
        count.
        """

    def open(self) -> None:
        """Acquire the device; raise ``DeviceAbsentError`` if none answers."""

    def close(self) -> None:
        """Release the device. Idempotent."""

    def probe(self) -> dict[str, str]:
        """Identity and configuration read back from the device."""

    def exchange(self, seq: int) -> tuple[int, int, int, int]:
        """Perform one exchange; return ``(t1, t2, t3, t4)`` [ns].

        Raise :class:`~timetransfer.errors.ExchangeLostError` if the exchange
        did not complete.
        """


@runtime_checkable
class Backend(Protocol):
    """A complete HAL instance: a timebase plus a source of exchanges."""

    @property
    def info(self) -> BackendInfo:
        """Backend identity, recorded with every exchange set."""

    @property
    def timebase(self) -> Timebase:
        """The backend's local timebase."""

    @property
    def is_open(self) -> bool:
        """``True`` between a successful :meth:`open` and :meth:`close`."""

    def open(self) -> None:
        """Acquire the backend; raise ``DeviceAbsentError`` if unavailable."""

    def close(self) -> None:
        """Release the backend. Idempotent."""

    def next_exchange(self, seq: int) -> ExchangeOutcome:
        """Attempt exchange number ``seq``."""

    def snapshot(self) -> dict[str, object]:
        """Complete state, sufficient for :meth:`restore` to undo a run."""

    def restore(self, snapshot: dict[str, object]) -> None:
        """Put the backend back into a state captured by :meth:`snapshot`."""
