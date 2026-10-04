"""Exception hierarchy for hardware-in-the-loop faults.

Every fault a HIL harness has to survive is represented by a distinct
exception type, so a test can assert on the specific fault rather than on a
message string. The six failure modes required by the Batch 04 Level 3
specification map one-to-one onto the types below:

================================  ===========================================
Failure mode                      Exception
================================  ===========================================
device absent                     :class:`DeviceAbsentError`
device disconnects mid-run        :class:`DeviceDisconnectedError`
sample dropped                    :class:`SampleDroppedError`
timebase steps backwards          :class:`TimebaseRegressionError`
actuator write rejected           :class:`WriteRejectedError`
loop overrun cascade              :class:`OverrunCascadeError`
================================  ===========================================
"""

from __future__ import annotations

__all__ = [
    "BackendError",
    "ConfigurationError",
    "DeviceAbsentError",
    "DeviceDisconnectedError",
    "DryRunViolationError",
    "HilForgeError",
    "OverrunCascadeError",
    "PreflightFailure",
    "RecoveryError",
    "SampleDroppedError",
    "TimebaseRegressionError",
    "WriteRejectedError",
]


class HilForgeError(Exception):
    """Base class for every error raised by :mod:`hilforge`."""


class ConfigurationError(HilForgeError, ValueError):
    """An input that cannot describe a runnable configuration."""


class BackendError(HilForgeError):
    """Base class for faults originating in a HAL backend."""


class DeviceAbsentError(BackendError):
    """The device backend was opened but no device answered.

    Raised by :meth:`hilforge.hal.Backend.open`. A harness must distinguish
    this from a mid-run disconnect: nothing has been commanded yet, so no
    recovery is required, only a clear diagnostic.
    """


class DeviceDisconnectedError(BackendError):
    """The device stopped answering part way through a run.

    Unlike :class:`DeviceAbsentError` this happens after commands may already
    have been issued, so the caller owes the hardware a recovery pass; see
    :mod:`hilforge.deploy`.
    """


class SampleDroppedError(BackendError):
    """A sensor read produced no sample within its budget.

    The loop's response is governed by
    :attr:`hilforge.loop.LoopConfig.on_dropped_sample`; the exception only
    escapes when that policy is ``"abort"``.
    """


class WriteRejectedError(BackendError):
    """An actuator refused a command.

    Carries ``reason`` so a caller can separate a saturation refusal from a
    bus-level refusal without parsing text.
    """

    def __init__(self, message: str, *, reason: str = "unspecified") -> None:
        super().__init__(message)
        self.reason = reason


class TimebaseRegressionError(BackendError):
    """A timebase returned a timestamp earlier than one already observed.

    A monotonic clock must never do this (POSIX ``CLOCK_MONOTONIC``). A
    settable clock can, and a loop that silently accepts it reports negative
    durations and under-counts overruns, so the harness refuses instead.
    """


class OverrunCascadeError(HilForgeError):
    """Consecutive deadline overruns reached the configured limit.

    The cascade is the failure mode that distinguishes a HIL run from a
    simulation: one late iteration delays the next, which is then late for a
    deadline it would otherwise have met.
    """

    def __init__(self, message: str, *, run_length: int, first_index: int) -> None:
        super().__init__(message)
        self.run_length = run_length
        self.first_index = first_index


class DryRunViolationError(HilForgeError):
    """A write reached hardware during a dry run.

    Raised by :class:`hilforge.backends.stubs.CountingActuator` when it is
    configured with ``forbid_writes=True``. This is the belt to the dry-run
    decorator's braces: the guard stops the write, and the stub proves it.
    """


class PreflightFailure(HilForgeError):
    """A pre-run check failed and the run must not start."""


class RecoveryError(HilForgeError):
    """The recovery procedure could not restore the pre-run state."""
