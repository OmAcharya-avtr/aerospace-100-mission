"""Exception types.

Every invalid input raises one of these rather than returning a sentinel,
because a silently degraded time estimate is indistinguishable from a good one
once it has left the function.
"""

from __future__ import annotations

__all__ = [
    "ConfigurationError",
    "DeviceAbsentError",
    "ExchangeLostError",
    "HardwareClaimError",
    "InsufficientDataError",
    "PreflightFailure",
    "RecoveryError",
    "TimebaseRegressionError",
    "TimeTransferError",
]


class TimeTransferError(Exception):
    """Base class for every error raised by this package."""


class ConfigurationError(TimeTransferError, ValueError):
    """An argument is outside the stated validity range of a model or estimator."""


class TimebaseRegressionError(TimeTransferError, RuntimeError):
    """A timebase returned a value earlier than one it had already returned.

    A clock going backwards is a fault, not a rounding artefact: the estimators
    in this package assume monotone non-decreasing local timestamps.
    """


class ExchangeLostError(TimeTransferError, RuntimeError):
    """One timestamp exchange did not complete.

    Raised by a transport when the link dropped the request or the reply. The
    acquisition layer counts these and keeps going until its retry budget is
    spent; it is not an estimation failure on its own.
    """


class InsufficientDataError(TimeTransferError, RuntimeError):
    """Fewer usable exchanges survived than the estimator needs.

    Carries ``n_usable`` and ``n_required`` so a recovery procedure can decide
    whether to extend the acquisition or abandon the estimate.
    """

    def __init__(self, message: str, *, n_usable: int, n_required: int) -> None:
        super().__init__(message)
        self.n_usable = int(n_usable)
        self.n_required = int(n_required)


class DeviceAbsentError(TimeTransferError, RuntimeError):
    """A device backend was opened and no device answered."""


class PreflightFailure(TimeTransferError, RuntimeError):
    """A pre-run check failed; carries the report that failed."""

    def __init__(self, message: str, *, report: object = None) -> None:
        super().__init__(message)
        self.report = report


class RecoveryError(TimeTransferError, RuntimeError):
    """A backend could not be returned to its pre-run state after an abort."""


class HardwareClaimError(TimeTransferError, RuntimeError):
    """A number measured off hardware was about to be labelled as hardware.

    This exists because the single most damaging thing this package could do is
    present a simulated timing figure as a board measurement. The guard is an
    exception rather than a warning so that it cannot be ignored.
    """
