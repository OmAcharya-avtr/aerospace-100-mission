"""Dry-run mode: the whole path, against a recording, with nothing driven.

What a dry run is for
---------------------
Before an acquisition window opens, the thing worth knowing is whether the
acquisition and estimation path works end to end -- feature extraction,
polynomial fit, Kalman filter, uncertainty budget, Allan table -- without
spending the window finding out. A dry run runs exactly that path against a
recorded exchange set.

Proved by counting, not by inspection
-------------------------------------
"Nothing was driven" is a claim about a count, so it is implemented as one. The
recorded driver is wrapped in a
:class:`~timetransfer.backends.stubs.GuardedDriver`, which counts every
operation that passes through it, and the result carries:

* ``driver_operations`` -- operations on the recording (file-level reads;
  expected to equal the number of attempted exchanges),
* ``hardware_operations`` -- operations on any driver whose
  :attr:`~timetransfer.hal.BackendInfo.is_hardware` is true, which is **zero
  by construction** because no such driver exists in this repository, and
* ``refused_operations`` -- from the negative control below.

The negative control is the part that makes the claim falsifiable:
:func:`dry_run_negative_control` runs the identical function with the guard set
to forbid exchanges, and the run must come back with zero completed exchanges
and a non-zero refusal count. A counter that cannot be made to fire is not
evidence, and that is checked in ``tests/test_dryrun.py`` and in
``validation/validate_dryrun.py``.

Recording
---------
:func:`record_exchanges` drives any backend and writes the result as JSON via
:meth:`~timetransfer.exchange.ExchangeSet.write_json`. A recording carries the
backend kind, driver and ``is_hardware`` flag of whatever produced it, so a
dry run over a recording can never be mistaken for a hardware measurement; the
recording shipped in ``fixtures/`` came from the simulated backend and says so.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .acquire import AcquisitionReport, acquire
from .backends.device import DeviceBackend
from .backends.stubs import GuardedDriver, RecordedDriver
from .errors import ConfigurationError, DeviceAbsentError, InsufficientDataError
from .exchange import ExchangeSet
from .hal import Backend
from .skew import LeastSquaresFit, fit_least_squares
from .twoway import TwoWayEstimate, two_way_estimate
from .uncertainty import AsymmetrySource, OffsetUncertainty, offset_uncertainty

__all__ = [
    "DryRunResult",
    "NegativeControlResult",
    "dry_run",
    "dry_run_negative_control",
    "record_exchanges",
]


def record_exchanges(
    backend: Backend, n_attempts: int, path: str | Path, *, min_exchanges: int = 8
) -> tuple[Path, AcquisitionReport]:
    """Drive ``backend`` and write the exchanges to ``path`` as JSON.

    Parameters
    ----------
    backend:
        An open backend.
    n_attempts:
        Exchanges to attempt.
    path:
        Destination file.
    min_exchanges:
        Minimum completed exchanges for the recording to be usable.

    Returns
    -------
    tuple
        ``(path, report)``.
    """
    report = acquire(backend, int(n_attempts), min_exchanges=int(min_exchanges))
    written = report.exchanges.write_json(path)
    return written, report


@dataclass(frozen=True)
class DryRunResult:
    """What a dry run produced, and what it touched.

    Attributes
    ----------
    report:
        The acquisition report from the replay.
    estimate:
        The symmetric-delay estimate over the replayed exchanges.
    fit:
        The least-squares clock-polynomial fit.
    uncertainty:
        The offset uncertainty budget.
    driver_operations:
        Operations that reached the recording.
    hardware_operations:
        Operations that reached a driver declaring itself hardware. Zero.
    refused_operations:
        Operations the guard refused.
    backend_driver:
        The driver name recorded on the backend, e.g. ``"guarded-recorded"``.
    is_hardware:
        ``False``; carried so that nothing downstream has to look it up.
    """

    report: AcquisitionReport
    estimate: TwoWayEstimate
    fit: LeastSquaresFit
    uncertainty: OffsetUncertainty
    driver_operations: int
    hardware_operations: int
    refused_operations: int
    backend_driver: str
    is_hardware: bool

    def as_text(self) -> str:
        """Human-readable summary, with the counts that make the claim."""
        return (
            f"replayed exchanges   : {self.report.exchanges.n} of "
            f"{self.report.n_attempted} attempted\n"
            f"driver operations    : {self.driver_operations}\n"
            f"hardware operations  : {self.hardware_operations} (must be 0)\n"
            f"refused operations   : {self.refused_operations}\n"
            f"backend driver       : {self.backend_driver}\n"
            f"is_hardware          : {self.is_hardware}\n"
            f"offset               : {self.uncertainty.offset_s:.9e} s\n"
            f"u combined           : {self.uncertainty.u_combined_s:.6e} s"
        )


def _run(
    recorded: ExchangeSet, *, allow_exchange: bool, min_exchanges: int
) -> tuple[GuardedDriver, DeviceBackend, AcquisitionReport | None]:
    """Shared body of the dry run and its negative control.

    Builds the guarded recorded backend, attempts every recorded sequence
    number, and returns the driver (for its counters), the backend (for its
    identity) and the report, or ``None`` when too few exchanges survived.
    """
    driver = GuardedDriver(RecordedDriver(recorded), allow_exchange=allow_exchange)
    backend = DeviceBackend(
        driver, name="dry-run", period_s=1.0, virtual_timebase=True, resolution_ns=1
    )
    backend.open()
    n_attempts = int(np.max(recorded.seq)) + 1
    try:
        report: AcquisitionReport | None = acquire(
            backend, n_attempts, min_exchanges=int(min_exchanges)
        )
    except InsufficientDataError:
        report = None
    finally:
        backend.close()
    return driver, backend, report


def dry_run(
    recorded: ExchangeSet | str | Path,
    *,
    order: int = 1,
    asymmetry_source: AsymmetrySource = AsymmetrySource.UNACCOUNTED,
    asymmetry_bound_s: float = 0.0,
    asymmetry_stderr_s: float = 0.0,
    min_exchanges: int = 8,
) -> DryRunResult:
    """Run acquisition, estimation and the uncertainty budget over a recording.

    Parameters
    ----------
    recorded:
        An exchange set, or a path to one written by
        :meth:`~timetransfer.exchange.ExchangeSet.write_json`.
    order:
        Polynomial order for the clock fit, 1 or 2.
    asymmetry_source, asymmetry_bound_s, asymmetry_stderr_s:
        Passed to :func:`timetransfer.uncertainty.offset_uncertainty`.
    min_exchanges:
        Minimum completed exchanges.

    Returns
    -------
    DryRunResult
    """
    data = (
        recorded
        if isinstance(recorded, ExchangeSet)
        else ExchangeSet.read_json(Path(recorded))
    )
    if data.is_hardware:
        raise ConfigurationError(
            "a recording marked is_hardware=true cannot be replayed as a dry run; "
            "replaying hardware data and labelling the result a dry run would mislabel both"
        )
    driver, backend, report = _run(
        data, allow_exchange=True, min_exchanges=int(min_exchanges)
    )
    if report is None:
        raise InsufficientDataError(
            "dry run replayed too few exchanges to estimate from",
            n_usable=0,
            n_required=int(min_exchanges),
        )
    estimate = two_way_estimate(report.exchanges)
    fit = fit_least_squares(estimate.midpoint_s, estimate.offset_s, order=int(order))
    unc = offset_uncertainty(
        fit,
        resolution_ns=1,
        asymmetry_source=asymmetry_source,
        asymmetry_bound_s=float(asymmetry_bound_s),
        asymmetry_stderr_s=float(asymmetry_stderr_s),
    )
    return DryRunResult(
        report=report,
        estimate=estimate,
        fit=fit,
        uncertainty=unc,
        driver_operations=int(driver.touch_count),
        hardware_operations=0 if not backend.info.is_hardware else int(driver.touch_count),
        refused_operations=int(driver.refused_exchanges),
        backend_driver=backend.info.driver,
        is_hardware=bool(backend.info.is_hardware),
    )


@dataclass(frozen=True)
class NegativeControlResult:
    """Result of the dry-run negative control.

    Attributes
    ----------
    completed:
        Exchanges that completed; must be 0.
    refused:
        Operations the guard refused; must be > 0.
    raised_insufficient:
        Whether the acquisition raised
        :class:`~timetransfer.errors.InsufficientDataError`, which it must,
        because an acquisition that silently returned nothing would be the
        defect this control exists to catch.
    """

    completed: int
    refused: int
    raised_insufficient: bool

    @property
    def passed(self) -> bool:
        """``True`` when the control fired as it must."""
        return self.completed == 0 and self.refused > 0 and self.raised_insufficient


def dry_run_negative_control(
    recorded: ExchangeSet | str | Path, *, min_exchanges: int = 8
) -> NegativeControlResult:
    """Run the dry-run path with device operations forbidden.

    The counters in :class:`DryRunResult` are only evidence if they can be made
    to fire, so this runs the identical backend with the guard refusing every
    exchange. Every recorded sequence number is attempted individually, and
    then the ordinary :func:`timetransfer.acquire.acquire` path is attempted
    once, so the refusal count is ``n_attempts + 1`` and the acquisition's own
    refusal is confirmed rather than assumed.
    """
    data = (
        recorded
        if isinstance(recorded, ExchangeSet)
        else ExchangeSet.read_json(Path(recorded))
    )
    driver = GuardedDriver(RecordedDriver(data), allow_exchange=False)
    backend = DeviceBackend(
        driver, name="dry-run-control", period_s=1.0, virtual_timebase=True, resolution_ns=1
    )
    backend.open()
    n_attempts = int(np.max(data.seq)) + 1
    completed = 0
    for seq in range(n_attempts):
        try:
            outcome = backend.next_exchange(seq)
        except DeviceAbsentError:
            continue
        if not outcome.lost:
            completed += 1
    raised = False
    try:
        acquire(backend, n_attempts, min_exchanges=int(min_exchanges))
    except (DeviceAbsentError, InsufficientDataError):
        raised = True
    backend.close()
    return NegativeControlResult(
        completed=completed,
        refused=int(driver.refused_exchanges),
        raised_insufficient=raised,
    )
