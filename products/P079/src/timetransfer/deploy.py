"""Deployment and recovery as executable checks, not prose.

Pre-flight
----------
:func:`preflight` runs a fixed list of checks against a backend and returns a
:class:`CheckReport`. Each check is a small executable assertion carrying a
measured value and the value it was required to have, so the report is
evidence and not a tick-list. The narrative version, for the Jetson Orin Nano
specifically, is ``docs/DEPLOYMENT.md``; this module is what that document
tells the operator to run.

One check exists because of a defect
------------------------------------
``state_unchanged`` is in the list because the checks themselves perturb the
thing they are checking: a probe acquisition advances the clock model's noise
stream and the link model's generator, so two backends that had been
pre-flighted differently would no longer agree and the parity check would fail
for a reason that had nothing to do with parity. :func:`preflight` snapshots
the backend on entry, restores it on exit, and reports the field-by-field
comparison rather than assuming the restore worked.

Recovery
--------
Two different failures, two different mechanisms.

* **The run raised.** :class:`RunGuard` snapshots the backend on entry and, if
  the body raises, restores the snapshot and verifies it field by field.
  :meth:`RunGuard.recovery_report` gives the compared values.
* **The exchange was lost mid-estimate.** This is not an exception; it is the
  normal end of a degraded acquisition. :func:`recover_truncated_estimate`
  takes whatever the acquisition returned and produces an estimate with its
  uncertainty computed from the exchanges that actually arrived, plus a
  :attr:`TruncatedEstimate.truncated` flag and the fraction of the planned
  window that was lost. The estimate is not thrown away and it is not
  presented as though the window had completed; the uncertainty is larger and
  the flag says why.

Scope
-----
These checks run against whatever backend they are given. Against a simulated
backend or a loopback stub they verify the harness. Against real hardware they
would verify the rig -- and only then would their results say anything about
hardware. Nothing here can tell the difference on its own;
:attr:`~timetransfer.hal.BackendInfo.is_hardware` is what records which case it
was, and ``expect_hardware`` makes the operator declare it so that a mismatch
is a failed check rather than a silent assumption.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import TracebackType
from typing import Self

import numpy as np

from .acquire import acquire
from .errors import ConfigurationError, PreflightFailure, RecoveryError
from .exchange import ExchangeSet
from .hal import Backend
from .skew import fit_least_squares
from .twoway import two_way_estimate
from .uncertainty import AsymmetrySource, OffsetUncertainty, offset_uncertainty

__all__ = [
    "CheckReport",
    "CheckResult",
    "RunGuard",
    "TruncatedEstimate",
    "preflight",
    "recover_truncated_estimate",
]


@dataclass(frozen=True)
class CheckResult:
    """One pre-flight check.

    Attributes
    ----------
    name:
        Short identifier, stable across runs so a report can be diffed.
    passed:
        Whether the check passed.
    measured:
        What was observed, as a string.
    expected:
        What was required, as a string.
    detail:
        One line of context.
    """

    name: str
    passed: bool
    measured: str
    expected: str
    detail: str = ""


@dataclass(frozen=True)
class CheckReport:
    """The result of :func:`preflight`.

    Attributes
    ----------
    checks:
        Every check, in the order it ran.
    backend_kind, backend_driver, is_hardware:
        Copied from the backend so the report cannot be separated from it.
    """

    checks: tuple[CheckResult, ...]
    backend_kind: str
    backend_driver: str
    is_hardware: bool

    @property
    def passed(self) -> bool:
        """``True`` only if every check passed."""
        return all(check.passed for check in self.checks)

    @property
    def failures(self) -> tuple[CheckResult, ...]:
        """The checks that failed."""
        return tuple(check for check in self.checks if not check.passed)

    def as_text(self) -> str:
        """Fixed-width report."""
        head = f"{'check':>22} {'result':>7} {'measured':>26} {'expected':>26}"
        rows = [
            f"{c.name:>22} {('PASS' if c.passed else 'FAIL'):>7} "
            f"{c.measured:>26} {c.expected:>26}"
            for c in self.checks
        ]
        tail = (
            f"backend {self.backend_kind}/{self.backend_driver or 'n/a'}  "
            f"is_hardware={self.is_hardware}  "
            f"{sum(c.passed for c in self.checks)}/{len(self.checks)} passed"
        )
        return "\n".join([head, *rows, tail])

    def raise_if_failed(self) -> None:
        """Raise :class:`~timetransfer.errors.PreflightFailure` on any failure."""
        if not self.passed:
            names = ", ".join(c.name for c in self.failures)
            raise PreflightFailure(f"pre-flight failed: {names}", report=self)


def _compare_snapshots(before: object, after: object, path: str = "") -> list[str]:
    """Field paths where two snapshots differ, recursing into dicts."""
    if isinstance(before, dict) and isinstance(after, dict):
        diffs: list[str] = []
        for key in sorted(set(before) | set(after)):
            child = f"{path}.{key}" if path else str(key)
            diffs.extend(_compare_snapshots(before.get(key), after.get(key), child))
        return diffs
    if isinstance(before, float) and isinstance(after, float):
        same = before == after or (np.isnan(before) and np.isnan(after))
        return [] if same else [path]
    return [] if before == after else [path]


def preflight(
    backend: Backend,
    *,
    expect_hardware: bool,
    probe_attempts: int = 24,
    max_loss_fraction: float = 0.25,
    max_resolution_ns: int = 1_000_000,
    monotone_reads: int = 200,
) -> CheckReport:
    """Run the pre-run checks against ``backend`` and report what was measured.

    The backend is snapshotted on entry and restored on exit, so running
    pre-flight does not change the state of the thing that is about to be run.

    Parameters
    ----------
    backend:
        The backend to check. It is opened and closed by this function.
    expect_hardware:
        What the operator believes. A mismatch against
        :attr:`~timetransfer.hal.BackendInfo.is_hardware` fails the
        ``hardware_declaration`` check, which is the one guard against a
        simulated run being written up as a board measurement.
    probe_attempts:
        Exchanges attempted in the probe acquisition, >= 8.
    max_loss_fraction:
        Largest acceptable probe loss fraction.
    max_resolution_ns:
        Largest acceptable timebase resolution [ns].
    monotone_reads:
        Timebase reads used for the monotonicity check, >= 8.

    Returns
    -------
    CheckReport
    """
    if int(probe_attempts) < 8:
        raise ConfigurationError(f"probe_attempts must be >= 8, got {probe_attempts!r}")
    if int(monotone_reads) < 8:
        raise ConfigurationError(f"monotone_reads must be >= 8, got {monotone_reads!r}")
    checks: list[CheckResult] = []
    info = backend.info
    snapshot_before = backend.snapshot()

    opened = True
    open_error = ""
    try:
        backend.open()
    except Exception as exc:  # noqa: BLE001 - a failed open is a reportable check
        opened = False
        open_error = f"{type(exc).__name__}"
    checks.append(
        CheckResult(
            name="backend_opens",
            passed=opened,
            measured="opened" if opened else open_error,
            expected="opened",
            detail="backend.open() must succeed before anything else is attempted",
        )
    )

    checks.append(
        CheckResult(
            name="hardware_declaration",
            passed=bool(info.is_hardware) == bool(expect_hardware),
            measured=f"is_hardware={info.is_hardware}",
            expected=f"is_hardware={bool(expect_hardware)}",
            detail="the operator must declare whether this run is on hardware",
        )
    )

    reads = [backend.timebase.now_ns() for _ in range(int(monotone_reads))]
    regressions = int(np.sum(np.diff(np.asarray(reads, dtype=np.int64)) < 0))
    checks.append(
        CheckResult(
            name="timebase_monotone",
            passed=regressions == 0,
            measured=f"{regressions} regressions in {len(reads)}",
            expected="0 regressions",
            detail="a clock that goes backwards invalidates every duration",
        )
    )

    resolution = int(backend.timebase.resolution_ns)
    checks.append(
        CheckResult(
            name="timebase_resolution",
            passed=resolution <= int(max_resolution_ns),
            measured=f"{resolution} ns",
            expected=f"<= {int(max_resolution_ns)} ns",
            detail="quantisation enters the offset uncertainty as delta/sqrt(12)",
        )
    )

    probe: ExchangeSet | None = None
    probe_error = ""
    if opened:
        try:
            probe = acquire(backend, int(probe_attempts), min_exchanges=8).exchanges
        except Exception as exc:  # noqa: BLE001 - a failed probe is a reportable check
            probe_error = f"{type(exc).__name__}"
    checks.append(
        CheckResult(
            name="probe_acquires",
            passed=probe is not None,
            measured=f"{probe.n} exchanges" if probe is not None else probe_error or "not run",
            expected=">= 8 exchanges",
            detail="a probe acquisition must complete before the real window opens",
        )
    )

    if probe is not None:
        ordering_bad = int(
            np.sum(probe.t4_ns < probe.t1_ns) + np.sum(probe.t3_ns < probe.t2_ns)
        )
        rtt = probe.round_trip_ns()
        checks.append(
            CheckResult(
                name="timestamp_ordering",
                passed=ordering_bad == 0,
                measured=f"{ordering_bad} out-of-order",
                expected="0 out-of-order",
                detail="t1 <= t4 and t2 <= t3 on every exchange",
            )
        )
        checks.append(
            CheckResult(
                name="round_trip_positive",
                passed=bool(np.all(rtt > 0)),
                measured=f"min {int(np.min(rtt))} ns",
                expected="> 0 ns",
                detail="a non-positive round trip means the timestamps are not what they claim",
            )
        )
        checks.append(
            CheckResult(
                name="loss_budget",
                passed=probe.loss_fraction <= float(max_loss_fraction),
                measured=f"{probe.loss_fraction:.4f}",
                expected=f"<= {float(max_loss_fraction):.4f}",
                detail="probe loss fraction over the attempted exchanges",
            )
        )
        estimate_ok = False
        estimate_detail = "not run"
        try:
            est = two_way_estimate(probe)
            fit = fit_least_squares(est.midpoint_s, est.offset_s, order=1)
            unc = offset_uncertainty(fit, resolution_ns=resolution)
            estimate_ok = bool(
                np.isfinite(unc.offset_s) and np.isfinite(unc.u_combined_s) and unc.u_combined_s > 0
            )
            estimate_detail = f"u={unc.u_combined_s:.3e} s"
        except Exception as exc:  # noqa: BLE001
            estimate_detail = f"{type(exc).__name__}"
        checks.append(
            CheckResult(
                name="estimate_finite",
                passed=estimate_ok,
                measured=estimate_detail,
                expected="finite offset and u > 0",
                detail="the full estimation path must run on the probe data",
            )
        )
    else:
        for name in (
            "timestamp_ordering",
            "round_trip_positive",
            "loss_budget",
            "estimate_finite",
        ):
            checks.append(
                CheckResult(
                    name=name,
                    passed=False,
                    measured="probe unavailable",
                    expected="probe acquired",
                    detail="skipped because the probe acquisition did not complete",
                )
            )

    backend.close()
    backend.restore(snapshot_before)
    snapshot_after = backend.snapshot()
    diffs = _compare_snapshots(snapshot_before, snapshot_after)
    checks.append(
        CheckResult(
            name="state_unchanged",
            passed=len(diffs) == 0,
            measured=f"{len(diffs)} fields differ",
            expected="0 fields differ",
            detail="pre-flight must leave the backend exactly as it found it",
        )
    )

    return CheckReport(
        checks=tuple(checks),
        backend_kind=info.kind,
        backend_driver=info.driver,
        is_hardware=bool(info.is_hardware),
    )


class RunGuard:
    """Context manager that restores a backend if the run inside it raises.

    Parameters
    ----------
    backend:
        The backend to protect.

    Examples
    --------
    ``with RunGuard(backend) as guard: ...`` -- on an exception the backend is
    restored and the exception propagates; :meth:`recovery_report` then says
    field by field what was put back.
    """

    def __init__(self, backend: Backend) -> None:
        self._backend = backend
        self._before: dict[str, object] | None = None
        self._after: dict[str, object] | None = None
        self.restored = False
        self.differing_fields: tuple[str, ...] = ()
        self.exception_type = ""

    def __enter__(self) -> Self:
        self._before = self._backend.snapshot()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        if exc_type is None or self._before is None:
            return False
        self.exception_type = exc_type.__name__
        try:
            self._backend.restore(self._before)
        except Exception as restore_exc:  # noqa: BLE001
            raise RecoveryError(
                f"backend could not be restored after {self.exception_type}: "
                f"{type(restore_exc).__name__}"
            ) from restore_exc
        self._after = self._backend.snapshot()
        self.differing_fields = tuple(_compare_snapshots(self._before, self._after))
        self.restored = len(self.differing_fields) == 0
        return False

    @property
    def n_snapshot_fields(self) -> int:
        """Number of leaf fields compared."""
        if self._before is None:
            return 0
        return _count_leaves(self._before)

    def recovery_report(self) -> str:
        """Human-readable statement of what was restored."""
        if not self.exception_type:
            return "no exception; nothing to recover"
        return (
            f"exception           : {self.exception_type}\n"
            f"restored            : {self.restored}\n"
            f"fields compared     : {self.n_snapshot_fields}\n"
            f"fields differing    : {len(self.differing_fields)}"
            + (f" ({', '.join(self.differing_fields)})" if self.differing_fields else "")
        )


def _count_leaves(value: object) -> int:
    """Leaf count of a nested snapshot dictionary."""
    if isinstance(value, dict):
        return sum(_count_leaves(v) for v in value.values())
    return 1


@dataclass(frozen=True)
class TruncatedEstimate:
    """An estimate from an acquisition that ended early, labelled as such.

    Attributes
    ----------
    uncertainty:
        The offset and its uncertainty budget from the exchanges that arrived.
    n_used, n_planned:
        Exchanges used and exchanges the window was planned for.
    truncated:
        ``True`` when fewer arrived than planned.
    completed_fraction:
        ``n_used / n_planned``.
    uncertainty_ratio_to_plan:
        ``sqrt(n_planned / n_used)`` -- how much larger the statistical term is
        than it would have been on a complete window, under the independence
        assumption. Reported as a *scaling expectation*, not as a correction:
        the actual uncertainty in :attr:`uncertainty` is measured from the data
        that arrived, serial correlation included.
    note:
        One line for an operator.
    """

    uncertainty: OffsetUncertainty
    n_used: int
    n_planned: int
    truncated: bool
    completed_fraction: float
    uncertainty_ratio_to_plan: float
    note: str = ""

    def as_text(self) -> str:
        """Human-readable summary."""
        return (
            f"truncated           : {self.truncated}\n"
            f"exchanges used      : {self.n_used} of {self.n_planned} planned "
            f"({self.completed_fraction:.4f})\n"
            f"expected u scaling  : x{self.uncertainty_ratio_to_plan:.3f} vs a full window\n"
            f"{self.uncertainty.as_text()}"
            + (f"\nnote                : {self.note}" if self.note else "")
        )


def recover_truncated_estimate(
    exchanges: ExchangeSet,
    n_planned: int,
    *,
    order: int = 1,
    resolution_ns: int = 1,
    asymmetry_source: AsymmetrySource = AsymmetrySource.UNACCOUNTED,
    asymmetry_bound_s: float = 0.0,
    asymmetry_stderr_s: float = 0.0,
) -> TruncatedEstimate:
    """Estimate from a truncated acquisition, with the truncation declared.

    Parameters
    ----------
    exchanges:
        Whatever the acquisition returned.
    n_planned:
        Exchanges the window was planned for, >= 1.
    order:
        Clock-polynomial order.
    resolution_ns:
        Timestamp resolution [ns].
    asymmetry_source, asymmetry_bound_s, asymmetry_stderr_s:
        Passed to :func:`timetransfer.uncertainty.offset_uncertainty`.

    Returns
    -------
    TruncatedEstimate

    Raises
    ------
    InsufficientDataError
        Too few exchanges to fit the requested polynomial order.
    """
    if int(n_planned) < 1:
        raise ConfigurationError(f"n_planned must be >= 1, got {n_planned!r}")
    estimate = two_way_estimate(exchanges)
    fit = fit_least_squares(estimate.midpoint_s, estimate.offset_s, order=int(order))
    unc = offset_uncertainty(
        fit,
        resolution_ns=int(resolution_ns),
        asymmetry_source=asymmetry_source,
        asymmetry_bound_s=float(asymmetry_bound_s),
        asymmetry_stderr_s=float(asymmetry_stderr_s),
    )
    used = int(exchanges.n)
    planned = int(n_planned)
    return TruncatedEstimate(
        uncertainty=unc,
        n_used=used,
        n_planned=planned,
        truncated=used < planned,
        completed_fraction=float(used) / planned,
        uncertainty_ratio_to_plan=float(np.sqrt(planned / used)) if used else float("inf"),
        note=(
            "window ended early; the statistical term is from the exchanges that arrived"
            if used < planned
            else ""
        ),
    )
