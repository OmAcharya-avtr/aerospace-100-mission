"""Deployment and recovery as executable checks, not prose.

Pre-run checks
--------------
:func:`preflight` runs a fixed list of checks against a backend and returns a
:class:`CheckReport`. Each check is a small executable assertion with a
measured value and an expected one, so the report is evidence rather than a
tick-list. The set covers the three things that make a run worth starting —
the device is there, the HAL contract is satisfied, the clock behaves — and
the three that make it safe to start: a dry-run rehearsal issues no write, the
backend's state can be captured and put back, and running the checks themselves
leaves the rig exactly as they found it.

That last one is a check and not an assumption because it was wrong once: the
check list reads a sensor sample, which advances the plant's noise stream, so
an early version of :func:`preflight` left the backend it was given one draw
ahead of an untouched one, and two backends that had been pre-flighted
differently no longer agreed. ``state_unchanged`` now snapshots on entry,
restores on exit and reports the field-by-field comparison.

Recovery
--------
:class:`RunGuard` snapshots the backend before a run and, if the run raises,
restores the snapshot and verifies the restoration field by field. A half-
finished HIL run leaves an actuator holding a command and a model part way
through its trajectory; recovery means those are back where they were, and
:meth:`RunGuard.recovery_report` says so with the compared values.

Scope
-----
These checks run against whatever backend they are given. Against a simulated
backend or a loopback stub they verify the harness. Against real hardware they
would verify the rig, and only then do their results say anything about
hardware. Nothing here can tell the difference on its own; the backend's
:attr:`hilforge.hal.BackendInfo.is_hardware` flag is what records which case
it was.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Self

import numpy as np

from .dryrun import DryRunActuator
from .errors import DeviceAbsentError, PreflightFailure, RecoveryError
from .hal import ChannelSpec
from .timing import MonotonicGuard

__all__ = [
    "CheckReport",
    "CheckResult",
    "RunGuard",
    "preflight",
    "restore_backend",
    "snapshot_backend",
    "verify_restored",
]


@dataclass(frozen=True)
class CheckResult:
    """One pre-run check.

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

    def as_line(self) -> str:
        """One fixed-width line for a report."""
        mark = "PASS" if self.passed else "FAIL"
        return (
            f"[{mark}] {self.name:<26} measured={self.measured:<24} "
            f"expected={self.expected}"
            + (f"  ({self.detail})" if self.detail else "")
        )


@dataclass
class CheckReport:
    """A list of :class:`CheckResult` with a pass/fail verdict."""

    results: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """``True`` only if every check passed."""
        return all(r.passed for r in self.results)

    @property
    def n_passed(self) -> int:
        """Number of checks that passed."""
        return sum(1 for r in self.results if r.passed)

    @property
    def failures(self) -> list[CheckResult]:
        """The checks that failed."""
        return [r for r in self.results if not r.passed]

    def add(self, result: CheckResult) -> None:
        """Append a result."""
        self.results.append(result)

    def as_text(self) -> str:
        """Full report, one line per check plus a verdict line."""
        lines = [r.as_line() for r in self.results]
        lines.append(
            f"verdict: {self.n_passed}/{len(self.results)} checks passed — "
            f"{'GO' if self.passed else 'NO-GO'}"
        )
        return "\n".join(lines)

    def raise_if_failed(self) -> None:
        """Raise :class:`~hilforge.errors.PreflightFailure` on any failure."""
        if not self.passed:
            names = ", ".join(r.name for r in self.failures)
            raise PreflightFailure(f"pre-run checks failed: {names}")


def _expected_specs(torque_limit_nm: float) -> dict[str, ChannelSpec]:
    return {
        "ahrs": ChannelSpec(name="ahrs", length=2, units="rad, rad/s", lower=-1.0e4, upper=1.0e4),
        "torque": ChannelSpec(
            name="torque", length=1, units="N*m", lower=-torque_limit_nm, upper=torque_limit_nm
        ),
    }


def preflight(
    backend: Any,
    *,
    period_s: float,
    torque_limit_nm: float = 1.5,
    n_clock_reads: int = 200,
    max_resolution_fraction: float = 0.01,
) -> CheckReport:
    """Run the pre-run checks against ``backend`` and return the report.

    Parameters
    ----------
    backend:
        Any backend satisfying :class:`hilforge.hal.Backend`. It is opened and
        left in the state it was found in.
    period_s:
        Intended loop period [s], > 0; used by the clock-resolution check.
    torque_limit_nm:
        Actuator range [N*m] the rig is expected to advertise.
    n_clock_reads:
        Number of timebase reads for the monotonicity check, >= 2.
    max_resolution_fraction:
        The timebase resolution must be at most this fraction of the period;
        0.01 by default, i.e. a 10 ms loop needs a clock finer than 100 us.

    Returns
    -------
    CheckReport
        Nine checks. Nothing is raised for a failed check — the caller
        decides, via :meth:`CheckReport.raise_if_failed`. The backend is left
        exactly as it was found: same open/closed state, same internal state,
        same counters.
    """
    if not (period_s > 0.0):
        raise ValueError(f"period_s must be > 0, got {period_s!r}")
    if n_clock_reads < 2:
        raise ValueError(f"n_clock_reads must be >= 2, got {n_clock_reads!r}")
    if not (0.0 < max_resolution_fraction <= 1.0):
        raise ValueError(
            f"max_resolution_fraction must be in (0, 1], got {max_resolution_fraction!r}"
        )
    report = CheckReport()
    was_open = bool(getattr(backend, "is_open", False))
    entry_snapshot: dict[str, object] | None = None

    # 1 — device present
    try:
        if not was_open:
            backend.open()
        try:
            entry_snapshot = snapshot_backend(backend)
        except RecoveryError:
            entry_snapshot = None
        report.add(
            CheckResult(
                name="device_present",
                passed=True,
                measured=f"open={backend.is_open}",
                expected="open=True",
                detail=f"driver={backend.info.driver or 'n/a'}",
            )
        )
    except DeviceAbsentError as exc:
        report.add(
            CheckResult(
                name="device_present",
                passed=False,
                measured="DeviceAbsentError",
                expected="open=True",
                detail=str(exc),
            )
        )
        return report

    # 2 — channels advertised
    expected = _expected_specs(torque_limit_nm)
    have = {**backend.sensors, **backend.actuators}
    missing = [n for n in expected if n not in have]
    report.add(
        CheckResult(
            name="channels_present",
            passed=not missing,
            measured=f"{sorted(have)}",
            expected=f"{sorted(expected)}",
            detail="missing: " + ", ".join(missing) if missing else "",
        )
    )

    # 3 — channel specs match
    mismatches = []
    for name, spec in expected.items():
        chan = have.get(name)
        if chan is None:
            continue
        got = chan.spec
        if (got.length, got.units) != (spec.length, spec.units):
            mismatches.append(f"{name}: {got.length}x{got.units} != {spec.length}x{spec.units}")
    report.add(
        CheckResult(
            name="channel_specs_match",
            passed=not mismatches,
            measured="mismatches=" + (str(len(mismatches))),
            expected="mismatches=0",
            detail="; ".join(mismatches),
        )
    )

    # 4 — timebase monotonic
    guard = MonotonicGuard()
    regression = ""
    try:
        for _ in range(n_clock_reads):
            guard.check(backend.timebase.now())
    except Exception as exc:  # noqa: BLE001 - the check reports any failure
        regression = str(exc)
    report.add(
        CheckResult(
            name="timebase_monotonic",
            passed=not regression,
            measured=f"reads={n_clock_reads}, regressions={1 if regression else 0}",
            expected="regressions=0",
            detail=regression,
        )
    )

    # 5 — clock resolution fine enough for the period
    res = float(backend.timebase.resolution_s)
    limit = max_resolution_fraction * period_s
    report.add(
        CheckResult(
            name="clock_resolution",
            passed=res <= limit,
            measured=f"{res:.3e} s",
            expected=f"<= {limit:.3e} s",
            detail=f"{max_resolution_fraction:.3g} x period {period_s:.3e} s",
        )
    )

    # 6 — dry-run rehearsal issues no write
    act = backend.actuators.get("torque")
    if act is None:
        report.add(
            CheckResult(
                name="dry_run_no_write",
                passed=False,
                measured="no torque channel",
                expected="write_count unchanged",
            )
        )
    else:
        before = act.write_count
        rehearsal = DryRunActuator(act)
        rehearsal.write(np.zeros(act.spec.length))
        after = act.write_count
        report.add(
            CheckResult(
                name="dry_run_no_write",
                passed=after == before,
                measured=f"write_count {before} -> {after}",
                expected=f"write_count stays {before}",
                detail=f"rehearsed={rehearsal.rehearsed}",
            )
        )

    # 7 — snapshot and restore round trip
    snap_ok = True
    snap_detail = ""
    try:
        snap = snapshot_backend(backend)
        restore_backend(backend, snap)
        verify_restored(backend, snap)
    except Exception as exc:  # noqa: BLE001 - the check reports any failure
        snap_ok = False
        snap_detail = str(exc)
    report.add(
        CheckResult(
            name="state_round_trip",
            passed=snap_ok,
            measured="restored" if snap_ok else "failed",
            expected="restored",
            detail=snap_detail,
        )
    )

    # 8 — a single sensor read succeeds
    sensor = backend.sensors.get("ahrs")
    if sensor is None:
        report.add(
            CheckResult(
                name="sensor_read",
                passed=False,
                measured="no ahrs channel",
                expected="one finite sample",
            )
        )
    else:
        try:
            sample = sensor.read()
            ok = bool(np.all(np.isfinite(sample)))
            report.add(
                CheckResult(
                    name="sensor_read",
                    passed=ok,
                    measured=np.array2string(sample, precision=6),
                    expected="finite sample of length 2",
                )
            )
        except Exception as exc:  # noqa: BLE001 - the check reports any failure
            report.add(
                CheckResult(
                    name="sensor_read",
                    passed=False,
                    measured=type(exc).__name__,
                    expected="finite sample of length 2",
                    detail=str(exc),
                )
            )

    # 9 — the checks themselves left the rig untouched
    if entry_snapshot is None:
        report.add(
            CheckResult(
                name="state_unchanged",
                passed=False,
                measured="no snapshot available",
                expected="0 fields differing",
                detail="backend does not implement snapshot()",
            )
        )
    else:
        restore_backend(backend, entry_snapshot)
        diffs = verify_restored(backend, entry_snapshot)
        report.add(
            CheckResult(
                name="state_unchanged",
                passed=not diffs,
                measured=f"{len(diffs)} of {len(_flatten(entry_snapshot))} differing",
                expected="0 fields differing",
                detail="; ".join(sorted(diffs)) if diffs else "restored on exit",
            )
        )

    if not was_open:
        backend.close()
    return report


def snapshot_backend(backend: Any) -> dict[str, object]:
    """Capture the backend's restorable state.

    Requires the backend to implement ``snapshot()``; a backend that cannot be
    snapshotted cannot be recovered, and :func:`preflight` reports that as a
    failed check rather than discovering it after an abort.
    """
    snap = getattr(backend, "snapshot", None)
    if snap is None:
        raise RecoveryError(
            f"backend {type(backend).__name__} has no snapshot(); it cannot be recovered"
        )
    return dict(snap())


def restore_backend(backend: Any, snapshot: dict[str, object]) -> None:
    """Restore a snapshot taken by :func:`snapshot_backend`."""
    restore = getattr(backend, "restore", None)
    if restore is None:
        raise RecoveryError(
            f"backend {type(backend).__name__} has no restore(); it cannot be recovered"
        )
    restore(snapshot)


def _flatten(snapshot: dict[str, object], prefix: str = "") -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in snapshot.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(_flatten(value, prefix=f"{path}."))
        else:
            out[path] = value
    return out


def verify_restored(backend: Any, snapshot: dict[str, object]) -> dict[str, tuple[Any, Any]]:
    """Check the backend's current state equals ``snapshot``, field by field.

    Returns
    -------
    dict
        Empty if everything matches. Otherwise maps each differing flattened
        key to ``(expected, actual)``.

    Raises
    ------
    RecoveryError
        If the backend cannot be snapshotted at all.
    """
    now = snapshot_backend(backend)
    want = _flatten(snapshot)
    have = _flatten(now)
    diffs: dict[str, tuple[Any, Any]] = {}
    for key, value in want.items():
        other = have.get(key, "<missing>")
        if isinstance(value, (bytes, bytearray)) or isinstance(other, (bytes, bytearray)):
            same = bytes(value) == bytes(other)  # type: ignore[arg-type]
        elif isinstance(value, (list, tuple, np.ndarray)) or isinstance(
            other, (list, tuple, np.ndarray)
        ):
            same = np.array_equal(np.asarray(value), np.asarray(other))
        else:
            same = value == other
        if not same:
            diffs[key] = (value, other)
    return diffs


class RunGuard:
    """Snapshot before a run; restore and verify if the run raises.

    Usage::

        guard = RunGuard(backend)
        with guard:
            record = HilLoop(backend, cfg).run()
        print(guard.recovery_report())

    The guard does **not** swallow the exception: a failed run must still
    fail. It records whether recovery succeeded, so a test can assert both
    that the run aborted and that the rig came back.

    Parameters
    ----------
    backend:
        Backend to guard; must implement ``snapshot()`` and ``restore()``.
    recover_on_success:
        If ``True``, restore even when the run completes. Default ``False``:
        a successful run's end state is usually what you want to keep.
    """

    def __init__(self, backend: Any, *, recover_on_success: bool = False) -> None:
        self.backend = backend
        self.recover_on_success = bool(recover_on_success)
        self.snapshot: dict[str, object] | None = None
        self.recovered = False
        self.diffs: dict[str, tuple[Any, Any]] = {}
        self.error: BaseException | None = None

    def __enter__(self) -> Self:
        self.snapshot = snapshot_backend(self.backend)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        assert self.snapshot is not None
        self.error = exc
        if exc is not None or self.recover_on_success:
            restore_backend(self.backend, self.snapshot)
            self.diffs = verify_restored(self.backend, self.snapshot)
            self.recovered = not self.diffs
        return False

    def recovery_report(self) -> str:
        """Text report of what recovery did and whether it verified."""
        if self.snapshot is None:
            return "guard was never entered"
        if self.error is None and not self.recover_on_success:
            return "run completed; no recovery attempted (recover_on_success=False)"
        lines = [
            f"abort cause        : {type(self.error).__name__ if self.error else 'none'}",
            f"fields compared    : {len(_flatten(self.snapshot))}",
            f"fields differing   : {len(self.diffs)}",
            f"recovery verified  : {self.recovered}",
        ]
        for key, (want, have) in sorted(self.diffs.items()):
            lines.append(f"  {key}: expected {want!r}, found {have!r}")
        return "\n".join(lines)
