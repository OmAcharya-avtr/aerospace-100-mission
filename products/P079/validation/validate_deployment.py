"""Check 9 -- deployment pre-flight and recovery, as executed checks.

Requirement (BATCH_08_SPEC, P079, L4 scaffolding item 4): "Documented
deployment and recovery procedure for the Jetson Orin Nano, including what to
do when the exchange is lost mid-estimate."

The narrative procedure is ``docs/DEPLOYMENT.md``. This script runs the checks
that document tells the operator to run, and measures the two recovery paths:

* **The run raised.** :class:`~timetransfer.deploy.RunGuard` restores the
  backend and verifies it field by field. Five different injected faults.
* **The exchange was lost mid-estimate.** The acquisition stops on a loss run,
  returns what arrived, and
  :func:`~timetransfer.deploy.recover_truncated_estimate` produces an estimate
  labelled as truncated with its uncertainty computed from the data that
  actually arrived.

Every check is also shown to be able to fail: a wrong hardware declaration, an
absent device, a loss budget that cannot be met and a resolution ceiling that
cannot be met each produce a named failure. A check list where nothing can fail
is a tick-list.

Run: ``python validation/validate_deployment.py``
"""

from __future__ import annotations

import sys

from _bootstrap import Tee, bootstrap, environment_lines, scaled

ROOT = bootstrap()

from timetransfer.acquire import acquire  # noqa: E402
from timetransfer.backends import (  # noqa: E402
    AbsentDriver,
    DeviceBackend,
    SimulatedBackend,
    SimulationSpec,
)
from timetransfer.clockmodel import ClockSpec  # noqa: E402
from timetransfer.deploy import RunGuard, preflight, recover_truncated_estimate  # noqa: E402
from timetransfer.errors import InsufficientDataError, PreflightFailure  # noqa: E402
from timetransfer.linkmodel import LinkSpec  # noqa: E402
from timetransfer.uncertainty import AsymmetrySource  # noqa: E402

SEED = 20261024
PLANNED = scaled(600, minimum=200)
OUTAGE_START_S = 120.0


def _spec(
    *,
    seed: int = SEED,
    loss_probability: float = 0.02,
    resolution_ns: int = 1,
    outage: tuple[tuple[float, float], ...] = (),
) -> SimulationSpec:
    return SimulationSpec(
        clock=ClockSpec(
            offset0_s=1.250e-3,
            skew_s_per_s=1.800e-6,
            drift_per_s=2.000e-12,
            q1_s2_per_s=1.0e-20,
            q2_per_s=1.0e-26,
        ),
        link=LinkSpec(
            floor_ab_s=5.4e-3,
            floor_ba_s=4.6e-3,
            jitter_scale_ab_s=2.4e-4,
            jitter_scale_ba_s=1.8e-4,
            loss_probability=loss_probability,
            outage_windows=outage,
        ),
        period_s=1.0,
        timestamp_noise_s=2.0e-8,
        resolution_ns=resolution_ns,
        seed=seed,
    )


def main() -> int:
    out = Tee("validate_deployment_output.txt")
    out("timetransfer validation 9 -- deployment pre-flight and recovery")
    out("=" * 78)
    for line in environment_lines():
        out(line)
    failures = 0

    out.rule("9a. pre-flight against the simulated backend")
    report = preflight(SimulatedBackend(_spec()), expect_hardware=False)
    out(report.as_text())
    out("")
    out(f"all checks passed : {report.passed}")
    if not report.passed:
        failures += 1
        out("FAIL: pre-flight did not pass on a healthy simulated backend")

    out.rule("9b. every check can fail -- the negative controls")
    out(f"{'control':>34} {'failed checks':>44}")
    controls = [
        (
            "operator declares hardware wrongly",
            preflight(SimulatedBackend(_spec()), expect_hardware=True),
        ),
        (
            "no device answers",
            preflight(DeviceBackend(AbsentDriver()), expect_hardware=False),
        ),
        (
            "loss budget cannot be met",
            preflight(
                SimulatedBackend(_spec(loss_probability=0.4, seed=5)),
                expect_hardware=False,
                probe_attempts=60,
                max_loss_fraction=0.05,
            ),
        ),
        (
            "resolution ceiling cannot be met",
            preflight(
                SimulatedBackend(_spec(resolution_ns=1_000_000)),
                expect_hardware=False,
                max_resolution_ns=1000,
            ),
        ),
    ]
    for label, control in controls:
        names = ", ".join(check.name for check in control.failures) or "(none)"
        out(f"{label:>34} {names[:44]:>44}")
        if control.passed:
            failures += 1
            out(f"FAIL: the control {label!r} did not fail any check")
    out("")
    out("raise_if_failed raises PreflightFailure and carries the report:")
    try:
        controls[0][1].raise_if_failed()
    except PreflightFailure as exc:
        out(f"  {type(exc).__name__}: {exc}")
        out(f"  report attached : {exc.report is controls[0][1]}")

    out.rule("9c. pre-flight leaves the backend exactly as it found it")
    spec = _spec()
    backend = SimulatedBackend(spec)
    before = backend.snapshot()
    preflight(backend, expect_hardware=False)
    backend.open()
    after_preflight = acquire(backend, scaled(200, minimum=20), min_exchanges=8).exchanges
    backend.close()
    fresh = SimulatedBackend(spec)
    fresh.open()
    untouched = acquire(fresh, scaled(200, minimum=20), min_exchanges=8).exchanges
    fresh.close()
    identical = after_preflight.timestamp_bytes() == untouched.timestamp_bytes()
    out(f"engine attempts before pre-flight : {before['engine']['attempts']}")
    out(f"digest after pre-flight           : {after_preflight.digest()}")
    out(f"digest on an untouched backend    : {untouched.digest()}")
    out(f"identical                         : {identical}")
    out("This is a check and not an assumption because it was wrong once: the probe")
    out("acquisition advances the clock and link generators, so a pre-flighted backend")
    out("would otherwise no longer agree with an untouched one and the parity check")
    out("would fail for a reason unrelated to parity.")
    if not identical:
        failures += 1
        out("FAIL")

    out.rule("9d. recovery after an injected abort")
    out(f"{'injected fault':>22} {'restored':>10} {'fields compared':>16} "
        f"{'fields differing':>18}")
    restored_count = 0
    faults: list[type[BaseException]] = [
        RuntimeError,
        ValueError,
        KeyError,
        ZeroDivisionError,
        TimeoutError,
    ]
    for index, fault in enumerate(faults):
        backend = SimulatedBackend(_spec(seed=SEED + index))
        backend.open()
        snapshot_before = backend.snapshot()
        guard = None
        try:
            with RunGuard(backend) as guard:
                acquire(backend, scaled(120, minimum=20), min_exchanges=8)
                raise fault("injected mid-acquisition fault")
        except Exception:  # noqa: BLE001 - the injected fault is expected
            pass
        assert guard is not None
        clean = guard.restored and backend.snapshot() == snapshot_before
        restored_count += int(clean)
        out(
            f"{fault.__name__:>22} {str(guard.restored):>10} "
            f"{guard.n_snapshot_fields:>16d} {len(guard.differing_fields):>18d}"
        )
    out("")
    out(f"restored cleanly : {restored_count}/{len(faults)}")
    if restored_count != len(faults):
        failures += 1
        out("FAIL: a backend was not restored after an abort")
    else:
        out("PASS")

    out.rule("9e. the exchange is lost mid-estimate")
    out(f"planned window        : {PLANNED} exchanges at 1 Hz")
    out(f"outage starts at      : {OUTAGE_START_S:.1f} s of local reference time")
    out("max consecutive losses: 10 (the acquisition stops rather than hammering a")
    out("                        dead link for the rest of the window)")
    out("")
    spec = _spec(loss_probability=0.0, outage=((OUTAGE_START_S, 10_000.0),))
    backend = SimulatedBackend(spec)
    backend.open()
    report = acquire(backend, PLANNED, min_exchanges=8, max_consecutive_losses=10)
    backend.close()
    out(report.as_text())
    out("")
    truncated = recover_truncated_estimate(
        report.exchanges,
        PLANNED,
        order=1,
        asymmetry_source=AsymmetrySource.DECLARED_BOUND,
        asymmetry_bound_s=1.6e-3,
    )
    out(truncated.as_text())
    out("")
    full_backend = SimulatedBackend(_spec(loss_probability=0.0))
    full_backend.open()
    full = acquire(full_backend, PLANNED, min_exchanges=8).exchanges
    full_backend.close()
    complete = recover_truncated_estimate(
        full,
        PLANNED,
        order=1,
        asymmetry_source=AsymmetrySource.DECLARED_BOUND,
        asymmetry_bound_s=1.6e-3,
    )
    out(f"{'window':>14} {'n used':>8} {'u statistical [s]':>20} {'u combined [s]':>18}")
    out(f"{'truncated':>14} {truncated.n_used:>8d} "
        f"{truncated.uncertainty.u_statistical_s:>20.6e} "
        f"{truncated.uncertainty.u_combined_s:>18.6e}")
    out(f"{'complete':>14} {complete.n_used:>8d} "
        f"{complete.uncertainty.u_statistical_s:>20.6e} "
        f"{complete.uncertainty.u_combined_s:>18.6e}")
    out("")
    ratio = truncated.uncertainty.u_statistical_s / complete.uncertainty.u_statistical_s
    out(f"measured statistical-term ratio : {ratio:.4f}")
    out(f"independence expectation        : {truncated.uncertainty_ratio_to_plan:.4f} "
        f"(sqrt of the planned-over-used count)")
    out("The measured ratio is not required to equal the expectation: the expectation")
    out("assumes independent samples and equal spans, and a truncated window is")
    out("shorter in time as well as smaller in count, which affects the fitted offset")
    out("differently. The expectation is printed so the reader can see the gap.")
    if not truncated.truncated or truncated.n_used >= PLANNED:
        failures += 1
        out("FAIL: the truncation was not labelled")
    if truncated.uncertainty.u_statistical_s <= complete.uncertainty.u_statistical_s:
        failures += 1
        out("FAIL: the truncated window did not report a larger statistical term")

    out.rule("9f. a window that loses too much is refused rather than estimated")
    spec = _spec(loss_probability=0.0, outage=((0.0, 10_000.0),))
    backend = SimulatedBackend(spec)
    backend.open()
    refused = False
    try:
        acquire(backend, PLANNED, min_exchanges=8, max_consecutive_losses=10)
    except InsufficientDataError as exc:
        refused = True
        out(f"{type(exc).__name__}: {exc}")
        out(f"  n_usable   : {exc.n_usable}")
        out(f"  n_required : {exc.n_required}")
    backend.close()
    out(f"verdict : {'PASS' if refused else 'FAIL'}")
    if not refused:
        failures += 1

    out.rule("9g. the operator's declaration is the only guard against a false claim")
    out("preflight(expect_hardware=...) makes the operator state what they believe,")
    out("and a mismatch against the backend's own is_hardware flag is a failed check.")
    out("Every backend in this repository reports is_hardware=False, so a run written")
    out("up as a board measurement would have had to pass expect_hardware=True and")
    out("ignore a failing check to do it.")
    out("")
    out(f"{'backend':>28} {'is_hardware':>13} {'declaration check with True':>30}")
    for label, backend_obj in (
        ("simulated", SimulatedBackend(_spec())),
        ("device over absent driver", DeviceBackend(AbsentDriver())),
    ):
        control = preflight(backend_obj, expect_hardware=True)
        declaration = next(
            check for check in control.checks if check.name == "hardware_declaration"
        )
        out(f"{label:>28} {str(backend_obj.info.is_hardware):>13} "
            f"{('PASS' if declaration.passed else 'FAIL'):>30}")

    out.rule("summary")
    out(f"planned window {PLANNED}, truncated at {truncated.n_used} exchanges "
        f"({truncated.completed_fraction:.4f} of plan)")
    out(f"hard failures : {failures}")
    out(f"saved to      : {out.save()}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
