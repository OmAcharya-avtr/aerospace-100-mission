"""Check 6 -- dry-run mode over a recorded exchange, with nothing driven.

Requirement (BATCH_08_SPEC, P079, L4 scaffolding item 3): "Dry-run mode that
exercises the full acquisition and estimation path against a recorded exchange
without touching a device."

Proved by counting
------------------
"Nothing was driven" is a claim about a count, so it is implemented and
reported as one. The recorded driver is wrapped in a
:class:`~timetransfer.backends.stubs.GuardedDriver`, which counts every
operation. The run reports operations on the recording, operations on any
driver declaring itself hardware (zero), and refusals.

The negative control is the part that makes the counts evidence: the identical
code is run with the guard forbidding every exchange, and it must come back
with zero completed exchanges and a non-zero refusal count. A counter that
cannot be made to fire is not evidence.

Run: ``python validation/validate_dryrun.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from _bootstrap import Tee, bootstrap, environment_lines, relative

ROOT = bootstrap()

from timetransfer.acquire import acquire  # noqa: E402
from timetransfer.backends import SimulatedBackend, SimulationSpec  # noqa: E402
from timetransfer.clockmodel import ClockSpec  # noqa: E402
from timetransfer.dryrun import dry_run, dry_run_negative_control, record_exchanges  # noqa: E402
from timetransfer.errors import ConfigurationError  # noqa: E402
from timetransfer.exchange import ExchangeSet  # noqa: E402
from timetransfer.linkmodel import LinkSpec  # noqa: E402
from timetransfer.twoway import two_way_estimate  # noqa: E402
from timetransfer.uncertainty import AsymmetrySource  # noqa: E402

FIXTURE = Path("fixtures") / "recorded_exchange.json"
SCRATCH = Path("fixtures") / "_scratch_recording.json"


def main() -> int:
    out = Tee("validate_dryrun_output.txt")
    out("timetransfer validation 6 -- dry-run mode over a recorded exchange")
    out("=" * 78)
    for line in environment_lines():
        out(line)
    failures = 0

    out.rule("6a. the committed recording")
    if not FIXTURE.exists():
        out(f"FAIL: {relative(FIXTURE)} is missing; run validation/regenerate_fixture.py")
        out(f"saved to : {out.save()}")
        return 1
    recorded = ExchangeSet.read_json(FIXTURE)
    out(f"path              : {relative(FIXTURE)}")
    out(f"exchanges         : {recorded.n} completed, {recorded.n_lost} lost")
    out(f"sequence span     : {int(recorded.seq.min())} to {int(recorded.seq.max())}")
    out(f"backend           : {recorded.backend_kind}/{recorded.backend_driver or 'n/a'}")
    out(f"is_hardware       : {recorded.is_hardware}")
    out(f"digest            : {recorded.digest()}")
    out(f"span              : {recorded.span_s():.6f} s")
    out(f"loss fraction     : {recorded.loss_fraction:.6f}")
    gaps = np.diff(recorded.seq)
    out(f"longest seq gap   : {int(np.max(gaps))} (an outage window is recorded in it)")

    out.rule("6b. the dry run")
    result = dry_run(recorded, order=2)
    out(result.as_text())
    out("")
    out("full estimation path exercised:")
    out(f"  two-way estimate : {result.estimate.n} per-exchange offsets")
    out(f"  ls fit order     : {result.fit.order}, dof {result.fit.dof}")
    out(f"  fitted skew      : {result.fit.skew_s_per_s:.9e} s/s")
    out(f"  u statistical    : {result.uncertainty.u_statistical_s:.6e} s")
    out(f"  u quantisation   : {result.uncertainty.u_quantisation_s:.6e} s")
    out(f"  u asymmetry      : {result.uncertainty.u_asymmetry_s:.6e} s "
        f"({result.uncertainty.asymmetry_source.value})")
    checks = {
        "hardware_operations == 0": result.hardware_operations == 0,
        "is_hardware is False": result.is_hardware is False,
        "driver is a stub": result.backend_driver == "guarded-recorded",
        "every recorded exchange replayed": result.report.exchanges.n == recorded.n,
        "uncertainty is finite and positive": bool(
            np.isfinite(result.uncertainty.u_combined_s) and result.uncertainty.u_combined_s > 0
        ),
    }
    out("")
    for name, passed in checks.items():
        out(f"  {('PASS' if passed else 'FAIL'):>4}  {name}")
        if not passed:
            failures += 1

    out.rule("6c. the replay reproduces the original estimate exactly")
    original = two_way_estimate(recorded)
    replayed = two_way_estimate(result.report.exchanges)
    max_abs = float(np.max(np.abs(original.offset_s - replayed.offset_s)))
    bytes_equal = recorded.timestamp_bytes() == result.report.exchanges.timestamp_bytes()
    out(f"timestamp bytes identical : {bytes_equal}")
    out(f"offset max |diff|         : {max_abs:.3e} s")
    if not bytes_equal or max_abs != 0.0:
        failures += 1
        out("FAIL: the replay did not reproduce the recording")
    else:
        out("PASS")

    out.rule("6d. negative control -- device operations forbidden")
    control = dry_run_negative_control(recorded)
    out(f"completed exchanges : {control.completed} (must be 0)")
    out(f"refused operations  : {control.refused} (must be > 0)")
    out(f"acquisition refused : {control.raised_insufficient} (must be True)")
    out(f"control passed      : {control.passed}")
    out("")
    out("The refusal count is the number of recorded sequence numbers plus one for the")
    out("acquisition path's own first attempt, so the counter is shown to fire on every")
    out("route into the driver and not just on one.")
    if not control.passed:
        failures += 1
        out("FAIL: the negative control did not fire")
    else:
        out("PASS")

    out.rule("6e. a recording marked as hardware is refused")
    marked = ExchangeSet.read_json(FIXTURE)
    marked.is_hardware = True
    refused = False
    try:
        dry_run(marked)
    except ConfigurationError as exc:
        refused = True
        out(f"refused with : {type(exc).__name__}: {exc}")
    out(f"verdict      : {'PASS' if refused else 'FAIL'}")
    if not refused:
        failures += 1

    out.rule("6f. a freshly recorded exchange round-trips through the file")
    spec = SimulationSpec(
        clock=ClockSpec(offset0_s=-2.0e-3, skew_s_per_s=-9.0e-7, q1_s2_per_s=1.0e-20),
        link=LinkSpec(floor_ab_s=1.2e-2, floor_ba_s=1.1e-2, loss_probability=0.06),
        period_s=0.5,
        timestamp_noise_s=2.0e-8,
        seed=4242,
    )
    backend = SimulatedBackend(spec)
    backend.open()
    path, report = record_exchanges(backend, 120, SCRATCH, min_exchanges=8)
    backend.close()
    fresh = dry_run(
        path,
        order=1,
        asymmetry_source=AsymmetrySource.DECLARED_BOUND,
        asymmetry_bound_s=4.0e-3,
    )
    out(f"recorded            : {relative(path)} ({report.exchanges.n} exchanges)")
    out(f"replayed            : {fresh.report.exchanges.n} exchanges")
    out(f"bytes identical     : "
        f"{report.exchanges.timestamp_bytes() == fresh.report.exchanges.timestamp_bytes()}")
    out(f"declared asymmetry  : {spec.link.static_asymmetry_s:+.6e} s "
        f"-> closed-form bias {spec.link.static_bias_s:+.6e} s")
    out(f"offset              : {fresh.uncertainty.offset_s:.9e} s")
    out(f"u combined          : {fresh.uncertainty.u_combined_s:.6e} s "
        f"(dominant: {fresh.uncertainty.dominant_term})")
    out(f"hardware operations : {fresh.hardware_operations}")
    if fresh.hardware_operations != 0:
        failures += 1
        out("FAIL: a fresh dry run reached a hardware driver")
    SCRATCH.unlink(missing_ok=True)
    out(f"scratch file removed: {not SCRATCH.exists()}")

    out.rule("6g. a live acquisition for comparison, to show the paths agree")
    backend = SimulatedBackend(spec)
    backend.open()
    live = acquire(backend, 120, min_exchanges=8).exchanges
    backend.close()
    out(f"live timestamp bytes equal the recording : "
        f"{live.timestamp_bytes() == report.exchanges.timestamp_bytes()}")
    out("A dry run is a replay, so this is a statement about the recording being")
    out("faithful rather than about the estimator.")

    out.rule("summary")
    out(f"hard failures : {failures}")
    out(f"saved to      : {out.save()}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
