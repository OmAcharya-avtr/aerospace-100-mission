"""Check 10 -- the benchmark harness, with the hardware rows empty and marked.

Requirement (BATCH_08_SPEC, P079, L4 scaffolding item 5): "Benchmark harness
that produces the timing and resource table, ready to run, with the table's
hardware rows left **empty and explicitly marked unmeasured** rather than
filled from simulation."

Two tables, one of them deliberately blank
------------------------------------------
The host table is measured in this process and is declared unstable: it is
wall-clock timing on a shared two-core container and every value moves between
runs. The Level 4 hardware table has five rows, every value is ``None`` and
every status is ``"unmeasured"``, and the guards that keep it that way are
exercised here rather than described:

* :class:`~timetransfer.bench.BenchmarkRecord` refuses construction with
  ``is_hardware=True``;
* it refuses a hardware row carrying a value or a status other than
  ``"unmeasured"``;
* :meth:`~timetransfer.bench.BenchmarkRecord.promote_to_hardware` always
  raises.

Each refusal is attempted here and the exception is printed, so "the rows
cannot be filled from simulation" is a demonstrated property and not a
promise.

Run: ``python validation/validate_benchmark_table.py``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from _bootstrap import Tee, bootstrap, environment_lines, relative, scaled

ROOT = bootstrap()

from timetransfer.acquire import acquire  # noqa: E402
from timetransfer.allan import allan_table  # noqa: E402
from timetransfer.asym import window_features  # noqa: E402
from timetransfer.backends import SimulatedBackend, SimulationSpec  # noqa: E402
from timetransfer.bench import (  # noqa: E402
    HARDWARE_STAGES,
    BenchmarkRecord,
    HardwareRow,
    run_benchmark,
)
from timetransfer.clockmodel import ClockSpec  # noqa: E402
from timetransfer.dryrun import dry_run  # noqa: E402
from timetransfer.errors import HardwareClaimError  # noqa: E402
from timetransfer.exchange import ExchangeSet  # noqa: E402
from timetransfer.linkmodel import LinkSpec  # noqa: E402
from timetransfer.skew import KalmanConfig, fit_least_squares, kalman_skew  # noqa: E402
from timetransfer.twoway import two_way_estimate  # noqa: E402
from timetransfer.uncertainty import AsymmetrySource, offset_uncertainty  # noqa: E402

RECORD_JSON = Path("validation") / "benchmark_record.json"
FIXTURE = Path("fixtures") / "recorded_exchange.json"
SEED = 20261024
N = scaled(600, minimum=40)

SPEC = SimulationSpec(
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
        loss_probability=0.03,
    ),
    period_s=1.0,
    timestamp_noise_s=2.0e-8,
    resolution_ns=1,
    seed=SEED,
)


def main() -> int:
    out = Tee("validate_benchmark_table_output.txt")
    out("timetransfer validation 10 -- the benchmark harness and the empty L4 table")
    out("=" * 78)
    for line in environment_lines():
        out(line)
    failures = 0

    backend = SimulatedBackend(SPEC)
    backend.open()
    exchanges = acquire(backend, N, min_exchanges=8).exchanges
    backend.close()
    estimate = two_way_estimate(exchanges)
    fit = fit_least_squares(estimate.midpoint_s, estimate.offset_s, order=2)
    residual = fit.detrended(estimate.midpoint_s, estimate.offset_s)
    recorded = ExchangeSet.read_json(FIXTURE) if FIXTURE.exists() else exchanges

    def stage_acquire() -> object:
        local = SimulatedBackend(SPEC)
        local.open()
        result = acquire(local, N, min_exchanges=8)
        local.close()
        return result

    stages = {
        "acquire_600_exchanges": stage_acquire,
        "two_way_estimate": lambda: two_way_estimate(exchanges),
        "fit_least_squares_order2": lambda: fit_least_squares(
            estimate.midpoint_s, estimate.offset_s, order=2
        ),
        "kalman_skew_2state": lambda: kalman_skew(
            estimate.midpoint_s,
            estimate.offset_s,
            config=KalmanConfig(q1_s2_per_s=1.0e-20, q2_per_s=1.0e-26),
        ),
        "allan_table": lambda: allan_table(residual, 1.0, noise_kind="white_fm"),
        "offset_uncertainty": lambda: offset_uncertainty(
            fit,
            resolution_ns=1,
            asymmetry_source=AsymmetrySource.DECLARED_BOUND,
            asymmetry_bound_s=1.6e-3,
        ),
        "window_features": lambda: window_features(exchanges),
        "dry_run_replay": lambda: dry_run(recorded, order=1),
    }

    out.rule("10a. the measured host table")
    repeats = scaled(9, minimum=3)
    record = run_benchmark(
        stages,
        label="P079 timetransfer, simulated backend",
        environment_note=(
            "shared two-core container, no attempt made to quiesce the machine; "
            "five build agents were active in the same container"
        ),
        backend_kind="simulated",
        backend_driver="",
        n_repeats=repeats,
        clock_samples=4000,
    )
    out(record.as_text())

    out.rule("10b. the Level 4 hardware table is empty and marked unmeasured")
    out(f"{'figure':>26} {'value':>8} {'status':>12} {'units':>7}  description")
    for row in record.hardware:
        out(f"{row.stage:>26} {str(row.value):>8} {row.status:>12} {row.units:>7}  "
            f"{row.description}")
    out("")
    out(f"rows                   : {len(record.hardware)} (expected {len(HARDWARE_STAGES)})")
    out(f"every value is None    : {all(row.value is None for row in record.hardware)}")
    out(f"every status unmeasured: {all(row.status == 'unmeasured' for row in record.hardware)}")
    out(f"reason recorded        : {record.hardware[0].reason}")
    if not all(row.value is None and row.status == "unmeasured" for row in record.hardware):
        failures += 1
        out("FAIL: a hardware row is not empty")

    out.rule("10c. the guards, exercised rather than described")
    attempts = 0
    caught = 0

    attempts += 1
    try:
        record.promote_to_hardware()
    except HardwareClaimError as exc:
        caught += 1
        out("promote_to_hardware() on a simulated record:")
        out(f"  {type(exc).__name__}: {exc}")

    attempts += 1
    try:
        BenchmarkRecord(
            label="forged",
            environment_note="a record that claims to be hardware",
            backend_kind="device",
            backend_driver="ixgbe-ptp",
            is_hardware=True,
            stages=(),
            tracemalloc_peak_bytes=0.0,
            maxrss_bytes=0.0,
            maxrss_note="n",
            clock_reported_resolution_ns=1,
            clock_measured_resolution_ns=1,
            clock_call_overhead_ns=0.0,
            offset_quantisation_uncertainty_s=0.0,
        )
    except HardwareClaimError as exc:
        caught += 1
        out("")
        out("BenchmarkRecord(is_hardware=True):")
        out(f"  {type(exc).__name__}: {exc}")

    attempts += 1
    try:
        BenchmarkRecord(
            label="forged",
            environment_note="a record with a filled hardware row",
            backend_kind="simulated",
            backend_driver="",
            is_hardware=False,
            stages=(),
            tracemalloc_peak_bytes=0.0,
            maxrss_bytes=0.0,
            maxrss_note="n",
            clock_reported_resolution_ns=1,
            clock_measured_resolution_ns=1,
            clock_call_overhead_ns=0.0,
            offset_quantisation_uncertainty_s=0.0,
            hardware=(
                HardwareRow(
                    stage="estimator_latency_s",
                    description="filled from simulation",
                    units="s",
                    reason="none",
                    status="measured",
                ),
            ),
        )
    except HardwareClaimError as exc:
        caught += 1
        out("")
        out("BenchmarkRecord with a hardware row marked measured:")
        out(f"  {type(exc).__name__}: {exc}")

    out("")
    out(f"guards fired : {caught}/{attempts}")
    if caught != attempts:
        failures += 1
        out("FAIL: a guard did not fire")
    else:
        out("PASS")

    out.rule("10d. what is still missing for Level 4")
    out("The five figures below have to be measured ON the Jetson Orin Nano, with")
    out("this same harness and this same record format, and captured as raw .txt and")
    out(".json output. Nothing else is missing.")
    out("")
    for stage, description in HARDWARE_STAGES:
        out(f"  {stage:<26} {description}")
    out("")
    out("Blocked on owner action: the board powered on and reachable, its hostname or")
    out("IP, and passwordless SSH from the Mac. No cloud session can move it. Until")
    out("then this product is labelled Level 3, hardware-pending, and it is never")
    out("labelled Level 4 on simulated numbers.")

    out.rule("10e. the record on disk")
    path = record.write(RECORD_JSON)
    payload = json.loads(path.read_text())
    out(f"written to            : {relative(path)}")
    out(f"is_hardware in json   : {payload['is_hardware']}")
    out(f"hardware rows in json : {len(payload['hardware'])}")
    out(f"all values null       : {all(row['value'] is None for row in payload['hardware'])}")
    out(f"all statuses          : "
        f"{sorted({row['status'] for row in payload['hardware']})}")
    out(f"stages measured       : {len(payload['stages'])}")
    out(f"repeats per stage     : {repeats}")
    if payload["is_hardware"] or any(row["value"] is not None for row in payload["hardware"]):
        failures += 1
        out("FAIL: the written record carries a hardware claim")

    out.rule("summary")
    out(f"host stages measured : {len(record.stages)}")
    out(f"hardware rows empty  : {len(record.hardware)}")
    out(f"tracemalloc peak     : {record.tracemalloc_peak_bytes:,.0f} B")
    out(f"maxrss               : {record.maxrss_bytes:,.0f} B")
    out(f"measured clock res.  : {record.clock_measured_resolution_ns} ns")
    out(f"u_q(offset)          : {record.offset_quantisation_uncertainty_s:.6e} s")
    out(f"finite timings       : "
        f"{all(np.isfinite(stage.mean_s) for stage in record.stages)}")
    out(f"hard failures        : {failures}")
    out(f"saved to             : {out.save()}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
