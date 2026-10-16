"""Known answers: the planted fixture and the worked example.

The planted fixture is built to carry exactly one instance of each of the
twelve checks, so the expected mapping of code to subject is written out here
by hand and compared against what the package produces. The
tolerance-exceeds-deadline defect the batch specification asks for by name is
checked explicitly, with its arithmetic printed.
"""

from __future__ import annotations

from _support import SPECS, Emitter
from watchdogspec import CHECKS, Margin, load_spec, run_checks

EXPECTED_PLANTED: dict[str, str] = {
    "WS001": "T-WS001",
    "WS002": "T-WS002",
    "WS003": "T-WS003",
    "WS004": "T-WS004",
    "WS005": "T-WS005",
    "WS006": "T-WS006",
    "WS007": "T-WS007",
    "WS008": "T-WS008",
    "WS009": "L-NORECOVER",
    "WS010": "L-DISORDER",
    "WS011": "L-COSTDROP",
    "WS012": "T-WS012",
}

EXPECTED_WORKED: set[tuple[str, str]] = {("WS003", "TASK-FDIR"), ("WS005", "TASK-TLM")}


def main() -> int:  # noqa: C901 - a linear sequence of independent checks
    """Run both known-answer cases. Returns a process exit code."""
    emit = Emitter(__file__)
    emit("watchdogspec validation: known answers")
    emit()

    emit("1. planted fixture, examples/specs/planted_defects.yaml")
    planted = run_checks(load_spec(SPECS / "planted_defects.yaml"))
    actual = {finding.code: finding.subject for finding in planted.findings}
    emit(f"  findings              : {len(planted.findings)}")
    emit(f"  codes that fired      : {len(actual)} of {len(CHECKS)}")
    emit(f"  severity split        : {planted.count_by_severity()}")
    emit(f"  exit code             : {planted.exit_code}")
    emit()
    emit(f"  {'code':<7} {'expected subject':<14} {'actual subject':<14} match")
    for code, expected in EXPECTED_PLANTED.items():
        got = actual.get(code, "(not raised)")
        emit(f"  {code:<7} {expected:<14} {got:<14} {'yes' if got == expected else 'NO'}")
    emit()
    emit.check(
        "every one of the twelve checks fired exactly once",
        planted.count_by_code() == dict.fromkeys(EXPECTED_PLANTED, 1),
        f"{len(planted.findings)} findings",
    )
    emit.check("code to subject mapping is as hand-written", actual == EXPECTED_PLANTED)
    emit.check(
        "severity split is 7 error and 5 warning",
        planted.count_by_severity() == {"error": 7, "warning": 5},
    )
    emit.check("exit code is 1", planted.exit_code == 1)
    emit.check(
        "the clean task carries no finding",
        all(finding.subject != "T-CLEAN" for finding in planted.findings),
    )

    emit()
    emit("2. the planted tolerance-exceeds-deadline defect, in detail")
    timing = planted.timings["T-WS002"]
    emit("  task                  : T-WS002")
    emit("  heartbeat period P    : 50 ms")
    emit("  jitter bound J        : 5 ms")
    emit(f"  beat window W = P + J : {timing.beat_window_ms:.6f} ms")
    emit("  missed beats k        : 4")
    emit(f"  tolerated silence k*W : {timing.tolerated_silence_ms:.6f} ms")
    emit(f"  liveness deadline D   : {timing.deadline_ms:.6f} ms")
    emit(f"  slack D - k*W         : {timing.tolerance_slack_ms:.6f} ms")
    emit(f"  margin                : {timing.tolerance_margin.value}")
    finding = next(f for f in planted.findings if f.code == "WS002")
    emit(f"  finding               : {finding.code} {finding.severity.value} {finding.subject}")
    emit.check(
        "tolerated silence is 220 ms", abs(timing.tolerated_silence_ms - 220.0) < 1e-9
    )
    emit.check("deadline is 200 ms", abs(timing.deadline_ms - 200.0) < 1e-9)
    emit.check("slack is -20 ms", abs(timing.tolerance_slack_ms + 20.0) < 1e-9)
    emit.check("margin is missed", timing.tolerance_margin is Margin.MISSED)
    emit.check(
        "WS002 was raised against T-WS002",
        finding.subject == "T-WS002"
        and "tolerance exceeds the deadline it is declared to protect" in finding.message,
    )

    emit()
    emit("3. the deliberate floating-point boundary case, T-WS004")
    boundary = planted.timings["T-WS004"]
    emit(f"  worst-case completion : {boundary.full_ladder_latency_ms!r} ms")
    emit(f"  declared deadline     : {boundary.deadline_ms!r} ms")
    emit(f"  slack                 : {boundary.ladder_slack_ms!r} ms")
    emit(f"  comparison tolerance  : {planted.tol_ms:.3e} ms")
    emit(f"  margin                : {boundary.ladder_margin.value}")
    emit.check(
        "the boundary case is reported as marginal, not met or missed",
        boundary.ladder_margin is Margin.MARGINAL,
    )
    emit.check(
        "a marginal final rung is a warning and does not set the exit code",
        next(f for f in planted.findings if f.code == "WS004").severity.value == "warning",
    )

    emit()
    emit("4. the clean specification, examples/specs/minimal_consistent.yaml")
    minimal = run_checks(load_spec(SPECS / "minimal_consistent.yaml"))
    emit(f"  findings              : {len(minimal.findings)}")
    emit(f"  exit code             : {minimal.exit_code}")
    for task_id in sorted(minimal.timings):
        timing = minimal.timings[task_id]
        emit(
            f"    {task_id:<10} WCDL {timing.detection_latency_ms:9.4f} ms   "
            f"ladder {timing.full_ladder_latency_ms:9.4f} ms   "
            f"D {timing.deadline_ms:9.4f} ms   slack {timing.ladder_slack_ms:9.4f} ms"
        )
    emit.check("the clean specification raises no finding", minimal.findings == ())
    emit.check("the clean specification exits 0", minimal.exit_code == 0)

    emit()
    emit("5. the worked example, examples/specs/smallsat_fc.yaml")
    worked = run_checks(load_spec(SPECS / "smallsat_fc.yaml"))
    got = {(finding.code, finding.subject) for finding in worked.findings}
    for finding in worked.findings:
        emit(f"  {finding.code}  {finding.severity.value:<7} {finding.message}")
    emit.check("the worked example ships exactly its two known findings", got == EXPECTED_WORKED)
    emit.check("the worked example exits 1", worked.exit_code == 1)

    emit()
    emit("6. hand-calculated escalation arithmetic, worked example")
    emit(
        f"  {'task':<10} {'W/ms':>9} {'WCDL/ms':>10} {'ladder/ms':>11} {'D/ms':>9} "
        f"{'slack/ms':>11} {'rungs':>7}"
    )
    hand = {
        # task: (W, WCDL, ladder completion, D, deepest rung)
        "TASK-GNC": (23.0, 75.5, 687.5, 800.0, 4),
        "TASK-IMU": (6.0, 18.5, 134.5, 300.0, 3),
        "TASK-TLM": (1250.0, 1256.5, 1372.5, 4000.0, 3),
        "TASK-FDIR": (115.0, 466.5, 1078.5, 500.0, 1),
    }
    for task_id, (window, wcdl, ladder, _deadline, rungs) in hand.items():
        timing = worked.timings[task_id]
        emit(
            f"  {task_id:<10} {timing.beat_window_ms:>9.4f} "
            f"{timing.detection_latency_ms:>10.4f} {timing.full_ladder_latency_ms:>11.4f} "
            f"{timing.deadline_ms:>9.4f} {timing.ladder_slack_ms:>11.4f} "
            f"{timing.deepest_rung_reachable:>5}/{len(timing.rungs)}"
        )
        emit.check(
            f"{task_id} beat window is {window} ms",
            abs(timing.beat_window_ms - window) < 1e-9,
        )
        emit.check(
            f"{task_id} worst-case detection latency is {wcdl} ms",
            abs(timing.detection_latency_ms - wcdl) < 1e-9,
        )
        emit.check(
            f"{task_id} ladder completion is {ladder} ms",
            abs(timing.full_ladder_latency_ms - ladder) < 1e-9,
        )
        emit.check(
            f"{task_id} deepest reachable rung is {rungs}",
            timing.deepest_rung_reachable == rungs,
        )

    return emit.finish()


if __name__ == "__main__":
    raise SystemExit(main())
