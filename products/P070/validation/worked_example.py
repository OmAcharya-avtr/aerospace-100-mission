"""The worked example, end to end, exactly as the README quotes it.

This is the README's worked-example block with the emitter in place of
``print``, so the committed output file and the README cannot drift apart.
"""

from __future__ import annotations

from _support import SPECS, Emitter
from watchdogspec import (
    format_timing_table,
    load_spec,
    run_checks,
    simulate_detection,
    spec_false_trip_rates,
)


def main() -> int:
    """Run the worked example. Returns a process exit code."""
    emit = Emitter(__file__)

    spec = load_spec(SPECS / "smallsat_fc.yaml")
    report = run_checks(spec)
    rates = spec_false_trip_rates(spec)

    emit(f"system            : {spec.system}")
    emit(f"tasks             : {len(spec.tasks)}")
    emit(f"ladders           : {len(spec.ladders)}")
    emit(f"supervisor        : poll {spec.supervisor.poll_period_ms:g} ms, "
         f"jitter {spec.supervisor.poll_jitter_ms:g} ms, "
         f"overhead {spec.supervisor.declaration_overhead_ms:g} ms")
    emit(f"observation delay : {spec.supervisor.observation_delay_ms:g} ms")
    emit(f"consistent        : {report.is_consistent}   exit code: {report.exit_code}")
    emit()
    emit(format_timing_table(report).rstrip("\n"))
    emit()
    emit("false-trip rate per task")
    for task_id in sorted(rates):
        rate = rates[task_id]
        emit(
            f"  {task_id:<10} threshold {rate.threshold_ms:9.3f} ms   "
            f"p {rate.per_gap:12.6g}   per hour {rate.per_hour:12.6g}"
        )
    emit()
    emit("closed form against the seeded simulation, 8 seeds of 500 trials")
    for task_id in sorted(report.timings):
        task = spec.task(task_id)
        assert task is not None
        closed = report.timings[task_id].detection_latency_ms
        adversarial = simulate_detection(
            task, spec.supervisor, seed=0, trials=1, adversarial=True
        )
        worst = max(
            simulate_detection(
                task, spec.supervisor, seed=seed, trials=500
            ).max_latency_from_receipt_ms
            for seed in range(8)
        )
        emit(
            f"  {task_id:<10} closed {closed:11.6f} ms   "
            f"adversarial {adversarial.max_latency_from_receipt_ms:11.6f} ms   "
            f"difference {adversarial.max_latency_from_receipt_ms - closed:+.2e} ms   "
            f"stochastic max {worst:11.6f} ms"
        )
    emit()
    emit("findings")
    for finding in report.findings:
        emit(f"  {finding.code}  {finding.severity.value:<7} {finding.message}")

    emit()
    emit.check("the worked example raises its two known findings", len(report.findings) == 2)
    emit.check("the worked example exits 1", report.exit_code == 1)
    return emit.finish()


if __name__ == "__main__":
    raise SystemExit(main())
