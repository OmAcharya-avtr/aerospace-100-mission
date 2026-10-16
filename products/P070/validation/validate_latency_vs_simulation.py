"""Closed-form worst-case detection latency against the discrete-event simulation.

Two separate claims are measured.

1. **The bound holds.** Over every shipped task and a grid of synthetic ones,
   across many seeds, no simulated detection latency exceeds equation (3).
2. **The bound is attained.** With the fault placed at a heartbeat receipt and
   the supervisor poll phase adversarial, the simulated latency equals the
   closed form to within floating-point rounding. "Equals" here means the
   difference is a few units in the last place of a sum of six terms, not that
   two doubles compare equal; the measured differences are printed.

Compute budget: the grid below is sized for two cores and the script reports
its own wall-clock time.
"""

from __future__ import annotations

import itertools
import time

from _support import SPECS, Emitter
from watchdogspec import (
    Supervisor,
    Task,
    load_spec,
    run_checks,
    simulate_detection,
    worst_case_detection_latency_ms,
)

SEEDS = 16
TRIALS = 1200
AGREEMENT_TOL_MS = 1e-6

PERIODS = (1.0, 5.0, 20.0, 100.0, 1000.0)
JITTERS = (0.0, 0.1, 3.0, 25.0)
TOLERANCES = (0, 1, 2, 5, 9)
SUPERVISORS = (
    Supervisor(poll_period_ms=1.0, poll_jitter_ms=0.0, declaration_overhead_ms=0.0),
    Supervisor(poll_period_ms=5.0, poll_jitter_ms=1.0, declaration_overhead_ms=0.5),
    Supervisor(poll_period_ms=10.0, poll_jitter_ms=2.0, declaration_overhead_ms=1.0),
    Supervisor(poll_period_ms=50.0, poll_jitter_ms=50.0, declaration_overhead_ms=3.0),
    # Deliberately outside the tightness condition Jq <= Q, so that the
    # over-estimate found by Hypothesis is measured here as well.
    Supervisor(poll_period_ms=2.0, poll_jitter_ms=6.0, declaration_overhead_ms=0.0),
)


def main() -> int:
    """Run both cross-checks. Returns a process exit code."""
    emit = Emitter(__file__)
    started = time.perf_counter()
    emit("watchdogspec validation: closed form against the discrete-event simulation")
    emit()
    emit("Equation (3): WCDL = (k + 1)(P + J) + Q + Jq + O, all in ms.")
    emit("The simulation generates beat arrivals, places the fault at a continuous")
    emit("instant inside a beat interval, steps the supervisor deadline chain and")
    emit("poll loop, and reports the latency it observed. It never consults")
    emit("equation (3) while running.")
    emit()

    emit("1. the shipped worked example, per task")
    spec = load_spec(SPECS / "smallsat_fc.yaml")
    report = run_checks(spec)
    emit(
        f"  {'task':<10} {'closed form/ms':>16} {'adversarial/ms':>16} "
        f"{'difference/ms':>14} {'ulp':>5} {'stochastic max/ms':>18} {'shortfall/ms':>13} "
        f"{'violations':>11}"
    )
    import math

    for task_id in ("TASK-GNC", "TASK-IMU", "TASK-TLM", "TASK-FDIR"):
        task = spec.task(task_id)
        assert task is not None
        closed = report.timings[task_id].detection_latency_ms
        adversarial = simulate_detection(
            task, spec.supervisor, seed=0, trials=1, adversarial=True
        )
        observed = adversarial.max_latency_from_receipt_ms
        difference = observed - closed
        ulps = abs(difference) / math.ulp(closed)
        worst = -math.inf
        violations = 0
        for seed in range(SEEDS):
            run = simulate_detection(task, spec.supervisor, seed=seed, trials=TRIALS)
            worst = max(worst, run.max_latency_from_receipt_ms)
            violations += run.violations
        emit(
            f"  {task_id:<10} {closed:>16.9f} {observed:>16.9f} {difference:>14.2e} "
            f"{ulps:>5.1f} {worst:>18.9f} {closed - worst:>13.6f} {violations:>11}"
        )
        emit.check(
            f"{task_id}: the adversarial trial attains the closed form",
            abs(difference) <= AGREEMENT_TOL_MS,
            f"difference {difference:.3e} ms, {ulps:.1f} ulp",
        )
        emit.check(
            f"{task_id}: no stochastic trial exceeded the bound over "
            f"{SEEDS * TRIALS} trials",
            violations == 0,
        )

    emit()
    emit("2. a grid of synthetic declarations")
    grid = [
        (period, jitter, k, supervisor)
        for period, jitter, k, supervisor in itertools.product(
            PERIODS, JITTERS, TOLERANCES, SUPERVISORS
        )
        if jitter < period
    ]
    emit(f"  configurations        : {len(grid)}")
    emit("  seeds per config      : 4")
    emit("  trials per seed       : 200")
    worst_difference = 0.0
    worst_ulps = 0.0
    worst_case = ""
    total_violations = 0
    tight = 0
    loose = 0
    worst_shortfall = 0.0
    for period, jitter, k, supervisor in grid:
        task = Task(
            id="grid",
            heartbeat_period_ms=period,
            missed_beats_tolerated=k,
            liveness_deadline_ms=1e9,
            escalation="L",
            heartbeat_jitter_ms=jitter,
        )
        closed = worst_case_detection_latency_ms(
            period,
            jitter,
            k,
            supervisor.poll_period_ms,
            supervisor.poll_jitter_ms,
            supervisor.declaration_overhead_ms,
        )
        adversarial = simulate_detection(task, supervisor, seed=0, trials=1, adversarial=True)
        difference = adversarial.max_latency_from_receipt_ms - closed
        # Equation (3) is tight only while the poll release jitter does not
        # exceed the poll period; see the module docstring of
        # watchdogspec.latency and tests/test_regression.py.
        if supervisor.poll_jitter_ms <= supervisor.poll_period_ms:
            tight += 1
            if abs(difference) > abs(worst_difference):
                worst_difference = difference
                worst_ulps = abs(difference) / math.ulp(closed)
                worst_case = (
                    f"P={period:g} J={jitter:g} k={k} Q={supervisor.poll_period_ms:g} "
                    f"Jq={supervisor.poll_jitter_ms:g} O={supervisor.declaration_overhead_ms:g}"
                )
        else:
            loose += 1
            worst_shortfall = max(worst_shortfall, -difference)
        for seed in range(4):
            run = simulate_detection(task, supervisor, seed=seed, trials=200)
            total_violations += run.violations
    emit(f"  configurations with Jq <= Q (bound claimed tight) : {tight}")
    emit(f"  configurations with Jq >  Q (bound only an upper bound) : {loose}")
    emit(f"  largest adversarial difference where tight        : {worst_difference:.3e} ms "
         f"({worst_ulps:.1f} ulp)")
    emit(f"  at                                                 : {worst_case}")
    emit(f"  largest shortfall where Jq > Q                     : {worst_shortfall:.6f} ms")
    emit(f"  total bound violations over {len(grid) * 4 * 200} trials : {total_violations}")
    emit.check(
        "the bound is never violated anywhere on the grid", total_violations == 0
    )
    emit.check(
        "the adversarial trial attains the closed form wherever Jq <= Q",
        abs(worst_difference) <= AGREEMENT_TOL_MS,
        f"worst {worst_difference:.3e} ms",
    )
    emit.check(
        "with Jq > Q the closed form over-estimates rather than under-estimates",
        worst_shortfall > 0.0,
        f"largest shortfall {worst_shortfall:.6f} ms; the bound still holds",
    )

    elapsed = time.perf_counter() - started
    emit()
    emit(f"  wall clock            : {elapsed:.1f} s on 2 cores")
    emit.check("the script finished inside the three-minute compute budget", elapsed < 180.0)
    return emit.finish()


if __name__ == "__main__":
    raise SystemExit(main())
