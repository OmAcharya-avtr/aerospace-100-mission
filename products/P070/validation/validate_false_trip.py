"""Closed-form false-trip rate against the event-driven simulation.

Equation (7) of :mod:`watchdogspec.falsetrip` is an analytic Gaussian tail.
The simulation draws heartbeat arrival times and steps the supervisor deadline
chain over them, counting declarations against a task that never fails. The
two are compared with a Wilson score interval, which is appropriate here
because the proportions are small (Wilson 1927; the Wald interval misbehaves
near zero).

A tail probability is only measurable when it is large enough to be sampled.
The configurations below were chosen to span about three decades of rate,
from one trip in four gaps down to one in five thousand, which is as deep
as the budget allows; the README says plainly that the far tail of equation
(7) is unvalidated by sampling and is only as good as the Gaussian assumption.
"""

from __future__ import annotations

import time

from _support import SPECS, Emitter
from watchdogspec import (
    Task,
    false_trip_probability,
    load_spec,
    simulate_false_trips,
    spec_false_trip_rates,
    wilson_interval,
)

# (label, period ms, jitter bound ms, sigma ms, k, gaps)
CONFIGS = [
    ("p ~ 2.4e-1", 20.0, 1.0, 1.0, 0, 200_000),
    ("p ~ 4.5e-2", 20.0, 2.4, 1.0, 0, 200_000),
    ("p ~ 6.7e-3", 20.0, 3.5, 1.0, 0, 300_000),
    ("p ~ 9.3e-4", 20.0, 4.4, 1.0, 0, 400_000),
    ("p ~ 2.0e-4", 20.0, 5.0, 1.0, 0, 900_000),
    ("k = 1", 10.0, 1.0, 5.0, 1, 300_000),
    ("k = 2", 10.0, 1.0, 9.0, 2, 300_000),
]

SEEDS = (20261016, 11, 12345)


def main() -> int:
    """Compare equation (7) with the simulation. Returns a process exit code."""
    emit = Emitter(__file__)
    started = time.perf_counter()
    emit("watchdogspec validation: false-trip rate")
    emit()
    emit("Equation (7): p = 0.5 * erfc((k P + (k+1) J) / (2 sigma)) per heartbeat gap.")
    emit("The simulation draws arrivals n P + d_n with d_n ~ N(0, sigma^2) and steps")
    emit("the deadline chain; a declaration against a healthy task is a false trip.")
    emit()
    emit(
        f"  {'config':<10} {'P/ms':>7} {'J/ms':>7} {'sig/ms':>7} {'k':>3} {'gaps':>9} "
        f"{'seed':>9} {'closed p':>12} {'sampled p':>12} {'Wilson low':>12} "
        f"{'Wilson high':>12} {'in':>4}"
    )
    inside = 0
    total = 0
    worst_relative = 0.0
    for label, period, jitter, sigma, k, gaps in CONFIGS:
        closed = false_trip_probability(period, jitter, sigma, k)
        task = Task(
            id=label,
            heartbeat_period_ms=period,
            missed_beats_tolerated=k,
            liveness_deadline_ms=1e9,
            escalation="L",
            heartbeat_jitter_ms=jitter,
            jitter_sigma_ms=sigma,
        )
        for seed in SEEDS:
            result = simulate_false_trips(task, seed=seed, gaps=gaps)
            low, high = wilson_interval(result.false_trips, result.gaps)
            contained = low <= closed <= high
            inside += int(contained)
            total += 1
            if closed > 0.0:
                worst_relative = max(
                    worst_relative, abs(result.empirical_p - closed) / closed
                )
            emit(
                f"  {label:<10} {period:>7.1f} {jitter:>7.1f} {sigma:>7.2f} {k:>3} "
                f"{gaps:>9} {seed:>9} {closed:>12.6g} {result.empirical_p:>12.6g} "
                f"{low:>12.6g} {high:>12.6g} {'yes' if contained else 'NO':>4}"
            )
    emit()
    emit(f"  closed form inside the 95 % Wilson interval : {inside} of {total}")
    emit(f"  largest relative deviation of the sample    : {worst_relative:.4f}")
    emit.check(
        "the closed form is inside the 95 % interval in at least 19 of every 20 "
        "comparisons",
        inside >= total - max(1, total // 20),
        f"{inside}/{total}",
    )

    emit()
    emit("2. the worked example's telemetry task, which exceeds its declared budget")
    spec = load_spec(SPECS / "smallsat_fc.yaml")
    task = spec.task("TASK-TLM")
    assert task is not None
    rate = spec_false_trip_rates(spec)["TASK-TLM"]
    emit(f"  P                      : {task.heartbeat_period_ms:g} ms")
    emit(f"  J                      : {task.heartbeat_jitter_ms:g} ms")
    emit(f"  sigma                  : {task.jitter_sigma_ms:g} ms")
    emit(f"  k                      : {task.missed_beats_tolerated}")
    emit(f"  trip threshold k P + (k+1) J : {rate.threshold_ms:g} ms")
    emit(f"  z = threshold / (sigma sqrt 2) : {rate.z:.6f}")
    emit(f"  closed-form p per gap  : {rate.per_gap:.8f}")
    emit(f"  closed-form per hour   : {rate.per_hour:.4f}")
    emit(f"  mean time between trips: {rate.mtbft_hours:.6f} hours")
    emit(f"  declared budget        : {task.false_trip_budget_per_hour:g} per hour")
    result = simulate_false_trips(task, seed=20261016, gaps=400_000)
    low, high = wilson_interval(result.false_trips, result.gaps)
    emit(f"  simulated p per gap    : {result.empirical_p:.8f} over {result.gaps} gaps")
    emit(f"  Wilson 95 % interval   : [{low:.8f}, {high:.8f}]")
    emit(f"  simulated hours        : {result.simulated_hours:.2f}")
    emit(f"  simulated trips/hour   : {result.empirical_per_hour:.4f}")
    emit.check(
        "the closed form for TASK-TLM is inside the 95 % interval",
        low <= rate.per_gap <= high,
    )
    emit.check(
        "the declared budget of 1 per hour is exceeded by about two orders of magnitude",
        rate.per_hour > 50.0 * task.false_trip_budget_per_hour,
        f"{rate.per_hour:.1f} against {task.false_trip_budget_per_hour:g}",
    )

    emit()
    emit("3. a declared sigma of zero is an exact zero, not a small number")
    exact = spec_false_trip_rates(spec)["TASK-GNC"]
    emit("  TASK-GNC sigma         : 0.9 ms (declared, non-zero)")
    emit(f"  TASK-GNC p per gap     : {exact.per_gap:g}  (erfc underflowed)")
    emit(f"  TASK-GNC z             : {exact.z:.4f} sigma of the gap")
    zero = false_trip_probability(20.0, 2.0, 0.0, 1)
    emit(f"  sigma = 0 gives p      : {zero!r}")
    emit.check("a declared sigma of zero gives exactly 0.0", zero == 0.0)
    emit.check(
        "a tail beyond the double range is reported as 0, not as a wrong small number",
        exact.per_gap == 0.0 and exact.z > 26.0,
        f"z = {exact.z:.2f}",
    )

    elapsed = time.perf_counter() - started
    emit()
    emit(f"  wall clock            : {elapsed:.1f} s on 2 cores")
    emit(f"  heartbeat gaps simulated : {sum(c[5] for c in CONFIGS) * len(SEEDS) + 400_000}")
    emit.check("the script finished inside the three-minute compute budget", elapsed < 180.0)
    return emit.finish()


if __name__ == "__main__":
    raise SystemExit(main())
