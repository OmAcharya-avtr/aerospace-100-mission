"""False-trip rate against the declared jitter statistics.

Equation (7) of :mod:`watchdogspec.falsetrip` has two levers: the declared
jitter bound ``J``, which sets the trip threshold, and the declared jitter
standard deviation ``sigma``, which sets how often that threshold is crossed.
This figure sweeps both for the worked example's telemetry task and marks the
declared operating point, so the sensitivity of the rate to a declaration
nobody measured very carefully is visible.

Writes ``screenshots/false_trip_surface.png``.
"""

from __future__ import annotations

import numpy as np

from _bootstrap import SCREENSHOTS, SPECS
from _plotstyle import MISSED, finish, plt
from watchdogspec import false_trip_probability, load_spec, simulate_false_trips, wilson_interval

GAPS = 60_000


def main() -> int:
    """Draw the sweep, check one point against the simulation. Returns 0."""
    spec = load_spec(SPECS / "smallsat_fc.yaml")
    task = spec.task("TASK-TLM")
    assert task is not None
    period = task.heartbeat_period_ms
    beats_per_hour = 3_600_000.0 / period

    bounds = np.linspace(50.0, 600.0, 120)
    sigmas = [30.0, 60.0, 90.0, 150.0]

    print(f"task                : {task.id}")
    print(f"declared J          : {task.heartbeat_jitter_ms:g} ms")
    print(f"declared sigma      : {task.jitter_sigma_ms:g} ms")
    print(f"declared k          : {task.missed_beats_tolerated}")
    print(f"beats per hour      : {beats_per_hour:g}")

    fig, axis = plt.subplots(figsize=(9.0, 5.0))
    for sigma in sigmas:
        rates = [
            false_trip_probability(period, bound, sigma, task.missed_beats_tolerated)
            * beats_per_hour
            for bound in bounds
        ]
        axis.semilogy(bounds, rates, label=f"sigma = {sigma:g} ms")

    declared_rate = (
        false_trip_probability(
            period,
            task.heartbeat_jitter_ms,
            task.jitter_sigma_ms,
            task.missed_beats_tolerated,
        )
        * beats_per_hour
    )
    axis.plot(
        [task.heartbeat_jitter_ms],
        [declared_rate],
        marker="o",
        markersize=9,
        color=MISSED,
        linestyle="none",
        label=f"declared point: {declared_rate:.3g} per hour",
    )
    budget = task.false_trip_budget_per_hour
    assert budget is not None
    axis.axhline(
        budget, color=MISSED, linestyle="--", linewidth=1.3, label=f"budget {budget:g} per hour"
    )
    axis.set_xlabel("declared jitter bound J (ms)")
    axis.set_ylabel("false trips per hour (log scale)")
    axis.set_title(
        f"False-trip rate against the declared jitter statistics, {task.id} "
        f"(k = {task.missed_beats_tolerated})",
        fontsize=10,
    )
    axis.grid(alpha=0.3, linestyle=":", which="both")
    axis.legend(fontsize=8)

    closed = false_trip_probability(
        period, task.heartbeat_jitter_ms, task.jitter_sigma_ms, task.missed_beats_tolerated
    )
    result = simulate_false_trips(task, seed=20261016, gaps=GAPS)
    low, high = wilson_interval(result.false_trips, result.gaps)
    inside = low <= closed <= high
    print(f"closed-form p       : {closed:.8f} per gap")
    print(f"simulated p         : {result.empirical_p:.8f} per gap over {GAPS} gaps")
    print(f"Wilson 95 % interval: [{low:.8f}, {high:.8f}]")
    print(f"closed form inside  : {inside}")

    path = finish(fig, SCREENSHOTS / "false_trip_surface.png")
    print(f"wrote screenshots/{path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
