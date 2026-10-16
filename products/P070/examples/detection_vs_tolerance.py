"""The missed-beat tolerance trade-off: detection latency against false trips.

Raising the tolerance ``k`` pushes the worst-case detection latency out by one
beat window per step and pushes the false-trip rate down the Gaussian tail.
Both curves are drawn for the worked example's telemetry task, together with
its liveness deadline and its declared false-trip budget, so the window of
admissible ``k`` can be read off directly.

Writes ``screenshots/detection_vs_tolerance.png``.
"""

from __future__ import annotations

import numpy as np

from _bootstrap import SCREENSHOTS, SPECS
from _plotstyle import ACCENT, MISSED, NEUTRAL, finish, plt
from watchdogspec import false_trip_probability, load_spec, worst_case_detection_latency_ms

MAX_K = 8


def main() -> int:
    """Draw the trade-off and print the admissible tolerance window. Returns 0."""
    spec = load_spec(SPECS / "smallsat_fc.yaml")
    task = spec.task("TASK-TLM")
    assert task is not None
    supervisor = spec.supervisor
    ladder = spec.ladder(task.escalation)
    assert ladder is not None

    ks = np.arange(0, MAX_K + 1)
    latencies = np.array(
        [
            worst_case_detection_latency_ms(
                task.heartbeat_period_ms,
                task.heartbeat_jitter_ms,
                int(k),
                supervisor.poll_period_ms,
                supervisor.poll_jitter_ms,
                supervisor.declaration_overhead_ms,
            )
            + ladder.total_latency_ms
            for k in ks
        ]
    )
    rates = np.array(
        [
            false_trip_probability(
                task.heartbeat_period_ms,
                task.heartbeat_jitter_ms,
                task.jitter_sigma_ms,
                int(k),
            )
            * 3_600_000.0
            / task.heartbeat_period_ms
            for k in ks
        ]
    )
    budget = task.false_trip_budget_per_hour
    assert budget is not None

    print(f"task                : {task.id} ({task.name})")
    print(f"beat window W       : {task.beat_window_ms:.3f} ms")
    print(f"liveness deadline D : {task.liveness_deadline_ms:.3f} ms")
    print(f"false-trip budget   : {budget:g} per hour")
    print(f"  {'k':>3} {'ladder completion/ms':>22} {'false trips/hour':>18} {'verdict':>12}")
    admissible = []
    for k, latency, rate in zip(ks, latencies, rates, strict=True):
        fits = latency <= task.liveness_deadline_ms
        quiet = rate <= budget
        if fits and quiet:
            admissible.append(int(k))
        verdict = "ok" if fits and quiet else ("late" if not fits else "noisy")
        print(f"  {k:>3} {latency:>22.3f} {rate:>18.6g} {verdict:>12}")
    if admissible:
        print(f"admissible k        : {admissible[0]} to {admissible[-1]}")
    else:
        print("admissible k        : none")

    fig, axis = plt.subplots(figsize=(9.0, 5.0))
    axis.plot(ks, latencies, marker="o", color=NEUTRAL, label="worst-case ladder completion")
    axis.axhline(
        task.liveness_deadline_ms,
        color=MISSED,
        linestyle="--",
        linewidth=1.5,
        label=f"liveness deadline D = {task.liveness_deadline_ms:g} ms",
    )
    axis.set_xlabel("missed beats tolerated, k")
    axis.set_ylabel("worst-case time from last heartbeat (ms)")
    axis.grid(alpha=0.3, linestyle=":")

    twin = axis.twinx()
    twin.semilogy(ks, rates, marker="s", color=ACCENT, label="false trips per hour")
    twin.axhline(
        budget,
        color=ACCENT,
        linestyle=":",
        linewidth=1.5,
        label=f"declared budget = {budget:g} per hour",
    )
    # Beyond k = 3 the Gaussian tail underflows the double range and the rate
    # is reported as exactly 0, which a log axis cannot draw. The axis is
    # clamped so that the readable part of the curve is readable; the printed
    # table above carries every value, including the zeros.
    twin.set_ylim(1e-35, 1e4)
    twin.set_ylabel("false trips per hour (log scale, clamped at 1e-35)")

    handles, labels = axis.get_legend_handles_labels()
    extra_handles, extra_labels = twin.get_legend_handles_labels()
    axis.legend(handles + extra_handles, labels + extra_labels, fontsize=8, loc="center right")
    axis.set_title(
        f"Missed-beat tolerance trade-off for {task.id}: detection latency against false trips",
        fontsize=10,
    )
    path = finish(fig, SCREENSHOTS / "detection_vs_tolerance.png")
    print(f"wrote screenshots/{path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
