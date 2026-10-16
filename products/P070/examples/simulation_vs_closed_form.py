"""Histogram the simulated detection latency against the closed-form bound.

The closed form is the worst case; the simulation samples the distribution
the worst case bounds. The point of the figure is the gap on the right of each
panel: the bound is attained only when the fault lands exactly at a heartbeat
receipt and the supervisor poll phase is adversarial, and a random run never
does both.

Writes ``screenshots/simulation_vs_closed_form.png``.
"""

from __future__ import annotations

import time

import numpy as np

from _bootstrap import SCREENSHOTS, SPECS
from _plotstyle import ACCENT, MISSED, NEUTRAL, finish, plt
from watchdogspec import load_spec, run_checks, simulate_detection

SEEDS = 12
TRIALS = 1500
TASK_IDS = ("TASK-GNC", "TASK-IMU", "TASK-FDIR")


def main() -> int:
    """Run the cross-check, draw the figure and print the numbers. Returns 0."""
    spec = load_spec(SPECS / "smallsat_fc.yaml")
    report = run_checks(spec)

    started = time.perf_counter()
    fig, axes = plt.subplots(1, len(TASK_IDS), figsize=(12.5, 4.2))
    print(f"seeds {SEEDS}, trials per seed {TRIALS}, {SEEDS * TRIALS} trials per task")
    print(
        f"  {'task':<10} {'closed form/ms':>15} {'adversarial/ms':>16} "
        f"{'difference/ms':>14} {'stochastic max/ms':>18} {'shortfall/ms':>13} "
        f"{'violations':>11}"
    )
    for axis, task_id in zip(axes, TASK_IDS, strict=True):
        task = spec.task(task_id)
        assert task is not None
        closed = report.timings[task_id].detection_latency_ms

        adversarial = simulate_detection(
            task, spec.supervisor, seed=0, trials=1, adversarial=True
        )
        latencies: list[float] = []
        violations = 0
        for seed in range(SEEDS):
            run = simulate_detection(
                task, spec.supervisor, seed=seed, trials=TRIALS, collect_trials=True
            )
            latencies.extend(run.latencies_from_receipt_ms)
            violations += run.violations
        samples = np.array(latencies)
        difference = adversarial.max_latency_from_receipt_ms - closed
        print(
            f"  {task_id:<10} {closed:>15.9f} "
            f"{adversarial.max_latency_from_receipt_ms:>16.9f} {difference:>14.2e} "
            f"{samples.max():>18.9f} {closed - samples.max():>13.6f} {violations:>11}"
        )

        axis.hist(samples, bins=60, color=NEUTRAL, alpha=0.8)
        axis.axvline(
            closed,
            color=MISSED,
            linestyle="--",
            linewidth=1.8,
            label=f"closed form {closed:.3f} ms",
        )
        axis.axvline(
            float(samples.max()),
            color=ACCENT,
            linestyle=":",
            linewidth=1.8,
            label=f"sampled max {samples.max():.3f} ms",
        )
        axis.set_title(f"{task_id}  ({len(samples)} trials)", fontsize=10)
        axis.set_xlabel("detection latency from last receipt (ms)")
        axis.legend(fontsize=7)
        axis.grid(alpha=0.25, linestyle=":")
    axes[0].set_ylabel("trials")
    fig.suptitle(
        "Closed-form worst-case detection latency against the seeded simulation",
        fontsize=11,
    )
    path = finish(fig, SCREENSHOTS / "simulation_vs_closed_form.png")
    print(f"elapsed             : {time.perf_counter() - started:.2f} s on 2 cores")
    print(f"wrote screenshots/{path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
