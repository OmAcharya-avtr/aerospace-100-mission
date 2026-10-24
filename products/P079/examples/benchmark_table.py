"""The benchmark harness: a measured host table and a deliberately blank one.

Writes ``screenshots/benchmark_table.png``. What to notice: the left panel is
measured on this machine and is unstable -- it is wall-clock timing on a shared
container. The right panel is the Level 4 table and it is **empty**: five rows,
no numbers, every status ``unmeasured``. It stays that way until the five
figures are measured on the Jetson Orin Nano itself, and this product is
labelled Level 3, hardware-pending until they are.
"""

from __future__ import annotations

import numpy as np
from _common import SERIES, bootstrap, save, style

ROOT = bootstrap()
style()

import matplotlib.pyplot as plt  # noqa: E402

from timetransfer import (  # noqa: E402
    AsymmetrySource,
    ClockSpec,
    KalmanConfig,
    LinkSpec,
    SimulatedBackend,
    SimulationSpec,
    acquire,
    allan_table,
    fit_least_squares,
    kalman_skew,
    offset_uncertainty,
    run_benchmark,
    two_way_estimate,
)
from timetransfer.asym import window_features  # noqa: E402

N = 600
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
    seed=20261024,
)


def main() -> None:
    backend = SimulatedBackend(SPEC)
    backend.open()
    exchanges = acquire(backend, N, min_exchanges=8).exchanges
    backend.close()
    estimate = two_way_estimate(exchanges)
    fit = fit_least_squares(estimate.midpoint_s, estimate.offset_s, order=2)
    residual = fit.detrended(estimate.midpoint_s, estimate.offset_s)

    def stage_acquire() -> object:
        local = SimulatedBackend(SPEC)
        local.open()
        result = acquire(local, N, min_exchanges=8)
        local.close()
        return result

    record = run_benchmark(
        {
            "acquire": stage_acquire,
            "two_way_estimate": lambda: two_way_estimate(exchanges),
            "fit_least_squares": lambda: fit_least_squares(
                estimate.midpoint_s, estimate.offset_s, order=2
            ),
            "kalman_skew": lambda: kalman_skew(
                estimate.midpoint_s,
                estimate.offset_s,
                config=KalmanConfig(q1_s2_per_s=1.0e-20, q2_per_s=1.0e-26),
            ),
            "allan_table": lambda: allan_table(residual, 1.0, noise_kind="white_pm"),
            "offset_uncertainty": lambda: offset_uncertainty(
                fit,
                resolution_ns=1,
                asymmetry_source=AsymmetrySource.DECLARED_BOUND,
                asymmetry_bound_s=1.6e-3,
            ),
            "window_features": lambda: window_features(exchanges),
        },
        label="example run",
        environment_note=(
            "shared two-core container; no attempt made to quiesce the machine"
        ),
        backend_kind="simulated",
        backend_driver="",
        n_repeats=7,
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3))
    left, right = axes
    names = [stage.stage for stage in record.stages]
    medians = np.asarray([stage.median_s for stage in record.stages]) * 1e3
    p95 = np.asarray([stage.p95_s for stage in record.stages]) * 1e3
    positions = np.arange(len(names))
    left.barh(positions - 0.18, medians, 0.34, color=SERIES[0], label="median")
    left.barh(positions + 0.18, p95, 0.34, color=SERIES[1], alpha=0.7, label="p95")
    left.set_yticks(positions)
    left.set_yticklabels(names, fontsize=7.6)
    left.invert_yaxis()
    left.set_xscale("log")
    left.set_xlabel("milliseconds per call (log scale)")
    left.set_title(
        f"host table, MEASURED and UNSTABLE\n{record.stages[0].n} repeats on "
        f"{record.host['cpu_count_available']} available cores, is_hardware="
        f"{record.is_hardware}"
    )
    left.legend(loc="lower right", fontsize=7.6)

    right.set_xlim(0, 1)
    right.set_ylim(0, 1)
    right.axis("off")
    right.set_title("Level 4 hardware table: EMPTY, UNMEASURED")
    rows = [(row.stage, row.units, row.status) for row in record.hardware]
    right.text(0.02, 0.94, f"{'figure':<26}{'value':>8}{'units':>7}{'status':>14}",
               family="monospace", fontsize=7.6, va="top")
    for index, (stage, units, status) in enumerate(rows):
        right.text(
            0.02,
            0.86 - 0.075 * index,
            f"{stage:<26}{'--':>8}{units:>7}{status:>14}",
            family="monospace",
            fontsize=7.6,
            va="top",
        )
    right.text(
        0.02,
        0.40,
        "No Jetson Orin Nano was reachable from the environment this\n"
        "product was built in. Filling any row above from simulation\n"
        "would be a false hardware claim, so the rows stay blank and\n"
        "BenchmarkRecord.promote_to_hardware() raises.\n\n"
        "Blocked on owner action: the board powered on and reachable,\n"
        "its hostname or IP, and passwordless SSH from the Mac.",
        fontsize=7.8,
        va="top",
    )

    fig.tight_layout()
    print(save(fig, "benchmark_table"))
    print(f"{'stage':>22} {'median ms':>12} {'p95 ms':>12}")
    for stage in record.stages:
        print(f"{stage.stage:>22} {stage.median_s * 1e3:>12.4f} {stage.p95_s * 1e3:>12.4f}")
    print(f"tracemalloc peak      : {record.tracemalloc_peak_bytes:,.0f} B")
    print(f"maxrss                : {record.maxrss_bytes:,.0f} B")
    print(f"measured clock res.   : {record.clock_measured_resolution_ns} ns")
    print(f"hardware rows         : {len(record.hardware)}, all values "
          f"{ {str(row.value) for row in record.hardware} }, all statuses "
          f"{ {row.status for row in record.hardware} }")
    print(f"is_hardware           : {record.is_hardware}")


if __name__ == "__main__":
    main()
