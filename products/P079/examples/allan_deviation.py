"""Allan deviation over tau with a confidence interval, against closed forms.

Writes ``screenshots/allan_deviation.png``. What to notice: the three
synthetic processes sit on their theoretical lines with the stated slopes, the
intervals widen as tau grows and the number of second differences falls, and
the right panel is the deviation of a real acquisition's detrended offset
residual, which is white PM: the per-exchange queueing jitter dominates the
clock's own noise at every tau this acquisition reaches, so the stability of
the *measurement* is what is being plotted, not the stability of the
oscillator.
"""

from __future__ import annotations

import numpy as np
from _common import SERIES, bootstrap, save, style

ROOT = bootstrap()
style()

import matplotlib.pyplot as plt  # noqa: E402

from timetransfer import (  # noqa: E402
    ClockSpec,
    LinkSpec,
    SimulatedBackend,
    SimulationSpec,
    acquire,
    allan_table,
    fit_least_squares,
    fit_log_slope,
    identify_noise,
    two_way_estimate,
)
from timetransfer.noise import (  # noqa: E402
    NOISE_KINDS,
    synthetic_phase,
    theoretical_adev,
    theory_slope,
)

LEVELS = {"white_pm": 1.0e-9, "white_fm": 1.0e-20, "random_walk_fm": 1.0e-24}
LABELS = {
    "white_pm": "white PM, slope -1",
    "white_fm": "white FM, slope -1/2",
    "random_walk_fm": "random walk FM, slope +1/2",
}
N = 20000


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.1))
    left, right = axes

    print(f"{'process':>16} {'theory':>8} {'fitted':>10} {'stderr':>9} {'worst rel err':>14}")
    for index, kind in enumerate(NOISE_KINDS):
        level = LEVELS[kind]
        phase = synthetic_phase(kind, N, tau0_s=1.0, level=level, seed=20261024)
        table = allan_table(phase, 1.0, noise_kind=kind)
        theory = theoretical_adev(kind, table.tau_s, level=level)
        fit = fit_log_slope(table.tau_s, table.adev)
        colour = SERIES[index]
        left.errorbar(
            table.tau_s,
            table.adev,
            yerr=[table.adev - table.lower, table.upper - table.adev],
            fmt="o",
            ms=4,
            color=colour,
            capsize=3,
            lw=1.0,
            label=f"{LABELS[kind]} (fitted {fit.slope:+.3f})",
        )
        left.plot(table.tau_s, theory, "-", color=colour, lw=1.2, alpha=0.8)
        print(
            f"{kind:>16} {theory_slope(kind):>8.2f} {fit.slope:>10.4f} "
            f"{fit.slope_stderr:>9.4f} "
            f"{float(np.max(np.abs(table.adev / theory - 1.0))):>14.5f}"
        )
    left.set_xscale("log")
    left.set_yscale("log")
    left.set_xlabel("tau [s]")
    left.set_ylabel("overlapping Allan deviation")
    left.set_title(
        f"three closed-form processes, N = {N}\npoints with 95 per cent intervals, "
        f"lines are theory"
    )
    left.legend(loc="lower left", fontsize=7.2)

    spec = SimulationSpec(
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
        ),
        period_s=1.0,
        timestamp_noise_s=2.0e-8,
        resolution_ns=1,
        seed=20261024,
    )
    backend = SimulatedBackend(spec)
    backend.open()
    exchanges = acquire(backend, 4000, min_exchanges=8).exchanges
    backend.close()
    estimate = two_way_estimate(exchanges)
    fit2 = fit_least_squares(estimate.midpoint_s, estimate.offset_s, order=2)
    residual = fit2.detrended(estimate.midpoint_s, estimate.offset_s)
    identified = identify_noise(residual)
    table = allan_table(residual, 1.0, noise_kind=identified.kind)
    measured_fit = fit_log_slope(table.tau_s, table.adev)

    right.errorbar(
        table.tau_s,
        table.adev,
        yerr=[table.adev - table.lower, table.upper - table.adev],
        fmt="o",
        ms=4,
        color=SERIES[0],
        capsize=3,
        lw=1.0,
        label=f"measured (fitted slope {measured_fit.slope:+.3f})",
    )
    reference = table.adev[0] * (table.tau_s / table.tau_s[0]) ** -1.0
    right.plot(table.tau_s, reference, "--", color=SERIES[1], lw=1.2,
               label="tau^-1 through the first point (white PM)")
    right.set_xscale("log")
    right.set_yscale("log")
    right.set_xlabel("tau [s]")
    right.set_ylabel("overlapping Allan deviation")
    right.set_title(
        f"detrended offset residual, {exchanges.n} exchanges\n"
        f"identified as {identified.kind} (alpha = {identified.alpha:+.2f})"
    )
    right.legend(loc="lower left", fontsize=7.6)

    fig.tight_layout()
    print(save(fig, "allan_deviation"))
    print(f"measured residual slope : {measured_fit.slope:+.4f} "
          f"+- {measured_fit.slope_stderr:.4f}")
    print(f"identified noise        : {identified.kind}, alpha {identified.alpha:+.4f}, "
          f"r1 {identified.r1:+.4f}")
    print(f"adev at tau = 1 s       : {table.adev[0]:.6e} "
          f"[{table.lower[0]:.6e}, {table.upper[0]:.6e}]")


if __name__ == "__main__":
    main()
