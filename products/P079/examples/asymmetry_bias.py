"""The closed-form asymmetry bias, measured against a declared asymmetry.

Writes ``screenshots/asymmetry_bias.png``. What to notice: the measured mean
estimation error falls on the line ``A/2`` with a fitted slope of one half, and
the right panel shows that averaging more exchanges shrinks the scatter and
leaves the bias exactly where it was.
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
    asymmetry_bias_s,
    two_way_estimate,
)

ASYMMETRIES_S = np.asarray(
    [-1.6e-3, -8.0e-4, -4.0e-4, -1.0e-4, 0.0, 1.0e-4, 4.0e-4, 8.0e-4, 1.6e-3]
)


def _spec(asymmetry_s: float, *, jitter_scale_s: float, seed: int) -> SimulationSpec:
    return SimulationSpec(
        clock=ClockSpec(offset0_s=1.0e-3, skew_s_per_s=0.0),
        link=LinkSpec(
            floor_ab_s=5.0e-3 + 0.5 * asymmetry_s,
            floor_ba_s=5.0e-3 - 0.5 * asymmetry_s,
            jitter_scale_ab_s=jitter_scale_s,
            jitter_scale_ba_s=jitter_scale_s,
        ),
        period_s=1.0,
        timestamp_noise_s=2.0e-8,
        resolution_ns=1,
        seed=seed,
    )


def _mean_error(spec: SimulationSpec, n: int) -> float:
    backend = SimulatedBackend(spec)
    backend.open()
    exchanges = acquire(backend, n, min_exchanges=8).exchanges
    backend.close()
    estimate = two_way_estimate(exchanges)
    truth = exchanges.true_offset_s
    assert truth is not None
    return float(np.mean(estimate.offset_s - truth))


def main() -> None:
    quiet = np.asarray([_mean_error(_spec(a, jitter_scale_s=0.0, seed=7), 300)
                        for a in ASYMMETRIES_S])
    noisy = np.asarray([_mean_error(_spec(a, jitter_scale_s=2.0e-4, seed=7), 300)
                        for a in ASYMMETRIES_S])
    slope, intercept = np.polyfit(ASYMMETRIES_S, quiet, 1)
    closed = np.asarray(asymmetry_bias_s(ASYMMETRIES_S))

    counts = (20, 50, 100, 200, 500, 1000, 2000)
    fixed_asymmetry = 8.0e-4
    spread = []
    bias = []
    for n in counts:
        errors = [
            _mean_error(_spec(fixed_asymmetry, jitter_scale_s=2.0e-4, seed=1000 + s), n)
            for s in range(12)
        ]
        bias.append(float(np.mean(errors)))
        spread.append(float(np.std(errors, ddof=1)))

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.9))
    left, right = axes

    left.plot(ASYMMETRIES_S * 1e6, closed * 1e6, "-", color=SERIES[2], lw=1.6,
              label="closed form A/2")
    left.plot(ASYMMETRIES_S * 1e6, quiet * 1e6, "o", color=SERIES[0], ms=5,
              label="measured, no queueing jitter")
    left.plot(ASYMMETRIES_S * 1e6, noisy * 1e6, "x", color=SERIES[1], ms=6,
              label="measured, 240 us jitter scale")
    left.set_xlabel("declared asymmetry A = d_ab - d_ba [us]")
    left.set_ylabel("mean offset error [us]")
    left.set_title(f"bias against declared asymmetry\nfitted slope {slope:.9f}")
    left.legend(loc="upper left", fontsize=7.6)

    right.errorbar(counts, np.asarray(bias) * 1e6, yerr=np.asarray(spread) * 1e6,
                   fmt="o-", color=SERIES[0], ms=4, lw=1.1, capsize=3,
                   label="mean +- sd over 12 seeds")
    right.axhline(fixed_asymmetry / 2.0 * 1e6, color=SERIES[2], lw=1.6,
                  label=f"A/2 = {fixed_asymmetry / 2.0 * 1e6:.0f} us")
    right.axhline(0.0, color=SERIES[4], lw=1.0, ls="--", label="zero")
    right.set_xscale("log")
    right.set_xlabel("exchanges averaged")
    right.set_ylabel("mean offset error [us]")
    right.set_title("averaging shrinks the scatter, not the bias")
    right.legend(loc="center right", fontsize=7.6)

    fig.tight_layout()
    print(save(fig, "asymmetry_bias"))
    print(f"fitted slope          : {slope:.12f} (closed form 0.5)")
    print(f"fitted intercept      : {intercept:.3e} s")
    print(f"worst |measured - A/2|: {float(np.max(np.abs(quiet - closed))):.3e} s")
    print(f"bias at n=20          : {bias[0]:+.6e} s  sd {spread[0]:.3e} s")
    print(f"bias at n=2000        : {bias[-1]:+.6e} s  sd {spread[-1]:.3e} s")
    print(f"sd ratio 20 -> 2000   : {spread[0] / spread[-1]:.3f}")


if __name__ == "__main__":
    main()
