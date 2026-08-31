"""Detumble time and cost against B-dot gain.

Two panels: the unsaturated sweep, where the analytic 1/k law holds and is
drawn against the simulation; and the realistic sweep with a 0.2 A m^2 dipole
limit, where saturation produces an interior optimum that the 1/k law does not
predict.

    python examples/gain_sweep.py
"""

from __future__ import annotations

import numpy as np
from _plotstyle import COLORS, save

import matplotlib.pyplot as plt  # noqa: E402

from detumblesim import (  # noqa: E402
    CircularOrbit,
    DetumbleConfig,
    FixedGainPolicy,
    Magnetorquer,
    detumble_time_first_order,
    inertia_from_diagonal,
    orbit_field_moments,
    simulate_detumble,
)

ORBIT = CircularOrbit(altitude_km=500.0, inclination_deg=97.4)
J_SCALAR = 0.05
INERTIA = inertia_from_diagonal(J_SCALAR, J_SCALAR, J_SCALAR)
OMEGA0 = np.radians([6.0, -5.0, 5.7])
TARGET = np.radians(1.0)


def sweep(gains, dipole_limit, duration):
    times, costs, sat = [], [], []
    for k in gains:
        cfg = DetumbleConfig(
            inertia=INERTIA, orbit=ORBIT,
            magnetorquer=Magnetorquer.isotropic(dipole_limit),
            omega0_rad_s=OMEGA0, duration_s=duration, control_dt_s=4.0, substeps=1,
            target_rate_rad_s=TARGET, stop_when_detumbled=True,
        )
        r = simulate_detumble(cfg, FixedGainPolicy(float(k)))
        times.append(r.detumble_time_s)
        costs.append(r.actuation_cost_a2m4s)
        sat.append(r.saturated_fraction)
    return np.array(times), np.array(costs), np.array(sat)


def main() -> None:
    moments = orbit_field_moments(ORBIT, 4000, 10.0 * ORBIT.period_s)
    rate0 = float(np.linalg.norm(OMEGA0))

    free_gains = np.geomspace(3.0e3, 4.8e4, 8)
    t_free, _, sat_free = sweep(free_gains, 50.0, 250000.0)
    slope = float(np.polyfit(np.log(free_gains), np.log(t_free), 1)[0])
    t_iso = np.array(
        [detumble_time_first_order(J_SCALAR, k, moments, rate0, TARGET, "isotropic")
         for k in free_gains]
    )
    t_fast = np.array(
        [detumble_time_first_order(J_SCALAR, k, moments, rate0, TARGET, "fastest")
         for k in free_gains]
    )
    t_slow = np.array(
        [detumble_time_first_order(J_SCALAR, k, moments, rate0, TARGET, "slowest")
         for k in free_gains]
    )

    real_gains = np.geomspace(1.0e4, 3.0e6, 14)
    t_real, q_real, sat_real = sweep(real_gains, 0.2, 40000.0)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))

    ax = axes[0]
    ax.fill_between(free_gains, t_fast, t_slow, color=COLORS["analytic"], alpha=0.10,
                    label="analytic modal bracket")
    ax.loglog(free_gains, t_iso, color=COLORS["analytic"], ls="--", lw=1.0,
              label=r"first-order model, $t \propto 1/k$")
    ax.loglog(free_gains, t_free, "o-", color=COLORS["bdot"], lw=1.3, ms=4,
              label=f"simulation (fitted slope {slope:.3f})")
    ax.set_xlabel(r"B-dot gain $k$  [A m$^2$ s / T]")
    ax.set_ylabel("detumble time to 1 deg/s  [s]")
    ax.set_title("Saturation disabled (50 A m$^2$ limit):\n"
                 "the 1/k law holds and the bracket contains every point", fontsize=9.5)
    ax.legend(fontsize=8, loc="upper right")
    ax.text(0.03, 0.06, f"max saturated fraction over the sweep: {sat_free.max():.3f}",
            transform=ax.transAxes, fontsize=8, color="#444444")

    ax = axes[1]
    finite = np.isfinite(t_real)
    ax.loglog(real_gains[finite], t_real[finite], "o-", color=COLORS["bdot"], lw=1.3,
              ms=4, label="detumble time [s]")
    if (~finite).any():
        ax.loglog(real_gains[~finite],
                  np.full((~finite).sum(), np.nanmax(t_real[finite]) * 1.6), "x",
                  color=COLORS["weak"], ms=7, label="target never reached")
    best = int(np.nanargmin(np.where(finite, t_real, np.inf)))
    ax.axvline(real_gains[best], color=COLORS["bdot"], lw=0.8, alpha=0.4)
    ax.text(real_gains[best] * 1.15, np.nanmin(t_real[finite]) * 1.05,
            f"fastest at k = {real_gains[best]:.2e}\n({t_real[best]:.0f} s)",
            fontsize=8, color=COLORS["bdot"])
    ax2 = ax.twinx()
    ax2.semilogx(real_gains, 100.0 * sat_real, color=COLORS["sat"], lw=1.2, ls="-")
    ax2.set_ylabel("control steps saturated [%]", color="#8a6d00")
    ax2.tick_params(axis="y", colors="#8a6d00")
    ax2.grid(False)
    ax.set_xlabel(r"B-dot gain $k$  [A m$^2$ s / T]")
    ax.set_ylabel("detumble time to 1 deg/s  [s]")
    ax.set_title("Realistic 0.2 A m$^2$ limit:\n"
                 "saturation puts an interior optimum where 1/k predicts none",
                 fontsize=9.5)
    ax.legend(fontsize=8, loc="upper center")

    path = save(fig, "gain_sweep.png")
    print(f"saved {path}")
    print(f"unsaturated log-log slope = {slope:.6f} (model predicts -1)")
    print(f"points inside the analytic bracket = "
          f"{int(np.sum((t_free >= t_fast) & (t_free <= t_slow)))} of {t_free.size}")
    print(f"saturated sweep: fastest gain = {real_gains[best]:.4e} at "
          f"{t_real[best]:.1f} s, {100 * sat_real[best]:.1f} % of steps saturated")
    print(f"saturated sweep: runs that never reached the target = {int((~finite).sum())}"
          f" of {real_gains.size}")


if __name__ == "__main__":
    main()
