"""The controllability gap: what a magnetorquer cannot torque about.

Left: orbit-averaged geometry factors against inclination, against the
isotropic value of 2/3 that a perfectly uniform field-direction history would
give.  Right: two identical spacecraft, identical gain, one on an equatorial
orbit and one sun-synchronous, with the residual rate resolved along the
weakest inertial direction.

    python examples/controllability_gap.py
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
    controllability_report,
    inertia_from_diagonal,
    residual_rate_along,
    simulate_detumble,
)

INCLINATIONS = np.array([0.0, 5.0, 10.0, 20.0, 30.0, 45.0, 60.0, 75.0, 90.0, 97.4])
GAIN = 1.5e5
DURATION = 40000.0


def well_damped_axis(rep):
    w = rep.weakest_direction_eci
    ref = np.array([0.0, 0.0, 1.0]) if abs(w[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    v = np.cross(w, ref)
    return v / np.linalg.norm(v)


def main() -> None:
    lam = []
    for inc in INCLINATIONS:
        orbit = CircularOrbit(altitude_km=500.0, inclination_deg=float(inc))
        rep = controllability_report(orbit, 6000, 10.0 * orbit.period_s)
        lam.append(rep.weighted_eigenvalues)
    lam = np.array(lam)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))

    ax = axes[0]
    ax.plot(INCLINATIONS, lam[:, 0], "o-", color=COLORS["weak"], lw=1.4, ms=4,
            label=r"$\lambda_{\min}$  (worst-damped axis)")
    ax.plot(INCLINATIONS, lam[:, 1], "s-", color="#7f7f7f", lw=1.0, ms=3,
            label=r"$\lambda_{\mathrm{mid}}$")
    ax.plot(INCLINATIONS, lam[:, 2], "^-", color=COLORS["strong"], lw=1.0, ms=3,
            label=r"$\lambda_{\max}$")
    ax.axhline(2.0 / 3.0, color=COLORS["analytic"], ls="--", lw=1.0)
    ax.text(50.0, 2.0 / 3.0 + 0.02, "isotropic value 2/3", fontsize=8)
    ax.set_xlabel("orbit inclination [deg]")
    ax.set_ylabel("geometry factor  (eigenvalue, sum = 2)")
    ax.set_title("Orbit-averaged damping anisotropy, 500 km, 10-orbit span\n"
                 "no magnetorquer can torque about B, so this is what geometry buys",
                 fontsize=9.5)
    ax.set_ylim(0.0, 1.15)
    ax.legend(fontsize=8, loc="center right")
    ax.annotate(
        f"equatorial: {lam[0, 0]:.3f}\n({(2 / 3) / lam[0, 0]:.1f}x slower than isotropic)",
        xy=(0.0, lam[0, 0]), xytext=(12.0, 0.20), fontsize=8, color=COLORS["weak"],
        arrowprops={"arrowstyle": "->", "color": COLORS["weak"], "lw": 0.8},
    )

    ax = axes[1]
    inertia = inertia_from_diagonal(0.05, 0.05, 0.05)
    summary = []
    for inc, style, key in ((0.0, "-", "weak"), (97.4, "-", "strong")):
        orbit = CircularOrbit(altitude_km=500.0, inclination_deg=inc)
        rep = controllability_report(orbit, 4000, 10.0 * orbit.period_s)
        cfg = DetumbleConfig(
            inertia=inertia, orbit=orbit, magnetorquer=Magnetorquer.isotropic(0.2),
            omega0_rad_s=np.radians([5.0, 5.0, 5.0]), duration_s=DURATION,
            control_dt_s=2.0, substeps=2, target_rate_rad_s=np.radians(1.0),
        )
        r = simulate_detumble(cfg, FixedGainPolicy(GAIN))
        weak = np.abs(residual_rate_along(r.omega_rad_s, rep.weakest_direction_eci, r.quat))
        good = np.abs(residual_rate_along(r.omega_rad_s, well_damped_axis(rep), r.quat))
        ax.semilogy(r.t_s / 60.0, np.degrees(weak), style, color=COLORS[key], lw=1.3,
                    label=f"i = {inc:.1f} deg, weak axis")
        ax.semilogy(r.t_s / 60.0, np.degrees(good), ":", color=COLORS[key], lw=1.0,
                    label=f"i = {inc:.1f} deg, well-damped plane")
        summary.append((inc, r, weak, good, rep))
    ax.axhline(1.0, color="#888888", ls=":", lw=1.0)
    ax.text(DURATION / 60.0, 1.1, " target 1 deg/s", ha="right", fontsize=8,
            color="#555555")
    ax.set_xlabel("time [minutes]")
    ax.set_ylabel("rate component [deg/s]")
    ax.set_ylim(1e-3, 20.0)
    ax.set_title("Same vehicle, same gain, two orbits:\n"
                 "on the equatorial orbit the residual rate is almost all weak-axis",
                 fontsize=9.5)
    ax.legend(fontsize=7.5, loc="upper right")

    path = save(fig, "controllability_gap.png")
    print(f"saved {path}")
    for inc, r, weak, _unused, rep in summary:
        t = f"{r.detumble_time_s:.1f} s" if r.detumbled else f"not reached in {DURATION:.0f} s"
        print(f"inclination {inc:5.1f} deg: lambda_min = {rep.weighted_eigenvalues[0]:.5f}, "
              f"detumble {t}")
        print(f"    final |omega| = {np.degrees(r.rate_norm_rad_s[-1]):.5f} deg/s, "
              f"weak-axis component = {np.degrees(weak[-1]):.5f} deg/s "
              f"({weak[-1] / r.rate_norm_rad_s[-1]:.3f} of it)")


if __name__ == "__main__":
    main()
