"""Rate-versus-time detumble curves for B-dot and cross-product control.

Runs one 3U-class spacecraft on a sun-synchronous orbit under both control
laws, plots the body-rate history against the analytic first-order envelope,
and shows where the magnetorquers are saturated.

    python examples/detumble_curve.py
"""

from __future__ import annotations

import numpy as np
from _plotstyle import COLORS, save

import matplotlib.pyplot as plt  # noqa: E402  (backend set in _plotstyle)

from detumblesim import (  # noqa: E402
    CircularOrbit,
    CrossProductController,
    DetumbleConfig,
    FixedGainPolicy,
    Magnetorquer,
    detumble_time_first_order,
    inertia_from_diagonal,
    orbit_field_moments,
    simulate_detumble,
)

ORBIT = CircularOrbit(altitude_km=500.0, inclination_deg=97.4)
INERTIA = inertia_from_diagonal(0.05, 0.06, 0.04)
OMEGA0_DEG_S = np.array([8.0, -6.0, 5.0])
TARGET_DEG_S = 1.0
BDOT_GAIN = 1.5e5


def run(controller, duration=20000.0):
    cfg = DetumbleConfig(
        inertia=INERTIA,
        orbit=ORBIT,
        magnetorquer=Magnetorquer.isotropic(0.2),
        omega0_rad_s=np.radians(OMEGA0_DEG_S),
        duration_s=duration,
        control_dt_s=2.0,
        substeps=2,
        target_rate_rad_s=np.radians(TARGET_DEG_S),
    )
    return simulate_detumble(cfg, controller)


def main() -> None:
    bdot = run(FixedGainPolicy(BDOT_GAIN))
    moments = orbit_field_moments(ORBIT, 4000, 10.0 * ORBIT.period_s)
    k_cross = BDOT_GAIN * moments.mean_b2_t2  # matched effective damping rate
    cross = run(CrossProductController(gain=k_cross))

    rate0 = float(np.linalg.norm(np.radians(OMEGA0_DEG_S)))
    j = float(np.mean(np.diag(INERTIA)))
    tau = 3.0 * j / (2.0 * BDOT_GAIN * moments.mean_b2_t2)
    t_env = np.linspace(0.0, bdot.t_s[-1], 400)
    envelope = np.degrees(rate0 * np.exp(-t_env / tau))

    fig, axes = plt.subplots(3, 1, figsize=(8.0, 8.4), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1.0, 1.0]})

    ax = axes[0]
    for r, key, label in (
        (bdot, "bdot", f"B-dot, k = {BDOT_GAIN:.2e} A m$^2$ s/T"),
        (cross, "cross", f"cross-product, k = {k_cross:.3e} N m s"),
    ):
        ax.semilogy(r.t_s / 60.0, np.degrees(r.rate_norm_rad_s), color=COLORS[key],
                    lw=1.2, label=label)
    ax.semilogy(t_env / 60.0, envelope, color=COLORS["analytic"], ls="--", lw=1.0,
                label=rf"first-order model, $\tau$ = {tau:.0f} s")
    ax.axhline(TARGET_DEG_S, color="#888888", ls=":", lw=1.0)
    ax.text(bdot.t_s[-1] / 60.0, TARGET_DEG_S * 1.12, f" target {TARGET_DEG_S} deg/s",
            ha="right", va="bottom", fontsize=8, color="#555555")
    floor = np.degrees(ORBIT.mean_motion_rad_s)
    ax.axhline(floor, color=COLORS["weak"], ls=":", lw=1.0)
    ax.text(bdot.t_s[-1] / 60.0, floor * 0.72,
            f" orbital rate {floor:.3f} deg/s: B-dot cannot go below about this",
            ha="right", va="top", fontsize=8, color=COLORS["weak"])
    if bdot.detumbled:
        ax.axvline(bdot.detumble_time_s / 60.0, color=COLORS["bdot"], lw=0.8, alpha=0.5)
        ax.text(bdot.detumble_time_s / 60.0, 20.0,
                f"  B-dot reaches target at {bdot.detumble_time_s:.0f} s",
                fontsize=8, color=COLORS["bdot"])
    ax.set_ylim(1e-4, 1e2)
    ax.set_ylabel(r"$|\omega|$  [deg/s]")
    ax.text(0.99, 0.22,
            "cross-product uses a rate estimate and keeps falling off the bottom of\n"
            "this axis; B-dot sees only dB/dt and floors near the orbital rate",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5,
            color="#444444")
    ax.set_title(
        "Detumbling a 3U-class spacecraft, 500 km sun-synchronous orbit\n"
        f"J = diag(0.05, 0.06, 0.04) kg m$^2$, 0.2 A m$^2$ per axis, "
        rf"$|\omega_0|$ = {np.degrees(rate0):.2f} deg/s",
        fontsize=10,
    )
    ax.legend(loc="lower left", fontsize=8)

    ax = axes[1]
    ax.plot(bdot.t_s / 60.0, np.abs(bdot.dipole_am2), lw=0.6, alpha=0.8,
            color=COLORS["bdot"])
    ax.axhline(0.2, color=COLORS["sat"], lw=1.2)
    ax.text(0.0, 0.205, " per-axis dipole limit 0.2 A m$^2$", fontsize=8,
            color="#8a6d00", va="bottom")
    ax.set_ylabel("|dipole| per axis\n[A m$^2$]")
    ax.set_ylim(0.0, 0.26)
    sat_frac = 100.0 * bdot.saturated_fraction
    ax.text(0.99, 0.9, f"saturated on {sat_frac:.1f} % of control steps",
            transform=ax.transAxes, ha="right", va="top", fontsize=8)

    ax = axes[2]
    ax.semilogy(bdot.t_s / 60.0, bdot.energy_j, color=COLORS["bdot"], lw=1.2,
                label="rotational kinetic energy [J]")
    ax.semilogy(bdot.t_s / 60.0, bdot.h_norm_nms, color=COLORS["cross"], lw=1.2,
                label=r"$|J\omega|$ [N m s]")
    ax.set_xlabel("time [minutes]")
    ax.set_ylabel("energy, momentum")
    ax.legend(loc="upper right", fontsize=8)

    path = save(fig, "detumble_curve.png")
    print(f"saved {path}")
    print(f"B-dot detumble time      = {bdot.detumble_time_s:.1f} s "
          f"({bdot.detumble_time_s / ORBIT.period_s:.2f} orbits)")
    print(f"cross-product detumble   = {cross.detumble_time_s:.1f} s")
    t_model = detumble_time_first_order(
        j, BDOT_GAIN, moments, rate0, np.radians(TARGET_DEG_S)
    )
    print(f"first-order model (no saturation) = {t_model:.1f} s")
    print(f"B-dot saturated fraction = {bdot.saturated_fraction:.4f}")
    print(f"final |omega|            = {np.degrees(bdot.rate_norm_rad_s[-1]):.4f} deg/s "
          f"(orbital rate {np.degrees(ORBIT.mean_motion_rad_s):.4f} deg/s)")


if __name__ == "__main__":
    main()
