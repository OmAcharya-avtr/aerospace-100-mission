"""Example 1 — run KF, EKF and UKF over one radar tracking truth and score them.

Saves ``../screenshots/estimator_bench.png``.

The linear KF is fed the measurement converted to Cartesian, which is the
common shortcut; the plot shows what that costs in both error and consistency.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from navbench import (  # noqa: E402
    ExtendedKalmanFilter,
    KalmanFilter,
    UnscentedKalmanFilter,
    chi2_bounds,
    compare_scores,
    constant_velocity_2d,
    nees,
    radar_jacobian,
    radar_measurement,
    score_run,
    simulate_radar_scenario,
)

DT = 1.0
N_STEPS = 200
Q_PSD = 0.05
SIGMA_R = 20.0
SIGMA_B = 0.01
SEED = 2026
BURN_IN = 20


def main() -> int:
    rng = np.random.default_rng(SEED)
    x0 = np.array([3000.0, -5.0, 3000.0, 3.0])
    truth, meas = simulate_radar_scenario(
        dt=DT, n_steps=N_STEPS, q_psd=Q_PSD, sigma_range=SIGMA_R,
        sigma_bearing=SIGMA_B, x0=x0, rng=rng,
    )
    f, q = constant_velocity_2d(DT, Q_PSD)
    r = np.diag([SIGMA_R**2, SIGMA_B**2])
    p0 = np.diag([SIGMA_R**2, 100.0, SIGMA_R**2, 100.0])
    x_init = np.array([truth[0, 0], 0.0, truth[0, 2], 0.0])

    ekf = ExtendedKalmanFilter(
        lambda x: f @ x, radar_measurement, q, r, x_init, p0,
        f_jac=lambda x: f, h_jac=radar_jacobian,
    )
    res_e = ekf.run(meas)
    ukf = UnscentedKalmanFilter(lambda x: f @ x, radar_measurement, q, r, x_init, p0)
    res_u = ukf.run(meas)
    z_cart = np.column_stack(
        [meas[:, 0] * np.cos(meas[:, 1]), meas[:, 0] * np.sin(meas[:, 1])]
    )
    h_lin = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    kf = KalmanFilter(f, h_lin, q, np.diag([SIGMA_R**2, SIGMA_R**2]), x_init, p0)
    res_k = kf.run(z_cart)

    runs = [
        ("EKF", res_e, "tab:blue"),
        ("UKF", res_u, "tab:orange"),
        ("KF (converted)", res_k, "tab:green"),
    ]
    scores = [
        score_run(name, truth, res.x_post, res.p_post, res.innovation,
                  res.innovation_cov, burn_in=BURN_IN)
        for name, res, _ in runs
    ]
    print(compare_scores(scores))

    t = np.arange(N_STEPS) * DT
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))

    ax = axes[0, 0]
    ax.plot(truth[:, 0] / 1e3, truth[:, 2] / 1e3, "k-", lw=1.8, label="truth")
    ax.plot(z_cart[:, 0] / 1e3, z_cart[:, 1] / 1e3, ".", ms=2.5, color="0.6",
            label="measurements (polar -> Cartesian)")
    for name, res, c in runs:
        ax.plot(res.x_post[:, 0] / 1e3, res.x_post[:, 2] / 1e3, "-", lw=1.0,
                color=c, label=name)
    ax.plot(0, 0, "r*", ms=12, label="sensor")
    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km]")
    ax.set_title("Radar tracking geometry")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")

    ax = axes[0, 1]
    for name, res, c in runs:
        e = np.hypot(truth[:, 0] - res.x_post[:, 0], truth[:, 2] - res.x_post[:, 2])
        ax.plot(t, e, color=c, lw=1.0, label=name)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("position error [m]")
    ax.set_title("Position error magnitude")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    lo1, hi1 = chi2_bounds(4, 1)
    for name, res, c in runs:
        ax.semilogy(t, nees(truth - res.x_post, res.p_post), color=c, lw=0.8, label=name)
    ax.axhline(4.0, color="k", ls="--", lw=1.2, label="E[NEES] = n = 4")
    ax.axhspan(lo1, hi1, color="k", alpha=0.08,
               label=f"single-sample 95 % band [{lo1:.2f}, {hi1:.2f}]")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("NEES")
    ax.set_title("NEES: does the filter know how wrong it is?")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3, which="both")

    ax = axes[1, 1]
    names = [s.name for s in scores]
    xpos = np.arange(len(names))
    ax.bar(xpos - 0.2, [s.rmse_total for s in scores], 0.4, label="total RMSE",
           color="tab:blue")
    ax2 = ax.twinx()
    ax2.bar(xpos + 0.2, [s.mean_nees for s in scores], 0.4, label="mean NEES",
            color="tab:red")
    ax2.axhline(4.0, color="k", ls="--", lw=1.2)
    ax.set_xticks(xpos)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("RMSE (state units)", color="tab:blue")
    ax2.set_ylabel("mean NEES (dashed line = dof 4)", color="tab:red")
    ax.set_title("Error and consistency side by side")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle(
        f"navbench estimator bench - radar tracking, seed {SEED}, {N_STEPS} steps",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = Path(__file__).resolve().parents[1] / "screenshots" / "estimator_bench.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"\nsaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
