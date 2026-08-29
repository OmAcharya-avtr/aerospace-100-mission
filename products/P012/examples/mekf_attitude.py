"""Example 3 — MEKF attitude and gyro-bias estimation with 3-sigma envelopes.

Saves ``../screenshots/mekf_attitude.png``.

Scenario: a rigid body under a small time-varying torque, a gyro with angle
random walk and rate random walk (IEEE Std 952-2020 terms), and a star tracker
supplying a full attitude quaternion every 4th step.
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
    GyroModel,
    MultiplicativeEKF,
    StarTrackerModel,
    arw_deg_per_sqrt_hour_to_si,
    attitude_trajectory,
    chi2_bounds,
    consistency_test,
    nees,
    quat_from_euler_zyx,
    quat_from_small_angle,
    quat_multiply,
    rrw_deg_per_hour_1p5_to_si,
)

DT = 0.5
N_STEPS = 800
SEED = 7
SIGMA_ST = 3e-5  # rad
ARW = 0.05  # deg/sqrt(hr)
RRW = 0.5  # deg/hr^1.5
SIG_A0 = 0.05  # rad
SIG_B0 = 2e-6  # rad/s
MEAS_EVERY = 4
BURN_IN = 100
RAD2ARCSEC = 180.0 / np.pi * 3600.0


def main() -> int:
    rng = np.random.default_rng(SEED)
    truth = attitude_trajectory(
        inertia=np.diag([10.0, 15.0, 20.0]),
        quat0=quat_from_euler_zyx(0.2, -0.1, 0.3),
        omega0=np.array([0.01, -0.02, 0.015]),
        dt=DT,
        n_steps=N_STEPS,
        torque_fn=lambda t, q, w: np.array([1e-5 * np.sin(0.01 * t), 0.0, 0.0]),
    )
    sigma_v = arw_deg_per_sqrt_hour_to_si(ARW)
    sigma_u = rrw_deg_per_hour_1p5_to_si(RRW)
    gyro = GyroModel(
        sigma_v=sigma_v, sigma_u=sigma_u, dt=DT, bias0=SIG_B0 * rng.standard_normal(3)
    )
    rates, biases = gyro.sample_series(truth.interval_rate(), rng)
    tracker = StarTrackerModel(sigma_rad=SIGMA_ST, reference_vectors=np.eye(3))
    q_meas = np.array([tracker.sample_quaternion(q, rng) for q in truth.quat[1:]])

    mekf = MultiplicativeEKF(
        sigma_v=sigma_v, sigma_u=sigma_u, dt=DT,
        quat0=quat_multiply(truth.quat[0], quat_from_small_angle(SIG_A0 * rng.standard_normal(3))),
        bias0=np.zeros(3),
        p0=np.diag([SIG_A0**2] * 3 + [SIG_B0**2] * 3),
    )
    res = mekf.run(rates, quat_meas=q_meas, sigma_rad=SIGMA_ST, measurement_every=MEAS_EVERY)
    err = res.error_state(truth.quat[1:], biases)
    sig = np.sqrt(np.diagonal(res.covariance, axis1=1, axis2=2))
    t = res.t

    nees_att = consistency_test(
        nees(err[BURN_IN:, :3], res.covariance[BURN_IN:, :3, :3]), 3,
        statistic="NEES", independent=False,
    )
    print(nees_att.summary())
    print(
        "  (single-run TIME average: successive steps are not independent, so the "
        "chi-squared\n   band above is indicative only. The defensible Monte Carlo "
        "form is in\n   validation/v4_mekf_quaternion.py PART E, where ANEES = 6.147 "
        "for dof 6.)"
    )
    print(f"attitude RMS after burn-in: "
          f"{np.sqrt(np.mean(np.sum(err[BURN_IN:, :3] ** 2, axis=1))) * RAD2ARCSEC:.2f} arcsec")
    print(f"max ||q| - 1| over the run: "
          f"{np.max(np.abs(np.linalg.norm(res.quat, axis=1) - 1.0)):.3e}")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    labels = ("x", "y", "z")
    colors = ("tab:blue", "tab:orange", "tab:green")

    ax = axes[0, 0]
    for i in range(3):
        ax.plot(t, err[:, i] * RAD2ARCSEC, color=colors[i], lw=0.8, label=f"error {labels[i]}")
        ax.plot(t, 3 * sig[:, i] * RAD2ARCSEC, color=colors[i], ls="--", lw=0.7)
        ax.plot(t, -3 * sig[:, i] * RAD2ARCSEC, color=colors[i], ls="--", lw=0.7)
    ax.set_ylim(-40, 40)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("attitude error [arcsec]")
    ax.set_title("Attitude error with filter 3-sigma envelope")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    for i in range(3):
        ax.plot(t, (biases[:, i] - res.bias[:, i]) * 1e6, color=colors[i], lw=0.8,
                label=f"bias error {labels[i]}")
        ax.plot(t, 3 * sig[:, 3 + i] * 1e6, color=colors[i], ls="--", lw=0.7)
        ax.plot(t, -3 * sig[:, 3 + i] * 1e6, color=colors[i], ls="--", lw=0.7)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("gyro bias error [micro-rad/s]")
    ax.set_title("Gyro bias estimation error with 3-sigma envelope")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    lo, hi = chi2_bounds(3, 1)
    ax.semilogy(t, nees(err[:, :3], res.covariance[:, :3, :3]), lw=0.6, color="tab:purple")
    ax.axhline(3.0, color="k", ls="--", lw=1.2, label="E[NEES] = 3")
    ax.axhspan(lo, hi, color="k", alpha=0.1, label=f"single-sample 95 % [{lo:.2f}, {hi:.2f}]")
    ax.axvline(BURN_IN * DT, color="r", ls=":", lw=1.0, label="burn-in ends")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("attitude-block NEES")
    ax.set_title(f"Attitude NEES — time mean {nees_att.mean:.3f} vs dof 3")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1, 1]
    upd = res.updated
    ax.semilogy(t[upd], res.reset_angle[upd] * RAD2ARCSEC, ".", ms=2.5, color="tab:red")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("reset angle |a_hat| [arcsec]")
    ax.set_title("Multiplicative reset angle folded into the reference quaternion")
    ax.grid(alpha=0.3, which="both")

    fig.suptitle(
        f"navbench MEKF — {N_STEPS} steps of {DT} s, star tracker every {MEAS_EVERY} steps "
        f"(sigma = {SIGMA_ST * RAD2ARCSEC:.1f} arcsec), gyro ARW {ARW} deg/sqrt(hr)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    out = Path(__file__).resolve().parents[1] / "screenshots" / "mekf_attitude.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
