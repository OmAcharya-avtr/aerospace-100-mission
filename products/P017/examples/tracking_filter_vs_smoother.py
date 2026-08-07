"""Example 1 — filter versus RTS smoother on a constant-velocity track.

Scenario: a 1-D constant-velocity target (continuous white-noise
acceleration, Bar-Shalom, Rong Li & Kirubarajan 2001, Ch. 6) observed by a
position sensor with 2 m standard deviation once per second for 300 s. The
same seeded data is passed through the forward Kalman filter and then
through the Rauch-Tung-Striebel fixed-interval smoother.

Run from the product root::

    PYTHONPATH=src python examples/tracking_filter_vs_smoother.py

Writes ``screenshots/tracking_filter_vs_smoother.png`` and prints the RMS
errors. Runtime: well under one second on the 2-core build environment.
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; never calls show()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from estimkit import KalmanFilter, constant_velocity_cwna, rts_smooth  # noqa: E402

DT = 1.0  # s
Q_PSD = 0.01  # m^2/s^3
R_VAR = 4.0  # m^2 (sigma = 2 m)
STEPS = 300
SEED = 2026

OUT = pathlib.Path(__file__).resolve().parent.parent / "screenshots"


def simulate() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (F, H, Q, R, truth, measurements) for the seeded scenario."""
    f, q = constant_velocity_cwna(DT, Q_PSD)
    h = np.array([[1.0, 0.0]])
    r = np.array([[R_VAR]])
    rng = np.random.default_rng(SEED)
    chol = np.linalg.cholesky(q)
    x = np.array([0.0, 10.0])  # 10 m/s
    truth = np.empty((STEPS, 2))
    for k in range(STEPS):
        x = f @ x + chol @ rng.standard_normal(2)
        truth[k] = x
    z = truth[:, 0:1] + np.sqrt(R_VAR) * rng.standard_normal((STEPS, 1))
    return f, h, q, r, truth, z


def main() -> None:
    f, h, q, r, truth, z = simulate()
    kf = KalmanFilter(f, h, q, r)
    res = kf.filter(np.array([0.0, 0.0]), np.diag([100.0, 100.0]), z)
    sm = rts_smooth(res)

    t = np.arange(STEPS) * DT
    err_f = res.x_post - truth
    err_s = sm.x - truth
    rms_f = np.sqrt(np.mean(err_f**2, axis=0))
    rms_s = np.sqrt(np.mean(err_s**2, axis=0))

    sigma_f = np.sqrt(np.array([np.diag(p) for p in res.p_post]))
    sigma_s = np.sqrt(np.array([np.diag(p) for p in sm.p]))

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)

    # Panel 1: position with the nominal 10 m/s ramp removed, otherwise the
    # 3 km of travel hides the metre-level differences under comparison.
    ramp = 10.0 * (t + DT)
    ax = axes[0]
    ax.plot(t, z[:, 0] - ramp, ".", ms=3, color="0.65", label="measurements")
    ax.plot(t, truth[:, 0] - ramp, "k-", lw=1.4, label="truth")
    ax.plot(t, res.x_post[:, 0] - ramp, "-", lw=1.0, color="tab:blue", label="filter")
    ax.plot(t, sm.x[:, 0] - ramp, "-", lw=1.2, color="tab:red", label="RTS smoother")
    ax.set_ylabel("position - 10 m/s ramp [m]")
    ax.set_title(
        "Constant-velocity tracking: forward filter vs RTS smoother "
        f"(seed {SEED}, sigma_z = {np.sqrt(R_VAR):.0f} m, {STEPS} steps)"
    )
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(t, err_f[:, 0], color="tab:blue", lw=0.9,
            label=f"filter error (RMS {rms_f[0]:.3f} m)")
    ax.plot(t, err_s[:, 0], color="tab:red", lw=0.9,
            label=f"smoother error (RMS {rms_s[0]:.3f} m)")
    ax.fill_between(t, -sigma_f[:, 0], sigma_f[:, 0], color="tab:blue", alpha=0.15,
                    label="filter +/- 1 sigma")
    ax.fill_between(t, -sigma_s[:, 0], sigma_s[:, 0], color="tab:red", alpha=0.15,
                    label="smoother +/- 1 sigma")
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_ylabel("position error [m]")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(t, err_f[:, 1], color="tab:blue", lw=0.9,
            label=f"filter error (RMS {rms_f[1]:.3f} m/s)")
    ax.plot(t, err_s[:, 1], color="tab:red", lw=0.9,
            label=f"smoother error (RMS {rms_s[1]:.3f} m/s)")
    ax.fill_between(t, -sigma_f[:, 1], sigma_f[:, 1], color="tab:blue", alpha=0.15)
    ax.fill_between(t, -sigma_s[:, 1], sigma_s[:, 1], color="tab:red", alpha=0.15)
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_ylabel("velocity error [m/s]")
    ax.set_xlabel("time [s]")
    # The first few steps carry the initialisation transient from P0 = 100;
    # clip the axis to the steady-state range so the comparison is readable.
    span = 1.3 * float(np.max(np.abs(err_f[5:, 1])))
    ax.set_ylim(-span, span)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    path = OUT / "tracking_filter_vs_smoother.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)

    print(f"seed                        : {SEED}")
    print(f"RMS position filter   [m]   : {rms_f[0]:.6f}")
    print(f"RMS position smoother [m]   : {rms_s[0]:.6f}"
          f"   ({100.0 * (1 - rms_s[0] / rms_f[0]):.2f} % reduction)")
    print(f"RMS velocity filter   [m/s] : {rms_f[1]:.6f}")
    print(f"RMS velocity smoother [m/s] : {rms_s[1]:.6f}"
          f"   ({100.0 * (1 - rms_s[1] / rms_f[1]):.2f} % reduction)")
    print(f"mean NIS (m = 1 dof)        : {float(np.mean(res.nis)):.4f}")
    print(f"figure                      : {path}")


if __name__ == "__main__":
    main()
