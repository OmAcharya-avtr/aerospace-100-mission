"""Example 5 — gyro Allan deviation against the IEEE Std 952-2020 analytic form.

Saves ``../screenshots/gyro_allan_deviation.png``.

Shows the two-term model implemented by :class:`navbench.GyroModel`:
angle random walk (slope −1/2) and rate random walk (slope +1/2), with the
minimum at ``tau* = sqrt(3) sigma_v / sigma_u``.  The absence of a flat
bias-instability (flicker) plateau is a documented limitation of the model,
not a plotting artefact.
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
    arw_deg_per_sqrt_hour_to_si,
    rrw_deg_per_hour_1p5_to_si,
)

DT = 0.01
N = 300000
SEED = 24680


def overlapping_allan(theta: np.ndarray, dt: float, m: int) -> float:
    """Overlapping Allan deviation at cluster size ``m`` (IEEE Std 952-2020 Eq. C.9)."""
    n = theta.size
    d = theta[2 * m :] - 2.0 * theta[m : n - m] + theta[: n - 2 * m]
    tau = m * dt
    return float(np.sqrt(np.sum(d * d) / (2.0 * tau * tau * d.size)))


def main() -> int:
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    specs = [
        ("navigation grade  (ARW 0.002 deg/sqrt(hr), RRW 0.002 deg/hr^1.5)", 0.002, 0.002),
        ("tactical grade    (ARW 0.05 deg/sqrt(hr),  RRW 0.5 deg/hr^1.5)", 0.05, 0.5),
        ("MEMS grade        (ARW 1.0 deg/sqrt(hr),   RRW 20 deg/hr^1.5)", 1.0, 20.0),
    ]
    ms = np.unique(np.round(np.logspace(0.3, 4.0, 22)).astype(int))
    ms = ms[ms >= 2]
    ax = axes[0]
    for label, arw, rrw in specs:
        sv = arw_deg_per_sqrt_hour_to_si(arw)
        su = rrw_deg_per_hour_1p5_to_si(rrw)
        gyro = GyroModel(sigma_v=sv, sigma_u=su, dt=DT, bias0=np.zeros(3))
        rates, _ = gyro.sample_series(np.zeros((N, 3)), rng)
        theta = np.cumsum(rates[:, 0]) * DT
        taus = ms * DT
        meas = np.array([overlapping_allan(theta, DT, int(m)) for m in ms if 2 * m < N])
        taus = taus[: meas.size]
        theo = np.sqrt(sv**2 / taus + su**2 * taus / 3.0)
        line, = ax.loglog(taus, meas, "o", ms=3.5, label=f"{label} — measured")
        ax.loglog(taus, theo, "-", lw=1.2, color=line.get_color(),
                  label="                          analytic IEEE 952 form")
        tau_star = np.sqrt(3.0) * sv / su
        if taus[0] <= tau_star <= taus[-1]:
            ax.axvline(tau_star, color=line.get_color(), ls=":", lw=0.8)
    ax.set_xlabel(r"cluster time $\tau$ [s]")
    ax.set_ylabel(r"Allan deviation $\sigma_A(\tau)$ [rad/s]")
    ax.set_title("Gyro Allan deviation: model vs analytic form")
    ax.legend(fontsize=6.5, loc="lower left")
    ax.grid(alpha=0.3, which="both")

    # Residual panel for the tactical-grade case.
    ax = axes[1]
    sv = arw_deg_per_sqrt_hour_to_si(0.05)
    su = rrw_deg_per_hour_1p5_to_si(0.5)
    gyro = GyroModel(sigma_v=sv, sigma_u=su, dt=DT, bias0=np.zeros(3))
    rates, biases = gyro.sample_series(np.zeros((N, 3)), rng)
    t = np.arange(N) * DT
    for i, c in enumerate(("tab:blue", "tab:orange", "tab:green")):
        ax.plot(t, biases[:, i] * 1e6, color=c, lw=0.8, label=f"bias axis {'xyz'[i]}")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("gyro bias [micro-rad/s]")
    ax.set_title(
        f"Bias random walk realisation (sigma_u = {su:.3e} rad/s^1.5)\n"
        f"theoretical std at t = {N * DT:.0f} s: "
        f"{su * np.sqrt(N * DT) * 1e6:.3f} micro-rad/s"
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(
        "navbench gyro model — IEEE Std 952-2020 angle and rate random walk", fontsize=12
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = Path(__file__).resolve().parents[1] / "screenshots" / "gyro_allan_deviation.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    measured_std = float(np.std(biases[-1]))
    print(f"final bias spread across axes: {measured_std * 1e6:.4f} micro-rad/s, "
          f"theory {su * np.sqrt(N * DT) * 1e6:.4f}")
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
