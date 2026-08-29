"""Example 4 — adaptive process-noise tuning: learned vs fixed vs classical Mehra.

Saves ``../screenshots/adaptive_q_tuning.png``.

The truth's acceleration PSD is drawn per run from 1/32 to 32 times the value
the filter was hand-tuned for.  All three tuners adjust the same single scalar
``lambda`` with ``Q = lambda * Q_nominal``.

The figure deliberately shows both the error metric and the consistency
metric, because on this benchmark they pick different winners.
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
    LearnedAdaptiveQ,
    chi2_bounds,
    constant_velocity_cwna,
    generate_adaptive_dataset,
    nees,
    run_adaptive_kf,
    simulate_linear_system,
)

SEED = 20260812
N_TRAIN = 120
N_TEST = 40
N_STEPS = 400
Q_NOM = 0.05
SIGMA_Z = 3.0
BURN_IN = 60
TUNERS = ("fixed", "mehra", "learned")
COLORS = {"fixed": "tab:gray", "mehra": "tab:orange", "learned": "tab:blue"}


def main() -> int:
    x_train, y_train, _ = generate_adaptive_dataset(
        n_runs=N_TRAIN, n_steps=N_STEPS, q_nominal_psd=Q_NOM, sigma_z=SIGMA_Z, seed=SEED
    )
    model = LearnedAdaptiveQ(n_members=5, random_state=SEED).fit(x_train, y_train)
    print(f"trained on {x_train.shape[0]} windows from {N_TRAIN} runs")

    f, q_nom = constant_velocity_cwna(1.0, Q_NOM)
    h = np.array([[1.0, 0.0]])
    r = np.array([[SIGMA_Z**2]])
    p0 = np.diag([100.0, 10.0])

    rmse = {t: [] for t in TUNERS}
    nees_m = {t: [] for t in TUNERS}
    lam = {t: [] for t in TUNERS}
    us = []
    trace_run = None
    for i in range(N_TEST):
        rng = np.random.default_rng(SEED + 100000 + i)
        u = float(rng.uniform(-1.5, 1.5))
        us.append(u)
        _, q_true = constant_velocity_cwna(1.0, Q_NOM * 10.0**u)
        truth, meas = simulate_linear_system(
            f, h, q_true, r, np.array([0.0, 1.0]), N_STEPS, rng
        )
        for t in TUNERS:
            res = run_adaptive_kf(
                f=f, h=h, q_nominal=q_nom, r=r, x0=np.zeros(2), p0=p0,
                measurements=meas, tuner=t, model=model if t == "learned" else None,
            )
            e = truth[BURN_IN:] - res.states[BURN_IN:]
            rmse[t].append(float(np.sqrt(np.mean(e[:, 0] ** 2))))
            nees_m[t].append(float(np.mean(nees(e, res.covariances[BURN_IN:]))))
            lam[t].append(float(res.scales[-1]))
            if i == 0:
                trace_run = trace_run or {}
                trace_run[t] = res
        if i == 0:
            trace_run["u"] = u  # type: ignore[index]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    ax = axes[0, 0]
    xpos = np.arange(len(TUNERS))
    ax.bar(xpos, [np.mean(rmse[t]) for t in TUNERS],
           yerr=[1.96 * np.std(rmse[t], ddof=1) / np.sqrt(N_TEST) for t in TUNERS],
           color=[COLORS[t] for t in TUNERS], capsize=4)
    ax.set_xticks(xpos)
    ax.set_xticklabels(TUNERS)
    ax.set_ylabel("position RMSE [m]")
    ax.set_title(f"Held-out position RMSE ({N_TEST} runs, 95 % CI on the mean)")
    ax.grid(alpha=0.3, axis="y")
    for i, t in enumerate(TUNERS):
        ax.text(i, np.mean(rmse[t]) * 1.01, f"{np.mean(rmse[t]):.3f}", ha="center", fontsize=9)

    ax = axes[0, 1]
    lo, hi = chi2_bounds(2, N_TEST)
    ax.bar(xpos, [np.mean(nees_m[t]) for t in TUNERS],
           color=[COLORS[t] for t in TUNERS])
    ax.axhspan(lo, hi, color="k", alpha=0.15, label=f"95 % band [{lo:.2f}, {hi:.2f}]")
    ax.axhline(2.0, color="k", ls="--", lw=1.2, label="E[NEES] = 2")
    ax.set_xticks(xpos)
    ax.set_xticklabels(TUNERS)
    ax.set_ylabel("mean NEES (dof 2)")
    ax.set_title("Held-out consistency — a different winner")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    for i, t in enumerate(TUNERS):
        ax.text(i, np.mean(nees_m[t]) * 1.01, f"{np.mean(nees_m[t]):.2f}", ha="center",
                fontsize=9)

    ax = axes[1, 0]
    u_arr = np.array(us)
    for t in ("mehra", "learned"):
        ax.plot(u_arr, np.log10(lam[t]), "o", ms=4, color=COLORS[t], label=t, alpha=0.75)
    lim = [-1.7, 1.7]
    ax.plot(lim, lim, "k--", lw=1.0, label="perfect recovery")
    ax.axhline(0.0, color="0.6", lw=0.8, ls=":", label="fixed tuner (always 0)")
    ax.set_xlim(lim)
    ax.set_xlabel("true log10(q_true / q_nominal)")
    ax.set_ylabel("estimated log10(lambda)")
    ax.set_title("Recovery of the true process-noise scale")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    assert trace_run is not None
    steps = np.arange(N_STEPS)
    for t in ("mehra", "learned"):
        ax.step(steps, trace_run[t].scales, where="post", color=COLORS[t], lw=1.2, label=t)
    ax.axhline(1.0, color="0.5", ls=":", lw=1.0, label="fixed (lambda = 1)")
    ax.axhline(10.0 ** trace_run["u"], color="k", ls="--", lw=1.2,
               label=f"truth 10^u = {10.0 ** trace_run['u']:.3f}")
    ax.set_yscale("log")
    ax.set_xlabel("filter step")
    ax.set_ylabel("lambda in force")
    ax.set_title("One held-out run: the scale each tuner applies over time")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    fig.suptitle(
        "navbench adaptive Q — learned tuner vs fixed hand-tuned Q vs classical "
        "Mehra IAE (held-out runs)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    out = Path(__file__).resolve().parents[1] / "screenshots" / "adaptive_q_tuning.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    for t in TUNERS:
        print(f"  {t:<9s} RMSE {np.mean(rmse[t]):.4f} m   mean NEES {np.mean(nees_m[t]):.4f}")
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
