"""Example 2 — Monte Carlo NEES/NIS with chi-squared bounds, correct vs mis-specified.

Saves ``../screenshots/nees_nis_consistency.png``.

This is the diagnostic most filter tools omit.  RMSE does rank the three
filters, but it does not say *which way* each is wrong: the ``Q`` too small
case is over-confident (NEES far above its bound) and the ``Q`` too large case
is under-confident (NEES below it).  Downstream fusion cares about that
distinction and RMSE never reveals it.
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
    KalmanFilter,
    constant_velocity_cwna,
    ensemble_consistency,
    nees,
    nis,
    simulate_linear_system,
)

DT = 1.0
Q_TRUE = 0.05
SIGMA_Z = 3.0
N_STEPS = 150
N_RUNS = 60
BURN_IN = 20
SEED = 90210


def ensemble(q_factor: float):
    f, q_true = constant_velocity_cwna(DT, Q_TRUE)
    _, q_filter = constant_velocity_cwna(DT, Q_TRUE * q_factor)
    h = np.array([[1.0, 0.0]])
    r = np.array([[SIGMA_Z**2]])
    p0 = np.diag([100.0, 10.0])
    nees_runs = np.zeros((N_RUNS, N_STEPS))
    nis_runs = np.zeros((N_RUNS, N_STEPS))
    rmse = np.zeros(N_RUNS)
    for i in range(N_RUNS):
        rng = np.random.default_rng(SEED + i)
        truth, meas = simulate_linear_system(
            f, h, q_true, r, np.array([0.0, 1.0]), N_STEPS, rng
        )
        kf = KalmanFilter(f, h, q_filter, r, np.zeros(2), p0)
        res = kf.run(meas)
        nees_runs[i] = nees(truth - res.x_post, res.p_post)
        nis_runs[i] = nis(res.innovation, res.innovation_cov)
        rmse[i] = float(np.sqrt(np.mean((truth[BURN_IN:, 0] - res.x_post[BURN_IN:, 0]) ** 2)))
    return nees_runs, nis_runs, rmse


def main() -> int:
    cases = [
        ("Q correct", 1.0, "tab:green"),
        ("Q too small (1/25)", 1.0 / 25.0, "tab:red"),
        ("Q too large (25x)", 25.0, "tab:blue"),
    ]
    t = np.arange(BURN_IN, N_STEPS) * DT
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    for ax, (idx, dof, label) in zip(
        axes[:2], [(0, 2, "NEES (dof 2)"), (1, 1, "NIS (dof 1)")], strict=True
    ):
        for name, qf, c in cases:
            nees_runs, nis_runs, _ = ensemble(qf)
            src = nees_runs if idx == 0 else nis_runs
            avg, lo, hi = ensemble_consistency(src[:, BURN_IN:], dof)
            ax.semilogy(t, avg, color=c, lw=1.1, label=f"{name}: mean {np.mean(avg):.2f}")
        ax.axhspan(lo, hi, color="k", alpha=0.12,
                   label=f"95 % band over M={N_RUNS} runs\n[{lo:.3f}, {hi:.3f}]")
        ax.axhline(dof, color="k", ls="--", lw=1.2, label=f"expectation = {dof}")
        ax.set_xlabel("time [s]")
        ax.set_ylabel(f"ensemble-average {label.split()[0]}")
        ax.set_title(f"{label} — {N_RUNS} independent runs")
        ax.legend(fontsize=7.5, loc="upper right")
        ax.grid(alpha=0.3, which="both")

    ax = axes[2]
    rmses = []
    for _name, qf, _c in cases:
        _, _, rm = ensemble(qf)
        rmses.append(rm)
    bp = ax.boxplot(rmses, tick_labels=[n for n, _, _ in cases], patch_artist=True)
    for patch, (_, _, c) in zip(bp["boxes"], cases, strict=True):
        patch.set_facecolor(c)
        patch.set_alpha(0.5)
    ax.set_ylabel("position RMSE [m]")
    ax.set_title("RMSE ranks them but hides the failure mode")
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(alpha=0.3, axis="y")
    for i, rm in enumerate(rmses):
        ax.text(i + 1, np.max(rm) * 1.02, f"{np.mean(rm):.3f}", ha="center", fontsize=8)

    fig.suptitle(
        "navbench — filter consistency: NEES/NIS against chi-squared bounds", fontsize=13
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = Path(__file__).resolve().parents[1] / "screenshots" / "nees_nis_consistency.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"saved {out}")
    for (name, _, _), rm in zip(cases, rmses, strict=True):
        print(f"  {name:<22s} mean position RMSE {np.mean(rm):.4f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
