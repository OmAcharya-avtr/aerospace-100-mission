"""Saturation: what each allocator does as the command leaves the attainable set.

Sweeps a commanded torque along a fixed direction from zero out to twice the
attainable-moment-set boundary, on the eight-thruster reference cluster, and
records what each method returns. Then repeats the sweep after two thrusters
have failed, to show the boundary move in.

Saves ``screenshots/saturation_sweep.png``.

Run: ``python examples/saturation_demo.py``
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from alloclab.allocation import METHODS, allocate  # noqa: E402
from alloclab.ams import attainable_moment_set  # noqa: E402
from alloclab.dataset import reference_thruster_cluster  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "screenshots" / "saturation_sweep.png"

DIRECTION = np.array([0.6, -0.5, 0.62])
DIRECTION /= np.linalg.norm(DIRECTION)
N_STEPS = 220
STYLE = {
    "pinv": ("#8c8c8c", "--"),
    "wpinv": ("#5b8ff9", "--"),
    "rpi": ("#d98324", "-."),
    "lp": ("#3f7d54", ":"),
    "qp": ("#a63a3a", "-"),
}


def sweep(eset, max_mag=None):
    ams = attainable_moment_set(eset)
    boundary = ams.boundary_scale(DIRECTION)
    mags = np.linspace(0.0, 2.0 * boundary if max_mag is None else max_mag, N_STEPS)
    resid = {m: np.zeros(N_STEPS) for m in METHODS}
    viol = {m: np.zeros(N_STEPS) for m in METHODS}
    for k, mag in enumerate(mags):
        tau = mag * DIRECTION
        for method in METHODS:
            res = allocate(eset, tau, method=method)
            resid[method][k] = res.residual_norm
            viol[method][k] = res.bound_violation
    return mags, boundary, resid, viol


def main() -> None:
    nominal = reference_thruster_cluster(max_thrust=1.0, arm=0.5)
    degraded = nominal.with_failures([0, 1])

    mags_n, bound_n, resid_n, viol_n = sweep(nominal)
    # Same magnitude axis on both panels, so the boundary moving in is visible.
    mags_d, bound_d, resid_d, viol_d = sweep(degraded, max_mag=mags_n[-1])

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.2), sharex="col")

    for col, (mags, bound, resid, viol, title) in enumerate(
        [
            (mags_n, bound_n, resid_n, viol_n, "nominal, 8 thrusters"),
            (mags_d, bound_d, resid_d, viol_d, "t1 and t2 (the +/-x couple) failed off"),
        ]
    ):
        ax = axes[0][col]
        for method in METHODS:
            colour, ls = STYLE[method]
            ax.plot(mags, resid[method], color=colour, ls=ls, lw=1.7, label=method)
        ax.axvline(bound, color="k", lw=1.0, ls="-")
        ax.text(
            bound, ax.get_ylim()[1] * 0.55, f"  AMS boundary\n  {bound:.4f} N$\\cdot$m",
            fontsize=8, va="top",
        )
        ax.set_ylabel("torque residual $\\|\\tau - Bu\\|$ [N$\\cdot$m]")
        ax.set_title(f"{title}", fontsize=11)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=2)

        ax = axes[1][col]
        for method in METHODS:
            colour, ls = STYLE[method]
            ax.plot(mags, viol[method], color=colour, ls=ls, lw=1.7, label=method)
        ax.axvline(bound, color="k", lw=1.0, ls="-")
        ax.set_xlabel("commanded torque magnitude [N$\\cdot$m]")
        ax.set_ylabel("max actuator bound violation [N]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=2)

    fig.suptitle(
        "Command swept along a fixed direction, out to twice the attainable boundary\n"
        "top: torque actually delivered falls short beyond the boundary; "
        "bottom: only pinv and wpinv answer with commands the actuators cannot execute",
        fontsize=11,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130)
    plt.close(fig)

    print(f"saved {OUT}")
    print(f"direction = {np.array2string(DIRECTION, precision=6)}")
    print(f"nominal AMS boundary along it  : {bound_n:.6f} N*m")
    print(f"degraded AMS boundary along it : {bound_d:.6f} N*m "
          f"({bound_d / bound_n:.4f} of nominal)")
    print()
    print(f"{'method':<8}{'max bound viol, nominal [N]':>30}"
          f"{'residual at 2x boundary [N*m]':>32}")
    for method in METHODS:
        print(f"{method:<8}{viol_n[method].max():>30.6e}{resid_n[method][-1]:>32.6e}")
    print()
    print("At 2x the boundary no in-box command can deliver the torque, so the")
    print("residual of lp/qp/rpi is the true shortfall; pinv and wpinv appear to")
    print("have zero residual only because their commands are not executable.")


if __name__ == "__main__":
    main()
