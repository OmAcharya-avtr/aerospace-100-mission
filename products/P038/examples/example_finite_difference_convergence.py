"""Finite-differenced range-rate against the analytic dot product: convergence and floor.

This is the picture of Level 1 validation check 1. The central difference is
second-order accurate, so the error falls as h^2 until round-off takes over
at about h = 0.03 s, below which it rises as 1/h. Both regimes are plotted,
with the reference slopes.

Writes ../screenshots/finite_difference_convergence.png.
Runtime: under 2 s on one core.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dopplerkit.analytic import CircularOverheadPass
from dopplerkit.constants import WGS84_A_M
from dopplerkit.geometry import range_rate_finite_difference_mps

ALT_M = 500.0e3
EPOCHS_S = (-300.0, -120.0, 40.0, 250.0)
OUT = Path(__file__).resolve().parent.parent / "screenshots" / "finite_difference_convergence.png"


def main() -> int:
    orbit = CircularOverheadPass(orbit_radius_m=WGS84_A_M + ALT_M)
    steps = np.logspace(np.log10(20.0), np.log10(1.0e-4), 60)

    fig, ax = plt.subplots(figsize=(9.0, 6.4))
    colors = ("#1f4e79", "#c07000", "#2e7d32", "#b03030")

    print(f"{'epoch [s]':>10} {'best step [s]':>14} {'best error [m/s]':>18} "
          f"{'order 20 s -> 1 s':>20}")
    for epoch, color in zip(EPOCHS_S, colors, strict=True):
        exact = orbit.range_rate_mps(epoch)
        errs = np.array([
            abs(range_rate_finite_difference_mps(orbit.range_m, epoch, float(h)) - exact)
            for h in steps
        ])
        ax.loglog(steps, np.maximum(errs, 1e-14), "o-", ms=2.6, lw=1.1, color=color,
                  label=f"t = {epoch:+.0f} s")
        best = int(np.argmin(errs))
        i20 = int(np.argmin(np.abs(steps - 20.0)))
        i1 = int(np.argmin(np.abs(steps - 1.0)))
        order = np.log(errs[i20] / errs[i1]) / np.log(steps[i20] / steps[i1])
        print(f"{epoch:10.1f} {steps[best]:14.6f} {errs[best]:18.4e} {order:20.4f}")

    ref_h = np.array([20.0, 0.03])
    ref_e = 5.0e-2 * (ref_h / 1.0) ** 2
    ax.loglog(ref_h, ref_e, "k--", lw=1.0, alpha=0.7, label=r"slope 2 (truncation $\propto h^2$)")
    ref_h2 = np.array([0.03, 1.0e-4])
    ref_e2 = 2.0e-8 * (0.03 / ref_h2)
    ax.loglog(ref_h2, ref_e2, "k:", lw=1.2, alpha=0.7,
              label=r"slope $-1$ (round-off $\propto \epsilon\rho/h$)")

    ax.set_xlabel("central-difference step $h$  [s]")
    ax.set_ylabel(r"$|\dot{\rho}_{\rm fd} - \dot{\rho}_{\rm analytic}|$  [m/s]")
    ax.set_title(
        f"Finite-differenced vs analytic range-rate, {ALT_M / 1e3:.0f} km overhead pass"
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax.invert_xaxis()

    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=130)
    plt.close(fig)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
