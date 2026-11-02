"""Equation (14): the single overrun threshold below which no miss needs looking at.

What to notice: the threshold falls like the logarithm of the reciprocal of the
input step, so a miss during a quiet stretch of a run can be an order of
magnitude longer than one during a transient and still be provably harmless.
The horizon barely matters, because the closed loop's worst amplification is
reached within ten sample instants.
"""

from __future__ import annotations

import sys  # noqa: E402

sys.dont_write_bytecode = True

import math  # noqa: E402

import numpy as np  # noqa: E402

from _common import CERTIFIED, NEUTRAL, TOLERANCE, UNDECIDED, bootstrap, save, style  # noqa: E402

ROOT = bootstrap()

import matplotlib.pyplot as plt  # noqa: E402

from hilverdict.certificate import (  # noqa: E402
    CertificateModel,
    certified_overrun_s,
    divergence_bound,
)
from hilverdict.plant import LoopModel  # noqa: E402

STEPS = 200


def main() -> int:
    style()
    model = LoopModel()
    constants = CertificateModel.build(model, STEPS)
    period = model.sampling.period_s
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))

    steps_deg = np.logspace(-2, 1.4, 160)
    for horizon, colour, dash in (
        (5, UNDECIDED, ":"),
        (20, CERTIFIED, "--"),
        (STEPS, NEUTRAL, "-"),
    ):
        thresholds = [
            certified_overrun_s(constants, math.radians(d), horizon, 1.0) * 1e3
            for d in steps_deg
        ]
        axes[0].plot(
            steps_deg, thresholds, dash, color=colour,
            label=f"horizon {horizon} samples",
        )
    axes[0].axhline(period * 1e3, color=TOLERANCE, lw=0.9, ls="-.")
    axes[0].text(
        0.012, period * 1e3 * 1.08, "one sampling period", color=TOLERANCE, fontsize=8
    )
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("input step held through the overrun [deg]")
    axes[0].set_ylabel("certified overrun, eq (14) [ms]")
    axes[0].set_title("below this, a miss CANNOT have mattered")
    axes[0].legend(loc="lower left")

    overruns = np.linspace(0.0, 0.03, 300)
    for degrees, colour in ((0.05, CERTIFIED), (0.5, UNDECIDED), (5.0, NEUTRAL)):
        bounds = [
            divergence_bound(constants, tau, math.radians(degrees), STEPS)
            for tau in overruns
        ]
        axes[1].plot(
            overruns * 1e3, bounds, color=colour, label=f"input step {degrees:g} deg"
        )
    axes[1].axhline(1.0, color=TOLERANCE, lw=1.1, ls="--")
    axes[1].text(0.4, 1.25, "declared tolerance", color=TOLERANCE, fontsize=8)
    axes[1].set_yscale("log")
    axes[1].set_ylim(1e-4, 1e4)
    axes[1].set_xlabel("overrun [ms]")
    axes[1].set_ylabel("divergence bound, eq (13)")
    axes[1].set_title("the bound, and where it crosses the tolerance")
    axes[1].legend(loc="lower right")

    figure.suptitle(
        "hilverdict: the closed-form certificate, "
        f"anisotropy {model.metric.anisotropy:.0f}, "
        f"mu_S {constants.mu_s:.2f} 1/s, gamma_sup "
        f"{constants.gamma_sup(STEPS):.3f}",
        fontsize=10,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    path = save(figure, "certificate_threshold")
    print(f"written: {path}")
    print(
        f"certified overrun at a 1 deg input step: "
        f"{certified_overrun_s(constants, math.radians(1.0), STEPS, 1.0) * 1e3:.6f} ms"
    )
    print(
        f"at a 0.05 deg input step: "
        f"{certified_overrun_s(constants, math.radians(0.05), STEPS, 1.0) * 1e3:.6f} ms"
    )
    print(f"mu_S = {constants.mu_s:.6f} 1/s, gamma_sup = "
          f"{constants.gamma_sup(STEPS):.6f}, c_free = {constants.free_factor:.6f}")
    print(f"np version {np.__version__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
