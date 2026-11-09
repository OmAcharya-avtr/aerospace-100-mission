"""Two different faults, one rate record: the unidentifiable subspace, drawn.

What to notice: the top panel shows the rate record of a wheel whose inertia has
risen by 8 per cent in logarithm and of a wheel whose other four parameters have
each fallen by the same amount. The two curves are the same curve. The bottom
panel is their difference, which for this pair is EXACTLY zero at every one of
the 2001 samples -- the two trajectories are bit-identical -- while the dashed
band is the declared sensor noise. Over the whole 120-case test population the
worst difference is two units in the last place of a 64-bit float, which
``validation/validate_attribution.py`` measures. No estimator can separate these
two faults from this record, and that is a proof rather than a difficulty.
"""

from __future__ import annotations

import sys  # noqa: E402

sys.dont_write_bytecode = True

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from _common import ANALYTIC, LEARNED, REFUSED, bootstrap, save, style  # noqa: E402

ROOT = bootstrap()

from twindiverge.plant import NOMINAL, PARAMETER_NAMES, perturb, simulate  # noqa: E402
from twindiverge.telemetry import RATE_SIGMA_RAD_S, TELEMETRY_SETS  # noqa: E402

AMOUNT = 0.08


def main() -> int:
    style()
    telemetry = TELEMETRY_SETS["rate_full"]
    campaign = telemetry.campaign
    first = np.array([AMOUNT, 0.0, 0.0, 0.0, 0.0])
    second = first - AMOUNT * np.ones(5)
    left = simulate(perturb(NOMINAL, first), campaign)
    right = simulate(perturb(NOMINAL, second), campaign)
    times = campaign.times_s()

    figure, axes = plt.subplots(
        2, 1, figsize=(9.5, 6.0), sharex=True, height_ratios=[2.2, 1.0]
    )
    axes[0].plot(
        times, left.rate_rad_s, color=ANALYTIC, linewidth=3.0,
        label=f"J raised by {AMOUNT:+.2f} in logarithm",
    )
    axes[0].plot(
        times, right.rate_rad_s, color=LEARNED, linewidth=1.2, linestyle="--",
        label=f"k_t, T_c, b, c_d each moved by {-AMOUNT:+.2f}",
    )
    axes[0].axvline(
        campaign.drive_duration_s, color=REFUSED, linewidth=0.8, linestyle=":"
    )
    axes[0].text(
        campaign.drive_duration_s + 2.0, float(left.rate_rad_s.max()),
        "current switched off", fontsize=7, color=REFUSED, va="top",
    )
    axes[0].set_ylabel("wheel rate [rad/s]")
    axes[0].set_title(
        "two physically different faults, one rate record (telemetry set rate_full)"
    )
    axes[0].legend(loc="lower right", fontsize=8)

    difference = left.rate_rad_s - right.rate_rad_s
    axes[1].plot(times, difference, color=ANALYTIC, linewidth=1.0)
    axes[1].axhline(RATE_SIGMA_RAD_S, color=REFUSED, linestyle="--", linewidth=1.0)
    axes[1].axhline(-RATE_SIGMA_RAD_S, color=REFUSED, linestyle="--", linewidth=1.0)
    axes[1].set_yscale("symlog", linthresh=1e-15)
    axes[1].set_ylim(-1.0, 1.0)
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("difference [rad/s]")
    axes[1].set_title(
        f"their difference, against the declared sensor noise of "
        f"{RATE_SIGMA_RAD_S} rad/s (dashed)"
    )
    path = save(figure, "nullspace_equivalence.png")
    print(f"written: {path}")
    worst = float(np.abs(difference).max())
    print(f"  perturbation one          : {np.array2string(first, precision=4)}")
    print(f"  perturbation two          : {np.array2string(second, precision=4)}")
    print(f"  parameter names           : {', '.join(PARAMETER_NAMES)}")
    print(f"  worst rate difference     : {worst:.6e} rad/s")
    print(f"  as a fraction of the noise: {worst / RATE_SIGMA_RAD_S:.6e}")
    print(f"  units in the last place   : "
          f"{worst / float(np.spacing(float(left.rate_rad_s.max()))):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
