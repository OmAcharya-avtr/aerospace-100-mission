"""Example 1: r0 and theta0 versus zenith angle for the three standard models.

Left panel  : Fried parameter r0 (cm) at 500 nm and 1550 nm.
Right panel : isoplanatic angle theta0 (urad) at the same two wavelengths.

Dashed grey curves show the analytic sec(zeta) laws anchored at the zenith
value - r0 ~ cos(zeta)^(3/5) and theta0 ~ cos(zeta)^(8/5) - so that the
computed points can be seen to follow the stated exponents rather than being
asserted to.

Saves ../screenshots/r0_theta0_vs_zenith.png. Runtime ~20 s.
"""

import math
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atmoprofile import (  # noqa: E402
    fried_parameter,
    hv57,
    isoplanatic_angle,
    slc_day,
    slc_night,
)

OUT = Path(__file__).resolve().parents[1] / "screenshots" / "r0_theta0_vs_zenith.png"

ZENITH_DEG = np.linspace(0.0, 70.0, 36)
WAVELENGTHS = ((500e-9, "500 nm", "-"), (1550e-9, "1550 nm", "--"))
PROFILES = (("HV5/7", hv57(), "tab:blue"), ("SLC-Day", slc_day(), "tab:red"),
            ("SLC-Night", slc_night(), "tab:green"))


def main() -> None:
    t0 = time.perf_counter()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    sec = 1.0 / np.cos(np.radians(ZENITH_DEG))

    with warnings.catch_warnings():
        # Beyond 60 deg the package warns that the flat-Earth airmass model is
        # an extrapolation; the plot shows that region deliberately, shaded.
        warnings.simplefilter("ignore", UserWarning)
        for name, profile, colour in PROFILES:
            for lam, lam_label, style in WAVELENGTHS:
                r0 = np.array(
                    [fried_parameter(profile, lam, zenith_rad=math.radians(z))
                     for z in ZENITH_DEG]
                )
                th = np.array(
                    [isoplanatic_angle(profile, lam, zenith_rad=math.radians(z))
                     for z in ZENITH_DEG]
                )
                if lam_label == "500 nm":
                    # analytic law drawn first, as a thick grey halo underneath
                    axes[0].plot(ZENITH_DEG, r0[0] * 100 * sec**-0.6, "-", color="0.65",
                                 lw=4.0, alpha=0.8, zorder=1)
                    axes[1].plot(ZENITH_DEG, th[0] * 1e6 * sec**-1.6, "-", color="0.65",
                                 lw=4.0, alpha=0.8, zorder=1)
                axes[0].plot(ZENITH_DEG, r0 * 100, style, color=colour, lw=1.4, zorder=3,
                             label=f"{name}, {lam_label}")
                axes[1].plot(ZENITH_DEG, th * 1e6, style, color=colour, lw=1.4, zorder=3,
                             label=f"{name}, {lam_label}")

    for ax, ylabel, title, law in (
        (axes[0], r"$r_0$  [cm]", "Fried parameter", r"$\propto \sec\zeta^{-3/5}$"),
        (axes[1], r"$\theta_0$  [$\mu$rad]", "Isoplanatic angle",
         r"$\propto \sec\zeta^{-8/5}$"),
    ):
        ax.axvspan(60.0, ZENITH_DEG[-1], color="0.9", zorder=0)
        ax.set_xlabel(r"zenith angle $\zeta$  [deg]")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}  {law}")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
        ax.text(60.5, ax.get_ylim()[0] * 1.15, "flat-Earth\nextrapolation",
                fontsize=7, color="0.35")

    fig.suptitle(
        "AtmoProfile 0.1.0 - zenith dependence of the coherence parameters "
        "(grey halo: analytic sec(zeta) law anchored at the zenith value, 500 nm)",
        fontsize=10,
    )
    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=140)
    plt.close(fig)
    sys.stdout.write(f"wrote {OUT} in {time.perf_counter() - t0:.1f} s\n")


if __name__ == "__main__":
    main()
