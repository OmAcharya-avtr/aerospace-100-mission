"""Example 2: comparison of the standard Cn^2 profile models.

Panel 1: Cn^2(h) for HV5/7, SLC-Day and SLC-Night (log-log).
Panel 2: the three weighting integrands that the metrics actually integrate,
         Cn^2, Cn^2 h^(5/6) and Cn^2 h^(5/3), normalised to unit area, showing
         which altitudes drive r0, the Rytov variance and theta0 respectively.
Panel 3: cumulative fraction of each integral versus altitude, with the
         effective turbulence height h_bar marked - this is the picture that
         explains why r0 is a ground-layer quantity while theta0 and
         scintillation are high-altitude quantities.

Saves ../screenshots/cn2_profile_comparison.png. Runtime ~10 s.
"""

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atmoprofile import (  # noqa: E402
    effective_turbulence_height,
    fried_parameter,
    hv57,
    slc_day,
    slc_night,
)

OUT = Path(__file__).resolve().parents[1] / "screenshots" / "cn2_profile_comparison.png"

PROFILES = (("HV5/7", hv57(), "tab:blue"), ("SLC-Day", slc_day(), "tab:red"),
            ("SLC-Night", slc_night(), "tab:green"))
WEIGHTS = ((0.0, r"$C_n^2$  ($r_0$)", "-"),
           (5 / 6, r"$C_n^2 h^{5/6}$  (Rytov)", "--"),
           (5 / 3, r"$C_n^2 h^{5/3}$  ($\theta_0$)", ":"))
LAM = 500e-9


def main() -> None:
    t0 = time.perf_counter()
    h = np.concatenate([np.linspace(0.5, 100.0, 400), np.geomspace(100.0, 20000.0, 1200)])
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.0))

    for name, profile, colour in PROFILES:
        cn2 = np.asarray(profile(h))
        r0 = fried_parameter(profile, LAM)
        axes[0].loglog(cn2, h, color=colour, label=f"{name}  ($r_0$ = {r0 * 100:.2f} cm)")

    axes[0].set_xlabel(r"$C_n^2$  [m$^{-2/3}$]")
    axes[0].set_ylabel("altitude  [m]")
    axes[0].set_title("Standard profile models")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=8, loc="upper right")

    # Panel 2: normalised integrands (HV5/7 and SLC-Night shown for clarity)
    for name, profile, colour in PROFILES:
        cn2 = np.asarray(profile(h))
        for power, label, style in WEIGHTS:
            integrand = cn2 * h**power
            area = np.trapezoid(integrand, h)
            # h * f(h) / area is the contribution per unit ln(h), which is the
            # honest quantity to plot against a logarithmic altitude axis.
            axes[1].semilogx(
                h, h * integrand / area, style, color=colour, lw=1.2,
                label=f"{name}: {label}" if name == "HV5/7" else None,
            )
    axes[1].set_xlabel("altitude  [m]")
    axes[1].set_ylabel(r"contribution per unit $\ln h$   [-]")
    axes[1].set_title("What each metric actually weights\n(colours as panel 1)")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(fontsize=8)

    # Panel 3: cumulative fractions
    for name, profile, colour in PROFILES:
        cn2 = np.asarray(profile(h))
        for power, _label, style in WEIGHTS:
            integrand = cn2 * h**power
            cumulative = np.concatenate(
                [[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(h))]
            )
            axes[2].semilogx(h, cumulative / cumulative[-1], style, color=colour, lw=1.2)
        hbar = effective_turbulence_height(profile)
        axes[2].axvline(hbar, color=colour, lw=0.8, alpha=0.5)
        axes[2].text(hbar * 1.06, 0.30, rf"$\bar h$={hbar:.0f} m", color=colour, fontsize=7,
                     rotation=90)
    axes[2].set_xlabel("altitude  [m]")
    axes[2].set_ylabel("cumulative fraction of the integral")
    axes[2].set_title("Cumulative contribution\n(solid $r_0$, dashed Rytov, dotted "
                      r"$\theta_0$)")
    axes[2].grid(True, which="both", alpha=0.3)
    axes[2].set_ylim(0.0, 1.02)

    fig.suptitle(
        "AtmoProfile 0.1.0 - Cn^2 profile models and their weighting integrals "
        f"(lambda = {LAM * 1e9:.0f} nm, vertical path 0-20 km)",
        fontsize=10,
    )
    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=140)
    plt.close(fig)
    sys.stdout.write(f"wrote {OUT} in {time.perf_counter() - t0:.1f} s\n")


if __name__ == "__main__":
    main()
