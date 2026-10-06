"""Aperture averaging: the factor, its two derived limits, and the error.

Panel 1: the averaging factor A(D) against D/rho_c for both covariance
shapes, with the small-D and large-D closed forms of the Gaussian case
overlaid. Panel 2: the relative error of each limit, so the reader can see
exactly where each one stops being usable instead of being told.

What to notice: the small-D form is within 1 % only up to about D/rho_c =
0.5, and the two-term large-D form is within 1e-4 from about D/rho_c = 20
upwards. Between those the integral has to be done. The two covariance
shapes also cross: at equal rho_c the exponential field averages less across
a small aperture (it is already decorrelated) and more across a large one.

Writes ../screenshots/aperture_averaging.png. Runtime about 10 s on one core.
"""

from __future__ import annotations

import pathlib
import sys

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from aperturediv.aperture import (  # noqa: E402
    aperture_averaging_factor,
    aperture_averaging_large_d,
    aperture_averaging_small_d,
    effective_scintillation_index,
    fresnel_scale,
)

OUT = pathlib.Path(__file__).resolve().parents[1] / "screenshots" / "aperture_averaging.png"
RHO_C = 1.0


def main() -> int:
    ratio = np.geomspace(0.02, 100.0, 220)
    a_gauss = np.array([aperture_averaging_factor(d, RHO_C) for d in ratio])
    a_exp = np.array([aperture_averaging_factor(d, RHO_C, "exponential") for d in ratio])
    a_small = np.array([aperture_averaging_small_d(d, RHO_C) for d in ratio])
    a_large = np.array([aperture_averaging_large_d(d, RHO_C, order=2) for d in ratio])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.0))

    ax1.loglog(ratio, a_gauss, lw=2.2, color="#1f3f7a", label="quadrature, Gaussian b(rho)")
    ax1.loglog(
        ratio, a_exp, lw=2.0, color="#8a5a00", label="quadrature, exponential b(rho)"
    )
    valid_small = a_small > 1e-3
    ax1.loglog(
        ratio[valid_small],
        a_small[valid_small],
        lw=1.6,
        ls="--",
        color="#2f8f4f",
        label=r"small-D: $1 - D^2/4\rho_c^2$",
    )
    ax1.loglog(
        ratio,
        a_large,
        lw=1.6,
        ls=":",
        color="#a01f1f",
        label=r"large-D: $4(\rho_c/D)^2 - (8/\sqrt{\pi})(\rho_c/D)^3$",
    )
    ax1.set_xlabel(r"aperture diameter / correlation scale, $D/\rho_c$")
    ax1.set_ylabel(r"aperture averaging factor $A = \sigma_I^2(D)/\sigma_I^2(0)$")
    ax1.set_ylim(2e-4, 2.0)
    ax1.set_title("Aperture averaging factor and its derived limits")
    ax1.grid(True, which="both", alpha=0.25)
    ax1.legend(fontsize=8.5, loc="lower left")

    err_small = np.abs(a_small - a_gauss) / a_gauss
    err_large = np.abs(a_large - a_gauss) / a_gauss
    ax2.loglog(ratio, err_small, lw=1.8, ls="--", color="#2f8f4f", label="small-D form")
    ax2.loglog(ratio, err_large, lw=1.8, ls=":", color="#a01f1f", label="large-D form, 2 terms")
    ax2.axhline(0.01, color="k", lw=1.0, alpha=0.6)
    ax2.text(0.025, 0.0125, "1 % relative error", fontsize=8.5)
    ax2.axhline(1e-4, color="k", lw=1.0, alpha=0.35)
    ax2.text(0.025, 1.25e-4, "0.01 %", fontsize=8.5)
    ax2.set_xlabel(r"$D/\rho_c$")
    ax2.set_ylabel("relative error against the quadrature")
    ax2.set_ylim(1e-9, 10.0)
    ax2.set_title("Where each closed form stops being usable")
    ax2.grid(True, which="both", alpha=0.25)
    ax2.legend(fontsize=9, loc="upper right")

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140)
    plt.close(fig)

    print("Aperture averaging factor, Gaussian irradiance covariance")
    print("  D/rho_c      A        small-D       rel err     large-D(2)     rel err")
    for d in (0.05, 0.2, 0.5, 1.0, 2.0, 5.0, 20.0, 100.0):
        a = aperture_averaging_factor(d, RHO_C)
        s = aperture_averaging_small_d(d, RHO_C)
        g = aperture_averaging_large_d(d, RHO_C, order=2)
        print(f"{d:9.3f} {a:10.6f} {s:12.6f} {abs(s - a) / a:12.3e} {g:13.6f} "
              f"{abs(g - a) / a:12.3e}")

    small_ok = ratio[err_small < 0.01].max()
    large_ok = ratio[err_large < 1e-4].min()
    print()
    print(f"small-D form within 1 % up to    D/rho_c = {small_ok:.3f}")
    print(f"large-D 2-term within 0.01 % from D/rho_c = {large_ok:.3f}")

    sign = np.sign(a_exp - a_gauss)
    flip = np.nonzero(np.diff(sign) != 0)[0]
    if flip.size:
        print(f"covariance shapes cross between D/rho_c = {ratio[flip[0]]:.3f} "
              f"and {ratio[flip[0] + 1]:.3f}")

    print()
    lam, path = 1.55e-6, 2000.0
    rho_c = fresnel_scale(lam, path)
    print(f"worked case: lambda = {lam:.3e} m, L = {path:.0f} m, "
          f"rho_c = sqrt(lambda L) = {rho_c:.6f} m, si(0) = 0.6")
    print("  D (m)     A(D)      si(D)")
    for d in (0.02, 0.05, 0.10, 0.20, 0.40):
        print(f"{d:7.3f} {aperture_averaging_factor(d, rho_c):10.6f} "
              f"{effective_scintillation_index(0.6, d, rho_c):10.6f}")
    print()
    print(f"wrote {OUT.parent.name}/{OUT.name}")
    print("research-grade output; not flight-qualified, not certified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
