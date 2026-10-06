"""Webb distribution against its Gaussian and Poisson limits.

Writes ``../screenshots/webb_limits.png``.

What to notice. Left: the Webb density is visibly skewed at F = 3 and collapses
onto the Gaussian as F approaches 1 --- the skewness is 3(F-1)/sqrt(mF), so it is
the excess noise factor alone that sets the departure. Right: the total variation
between the integer-binned Webb density at F = 1 and the Poisson pmf of the same
mean falls exactly as m^(-1/2), the straight line of slope -1/2 on log-log axes.
That is the honest Poisson limit: a continuous density cannot do better, and an
implementation that appeared to would be wrong.

Runtime: about 10 s on one core.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    _SRC = Path(__file__).resolve().parents[1] / "src"
    if _SRC.is_dir() and str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from photoncount.webb import (  # noqa: E402
    WebbParameters,
    binned_pmf,
    gaussian_limit_pdf,
    moments,
    pdf,
)

OUT = Path(__file__).resolve().parents[1] / "screenshots" / "webb_limits.png"


def main() -> int:
    mean = 60.0
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12.5, 4.8))

    grid = np.linspace(0.0, 160.0, 2001)
    for f, colour in zip((3.0, 2.0, 1.4, 1.05), ("#1b4965", "#5fa8d3", "#bee9e8", "#62b6cb"),
                         strict=True):
        params = WebbParameters(mean, f)
        skew = moments(params)["skewness"]
        ax_left.plot(grid, pdf(grid, params), color=colour, lw=1.8,
                     label=f"Webb, F = {f:.2f} (skew {skew:.3f})")
    ax_left.plot(
        grid,
        gaussian_limit_pdf(grid, WebbParameters(mean, 1.0)),
        "k--",
        lw=1.4,
        label="Gaussian limit N(m, m)",
    )
    ax_left.plot(
        np.arange(0, 161),
        stats.poisson.pmf(np.arange(0, 161), mean),
        color="#c1121f",
        ls=":",
        lw=1.6,
        label=f"Poisson(m = {mean:.0f})",
    )
    ax_left.set_xlabel("gain-normalised output y (primary photoelectron equivalents)")
    ax_left.set_ylabel("density")
    ax_left.set_title(f"Webb density, m = {mean:.0f}: excess noise sets the skew")
    ax_left.legend(fontsize=8)
    ax_left.grid(alpha=0.3)
    ax_left.set_xlim(0, 160)

    means = np.array([30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0])
    tvs = []
    for m in means:
        k_max = int(m + 14 * np.sqrt(m))
        probs = binned_pmf(WebbParameters(m, 1.0), k_max)
        poisson = stats.poisson.pmf(np.arange(k_max + 1), m)
        tvs.append(0.5 * float(np.abs(probs - poisson).sum()))
    tvs_arr = np.asarray(tvs)
    ax_right.loglog(means, tvs_arr, "o-", color="#1b4965", lw=1.8,
                    label="measured total variation")
    reference = tvs_arr[0] * np.sqrt(means[0] / means)
    ax_right.loglog(means, reference, "k--", lw=1.3, label="slope -1/2 reference")
    for m, tv in zip(means, tvs_arr, strict=True):
        ax_right.annotate(f"{tv * np.sqrt(m):.4f}", (m, tv), textcoords="offset points",
                          xytext=(4, 6), fontsize=7)
    ax_right.set_xlabel("mean primary photoelectrons m")
    ax_right.set_ylabel("total variation to Poisson(m)")
    ax_right.set_title("Poisson limit at F = 1: TV falls as m$^{-1/2}$\n"
                       "(labels are TV x sqrt(m), a constant)")
    ax_right.legend(fontsize=8)
    ax_right.grid(alpha=0.3, which="both")

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140)
    plt.close(fig)

    print("webb_limits.py")
    print(f"   wrote screenshots/{OUT.name}")
    print(f"   {'m':>8} {'total variation':>17} {'TV * sqrt(m)':>14}")
    for m, tv in zip(means, tvs_arr, strict=True):
        print(f"   {m:8.0f} {tv:17.8e} {tv * np.sqrt(m):14.6f}")
    print("   TV * sqrt(m) is constant to within the quadrature error, which is the")
    print("   statement that the Poisson limit is reached at rate m^-1/2 and no faster")
    return 0


if __name__ == "__main__":
    sys.exit(main())
