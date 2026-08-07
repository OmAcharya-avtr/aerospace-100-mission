"""Example 1 - spiral and raster acquisition scan visualisations.

Generates ``screenshots/ex01_scan_patterns.png``: the Archimedean spiral and
the serpentine raster over the same 2-D Gaussian uncertainty region, with the
1/2/3-sigma contours and the design containment radius overlaid, plus a
coverage-vs-overlap curve.

Run from the product root:  python examples/ex01_scan_patterns.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trackbench.scan import (  # noqa: E402
    GaussianUncertainty,
    coverage_fraction,
    raster_scan,
    spiral_scan,
    track_spacing,
)

OUT = Path(__file__).resolve().parents[1] / "screenshots" / "ex01_scan_patterns.png"
SIGMA = 3.0e-4
BEAM = 2.0e-5
CONTAINMENT = 0.995


def draw_pattern(ax, pattern, u: GaussianUncertainty, title: str) -> None:
    """Plot one scan pattern with sigma contours."""
    urad = 1e6
    ax.plot(pattern.points[:, 0] * urad, pattern.points[:, 1] * urad,
            lw=0.4, color="#1f77b4", alpha=0.85)
    theta = np.linspace(0, 2 * np.pi, 361)
    for k, style in ((1, ":"), (2, "--"), (3, "-.")):
        ax.plot(k * SIGMA * np.cos(theta) * urad, k * SIGMA * np.sin(theta) * urad,
                style, color="0.35", lw=0.9, label=f"{k}$\\sigma$")
    r = pattern.max_radius * urad
    ax.plot(r * np.cos(theta), r * np.sin(theta), color="crimson", lw=1.4,
            label=f"{CONTAINMENT:.1%} containment")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("azimuth offset [$\\mu$rad]")
    ax.set_ylabel("elevation offset [$\\mu$rad]")
    ax.set_aspect("equal")
    lim = 1.15 * r
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.grid(alpha=0.25)


def main() -> int:
    """Build the figure and save it."""
    u = GaussianUncertainty(SIGMA)
    sp = spiral_scan(u, BEAM, overlap=0.25, containment=CONTAINMENT, dwell_time=1e-3)
    ra = raster_scan(u, BEAM, overlap=0.25, containment=CONTAINMENT, dwell_time=1e-3)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    draw_pattern(axes[0], sp, u,
                 f"Archimedean spiral\n{sp.n_points} dwells, "
                 f"{sp.scan_time:.2f} s, pitch {sp.track_spacing * 1e6:.0f} $\\mu$rad")
    draw_pattern(axes[1], ra, u,
                 f"Serpentine raster\n{ra.n_points} dwells, "
                 f"{ra.scan_time:.2f} s, pitch {ra.track_spacing * 1e6:.0f} $\\mu$rad")
    axes[0].legend(fontsize=7, loc="upper right")

    overlaps = np.linspace(0.0, 0.6, 13)
    cov_sp, cov_ra, cost = [], [], []
    for ov in overlaps:
        a = spiral_scan(u, BEAM, overlap=ov, containment=CONTAINMENT)
        b = raster_scan(u, BEAM, overlap=ov, containment=CONTAINMENT)
        cov_sp.append(coverage_fraction(a, u, 20000, np.random.default_rng(1)))
        cov_ra.append(coverage_fraction(b, u, 20000, np.random.default_rng(1)))
        cost.append(a.n_points)
    ax = axes[2]
    ax.plot(overlaps, cov_sp, "o-", label="spiral coverage")
    ax.plot(overlaps, cov_ra, "s--", label="raster coverage")
    ax.axhline(CONTAINMENT, color="crimson", lw=1.0,
               label=f"design containment {CONTAINMENT}")
    ax.set_xlabel("track overlap factor")
    ax.set_ylabel("covered probability mass")
    ax.set_ylim(0.975, 1.001)
    ax.grid(alpha=0.25)
    ax2 = ax.twinx()
    ax2.plot(overlaps, cost, "^:", color="darkgreen", label="spiral dwell count")
    ax2.set_ylabel("spiral dwell count", color="darkgreen")
    ax.set_title("Coverage and cost vs overlap\n"
                 f"(s = 2R(1-overlap), R = {BEAM * 1e6:.0f} $\\mu$rad)", fontsize=10)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="lower right")

    fig.suptitle("TrackBench - acquisition scan patterns over a 2-D Gaussian "
                 f"uncertainty region ($\\sigma$ = {SIGMA * 1e6:.0f} $\\mu$rad)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=130)
    plt.close(fig)

    sys.stdout.write(f"spiral: {sp.n_points} dwells, {sp.scan_time:.3f} s, "
                     f"pitch {track_spacing(BEAM, 0.25) * 1e6:.1f} urad\n")
    sys.stdout.write(f"raster: {ra.n_points} dwells, {ra.scan_time:.3f} s\n")
    sys.stdout.write(f"coverage (overlap 0.25): spiral {cov_sp[5]:.5f}, "
                     f"raster {cov_ra[5]:.5f}\n")
    sys.stdout.write(f"saved {OUT}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
