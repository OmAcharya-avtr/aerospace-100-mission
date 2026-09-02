"""The synthetic catalogue and its pair table, as a picture.

Four panels:

1. the sky, in an equal-area Mollweide projection, with the brightest stars
   marked -- the point is that there is no structure, because the model has
   none;
2. cumulative star counts against Eq. C1, over five magnitude limits;
3. the distribution of pair separations inside the field against its analytic
   density ``sin(theta)``, which is the shape every range query samples from;
4. pair-table size and build time against magnitude limit, against Eq. P1.

Saves ``screenshots/catalogue_overview.png``.

Run: ``python examples/catalogue_overview.py``   (about 25 s)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skymatch.camera import CameraModel
from skymatch.catalogue import DEFAULT_SLOPE, generate_catalogue, predicted_count
from skymatch.pairtable import PairTable, expected_pair_count

OUT = Path(__file__).resolve().parents[1] / "screenshots" / "catalogue_overview.png"
SEED = 20260902
MAG_LIMITS = (5.0, 5.5, 6.0, 6.5, 7.0)


def main() -> None:
    cam = CameraModel()
    cat = generate_catalogue(6.0, seed=SEED)
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0))

    ax = axes[0, 0]
    ra = np.mod(cat.ra + np.pi, 2.0 * np.pi) - np.pi
    bright = cat.magnitude < 4.0
    ax.scatter(ra[~bright], np.sin(cat.dec[~bright]), s=1.5, c="#5b7fa6", alpha=0.35,
               linewidths=0, label=f"mag 4.0-6.0 ({int((~bright).sum())})")
    ax.scatter(ra[bright], np.sin(cat.dec[bright]), s=14.0, c="#c0392b",
               label=f"brighter than 4.0 ({int(bright.sum())})")
    ax.set_xlabel("right ascension - pi [rad]")
    ax.set_ylabel("sin(declination)  (equal area)")
    ax.set_title(f"Synthetic sky, magnitude limit 6.0, {cat.n_stars} stars, seed {SEED}")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-1.0, 1.0)

    ax = axes[0, 1]
    grid = np.linspace(-1.49, 7.0, 300)
    ax.semilogy(grid, [predicted_count(m) for m in grid], "k--", lw=1.4,
                label=r"Eq. C1  $N(<m)=4800\cdot10^{0.52(m-6)}$")
    for limit in MAG_LIMITS:
        c = generate_catalogue(limit, seed=SEED)
        edges = np.linspace(-1.5, limit, 60)
        counts = np.searchsorted(c.magnitude, edges)
        ax.semilogy(edges, np.maximum(counts, 0.5), lw=1.2, alpha=0.85,
                    label=f"sampled, limit {limit:.1f} ({c.n_stars})")
    ax.set_xlabel("magnitude")
    ax.set_ylabel("stars brighter than m (whole sky)")
    ax.set_title("Magnitude distribution against the model it was drawn from")
    ax.legend(fontsize=7.5, loc="upper left")
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    table = PairTable(cat, cam.max_separation_rad)
    theta = np.degrees(table.separations)
    ax.hist(theta, bins=90, color="#5b7fa6", alpha=0.8, label=f"{table.n_pairs} pairs")
    edges = np.linspace(0.0, np.degrees(cam.max_separation_rad), 91)
    centres = 0.5 * (edges[:-1] + edges[1:])
    density = np.sin(np.radians(centres))
    scale = table.n_pairs * (edges[1] - edges[0]) / np.trapezoid(
        np.sin(np.radians(centres)), centres
    )
    ax.plot(centres, density * scale, "k--", lw=1.5,
            label=r"$\propto\sin\theta$ (uniform sphere)")
    ax.axvline(np.degrees(cam.fov_rad), color="#c0392b", ls=":", lw=1.3,
               label=f"field width {cam.fov_deg:.0f} deg")
    ax.set_xlabel("pair separation [deg]")
    ax.set_ylabel("pairs per bin")
    ax.set_title(f"Pair separations inside the {np.degrees(cam.max_separation_rad):.1f} deg "
                 "search radius")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    n_stars, n_pairs, predicted, builds = [], [], [], []
    for limit in MAG_LIMITS:
        c = generate_catalogue(limit, seed=SEED)
        t0 = time.perf_counter()
        t = PairTable(c, cam.max_separation_rad)
        builds.append(time.perf_counter() - t0)
        n_stars.append(c.n_stars)
        n_pairs.append(t.n_pairs)
        predicted.append(expected_pair_count(c.n_stars, cam.max_separation_rad))
    ax.loglog(n_stars, n_pairs, "o-", color="#5b7fa6", label="pairs stored")
    ax.loglog(n_stars, predicted, "k--", lw=1.3, label="Eq. P1")
    ax.set_xlabel("catalogue stars")
    ax.set_ylabel("pairs within the search radius")
    ax.set_title("The pair table is quadratic: the real cost of a magnitude limit")
    twin = ax.twinx()
    twin.loglog(n_stars, builds, "s-", color="#c0392b", ms=4, label="build time")
    twin.set_ylabel("build time [s]", color="#c0392b")
    for x, y, limit in zip(n_stars, n_pairs, MAG_LIMITS, strict=True):
        ax.annotate(f"m={limit:.1f}", (x, y), textcoords="offset points",
                    xytext=(6, -11), fontsize=7.5)
    lines = ax.get_lines() + twin.get_lines()
    ax.legend(lines, [ln.get_label() for ln in lines], fontsize=8, loc="upper left")
    ax.grid(alpha=0.25, which="both")

    print(f"growth per magnitude: stars x{10 ** DEFAULT_SLOPE:.2f}, "
          f"pairs x{10 ** (2 * DEFAULT_SLOPE):.2f}")
    for limit, s_n, p_n, b in zip(MAG_LIMITS, n_stars, n_pairs, builds, strict=True):
        print(f"  mag {limit:.1f}: {s_n:6d} stars, {p_n:8d} pairs, {b:6.3f} s to build")

    fig.suptitle("SkyMatch synthetic catalogue -- generated, not measured "
                 "(see DATASET_CARD.md)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=125)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
