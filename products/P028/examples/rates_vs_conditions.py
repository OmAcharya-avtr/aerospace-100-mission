"""Both error rates against centroid noise, false stars, and magnitude limit.

The point of this figure is the second row. The top row is the identification
rate, the number star-identification papers plot. The bottom row is the
false-identification rate, and it is what separates a matcher you can believe
from one you cannot: the triangle rule's identification rate looks respectable
right up to the point where a third of its answers are wrong.

The dashed grey line in the false-star panels is the *ceiling* -- the fraction
of frames whose candidate list contained the truth at all. No decision rule can
be above it, so the gap between a curve and that line is the part a better
decision rule could recover, and the gap below it is not recoverable at all.

Saves ``screenshots/rates_vs_conditions.png``.

Run: ``python examples/rates_vs_conditions.py``   (2-3 minutes on 2 cores)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skymatch.benchmark import run_trials
from skymatch.camera import CameraModel
from skymatch.catalogue import generate_catalogue
from skymatch.pairtable import PairTable
from skymatch.scene import SceneConfig

OUT = Path(__file__).resolve().parents[1] / "screenshots" / "rates_vs_conditions.png"
SEED = 20260902
TRIALS = 100
SIGMAS = (1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 60.0)
FALSE_COUNTS = (0, 2, 4, 6, 8, 10, 12, 16, 20)
MAG_LIMITS = (5.0, 5.5, 6.0, 6.5)
STYLE = {"triangle": ("#c0392b", "s--"), "pyramid": ("#1f6f4a", "o-")}


def _plot(ax_top, ax_bottom, xs, points, xlabel, logx: bool) -> None:
    for name, (colour, marker) in STYLE.items():
        ident = [p.methods[name].identification_rate for p in points]
        false = [p.methods[name].false_identification_rate for p in points]
        hi = [p.methods[name].false_identification_ci[1] for p in points]
        ax_top.plot(xs, ident, marker, color=colour, label=name, ms=5)
        ax_bottom.plot(xs, false, marker, color=colour, label=name, ms=5)
        ax_bottom.plot(xs, hi, ":", color=colour, lw=1.0, alpha=0.7)
    ceiling = [p.ceiling for p in points]
    ax_top.plot(xs, ceiling, "-", color="0.55", lw=1.4, label="search ceiling")
    for ax in (ax_top, ax_bottom):
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.25)
        if logx:
            ax.set_xscale("log")
    ax_top.set_ylim(-0.03, 1.05)
    ax_bottom.set_ylim(-0.02, 0.55)


def main() -> None:
    cam = CameraModel()
    cat = generate_catalogue(6.0, seed=SEED)
    table = PairTable(cat, cam.max_separation_rad)
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.0), sharey="row")

    noise = [
        run_trials(cat, table, cam, SceneConfig(camera=cam, centroid_sigma_arcsec=s),
                   TRIALS, SEED + 1, with_attitude=False)
        for s in SIGMAS
    ]
    _plot(axes[0, 0], axes[1, 0], SIGMAS, noise, "centroid noise sigma [arcsec]", True)
    axes[0, 0].set_title("Centroid noise, clean sky")

    false_pts = [
        run_trials(cat, table, cam,
                   SceneConfig(camera=cam, centroid_sigma_arcsec=5.0, n_false_stars=n),
                   TRIALS, SEED + 2, with_attitude=False)
        for n in FALSE_COUNTS
    ]
    _plot(axes[0, 1], axes[1, 1], FALSE_COUNTS, false_pts, "false detections in the frame", False)
    axes[0, 1].set_title("False stars, sigma = 5 arcsec")
    axes[0, 1].axvline(cat.expected_in_solid_angle(cam.solid_angle_sr), color="0.3",
                       ls=":", lw=1.2)
    axes[0, 1].annotate("real stars\nin the field", (16.7, 0.55), fontsize=7.5, color="0.3")

    mag_pts = []
    for limit in MAG_LIMITS:
        c = generate_catalogue(limit, seed=SEED)
        t = PairTable(c, cam.max_separation_rad)
        mag_pts.append(
            run_trials(c, t, cam,
                       SceneConfig(camera=cam, centroid_sigma_arcsec=5.0, n_false_stars=4),
                       TRIALS, SEED + 3, with_attitude=False)
        )
    _plot(axes[0, 2], axes[1, 2], MAG_LIMITS, mag_pts, "catalogue magnitude limit", False)
    axes[0, 2].set_title("Magnitude limit, sigma = 5 arcsec, 4 false stars")

    axes[0, 0].set_ylabel("identification rate")
    axes[1, 0].set_ylabel("FALSE identification rate")
    axes[0, 0].legend(fontsize=8, loc="lower left")
    axes[1, 0].legend(fontsize=8, loc="upper left")
    axes[1, 1].annotate("dotted: 95% Wilson upper bound\non a measured zero",
                        (0.45, 0.72), xycoords="axes fraction", fontsize=7.5, color="0.3")

    for label, xs, points in (("noise", SIGMAS, noise), ("false", FALSE_COUNTS, false_pts),
                              ("mag", MAG_LIMITS, mag_pts)):
        print(f"{label}: " + ", ".join(
            f"{x:g}->tri {p.methods['triangle'].identification_rate:.3f}/"
            f"{p.methods['triangle'].false_identification_rate:.3f} "
            f"pyr {p.methods['pyramid'].identification_rate:.3f}/"
            f"{p.methods['pyramid'].false_identification_rate:.3f}"
            for x, p in zip(xs, points, strict=True)))

    fig.suptitle(f"SkyMatch: identification and FALSE identification, {TRIALS} trials per point, "
                 f"magnitude limit 6.0 unless shown", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=125)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
