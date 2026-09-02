"""The learned ranker against the classical Pyramid rule, as a picture.

A reduced-size rerun of ``validation/validate_ml_vs_classical.py`` -- fewer
training frames, fewer trials -- so the figure finishes in a couple of minutes.
Read the shape; the numbers of record are in
``validation/ml_vs_classical_output.txt``.

Four panels:

1. **the operating-point curve.** The classical rule is a single point: it has
   no threshold. The learned ranker is a curve, and where the curve passes
   above and to the left of that point, it is strictly better on both error
   rates at once. The grey line is the search ceiling, which neither can pass.
2. **identification rate against false-star count**, with the ceiling.
3. **false-identification rate against false-star count.** The triangle rule
   is there to show what the axis means.
4. **the mis-sized gate.** With a tolerance sized for 12x the actual centroid
   noise, the Pyramid rule's uniqueness test stops discriminating and it starts
   producing confidently wrong answers. The learned ranker does not, on these
   frames.

Saves ``screenshots/learned_vs_pyramid.png``.

Run: ``python examples/learned_vs_pyramid.py``   (2-3 minutes on 2 cores)
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

from skymatch.benchmark import run_trials
from skymatch.camera import CameraModel
from skymatch.dataset import DEFAULT_GRID, build_catalogue_tables, generate_candidate_dataset
from skymatch.ranker import LearnedRanker
from skymatch.scene import SceneConfig

OUT = Path(__file__).resolve().parents[1] / "screenshots" / "learned_vs_pyramid.png"
SEED = 20260902
TRAIN_FRAMES = 320
TRIALS = 120
THRESHOLDS = (0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9, 0.97, 0.995)
FALSE_COUNTS = (0, 4, 6, 8, 10, 12, 16)
PYRAMID_STYLE = {"color": "#1f6f4a", "marker": "o", "ms": 8}


def main() -> None:
    cam = CameraModel()
    tables = build_catalogue_tables(tuple(p.magnitude_limit for p in DEFAULT_GRID), cam, SEED)
    t0 = time.perf_counter()
    train = generate_candidate_dataset(TRAIN_FRAMES, 1234, camera=cam, tables=tables)
    ranker = LearnedRanker(random_state=0).fit(train.features, train.labels)
    print(f"trained on {train.n_rows} rows from {train.n_frames} frames in "
          f"{time.perf_counter() - t0:.1f} s")
    cat, table = tables[6.0]

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.0))

    ax = axes[0, 0]
    cfg = SceneConfig(camera=cam, centroid_sigma_arcsec=5.0, n_false_stars=8)
    point = run_trials(cat, table, cam, cfg, 250, SEED + 1, ranker=ranker,
                       thresholds=THRESHOLDS, with_attitude=False)
    ml_false = [point.methods[f"ranker@{t:g}"].false_identification_rate for t in THRESHOLDS]
    ml_ident = [point.methods[f"ranker@{t:g}"].identification_rate for t in THRESHOLDS]
    pyr = point.methods["pyramid"]
    tri = point.methods["triangle"]
    ax.plot(ml_false, ml_ident, "-", color="#7b4fa0", lw=1.6, marker=".", ms=9,
            label="learned ranker, threshold swept")
    for t, fx, iy in zip(THRESHOLDS, ml_false, ml_ident, strict=True):
        if t in (0.02, 0.5, 0.995):
            ax.annotate(f"{t:g}", (fx, iy), textcoords="offset points", xytext=(5, -10),
                        fontsize=8, color="#7b4fa0")
    ax.plot([pyr.false_identification_rate], [pyr.identification_rate], label="pyramid rule",
            ls="none", **PYRAMID_STYLE)
    ax.plot([tri.false_identification_rate], [tri.identification_rate], ls="none",
            color="#c0392b", marker="s", ms=8, label="triangle rule")
    ax.axhline(point.ceiling, color="0.55", lw=1.4, ls="-",
               label=f"search ceiling {point.ceiling:.3f}")
    ax.set_xlabel("FALSE identification rate  (lower is better)")
    ax.set_ylabel("identification rate  (higher is better)")
    ax.set_title(f"Operating points, 8 false stars, sigma 5 arcsec, {point.n_trials} trials")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25)

    points = []
    for n_false in FALSE_COUNTS:
        cfg = SceneConfig(camera=cam, centroid_sigma_arcsec=5.0, n_false_stars=n_false)
        points.append(run_trials(cat, table, cam, cfg, TRIALS, SEED + 2, ranker=ranker,
                                 thresholds=(0.5,), with_attitude=False))
    ceiling = [p.ceiling for p in points]

    ax = axes[0, 1]
    ax.plot(FALSE_COUNTS, ceiling, "-", color="0.55", lw=1.5, label="search ceiling")
    ax.plot(FALSE_COUNTS, [p.methods["pyramid"].identification_rate for p in points], "o-",
            color="#1f6f4a", label="pyramid rule")
    ax.plot(FALSE_COUNTS, [p.methods["ranker@0.5"].identification_rate for p in points], "^-",
            color="#7b4fa0", label="learned ranker, threshold 0.5")
    ax.fill_between(
        FALSE_COUNTS,
        [p.methods["pyramid"].identification_rate for p in points],
        [p.methods["ranker@0.5"].identification_rate for p in points],
        color="#7b4fa0", alpha=0.15,
    )
    ax.set_xlabel("false detections in the frame")
    ax.set_ylabel("identification rate")
    ax.set_title("What the learned decision recovers, and what it cannot")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    for name, colour, marker, label in (
        ("triangle", "#c0392b", "s--", "triangle rule"),
        ("pyramid", "#1f6f4a", "o-", "pyramid rule"),
        ("ranker@0.5", "#7b4fa0", "^-", "learned ranker @ 0.5"),
    ):
        ax.plot(FALSE_COUNTS, [p.methods[name].false_identification_rate for p in points],
                marker, color=colour, label=label, ms=5)
        ax.plot(FALSE_COUNTS, [p.methods[name].false_identification_ci[1] for p in points],
                ":", color=colour, lw=1.0, alpha=0.7)
    ax.set_xlabel("false detections in the frame")
    ax.set_ylabel("FALSE identification rate")
    ax.set_title("Dotted: 95% Wilson upper bound (a measured zero is not a zero)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    cfg = SceneConfig(camera=cam, centroid_sigma_arcsec=5.0, n_false_stars=4)
    bad = run_trials(cat, table, cam, cfg, 200, SEED + 3, ranker=ranker, thresholds=(0.5,),
                     tolerance_sigma_arcsec=60.0, with_attitude=False)
    names = ["triangle", "pyramid", "ranker@0.5"]
    x = np.arange(len(names))
    ident = [bad.methods[n].identification_rate for n in names]
    false = [bad.methods[n].false_identification_rate for n in names]
    ax.bar(x - 0.19, ident, 0.36, color="#1f6f4a", label="identification rate")
    ax.bar(x + 0.19, false, 0.36, color="#c0392b", label="FALSE identification rate")
    for xi, (a, b) in enumerate(zip(ident, false, strict=True)):
        ax.text(xi - 0.19, a + 0.015, f"{a:.3f}", ha="center", fontsize=8)
        ax.text(xi + 0.19, b + 0.015, f"{b:.3f}", ha="center", fontsize=8)
    ax.axhline(bad.ceiling, color="0.55", lw=1.4, label=f"search ceiling {bad.ceiling:.3f}")
    ax.set_xticks(x, ["triangle", "pyramid", "learned @ 0.5"])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("rate")
    ax.set_title("Tolerance sized for 60 arcsec on a 5 arcsec frame, 4 false stars")
    ax.legend(fontsize=8, loc="upper left")

    print(f"8 false stars: pyramid {pyr.identification_rate:.3f} ident / "
          f"{pyr.false_identification_rate:.3f} false; ceiling {point.ceiling:.3f}")
    print("  learned threshold curve (threshold, ident, false):")
    for t, iy, fx in zip(THRESHOLDS, ml_ident, ml_false, strict=True):
        print(f"    {t:6g} {iy:.4f} {fx:.4f}")
    print("mis-sized gate: " + ", ".join(
        f"{n} {bad.methods[n].identification_rate:.3f}/"
        f"{bad.methods[n].false_identification_rate:.3f}" for n in names))

    fig.suptitle("SkyMatch: a learned decision rule on the classical algorithm's own "
                 "candidate lists", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=125)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
