"""The learned detector against the analytic ones, at equal measured ARL0.

The analytic detectors are built and calibrated first. The forest is then
trained on windowed features, its probability threshold is moved until its
measured ARL0 matches, and only then are the delays compared. Comparing at any
other operating point would be comparing false-alarm rates.

Writes ../screenshots/learned_vs_analytic.png.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from telemdrift.benchmark import (  # noqa: E402
    DETECTOR_LABELS,
    STANDARD,
    calibrate_all_analytic,
    calibrate_learned_threshold,
    measure_change_response,
)
from telemdrift.detectors import ANALYTIC_DETECTORS  # noqa: E402
from telemdrift.learned import build_training_set, train_learned_detector  # noqa: E402
from telemdrift.streams import ChangeSpec  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "screenshots" / "learned_vs_analytic.png"
REPLICATES = 150
CHANGES = (
    ("mean step +1.0", ChangeSpec("mean_step", 1.0)),
    ("mean step +0.5", ChangeSpec("mean_step", 0.5)),
    ("variance x2.0", ChangeSpec("variance_step", 2.0)),
    ("drift 0.02/sample", ChangeSpec("drift_ramp", 0.02)),
)


def main() -> None:
    cals = calibrate_all_analytic(STANDARD)
    training = build_training_set(seeds=range(58_001, 58_009))
    print(f"training set: {training.summary()}")
    model = train_learned_detector(training, n_estimators=100)
    learned = calibrate_learned_threshold(model, STANDARD)
    print()
    for cal in cals.values():
        print(f"  {cal.summary()}")
    print(f"  {learned.summary()}")

    keys = list(ANALYTIC_DETECTORS) + ["learned"]
    thresholds = {k: cals[k].threshold for k in ANALYTIC_DETECTORS}
    thresholds["learned"] = learned.threshold
    table: dict[str, list[float]] = {}
    errs: dict[str, list[float]] = {}
    print()
    print("ARL1 at equal measured ARL0, lower is better")
    header = "change              " + "".join(f"{DETECTOR_LABELS[k]:>15s}" for k in keys)
    print(header)
    for label, spec in CHANGES:
        row, err = [], []
        for k in keys:
            r = measure_change_response(k, thresholds[k], spec, STANDARD,
                                        replicates=REPLICATES, model=model)
            row.append(r.arl1)
            err.append(r.sem)
        table[label] = row
        errs[label] = err
        print(f"{label:20s}" + "".join(
            f"{v:10.1f}+/-{e:<4.1f}" for v, e in zip(row, err, strict=True)))

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    x = np.arange(len(CHANGES))
    width = 0.8 / len(keys)
    colours = ["#31506e", "#b4452f", "#4f7a4a", "#7a4f8a", "#9c6b4f", "#2f7f8f"]
    for i, k in enumerate(keys):
        vals = [table[label][i] for label, _ in CHANGES]
        es = [errs[label][i] for label, _ in CHANGES]
        ax.bar(x + i * width, vals, width, yerr=es, capsize=2.5,
               label=DETECTOR_LABELS[k], color=colours[i], edgecolor="0.2",
               linewidth=0.5)
    ax.set_xticks(x + width * (len(keys) - 1) / 2)
    ax.set_xticklabels([label for label, _ in CHANGES], fontsize=9)
    ax.set_ylabel("ARL1: mean detection delay, samples")
    ax.set_title(f"Detection delay at equal measured ARL0 "
                 f"(~{STANDARD.target_arl0:.0f} samples), lower is better", fontsize=11)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8, ncol=3)
    fig.text(0.01, -0.04, "Error bars are Monte Carlo standard errors. The learned "
             "detector is the rightmost bar in each group.", fontsize=7.5, color="0.35")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
