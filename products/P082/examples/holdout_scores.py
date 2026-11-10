"""Held-out SITES and held-out TIME, side by side and never merged.

The figure that carries this product's research finding: a spatio-temporal model
can pass one hold-out and fail the other, and on this record it does. Writes
``screenshots/holdout_scores.png``.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

from _common import (  # noqa: E402
    EMPIRICAL,
    FITTED,
    INDEPENDENT,
    LEARNED,
    NEUTRAL,
    bootstrap,
    save,
    style,
)

ROOT = bootstrap()

import matplotlib.pyplot as plt  # noqa: E402

from sitediverse.field import RECORD_SPECS, generate_record  # noqa: E402
from sitediverse.splits import CELL_NAMES  # noqa: E402
from sitediverse.study import MODEL_NAMES, run_study  # noqa: E402

COLOURS = {
    "independent": INDEPENDENT,
    "exponential": FITTED,
    "per_regime": EMPIRICAL,
    "learned": LEARNED,
}


def main() -> int:
    style()
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    printed = []
    for index, name in enumerate(("isotropic", "anisotropic")):
        record = generate_record(RECORD_SPECS[name])
        study = run_study(record, name, margin_replicates=200, margin_block=64)
        axis = axes[index]
        positions = np.arange(len(CELL_NAMES))
        width = 0.2
        for offset, model in enumerate(MODEL_NAMES):
            values = [study.scores[cell].joint_mae[model] for cell in CELL_NAMES]
            axis.bar(
                positions + (offset - 1.5) * width,
                values,
                width,
                color=COLOURS[model],
                label=model,
            )
        for result in study.margins:
            if result.baseline != study.strongest_baseline:
                continue
            cell_index = CELL_NAMES.index(result.cell)
            axis.errorbar(
                cell_index + 1.5 * width,
                study.scores[result.cell].joint_mae["learned"],
                yerr=result.reference_half_width,
                color=NEUTRAL,
                capsize=4,
                linestyle="none",
                marker="none",
            )
        axis.set_xticks(positions)
        axis.set_xticklabels(CELL_NAMES, rotation=12, ha="right")
        axis.set_ylabel("mean absolute error in joint unavailability")
        correct = (
            "baseline CORRECTLY specified"
            if name == "isotropic"
            else "baseline MIS-specified"
        )
        axis.set_title(f"{name} record -- {correct}")
        if index == 0:
            axis.legend(loc="upper left")
        printed.append((name, study))

    path = save(figure, "holdout_scores.png")
    print(f"written: {path}")
    print("  the grey bars are the bootstrap half-width on the empirical reference")
    for name, study in printed:
        print(f"  {name}:")
        for cell in CELL_NAMES:
            score = study.scores[cell]
            print(f"    {cell:16s} "
                  + " ".join(
                      f"{model}={score.joint_mae[model]:.6f}"
                      for model in MODEL_NAMES
                  ))
        for result in study.margins:
            if result.baseline != study.strongest_baseline:
                continue
            print(f"    margin {result.cell:16s} = {result.margin:+.6f}, "
                  f"reference half-width {result.reference_half_width:.6f}, "
                  f"inside: {result.inside_reference_half_width}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
