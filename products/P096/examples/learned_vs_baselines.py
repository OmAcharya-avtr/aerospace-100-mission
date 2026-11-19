"""The learned component beside both baselines, on both held-out splits.

Baseline 2, the analytic fluid closed form, wins on both splits. That is the
product's finding and this figure is it: nothing here is arranged to flatter the
learned component and nothing was retuned after the measurement.

Three panels. The first two are dominance accuracy with event-level bootstrap
intervals, one per split. The third is the learned component's uncertainty output
against its nominal level, because an uncertainty output that is not calibrated
is still useful and only if its miscalibration is published.

Writes ``screenshots/learned_vs_baselines.png``.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

import _common  # noqa: E402

ROOT = _common.bootstrap()

from handoverloss.corpus import station_pair_split, time_split  # noqa: E402
from handoverloss.predictor import (  # noqa: E402
    HandoverLossPredictor,
    score_baselines,
    score_learned,
)
from handoverloss.sequence import MECHANISMS  # noqa: E402

RESAMPLES = 200 if _common.FAST else 2000
TREES = 80 if _common.FAST else None

SHORT = {
    "baseline1_acquisition_only": "baseline 1\nacquisition only",
    "baseline2_analytic_fluid": "baseline 2\nanalytic fluid",
    "learned_random_forest": "learned\nrandom forest",
}


def main() -> int:
    corpus = _common.corpus()
    handle, save = _common.figure("learned_vs_baselines", size=(13.5, 5.0))
    axes = handle.subplots(1, 3)

    printed: list[str] = []
    coverage_rows = []
    for panel, cut in enumerate((station_pair_split(corpus), time_split(corpus))):
        scores = list(score_baselines(corpus, cut, resamples=RESAMPLES))
        predictor = None if TREES is None else HandoverLossPredictor(n_estimators=TREES)
        learned, prediction, _ = score_learned(
            corpus, cut, predictor=predictor, resamples=RESAMPLES
        )
        scores.append(learned)

        positions = np.arange(len(scores))
        values = np.array([s.dominance_accuracy for s in scores])
        low = np.array([s.dominance_accuracy_low for s in scores])
        high = np.array([s.dominance_accuracy_high for s in scores])
        colours = ["#b3261e", "#36618e", "#6a7f23"]
        axes[panel].bar(
            positions, values,
            yerr=np.vstack([values - low, high - values]),
            capsize=6, color=colours,
        )
        # The floor any predictor must beat: the rate at which acquisition
        # genuinely dominates this split, which is baseline 1's accuracy by
        # construction.
        floor = float(
            np.mean(np.argmax(corpus.targets[cut.test], axis=1) == 0)
        )
        axes[panel].axhline(
            floor, color="k", linestyle="--", linewidth=1.0,
            label=f"always-acquisition floor {floor:.3f}",
        )
        axes[panel].set_xticks(positions)
        axes[panel].set_xticklabels([SHORT[s.predictor] for s in scores], fontsize=8)
        axes[panel].set_ylim(0.0, 1.0)
        axes[panel].set_ylabel("dominance accuracy")
        axes[panel].set_title(f"held-out {cut.name}\nn={cut.test.size}")
        axes[panel].legend(fontsize=7, loc="upper left")
        axes[panel].grid(axis="y", alpha=0.3)

        best = max(scores, key=lambda s: s.dominance_accuracy)
        printed.append(f"  SPLIT {cut.name}: n={cut.test.size}, held out {cut.held_out}")
        for score in scores:
            printed.append(
                f"    {score.predictor:<30} accuracy={score.dominance_accuracy:.6f} "
                f"[{score.dominance_accuracy_low:.6f}, {score.dominance_accuracy_high:.6f}] "
                f"log10MAE={score.log_mae_total:.6f}"
            )
        printed.append(f"    WINNER: {best.predictor} at {best.dominance_accuracy:.6f}")

        inside = (
            (corpus.targets[cut.test] >= prediction.low_bits)
            & (corpus.targets[cut.test] <= prediction.high_bits)
        )
        coverage_rows.append((cut.name, [float(inside[:, c].mean()) for c in range(3)]))

    width = 0.36
    base = np.arange(3)
    for offset, (name, values) in zip((-width / 2, width / 2), coverage_rows, strict=True):
        axes[2].bar(base + offset, values, width=width, label=f"held-out {name}")
    axes[2].axhline(0.90, color="k", linestyle="--", linewidth=1.0, label="nominal 0.90")
    axes[2].set_xticks(base)
    axes[2].set_xticklabels(MECHANISMS)
    axes[2].set_ylim(0.0, 1.0)
    axes[2].set_ylabel("achieved coverage of the 90% band")
    axes[2].set_title(
        "The learned component's uncertainty output\n"
        "across-tree spread, expected to undercover"
    )
    axes[2].legend(fontsize=7, loc="lower right")
    axes[2].grid(axis="y", alpha=0.3)

    handle.suptitle(
        "An analytic baseline beats the learned component on both held-out splits. "
        "That is the finding."
    )

    path = save()
    print("WHAT TO NOTICE")
    print(
        "  Baseline 2's bar is the tallest in both left-hand panels, and on the\n"
        "  station-pair split its interval does not overlap the learned component's.\n"
        "  The dashed line is the always-acquisition floor, which is baseline 1's\n"
        "  accuracy by construction: on the station-pair split the learned component\n"
        "  does not reach it. The right panel is why the uncertainty output is reported\n"
        "  with its measured coverage rather than its nominal level."
    )
    print("")
    print(f"  corpus sha256 {corpus.sha256}")
    for line in printed:
        print(line)
    print("")
    for name, values in coverage_rows:
        rendered = "  ".join(
            f"{mechanism}={value:.6f}"
            for mechanism, value in zip(MECHANISMS, values, strict=True)
        )
        print(f"  coverage, held-out {name:<14} {rendered}   (nominal 0.900000)")
    print("")
    print(f"figure written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
