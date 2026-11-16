"""Classify every corpus script with the two-arrangement runner, and plot it.

The mechanism, demonstrated on ground truth. Each script is run twice in two
PRISTINE copies and twice in ONE reused tree. A verdict that changes between
two pristine runs is intrinsic non-determinism; one that changes between run 1
and run 2 of the same tree while the pristine pair agrees is a self-polluting
check. Neither arrangement alone says which defect it is.

Writes ``screenshots/arrangement_classifier.png``.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()
_bootstrap.style()

import matplotlib.pyplot as plt  # noqa: E402

from seedaudit.corpus import (  # noqa: E402
    INJECTED_RATES,
    ITERATION_ORDER_SCRIPT,
    NEGATIVE_CONTROLS,
    SELF_POLLUTING_SCRIPT,
    SUB_PRECISION_SCRIPT,
    generate_corpus,
)
from seedaudit.dynamic import (  # noqa: E402
    INTRINSIC,
    INTRINSIC_MASKED,
    NO_EVIDENCE,
    SELF_POLLUTING,
    two_arrangements,
)
from seedaudit.power import (  # noqa: E402
    self_pollution_misclassification,
    worst_case_misclassification,
    worst_case_misclassification_rate,
)

#: Runs in each arrangement. Two is Baseline 1's budget and is the point.
REPEATS = 2

#: The order the classifications are drawn in.
LABELS = (SELF_POLLUTING, INTRINSIC, INTRINSIC_MASKED, NO_EVIDENCE)

#: Colours, one per classification.
COLOURS = {
    SELF_POLLUTING: "#b2182b",
    INTRINSIC: "#ef8a62",
    INTRINSIC_MASKED: "#f7c9a8",
    NO_EVIDENCE: "#92c5de",
}


def artifact_for(name: str) -> str | None:
    """The output file a corpus script writes, if it writes one."""
    return "self_polluting_output.txt" if name == SELF_POLLUTING_SCRIPT else None


def main() -> int:
    """Classify every corpus script and write the figure."""
    print(f"runs per arrangement: {REPEATS}")
    print(f"{'script':>38}  {'pristine':>9}  {'reused':>7}  classification")
    names: list[str] = []
    classifications: list[str] = []
    pristine_agreement: list[bool] = []
    reused_agreement: list[bool] = []
    with tempfile.TemporaryDirectory(prefix="seedaudit-arrangement-") as temporary:
        manifest = generate_corpus(Path(temporary) / "corpus")
        ordered = (
            [manifest.rate_scripts[rate] for rate in INJECTED_RATES]
            + list(NEGATIVE_CONTROLS)
            + [SELF_POLLUTING_SCRIPT, ITERATION_ORDER_SCRIPT, SUB_PRECISION_SCRIPT]
        )
        for name in ordered:
            result = two_arrangements(
                manifest.root,
                name,
                artifact_for(name),
                timeout=60.0,
                pristine_repeats=REPEATS,
                reused_repeats=REPEATS,
            )
            names.append(name.removesuffix(".py"))
            classifications.append(result.classification)
            pristine_agreement.append(result.pristine_agree)
            reused_agreement.append(result.reused_agree)
            print(
                f"{name:>38}  {str(result.pristine_agree):>9}  "
                f"{str(result.reused_agree):>7}  {result.classification}"
            )

    print()
    print("THE CLASSIFIER'S OWN ERROR RATE, which a run of this example forced into")
    print("the product. A script with intrinsic rate p and NO self-pollution is")
    print("called self-polluting whenever its pristine pair happens to agree and its")
    print("reused pair happens to differ, which has probability q(1-q) with")
    print("q = p^n + (1-p)^n.")
    print()
    print(f"{'n':>4}  {'worst-case rate':>16}  {'worst-case probability':>23}"
          f"  {'at p = 0.5':>11}")
    for runs in (2, 3, 5, 10, 20):
        print(
            f"{runs:>4}  {worst_case_misclassification_rate(runs):>16.9f}"
            f"  {worst_case_misclassification(runs):>23.9f}"
            f"  {self_pollution_misclassification(0.5, runs):>11.9f}"
        )
    print()
    print("raising the runs per arrangement does NOT lower the worst case: it stays")
    print("at 0.25 and only the rate at which it happens moves. The mitigation is")
    print("independent repetitions of the WHOLE experiment, which multiply:")
    for trials in (1, 2, 3, 4):
        print(f"  {trials} trials: worst case {worst_case_misclassification(2, trials)!r}")

    figure, axes = plt.subplots(1, 2, figsize=(10.6, 4.2), width_ratios=[1.45, 1.0])

    left = axes[0]
    positions = np.arange(len(names))
    for index, label in enumerate(classifications):
        left.barh(positions[index], 1.0, color=COLOURS[label], height=0.72)
    left.set_yticks(positions)
    left.set_yticklabels(names, fontsize=7.5)
    left.invert_yaxis()
    left.set_xticks([])
    left.set_xlim(0, 1)
    left.grid(False)
    left.set_title(f"Classification of every corpus script, n = {REPEATS} per arrangement")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOURS[label]) for label in LABELS]
    left.legend(
        handles,
        LABELS,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.04),
        ncols=2,
        fontsize=7.5,
    )

    right = axes[1]
    counts = np.zeros((2, 2))
    for agree_pristine, agree_reused in zip(
        pristine_agreement, reused_agreement, strict=True
    ):
        counts[0 if agree_pristine else 1, 0 if agree_reused else 1] += 1
    right.imshow(counts, cmap="Blues", vmin=0, vmax=max(counts.max(), 1.0))
    cell_labels = (
        (NO_EVIDENCE, SELF_POLLUTING),
        (INTRINSIC_MASKED, INTRINSIC),
    )
    for row in range(2):
        for column in range(2):
            right.text(
                column,
                row,
                f"{int(counts[row, column])}\n{cell_labels[row][column]}",
                ha="center",
                va="center",
                fontsize=7.5,
                color="0.1",
            )
    right.set_xticks([0, 1])
    right.set_xticklabels(["reused tree\nagrees", "reused tree\ndiffers"], fontsize=8)
    right.set_yticks([0, 1])
    right.set_yticklabels(["pristine pair\nagrees", "pristine pair\ndiffers"], fontsize=8)
    right.grid(False)
    right.set_title("The cross-tabulation, counted")

    figure.tight_layout()
    target = ROOT / _bootstrap.SCREENSHOTS / "arrangement_classifier.png"
    figure.savefig(target)
    plt.close(figure)
    print()
    print(f"figure written to {_bootstrap.relative(target)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
