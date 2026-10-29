"""The learned model's decision margin, and where it is confident and wrong.

What to notice: the margin separates correct predictions from wrong ones, so it
carries information. But on the subset where equation (27) says the analytic
continuity test cannot resolve the declared threshold, the model still answers
with a high margin, and some of those answers are wrong. The analytic screen's
answer on the same records is that it cannot tell.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import BENIGN, SILENT, THRESHOLD, UNDETERMINABLE, bootstrap, save, style

bootstrap()

import matplotlib.pyplot as plt  # noqa: E402

from tlecheck import features  # noqa: E402
from tlecheck.learned import LearnedClassifier  # noqa: E402

CONFIGURATION = "elements_and_history"


def main() -> None:
    style()
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "validation"))
    from _bootstrap import prepare_splits

    splits = prepare_splits()
    train, test, calibration = splits["train"], splits["test"], splits["calibration"]
    model = LearnedClassifier().fit(train.matrices[CONFIGURATION], train.measured_classes)

    figure, axes = plt.subplots(1, 3, figsize=(12.0, 3.9))

    prediction = model.predict(calibration.matrices[CONFIGURATION])
    correct = prediction.labels == calibration.measured_classes
    bins = np.linspace(0.0, 1.0, 21)
    axes[0].hist(
        prediction.margin[correct], bins=bins, color=BENIGN, alpha=0.8, label="correct"
    )
    axes[0].hist(
        prediction.margin[~correct], bins=bins, color=SILENT, alpha=0.8, label="wrong"
    )
    axes[0].set_xlabel("equation (20) decision margin")
    axes[0].set_ylabel("element sets")
    axes[0].set_title("Calibration split, margin by outcome", loc="left")
    axes[0].legend(fontsize=7.5)

    axes[1].hist(
        prediction.dispersion[correct], bins=20, color=BENIGN, alpha=0.8, label="correct"
    )
    axes[1].hist(
        prediction.dispersion[~correct], bins=20, color=SILENT, alpha=0.8, label="wrong"
    )
    axes[1].set_xlabel("equation (19) ensemble dispersion")
    axes[1].set_title("Calibration split, dispersion by outcome", loc="left")
    axes[1].legend(fontsize=7.5)

    names = features.feature_names(CONFIGURATION)
    test_prediction = model.predict(test.matrices[CONFIGURATION])
    resolves = (
        test.matrices[CONFIGURATION][:, names.index("continuity_resolves_threshold")] == 1.0
    )
    wrong = test_prediction.labels != test.measured_classes
    groups = [
        ("resolves\nthreshold", resolves),
        ("does NOT\nresolve", ~resolves),
    ]
    positions = np.arange(len(groups))
    confident = []
    other = []
    for _label, mask in groups:
        subset = wrong & mask
        margins = test_prediction.margin[subset]
        confident.append(int((margins > 0.5).sum()))
        other.append(int((margins <= 0.5).sum()))
    axes[2].bar(positions, other, color=UNDETERMINABLE, label="margin at or below 0.5")
    axes[2].bar(
        positions, confident, bottom=other, color=THRESHOLD, label="margin above 0.5"
    )
    for index, (low, high) in enumerate(zip(other, confident, strict=True)):
        axes[2].text(index, low + high + 0.4, f"{high} of {low + high}", ha="center", fontsize=7.5)
    axes[2].set_xticks(positions)
    axes[2].set_xticklabels([label for label, _mask in groups], fontsize=7.6)
    axes[2].set_ylabel("wrong predictions on the test split")
    axes[2].set_title("Confident and wrong, by equation (27)", loc="left")
    axes[2].legend(fontsize=7.2, loc="upper left")

    figure.subplots_adjust(wspace=0.32)
    print(save(figure, "learned_confidence"))


if __name__ == "__main__":
    main()
