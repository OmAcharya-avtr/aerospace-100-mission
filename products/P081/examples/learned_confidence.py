"""The headline: confident attribution exactly where the analytic method refuses.

What to notice: the left panel is the specified learned classifier's confidence
on two populations. On ordinary single-parameter faults it is confident and
right. Inside the unidentifiable subspace -- where each case produces telemetry
identical to a single-parameter fault, so no single-parameter answer can be
correct -- it is just as confident and necessarily wrong. The right panel puts
the accuracies side by side with the analytic path's, under a scoring convention
in which "the telemetry cannot tell" is the correct answer wherever it is true.
"""

from __future__ import annotations

import sys  # noqa: E402

sys.dont_write_bytecode = True

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from _common import ACCENT, ANALYTIC, LEARNED, REFUSED, bootstrap, save, style  # noqa: E402

ROOT = bootstrap()

from twindiverge.identifiability import analyse  # noqa: E402
from twindiverge.learned import ABSTAIN_THRESHOLD, ResidualClassifier  # noqa: E402
from twindiverge.pipeline import analyse_case, learned_study  # noqa: E402
from twindiverge.plant import NOMINAL, simulate_with_sensitivities  # noqa: E402
from twindiverge.synthetic import generate_cases  # noqa: E402
from twindiverge.telemetry import TELEMETRY_SETS  # noqa: E402

TRAIN = tuple(range(1000, 1240))
TEST = tuple(range(20000, 20120))
NULL = tuple(range(30000, 30120))
SET_NAME = "rate_full"


def main() -> int:
    style()
    telemetry = TELEMETRY_SETS[SET_NAME]
    report = analyse(telemetry, NOMINAL)
    twin = simulate_with_sensitivities(NOMINAL, telemetry.campaign)
    train = [
        analyse_case(telemetry, case, report, twin)
        for case in generate_cases(TRAIN, "single")
    ]
    model = ResidualClassifier("parameter").fit(
        np.vstack([item.features for item in train]),
        np.array([item.case.label for item in train]),
    )
    confidences = {}
    for kind in ("single", "equivalent", "scale"):
        seeds = NULL if kind == "scale" else TEST
        values = []
        for case in generate_cases(seeds, kind):
            analysis = analyse_case(telemetry, case, report, twin)
            values.append(model.predict_one(analysis.features).confidence)
        confidences[kind] = np.array(values)

    study = learned_study(telemetry, TRAIN, TEST, NULL)

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    bins = np.linspace(0.2, 1.0, 17)
    labels = {
        "single": ("single-parameter faults (answerable)", ACCENT),
        "equivalent": ("inside the unidentifiable subspace", LEARNED),
        "scale": ("pure scale change: residual is noise only", ANALYTIC),
    }
    for kind, (label, colour) in labels.items():
        axes[0].hist(
            confidences[kind], bins=bins, histtype="step", linewidth=1.8,
            color=colour, label=f"{label}\nmean {confidences[kind].mean():.4f}",
        )
    axes[0].axvline(
        ABSTAIN_THRESHOLD, color=REFUSED, linestyle="--", linewidth=1.0,
    )
    axes[0].text(
        ABSTAIN_THRESHOLD + 0.01, axes[0].get_ylim()[1] * 0.95,
        f"declared abstention threshold {ABSTAIN_THRESHOLD:.2f}",
        fontsize=7, color=REFUSED, va="top",
    )
    axes[0].set_xlabel("learned confidence (largest class probability)")
    axes[0].set_ylabel("cases")
    axes[0].set_title("the classifier's confidence, by population")
    axes[0].legend(fontsize=7, loc="upper left")

    populations = ("single", "weak", "equivalent", "scale")
    methods = (
        ("analytic-gated", "analytic, gated by baseline 1", ANALYTIC),
        ("analytic-constrained", "analytic, eq (15) assuming sparsity", ACCENT),
        ("learned-parameter", "learned, five classes", LEARNED),
    )
    width = 0.26
    positions = np.arange(len(populations))
    for index, (key, label, colour) in enumerate(methods):
        values = [
            study.population(name).methods[key].accuracy.value for name in populations
        ]
        axes[1].bar(
            positions + (index - 1) * width, values, width, color=colour,
            edgecolor="black", linewidth=0.4, label=label,
        )
        for position, value in zip(positions, values, strict=True):
            axes[1].text(
                position + (index - 1) * width, value + 0.015, f"{value:.3f}",
                ha="center", fontsize=6,
            )
    axes[1].set_xticks(positions, populations)
    axes[1].set_ylim(0.0, 1.15)
    axes[1].set_ylabel("accuracy under the declared scoring convention")
    axes[1].set_title(f"accuracy per population, telemetry set {SET_NAME}")
    axes[1].legend(fontsize=7, loc="upper right")
    figure.suptitle(
        "twindiverge: the learned component is confident where the analytic one refuses",
        fontsize=11,
    )
    path = save(figure, "learned_confidence.png")
    print(f"written: {path}")
    for kind, values in confidences.items():
        print(
            f"  {kind:<12s} n={values.size:4d} mean confidence {values.mean():.6f} "
            f"least {values.min():.6f} greatest {values.max():.6f} "
            f"abstentions {int((values < ABSTAIN_THRESHOLD).sum())}"
        )
    identical = bool(
        np.array_equal(confidences["single"], confidences["equivalent"])
    )
    print(
        f"  the single and equivalent confidence distributions are identical: "
        f"{identical}"
    )
    print(
        "  they must be: the two populations are the same observations to within "
        "two units"
    )
    print(
        "  in the last place, so the classifier cannot answer them differently, and "
        "their"
    )
    print("  true causes are nevertheless different.")
    for name in populations:
        block = study.population(name)
        print(f"  population {name}:")
        for key, label, _ in methods:
            score = block.methods[key]
            print(
                f"    {label:<36s} accuracy {score.accuracy.value:.6f} "
                f"confident and wrong {score.confident_error.value:.6f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
