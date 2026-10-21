"""Learned screen against the analytic screens, at matched false-positive rates.

What to notice, left panel: on the broad "nominal" population the Mahalanobis
screen is the worst of the three analytic screens, not the best, because it is
invariant to the hard-body size and the covariance scale, and those vary over
decades here. The learned model wins, but the one-line small-hard-body closed
form, which is given exactly the same features, comes within a few points of it.

Right panel: on the "homogeneous" population, where the hard body and the
covariance barely vary, the Mahalanobis screen recovers almost all of its
recall. The reason the baseline looks weak on the left is the spread of
R / sigma_eff in the population, and that is measured here rather than argued.
"""

from __future__ import annotations

import numpy as np

from _common import (
    LEARNED,
    MAHALANOBIS_SCREEN,
    MISS_SCREEN,
    SMALL_BODY,
    bootstrap,
    save,
    style,
)

bootstrap()

from conjprob.dataset import generate_population  # noqa: E402
from conjprob.screening import (  # noqa: E402
    ScreeningClassifier,
    brier_score,
    compare_screens,
    recall_at_fpr,
)
from conjprob.screens import all_screen_scores  # noqa: E402

N_TRAIN = 24_000
N_TEST = 12_000
SEED = 0
FPR_GRID = np.logspace(-3.0, -0.3, 22)
COLOURS = {
    "miss_distance": MISS_SCREEN,
    "mahalanobis": MAHALANOBIS_SCREEN,
    "small_body": SMALL_BODY,
    "learned": LEARNED,
}
STYLES = {
    "miss_distance": "-",
    "mahalanobis": "-",
    "small_body": "--",
    "learned": "-",
}


def curves(labels, scores: dict) -> dict:
    """Recall against achieved false-positive rate for every score."""
    out = {}
    for name, values in scores.items():
        points = [recall_at_fpr(labels, values, target, name=name) for target in FPR_GRID]
        out[name] = (
            np.array([point.fpr_achieved for point in points]),
            np.array([point.recall for point in points]),
        )
    return out


def main() -> None:
    style()
    import matplotlib.pyplot as plt

    train = generate_population(N_TRAIN, seed=SEED, regime="nominal")
    model = ScreeningClassifier(random_state=SEED).fit(train.features, train.labels)

    figure, axes = plt.subplots(1, 2, figsize=(11.6, 4.4), sharey=True)
    for axis, regime in zip(axes, ("nominal", "homogeneous"), strict=True):
        test = generate_population(N_TEST, seed=SEED + 1000, regime=regime)
        scores = dict(all_screen_scores(test.features))
        scores["learned"] = model.score_samples(test.features)
        for name, (fpr, recall) in curves(test.labels, scores).items():
            axis.semilogx(
                np.maximum(fpr, 1e-5),
                recall,
                marker="o",
                markersize=2.6,
                linewidth=1.4,
                linestyle=STYLES[name],
                color=COLOURS[name],
                label=name,
            )
        axis.set_xlabel("achieved false-positive rate")
        axis.set_title(
            f"test regime {regime}: {len(test)} conjunctions, "
            f"{int(test.labels.sum())} above Pc = {test.threshold:.0e}"
        )
        axis.set_ylim(0.0, 1.03)
    axes[0].set_ylabel("recall")
    axes[0].legend(loc="lower right")
    figure.suptitle(
        "Screening recall at matched false-positive rate; trained on the nominal regime only",
        y=1.02,
    )
    print(f"wrote {save(figure, 'screening_performance.png')}")

    for regime in ("nominal", "homogeneous", "dilution"):
        test = generate_population(N_TEST, seed=SEED + 1000, regime=regime)
        probabilities = model.predict_proba(test.features)
        comparison = compare_screens(
            test.labels, probabilities, test.features, (0.01, 0.05, 0.10)
        )
        base = float(test.labels.mean())
        print(
            f"\nregime {regime}: n = {len(test)}, positives = {int(test.labels.sum())}, "
            f"base rate {base:.4f}, Brier {brier_score(test.labels, probabilities):.6f} "
            f"against {base * (1.0 - base):.6f} for predicting the base rate"
        )
        for name in ("miss_distance", "mahalanobis", "small_body", "learned"):
            for point in comparison[name]:
                print(
                    f"  {name:<14s} fpr target {point.fpr_target:.2f} achieved "
                    f"{point.fpr_achieved:.5f}  recall {point.recall:.5f}  "
                    f"precision {point.precision:.5f}"
                )


if __name__ == "__main__":
    main()
