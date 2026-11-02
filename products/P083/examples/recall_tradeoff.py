"""The learned classifier's whole recall-against-cost curve, and the certificate's point.

What to notice: the certificate sits at recall 1 by proof, marked with a
square. The learned curve is measured on held-out runs at a threshold chosen
on the training runs, and it reaches recall 1 only by flagging most of the
population. The dotted curve is the ORACLE: the same model scored at a
threshold chosen with the test labels in hand, which no user has. The gap
between the two curves is the part of the apparent performance that comes from
knowing the answer.
"""

from __future__ import annotations

import sys  # noqa: E402

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

from _common import (  # noqa: E402
    CERTIFIED,
    LEARNED,
    NEUTRAL,
    TOLERANCE,
    bootstrap,
    save,
    style,
)

ROOT = bootstrap()

import matplotlib.pyplot as plt  # noqa: E402

from hilverdict.certificate import certify_run  # noqa: E402
from hilverdict.features import feature_matrix  # noqa: E402
from hilverdict.learned import MissClassifier  # noqa: E402
from hilverdict.resimulate import resimulate_run  # noqa: E402
from hilverdict.scoring import operating_points, threshold_at_recall  # noqa: E402
from hilverdict.synthetic import CAMPAIGNS, generate_run  # noqa: E402

STEPS = 200
TRAIN = tuple(range(8200, 8254))
TEST = tuple(range(8300, 8336))


def gather(seeds: tuple[int, ...]):
    features = []
    labels = []
    bounds = []
    for position, seed in enumerate(seeds):
        profile, amplitude = CAMPAIGNS[position % len(CAMPAIGNS)]
        log = generate_run(
            seed=seed, profile=profile, amplitude_deg=amplitude, steps=STEPS
        )
        certificate = certify_run(log, 1.0)
        matrix, events = feature_matrix(log)
        if not events:
            continue
        results = resimulate_run(log, 1.0)
        features.append(matrix)
        labels.append(np.array([r.consequential for r in results], dtype=bool))
        bounds.append(np.array([c.bound for c in certificate.certificates]))
    return np.vstack(features), np.concatenate(labels), np.concatenate(bounds)


def main() -> int:
    style()
    train_features, train_labels, train_bounds = gather(TRAIN)
    test_features, test_labels, test_bounds = gather(TEST)
    classifier = MissClassifier().fit(train_features, train_labels)
    probability = classifier.predict_proba(test_features)
    _thresholds, recall, flag_rate = operating_points(probability, test_labels)
    chosen, _train_recall, _train_flag = threshold_at_recall(
        classifier.predict_proba(train_features), train_labels, 1.0
    )
    honest_flag = float(np.mean(probability >= chosen))
    honest_recall = float(
        np.count_nonzero((probability >= chosen) & test_labels)
        / np.count_nonzero(test_labels)
    )
    certificate_flag = float(np.mean(test_bounds >= 1.0))

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].plot(flag_rate, recall, color=LEARNED, lw=1.4,
                 label="learned, every threshold (oracle curve)")
    axes[0].scatter([honest_flag], [honest_recall], s=70, marker="o", c=LEARNED,
                    label="learned at the TRAIN-chosen threshold")
    axes[0].scatter([certificate_flag], [1.0], s=90, marker="s", c=CERTIFIED,
                    label="analytic certificate (recall 1 by proof)")
    axes[0].axhline(1.0, color=TOLERANCE, lw=1.0, ls="--")
    axes[0].set_xlabel("fraction of misses sent to re-simulation")
    axes[0].set_ylabel("recall of consequential misses")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_title(
        f"{test_labels.shape[0]} held-out misses, "
        f"{int(np.count_nonzero(test_labels))} consequential"
    )
    axes[0].legend(loc="lower right", fontsize=7.5)

    residual = test_bounds >= 1.0
    train_residual = train_bounds >= 1.0
    classes = set(np.unique(train_labels[train_residual]).tolist())
    if classes == {False, True}:
        residual_model = MissClassifier().fit(
            train_features[train_residual], train_labels[train_residual]
        )
        residual_probability = residual_model.predict_proba(test_features[residual])
        _rt, residual_recall, residual_flag = operating_points(
            residual_probability, test_labels[residual]
        )
        axes[1].plot(residual_flag, residual_recall, color=LEARNED, lw=1.4,
                     label="residual-only model, oracle curve")
        residual_chosen, _r, _f = threshold_at_recall(
            residual_model.predict_proba(train_features[train_residual]),
            train_labels[train_residual],
            1.0,
        )
        flagged = residual_probability >= residual_chosen
        axes[1].scatter(
            [float(np.mean(flagged))],
            [float(np.count_nonzero(flagged & test_labels[residual])
                   / max(np.count_nonzero(test_labels[residual]), 1))],
            s=70, marker="o", c=LEARNED,
            label="at the TRAIN-chosen threshold",
        )
    axes[1].scatter([1.0], [1.0], s=90, marker="s", c=NEUTRAL,
                    label="re-simulate the whole residual")
    axes[1].axhline(1.0, color=TOLERANCE, lw=1.0, ls="--")
    axes[1].set_xlabel("fraction of the RESIDUAL re-simulated")
    axes[1].set_ylabel("recall of consequential misses in the residual")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_title(
        f"the residual: {int(np.count_nonzero(residual))} misses, "
        f"{int(np.count_nonzero(test_labels[residual]))} consequential"
    )
    axes[1].legend(loc="lower right", fontsize=7.5)

    figure.suptitle(
        "hilverdict: recall against cost, scored on held-out runs", fontsize=10
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    path = save(figure, "recall_tradeoff")
    print(f"written: {path}")
    print(f"certificate: recall 1.000000 at flag rate {certificate_flag:.6f}")
    print(f"learned at the train-chosen threshold: recall {honest_recall:.6f} at "
          f"flag rate {honest_flag:.6f}")
    print(f"test misses {test_labels.shape[0]}, consequential "
          f"{int(np.count_nonzero(test_labels))}")
    print(f"np version {np.__version__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
