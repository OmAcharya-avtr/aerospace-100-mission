"""Held-out splits and the scoring loop that compares every method fairly.

TWO SPLITS, SCORED SEPARATELY AND NEVER AVERAGED TOGETHER
---------------------------------------------------------
- **Held-out ARTIFACTS.** ``sklearn.model_selection.GroupKFold`` with the
  subject artifact as the group, so every evidence item about an artifact is on
  the same side of the split. Without this, two evidence items about the same
  artifact -- which go stale together -- land on both sides and the score is
  optimistic for a reason that has nothing to do with generalisation.
- **Held-out TIME.** A forward-chaining split on the production revision: train
  on evidence produced at or before a cutoff, test on evidence produced after
  it. Three cutoffs at the 50th, 65th and 80th percentiles of the production
  revision, each scored on its own test set.

The two answer different questions -- "does this transfer to an artifact the
model has not seen" and "does this transfer to evidence produced later" -- and a
single averaged figure would answer neither.

EVERY METHOD IS SCORED ON THE SAME TEST ITEMS. The baselines are deterministic
functions of the corpus, so they are evaluated on the test subset of each fold
rather than refitted; that is what makes the comparison a comparison.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import GroupKFold

from .metrics import ErrorDirections, brier_score, error_directions, roc_auc


@dataclass(frozen=True)
class Split:
    """One train/test split, as index arrays into the evidence sequence.

    Attributes
    ----------
    kind:
        ``"artifact"`` or ``"time"``.
    label:
        Human-readable fold label used in reports.
    train, test:
        Index arrays into the evidence sequence.
    """

    kind: str
    label: str
    train: np.ndarray
    test: np.ndarray


@dataclass(frozen=True)
class Score:
    """Scores of one method on one split.

    ``auc`` and ``brier`` are ``nan`` for a method with no probabilistic output
    or for a test set with one class only, and the ``nan`` is reported rather
    than replaced.
    """

    method: str
    split_kind: str
    split_label: str
    n_test: int
    n_positive: int
    auc: float
    brier: float
    errors: ErrorDirections

    @property
    def agreement(self) -> float:
        """Fraction of test items on which the method agrees with the truth."""
        return self.errors.agreement


def artifact_splits(groups: Sequence[str], n_splits: int = 5) -> list[Split]:
    """Grouped folds with the subject artifact as the group."""
    distinct = len(set(groups))
    if distinct < n_splits:
        raise ValueError(
            f"GroupKFold with n_splits={n_splits} needs at least that many distinct "
            f"subject artifacts, got {distinct}. Lower n_splits or use a corpus with "
            f"more subjects."
        )
    splitter = GroupKFold(n_splits=n_splits)
    dummy = np.zeros((len(groups), 1), dtype=np.float64)
    out: list[Split] = []
    for fold, (train, test) in enumerate(
        splitter.split(dummy, np.zeros(len(groups)), groups=list(groups))
    ):
        out.append(
            Split(
                kind="artifact",
                label=f"groupkfold_fold{fold}",
                train=np.asarray(train, dtype=np.int64),
                test=np.asarray(test, dtype=np.int64),
            )
        )
    return out


def time_splits(
    produced_at: Sequence[int], quantiles: Sequence[float] = (0.50, 0.65, 0.80)
) -> list[Split]:
    """Forward-chaining splits on the production revision at each quantile."""
    revisions = np.asarray(list(produced_at), dtype=np.int64)
    out: list[Split] = []
    for quantile in quantiles:
        cutoff = int(np.quantile(revisions, quantile))
        train = np.flatnonzero(revisions <= cutoff)
        test = np.flatnonzero(revisions > cutoff)
        if train.size == 0 or test.size == 0:
            raise ValueError(
                f"the time split at quantile {quantile} puts every item on one side "
                f"(cutoff revision {cutoff}); the production revisions are too "
                f"concentrated for this quantile."
            )
        out.append(
            Split(
                kind="time",
                label=f"produced_at<={cutoff}_q{quantile:.2f}",
                train=train,
                test=test,
            )
        )
    return out


def score_binary(
    method: str,
    split: Split,
    decisions: Sequence[int],
    truth: Sequence[int],
) -> Score:
    """Score a deterministic 0/1 method on the test part of ``split``.

    AUC is reported from the binary decisions, which is the honest reading for
    a method with no score to rank by: for a binary predictor the area equals
    ``(sensitivity + specificity) / 2``.
    """
    test_decisions = [int(decisions[index]) for index in split.test]
    test_truth = [int(truth[index]) for index in split.test]
    errors = error_directions(test_decisions, test_truth)
    return Score(
        method=method,
        split_kind=split.kind,
        split_label=split.label,
        n_test=len(test_truth),
        n_positive=sum(test_truth),
        auc=roc_auc([float(value) for value in test_decisions], test_truth),
        brier=brier_score([float(value) for value in test_decisions], test_truth),
        errors=errors,
    )


def score_probabilistic(
    method: str,
    split: Split,
    probabilities: Sequence[float],
    truth: Sequence[int],
    threshold: float = 0.5,
) -> Score:
    """Score a probabilistic method on the test part of ``split``.

    ``probabilities`` is indexed by TEST POSITION, not by evidence position,
    because a fitted model only produces values for its own test rows.
    """
    test_truth = [int(truth[index]) for index in split.test]
    if len(probabilities) != len(test_truth):
        raise ValueError(
            f"probabilities has {len(probabilities)} entries and the test set has "
            f"{len(test_truth)} items; pass the model's predictions for the test rows "
            f"of this split, in test order."
        )
    values = [float(value) for value in probabilities]
    decisions = [int(value >= threshold) for value in values]
    return Score(
        method=method,
        split_kind=split.kind,
        split_label=split.label,
        n_test=len(test_truth),
        n_positive=sum(test_truth),
        auc=roc_auc(values, test_truth),
        brier=brier_score(values, test_truth),
        errors=error_directions(decisions, test_truth),
    )


def summarise(scores: Sequence[Score]) -> dict[str, float]:
    """Mean of each reported figure over ``scores``, ignoring ``nan`` entries.

    The count of ``nan`` entries ignored is returned as ``nan_auc`` and
    ``nan_brier`` so a mean over fewer folds cannot be read as a mean over all.
    """
    if not scores:
        raise ValueError("summarise() needs at least one score")
    aucs = [score.auc for score in scores if not np.isnan(score.auc)]
    briers = [score.brier for score in scores if not np.isnan(score.brier)]
    agreements = [
        score.agreement for score in scores if not np.isnan(score.agreement)
    ]
    missed = [
        score.errors.missed_staleness_rate
        for score in scores
        if not np.isnan(score.errors.missed_staleness_rate)
    ]
    alarms = [
        score.errors.false_alarm_rate
        for score in scores
        if not np.isnan(score.errors.false_alarm_rate)
    ]
    return {
        "folds": float(len(scores)),
        "auc": float(np.mean(aucs)) if aucs else float("nan"),
        "brier": float(np.mean(briers)) if briers else float("nan"),
        "agreement": float(np.mean(agreements)) if agreements else float("nan"),
        "missed_staleness_rate": float(np.mean(missed)) if missed else float("nan"),
        "false_alarm_rate": float(np.mean(alarms)) if alarms else float("nan"),
        "nan_auc": float(len(scores) - len(aucs)),
        "nan_brier": float(len(scores) - len(briers)),
    }
