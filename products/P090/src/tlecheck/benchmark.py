"""Scoring: per-class precision and recall, and recall at a fixed false-positive rate.

The headline comparison this product exists to report is **recall at a fixed
false-positive rate on the silently-wrong class**, because a silently bad
element set that passes screening is the whole failure the product addresses
and a screen that catches it by flagging everything has not helped.

Three predictors are scored on the same held-out split with the same features
and the same labels:

1. the rule baseline of :mod:`tlecheck.verdict`, at its own operating point
   **and** as a swept curve over equation (15), so it is not understated;
2. the learned classifier of :mod:`tlecheck.learned`, in each of its three
   declared feature configurations;
3. :class:`tlecheck.learned.MajorityClassifier`, whose curve is the chance
   diagonal.

Equations
---------
(21) Per-class precision and recall, with the convention that an empty
     denominator is reported as ``nan`` rather than as 0 or 1:

         precision_c = TP_c / (TP_c + FP_c),   recall_c = TP_c / (TP_c + FN_c)

(22) Recall at a fixed false-positive rate. Over the score thresholds, take
     the operating point of greatest recall whose false-positive rate does not
     exceed the target. The achieved false-positive rate is reported beside the
     recall, because it is usually below the target and quoting the target as
     if it were achieved would overstate the result.

(23) Area under the operating characteristic, by the trapezoidal rule over the
     swept points. Reported for every predictor, with the majority baseline's
     value as the reference for chance.

Units
-----
All quantities here are dimensionless rates and counts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import features as features_module
from . import screens as screens_module
from .consequence import CLASSES, SILENTLY_WRONG, measure
from .dataset import Record
from .verdict import classify


@dataclass(frozen=True)
class ClassScore:
    """Equation (21) for one class."""

    label: str
    precision: float
    recall: float
    support: int
    true_positive: int
    false_positive: int
    false_negative: int


@dataclass(frozen=True)
class OperatingPoint:
    """One point of an operating characteristic."""

    threshold: float
    recall: float
    false_positive_rate: float
    flagged: int


@dataclass(frozen=True)
class Prepared:
    """One split, screened, measured and featurised.

    Attributes
    ----------
    records:
        The split.
    evidences:
        Screen evidence per record, in the same order.
    measured_classes:
        Measured consequence class per record. This is the label.
    measured_max_error_km:
        Equation (11) maximum per record, ``nan`` for a loud case.
    rule_classes:
        The rule baseline's predicted class per record.
    rule_scores:
        Equation (15) per record.
    matrices:
        Design matrix per declared feature configuration.
    """

    records: list[Record]
    evidences: list[screens_module.ScreenEvidence]
    measured_classes: np.ndarray
    measured_max_error_km: np.ndarray
    rule_classes: np.ndarray
    rule_scores: np.ndarray
    matrices: dict[str, np.ndarray]


def prepare(
    records: list[Record],
    epoch_requirement_days: float = 7.0,
    history_threshold: float = 4.0,
    divergence_threshold_km: float = 1.0,
    continuity_threshold: float = 4.0,
    horizon_days: float = 1.0,
    samples: int = 25,
    silent_threshold_km: float = 1.0,
    use_offset_days: float = 1.0,
) -> Prepared:
    """Screen, measure and featurise a split.

    The measured consequence class is produced by actually propagating each
    candidate and its uncorrupted reference with ``sgp4`` and differencing the
    positions. No label in this function is assumed from the corruption family.

    Raises
    ------
    ValueError
        If ``records`` is empty.
    """
    if not records:
        raise ValueError("records must be non-empty")
    evidences: list[screens_module.ScreenEvidence] = []
    measured: list[str] = []
    errors: list[float] = []
    rule_labels: list[str] = []
    rule_scores: list[float] = []
    for record in records:
        evidence = screens_module.run(
            record.candidate,
            history=list(record.history),
            other_sources=list(record.other_sources),
            reference_jd=record.use_jd,
            epoch_requirement_days=epoch_requirement_days,
            history_threshold=history_threshold,
            divergence_threshold_km=divergence_threshold_km,
            continuity_threshold=continuity_threshold,
            divergence_horizon_days=horizon_days,
            use_offset_days=use_offset_days,
            silent_threshold_km=silent_threshold_km,
        )
        evidences.append(evidence)
        consequence = measure(
            record.candidate,
            record.reference,
            record.use_jd,
            horizon_days=horizon_days,
            samples=samples,
            silent_threshold_km=silent_threshold_km,
        )
        measured.append(consequence.consequence_class)
        errors.append(consequence.max_error_km)
        predicted, score, _problems, _intact = classify(evidence, silent_threshold_km)
        rule_labels.append(predicted)
        rule_scores.append(score)

    matrices = {
        name: features_module.build(records, evidences, name)
        for name in features_module.CONFIGURATIONS
    }
    return Prepared(
        records=records,
        evidences=evidences,
        measured_classes=np.array(measured, dtype=object),
        measured_max_error_km=np.array(errors, dtype=float),
        rule_classes=np.array(rule_labels, dtype=object),
        rule_scores=np.array(rule_scores, dtype=float),
        matrices=matrices,
    )


def per_class(truth: np.ndarray, predicted: np.ndarray) -> dict[str, ClassScore]:
    """Equation (21) for every class in :data:`tlecheck.consequence.CLASSES`.

    Raises
    ------
    ValueError
        If the two arrays differ in length.
    """
    truth = np.asarray(truth, dtype=object)
    predicted = np.asarray(predicted, dtype=object)
    if truth.shape != predicted.shape:
        raise ValueError(f"shapes differ: {truth.shape} and {predicted.shape}")
    scores: dict[str, ClassScore] = {}
    for label in CLASSES:
        is_truth = truth == label
        is_predicted = predicted == label
        true_positive = int(np.sum(is_truth & is_predicted))
        false_positive = int(np.sum(~is_truth & is_predicted))
        false_negative = int(np.sum(is_truth & ~is_predicted))
        precision = (
            true_positive / (true_positive + false_positive)
            if (true_positive + false_positive) > 0
            else float("nan")
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if (true_positive + false_negative) > 0
            else float("nan")
        )
        scores[label] = ClassScore(
            label=label,
            precision=precision,
            recall=recall,
            support=int(np.sum(is_truth)),
            true_positive=true_positive,
            false_positive=false_positive,
            false_negative=false_negative,
        )
    return scores


def accuracy(truth: np.ndarray, predicted: np.ndarray) -> float:
    """Fraction of records whose class was predicted correctly."""
    truth = np.asarray(truth, dtype=object)
    predicted = np.asarray(predicted, dtype=object)
    if truth.size == 0:
        return float("nan")
    return float(np.mean(truth == predicted))


def operating_characteristic(
    scores: np.ndarray, positive: np.ndarray
) -> list[OperatingPoint]:
    """Every distinct operating point of a score, ordered by threshold descending.

    A record is flagged when ``score >= threshold``. The thresholds are the
    distinct scores plus an endpoint above the maximum, so both the
    flag-nothing and flag-everything corners are present.

    Raises
    ------
    ValueError
        If the arrays differ in length, or there is no positive or no negative
        example, because a false-positive rate is undefined without negatives.
    """
    scores = np.asarray(scores, dtype=float)
    positive = np.asarray(positive, dtype=bool)
    if scores.shape != positive.shape:
        raise ValueError(f"shapes differ: {scores.shape} and {positive.shape}")
    negatives = int(np.sum(~positive))
    positives = int(np.sum(positive))
    if negatives == 0 or positives == 0:
        raise ValueError(
            f"an operating characteristic needs both classes; got {positives} "
            f"positive and {negatives} negative examples"
        )
    finite = scores[np.isfinite(scores)]
    top = (float(np.max(finite)) + 1.0) if finite.size else 1.0
    thresholds = np.unique(np.concatenate([scores[np.isfinite(scores)], [top]]))
    if np.isinf(scores).any():
        thresholds = np.concatenate([thresholds, [np.inf]])
    points: list[OperatingPoint] = []
    for threshold in np.sort(thresholds)[::-1]:
        flagged = scores >= threshold
        points.append(
            OperatingPoint(
                threshold=float(threshold),
                recall=float(np.sum(flagged & positive)) / positives,
                false_positive_rate=float(np.sum(flagged & ~positive)) / negatives,
                flagged=int(np.sum(flagged)),
            )
        )
    return points


def recall_at_fpr(
    scores: np.ndarray, positive: np.ndarray, target_fpr: float = 0.05
) -> OperatingPoint:
    """Equation (22): best recall whose false-positive rate is at or below target.

    Raises
    ------
    ValueError
        If ``target_fpr`` is outside [0, 1].
    """
    if not 0.0 <= target_fpr <= 1.0:
        raise ValueError(f"target_fpr must be in [0, 1], got {target_fpr!r}")
    points = operating_characteristic(scores, positive)
    eligible = [point for point in points if point.false_positive_rate <= target_fpr]
    if not eligible:
        return OperatingPoint(float("inf"), 0.0, 0.0, 0)
    return max(eligible, key=lambda point: (point.recall, -point.false_positive_rate))


def area_under_curve(scores: np.ndarray, positive: np.ndarray) -> float:
    """Equation (23), by the trapezoidal rule over the swept points."""
    points = operating_characteristic(scores, positive)
    fpr = np.array([0.0] + [point.false_positive_rate for point in points] + [1.0])
    tpr = np.array([0.0] + [point.recall for point in points] + [1.0])
    order = np.argsort(fpr, kind="stable")
    return float(np.trapezoid(tpr[order], fpr[order]))


def point_operating_point(
    predicted: np.ndarray, positive: np.ndarray, label: str = SILENTLY_WRONG
) -> OperatingPoint:
    """The single operating point of a predictor that returns a class, not a score."""
    predicted = np.asarray(predicted, dtype=object)
    positive = np.asarray(positive, dtype=bool)
    flagged = predicted == label
    positives = int(np.sum(positive))
    negatives = int(np.sum(~positive))
    return OperatingPoint(
        threshold=float("nan"),
        recall=float(np.sum(flagged & positive)) / positives if positives else float("nan"),
        false_positive_rate=(
            float(np.sum(flagged & ~positive)) / negatives if negatives else float("nan")
        ),
        flagged=int(np.sum(flagged)),
    )
