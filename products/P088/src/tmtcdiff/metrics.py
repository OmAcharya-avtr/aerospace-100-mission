"""Per-class detection metrics and a binomial interval for a measured rate.

Precision, recall and F1 follow the standard definitions (van Rijsbergen,
*Information Retrieval*, 2nd edition, Butterworths, 1979, chapter 7). The
conventions here matter for reading the numbers:

- A detection is matched to ground truth on the pair ``(change class, name)``.
  A change detected on the right item under the wrong class is a false
  positive for the class detected *and* a false negative for the class
  injected. Nothing is forgiven for being close.
- Precision with no detections is reported as ``nan``, not as 1.0, and recall
  with no ground truth likewise. A metric with an empty denominator is not a
  perfect score and will not be printed as one.
- The interval on a measured rate is Clopper and Pearson's exact binomial
  interval (*The use of confidence or fiducial limits illustrated in the case
  of the binomial*, Biometrika 26(4), 1934), which is the right one here
  because the counts of interest are small and often zero, where a normal
  approximation gives an interval of zero width.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from .changes import ChangeClass


@dataclass(frozen=True)
class ClassMetrics:
    """Detection counts and derived metrics for one change class."""

    change_class: ChangeClass
    true_positives: int
    false_positives: int
    false_negatives: int

    @property
    def detections(self) -> int:
        """Number of changes the tool reported in this class."""
        return self.true_positives + self.false_positives

    @property
    def injected(self) -> int:
        """Number of changes of this class that were injected."""
        return self.true_positives + self.false_negatives

    @property
    def precision(self) -> float:
        """True positives over detections. ``nan`` when nothing was detected."""
        if self.detections == 0:
            return math.nan
        return self.true_positives / self.detections

    @property
    def recall(self) -> float:
        """True positives over injected. ``nan`` when nothing was injected."""
        if self.injected == 0:
            return math.nan
        return self.true_positives / self.injected

    @property
    def f1(self) -> float:
        """Harmonic mean of precision and recall. ``nan`` when either is ``nan`` or both are 0."""
        precision = self.precision
        recall = self.recall
        if math.isnan(precision) or math.isnan(recall) or (precision + recall) == 0.0:
            return math.nan
        return 2.0 * precision * recall / (precision + recall)


def per_class_metrics(
    truth: Iterable[tuple[ChangeClass, str]],
    detected: Iterable[tuple[ChangeClass, str]],
) -> dict[ChangeClass, ClassMetrics]:
    """Match detections to ground truth on ``(class, name)`` and count per class.

    Both arguments are iterables of keys. Duplicates are counted as multisets,
    so two injections of the same class on the same item are two true
    positives when both are detected.
    """
    truth_counts: dict[tuple[ChangeClass, str], int] = {}
    for key in truth:
        truth_counts[key] = truth_counts.get(key, 0) + 1
    detected_counts: dict[tuple[ChangeClass, str], int] = {}
    for key in detected:
        detected_counts[key] = detected_counts.get(key, 0) + 1

    classes = sorted(
        {key[0] for key in truth_counts} | {key[0] for key in detected_counts},
        key=lambda c: c.value,
    )
    out: dict[ChangeClass, ClassMetrics] = {}
    for change_class in classes:
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        keys = {k for k in truth_counts if k[0] is change_class} | {
            k for k in detected_counts if k[0] is change_class
        }
        for key in keys:
            injected = truth_counts.get(key, 0)
            found = detected_counts.get(key, 0)
            true_positives += min(injected, found)
            false_positives += max(0, found - injected)
            false_negatives += max(0, injected - found)
        out[change_class] = ClassMetrics(
            change_class=change_class,
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives,
        )
    return out


def aggregate(metrics: Iterable[ClassMetrics]) -> ClassMetrics:
    """Micro-averaged metrics over several classes, filed under the first class seen."""
    entries = list(metrics)
    if not entries:
        raise ValueError("aggregate needs at least one ClassMetrics")
    return ClassMetrics(
        change_class=entries[0].change_class,
        true_positives=sum(entry.true_positives for entry in entries),
        false_positives=sum(entry.false_positives for entry in entries),
        false_negatives=sum(entry.false_negatives for entry in entries),
    )


def clopper_pearson(successes: int, trials: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact binomial interval for ``successes`` of ``trials`` at level ``1 - alpha``.

    Returns ``(low, high)``. ``(0.0, 0.0)`` for zero trials, which is the only
    honest answer: an interval on no data has no width and no location.
    """
    if trials < 0 or successes < 0:
        raise ValueError("counts must be non-negative")
    if successes > trials:
        raise ValueError(f"successes {successes} exceeds trials {trials}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha!r}")
    if trials == 0:
        return (0.0, 0.0)
    from scipy.stats import beta

    low = 0.0 if successes == 0 else float(beta.ppf(alpha / 2.0, successes, trials - successes + 1))
    high = (
        1.0
        if successes == trials
        else float(beta.ppf(1.0 - alpha / 2.0, successes + 1, trials - successes))
    )
    return (low, high)
