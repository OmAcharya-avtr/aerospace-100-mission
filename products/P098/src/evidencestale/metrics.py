"""Error directions, agreement counts and scoring metrics.

The two error directions of a staleness verdict are never summed into one
accuracy number, because they are different failures:

- a **missed staleness** cites evidence that no longer describes the system;
- a **false alarm** sends an auditor to re-run work that did not need it.

An accuracy figure that trades one against the other hides which one the method
makes, so :class:`ErrorDirections` reports both rates with their denominators.

The probabilistic metrics are the area under the receiver operating
characteristic curve and the Brier score. AUC is computed from the
Mann-Whitney U statistic with the mid-rank treatment of ties, which is the
identity in D. Bamber, "The area above the ordinal dominance graph and the area
below the receiver operating characteristic graph", *Journal of Mathematical
Psychology*, 12(4):387--415, 1975. The Brier score is G. W. Brier,
"Verification of forecasts expressed in terms of probability", *Monthly Weather
Review*, 78(1):1--3, 1950.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ErrorDirections:
    """Confusion counts of one method against the truth, with both rates.

    Attributes
    ----------
    true_positive:
        Items the method called stale that are stale.
    false_negative:
        Items the method called current that are stale -- MISSED STALENESS.
    false_positive:
        Items the method called stale that are current -- FALSE ALARM.
    true_negative:
        Items the method called current that are current.
    """

    true_positive: int
    false_negative: int
    false_positive: int
    true_negative: int

    @property
    def total(self) -> int:
        """Items compared."""
        return (
            self.true_positive
            + self.false_negative
            + self.false_positive
            + self.true_negative
        )

    @property
    def actually_stale(self) -> int:
        """Items that are truly stale."""
        return self.true_positive + self.false_negative

    @property
    def actually_current(self) -> int:
        """Items that are truly current."""
        return self.false_positive + self.true_negative

    @property
    def missed_staleness_rate(self) -> float:
        """False-negative rate: missed / truly stale. ``nan`` with no positives."""
        if self.actually_stale == 0:
            return float("nan")
        return self.false_negative / self.actually_stale

    @property
    def false_alarm_rate(self) -> float:
        """False-positive rate: false alarms / truly current. ``nan`` with none."""
        if self.actually_current == 0:
            return float("nan")
        return self.false_positive / self.actually_current

    @property
    def agreement(self) -> float:
        """Fraction of items on which the method and the truth agree.

        Reported alongside the two rates and never instead of them.
        """
        if self.total == 0:
            return float("nan")
        return (self.true_positive + self.true_negative) / self.total


def error_directions(predicted: Sequence[int], truth: Sequence[int]) -> ErrorDirections:
    """Confusion counts of 0/1 ``predicted`` against 0/1 ``truth``."""
    if len(predicted) != len(truth):
        raise ValueError(
            f"predicted has {len(predicted)} entries and truth has {len(truth)}; the "
            f"two must be aligned item by item in the same order."
        )
    for name, values in (("predicted", predicted), ("truth", truth)):
        bad = [value for value in values if value not in (0, 1, True, False)]
        if bad:
            raise ValueError(
                f"{name} must contain only 0 or 1, found {bad[:4]}. Threshold a "
                f"probability before passing it here, and report the threshold."
            )
    true_positive = sum(1 for p, t in zip(predicted, truth, strict=True) if p and t)
    false_negative = sum(1 for p, t in zip(predicted, truth, strict=True) if not p and t)
    false_positive = sum(1 for p, t in zip(predicted, truth, strict=True) if p and not t)
    true_negative = sum(
        1 for p, t in zip(predicted, truth, strict=True) if not p and not t
    )
    return ErrorDirections(true_positive, false_negative, false_positive, true_negative)


def roc_auc(scores: Sequence[float], truth: Sequence[int]) -> float:
    """Area under the ROC curve from the Mann-Whitney U statistic, ties mid-ranked.

    Returns ``nan`` when either class is empty, because the area is undefined
    then; the caller reports the ``nan`` rather than substituting 0.5.
    """
    if len(scores) != len(truth):
        raise ValueError(
            f"scores has {len(scores)} entries and truth has {len(truth)}; align them."
        )
    positives = [s for s, t in zip(scores, truth, strict=True) if t]
    negatives = [s for s, t in zip(scores, truth, strict=True) if not t]
    if not positives or not negatives:
        return float("nan")
    wins = 0.0
    for p in positives:
        for n in negatives:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def brier_score(probabilities: Sequence[float], truth: Sequence[int]) -> float:
    """Mean squared error of probabilistic forecasts against 0/1 outcomes."""
    if len(probabilities) != len(truth):
        raise ValueError(
            f"probabilities has {len(probabilities)} entries and truth has "
            f"{len(truth)}; align them."
        )
    if not probabilities:
        raise ValueError("brier_score() needs at least one forecast")
    for value in probabilities:
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(
                f"probabilities must lie in [0, 1], got {value!r}. A model that emits "
                f"a score outside the unit interval is not emitting a probability."
            )
    return sum(
        (float(p) - float(t)) ** 2 for p, t in zip(probabilities, truth, strict=True)
    ) / len(probabilities)
