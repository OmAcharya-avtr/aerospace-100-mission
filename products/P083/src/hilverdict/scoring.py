"""Scoring, with recall of consequential misses as the quantity that decides.

The asymmetry, stated once
--------------------------
Dismissing a miss that mattered and re-simulating one that did not are not the
same mistake. The first publishes a hardware-in-the-loop result whose timing
fidelity was never established; the second costs some seconds of compute. So
every method in this package is scored on **recall of consequential misses**
first, and on how much re-simulation it saves second. Flag rate is reported in
its own column and never enters the verdict.

The exact interval
------------------
A recall measured on a few dozen consequential misses has a sampling error
that swamps most of the differences anyone wants to argue about, so recall is
always reported with a Clopper-Pearson exact binomial interval (Clopper &
Pearson 1934, "The use of confidence or fiducial limits illustrated in the
case of the binomial", *Biometrika* 26(4), 404-413) rather than with a normal
approximation that is wrong at exactly the end of the range that matters: a
recall of 1.0 out of 48 has a normal standard error of zero, which is the one
answer that is certainly false.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import beta

__all__ = [
    "Confusion",
    "DecisionScore",
    "clopper_pearson",
    "confusion",
    "operating_points",
    "score_decision",
    "threshold_at_recall",
]


def clopper_pearson(
    successes: int, trials: int, confidence: float = 0.95
) -> tuple[float, float]:
    """Exact two-sided binomial interval for a rate, dimensionless.

    Returns ``(lower, upper)``. At ``successes == trials`` the upper limit is
    exactly 1 and the lower limit is ``alpha_half ** (1/trials)``, which is the
    honest statement that a perfect score on a finite sample is not a
    guarantee. ``trials == 0`` gives ``(0.0, 1.0)``: nothing was measured.
    """
    if trials < 0:
        raise ValueError(f"trials must be non-negative, got {trials}")
    if not 0 <= successes <= trials:
        raise ValueError(
            f"successes must lie in [0, {trials}], got {successes}"
        )
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must lie in (0, 1), got {confidence}")
    if trials == 0:
        return 0.0, 1.0
    half = 0.5 * (1.0 - confidence)
    lower = 0.0 if successes == 0 else float(beta.ppf(half, successes, trials - successes + 1))
    upper = (
        1.0
        if successes == trials
        else float(beta.ppf(1.0 - half, successes + 1, trials - successes))
    )
    return lower, upper


@dataclass(frozen=True)
class Confusion:
    """Counts of a binary decision against the re-simulated truth.

    ``positive`` means *consequential*: the miss measurably changed the
    trajectory. ``flagged`` means the method refused to dismiss it.
    """

    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int

    @property
    def total(self) -> int:
        """Number of misses scored."""
        return (
            self.true_positive
            + self.false_positive
            + self.false_negative
            + self.true_negative
        )

    @property
    def positives(self) -> int:
        """Number of consequential misses in the sample."""
        return self.true_positive + self.false_negative

    @property
    def recall(self) -> float:
        """Fraction of consequential misses the method refused to dismiss."""
        return self.true_positive / self.positives if self.positives else float("nan")

    @property
    def flag_rate(self) -> float:
        """Fraction of all misses the method sends to re-simulation."""
        if not self.total:
            return float("nan")
        return (self.true_positive + self.false_positive) / self.total

    @property
    def precision(self) -> float:
        """Fraction of flagged misses that were consequential."""
        flagged = self.true_positive + self.false_positive
        return self.true_positive / flagged if flagged else float("nan")

    @property
    def specificity(self) -> float:
        """Fraction of inconsequential misses the method disposed of."""
        negatives = self.true_negative + self.false_positive
        return self.true_negative / negatives if negatives else float("nan")

    @property
    def dismissal_rate(self) -> float:
        """Fraction of all misses dismissed without re-simulation."""
        if not self.total:
            return float("nan")
        return (self.true_negative + self.false_negative) / self.total


def confusion(flagged: np.ndarray, consequential: np.ndarray) -> Confusion:
    """Count a decision against the truth. Both inputs are boolean arrays."""
    decision = np.asarray(flagged, dtype=bool)
    truth = np.asarray(consequential, dtype=bool)
    if decision.shape != truth.shape:
        raise ValueError(
            f"shapes must match, got {decision.shape} and {truth.shape}"
        )
    return Confusion(
        true_positive=int(np.count_nonzero(decision & truth)),
        false_positive=int(np.count_nonzero(decision & ~truth)),
        false_negative=int(np.count_nonzero(~decision & truth)),
        true_negative=int(np.count_nonzero(~decision & ~truth)),
    )


@dataclass(frozen=True)
class DecisionScore:
    """One method's score, with recall first and the interval attached.

    Attributes
    ----------
    method
        Declared name.
    counts
        The confusion counts.
    recall_interval
        Clopper-Pearson interval on the recall at the declared confidence.
    sound
        True when the method's dismissals are backed by a proof rather than by
        a measurement. Only the analytic certificate sets this.
    note
        One line a report prints beside the numbers.
    """

    method: str
    counts: Confusion
    recall_interval: tuple[float, float]
    sound: bool
    note: str = ""

    @property
    def missed_consequential(self) -> int:
        """Consequential misses this method dismissed. The asymmetric error."""
        return self.counts.false_negative


def score_decision(
    method: str,
    flagged: np.ndarray,
    consequential: np.ndarray,
    sound: bool = False,
    confidence: float = 0.95,
    note: str = "",
) -> DecisionScore:
    """Score one decision rule against the re-simulated truth."""
    counts = confusion(flagged, consequential)
    interval = clopper_pearson(counts.true_positive, counts.positives, confidence)
    return DecisionScore(
        method=method,
        counts=counts,
        recall_interval=interval,
        sound=sound,
        note=note,
    )


def operating_points(
    scores: np.ndarray, consequential: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recall and flag rate at every distinct threshold of ``scores``.

    A miss is flagged when ``score >= threshold``. The thresholds returned are
    the distinct scores plus one above the maximum, so the all-dismissed corner
    is represented. Returns ``(thresholds, recall, flag_rate)``.
    """
    values = np.asarray(scores, dtype=float)
    truth = np.asarray(consequential, dtype=bool)
    if values.shape != truth.shape:
        raise ValueError(f"shapes must match, got {values.shape} and {truth.shape}")
    if values.size == 0:
        raise ValueError("no misses to score")
    thresholds = np.concatenate(
        [np.unique(values), [np.nextafter(values.max(), np.inf)]]
    )
    positives = int(np.count_nonzero(truth))
    recall = np.empty(thresholds.shape[0], dtype=float)
    flag_rate = np.empty(thresholds.shape[0], dtype=float)
    for position, threshold in enumerate(thresholds):
        flagged = values >= threshold
        recall[position] = (
            np.count_nonzero(flagged & truth) / positives if positives else np.nan
        )
        flag_rate[position] = np.count_nonzero(flagged) / values.shape[0]
    return thresholds, recall, flag_rate


def threshold_at_recall(
    scores: np.ndarray, consequential: np.ndarray, target_recall: float
) -> tuple[float, float, float]:
    """The threshold with the smallest flag rate that still reaches ``target_recall``.

    Returns ``(threshold, recall, flag_rate)``. Raises when no threshold
    reaches the target, which happens only if the target exceeds 1.
    """
    if not 0.0 < target_recall <= 1.0:
        raise ValueError(
            f"target_recall must lie in (0, 1], got {target_recall}"
        )
    thresholds, recall, flag_rate = operating_points(scores, consequential)
    feasible = np.nonzero(recall >= target_recall - 1e-12)[0]
    if feasible.size == 0:
        finite = recall[np.isfinite(recall)]
        best = float(np.max(finite)) if finite.size else float("nan")
        raise ValueError(
            f"no threshold reaches a recall of {target_recall}; the best "
            f"available is {best}, and a recall is undefined when the sample "
            "contains no consequential miss"
        )
    best = feasible[int(np.argmin(flag_rate[feasible]))]
    return float(thresholds[best]), float(recall[best]), float(flag_rate[best])
