"""Scores, with intervals, and the scores that matter inside the null space.

Three groups of measurement live here.

**Accuracy with an exact interval.** Every proportion reported by this package
carries a Clopper-Pearson interval (Clopper & Pearson 1934, "The use of
confidence or fiducial limits illustrated in the case of the binomial",
Biometrika 26(4), 404-413) rather than a normal approximation, because the
proportions in question reach 0 and 1 -- an accuracy of 60 out of 60 has a
normal standard error of exactly zero, which is the one answer that is
certainly wrong.

**Calibration of the confidence output.** A classifier that is wrong is
tolerable if it says so. The Brier score (Brier 1950, "Verification of
forecasts expressed in terms of probability", Monthly Weather Review 78(1),
1-3) in its multi-class form, and a binned reliability table of predicted
confidence against observed accuracy, are both reported. Calibration is the
quantitative form of the question this product is built around: does the
learned component know when it does not know?

**Confident error inside the unidentifiable subspace.** The measurement the
specification singles out. For a case population whose true cause lies in the
null space of the telemetry set, no single-parameter answer is correct, so
accuracy is zero by construction and tells nothing. What tells something is
how *confidently* the wrong answer is given:
:func:`confident_error_rate` is the fraction of such cases answered with a
confidence at or above a declared threshold. The analytic path's comparable
number is its refusal rate, which should be 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from scipy import stats

__all__ = [
    "CONFIDENCE_BINS",
    "CONFIDENT_THRESHOLD",
    "Proportion",
    "brier_score",
    "calibration_table",
    "confident_error_rate",
    "confusion_matrix",
    "proportion",
]

#: Declared confidence at or above which an answer is called confident. 0.5 is
#: the point at which a five-class answer is more likely than everything else
#: put together, which is the weakest reading of the word.
CONFIDENT_THRESHOLD: Final[float] = 0.5

#: Declared bin edges of the reliability table.
CONFIDENCE_BINS: Final[tuple[float, ...]] = (0.2, 0.4, 0.6, 0.8, 0.9, 1.0 + 1e-12)


@dataclass(frozen=True)
class Proportion:
    """A count out of a total, with an exact two-sided interval."""

    count: int
    total: int
    level: float
    lower: float
    upper: float

    @property
    def value(self) -> float:
        """The proportion itself, or ``nan`` when the total is zero."""
        return self.count / self.total if self.total else float("nan")

    def __str__(self) -> str:
        if not self.total:
            return "n/a (0 cases)"
        return (
            f"{self.value:.6f} ({self.count}/{self.total}, "
            f"{self.level:.2f} CP [{self.lower:.6f}, {self.upper:.6f}])"
        )


def proportion(count: int, total: int, level: float = 0.95) -> Proportion:
    """A Clopper-Pearson interval for ``count`` successes out of ``total``.

    The interval is the exact binomial one: the lower limit is the Beta
    quantile with shape ``(count, total - count + 1)`` and the upper the Beta
    quantile with shape ``(count + 1, total - count)``, with the degenerate
    ends set to 0 and 1 respectively.
    """
    if total < 0 or count < 0 or count > total:
        raise ValueError(f"need 0 <= count <= total, got {count} of {total}")
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must lie in (0, 1), got {level!r}")
    if total == 0:
        return Proportion(0, 0, level, float("nan"), float("nan"))
    tail = 0.5 * (1.0 - level)
    lower = 0.0 if count == 0 else float(stats.beta.ppf(tail, count, total - count + 1))
    upper = (
        1.0 if count == total else float(stats.beta.ppf(1.0 - tail, count + 1, total - count))
    )
    return Proportion(int(count), int(total), float(level), lower, upper)


def confusion_matrix(
    truth: np.ndarray, predicted: np.ndarray, labels: tuple[int, ...]
) -> np.ndarray:
    """Counts of ``(true label, predicted label)``, shape ``(labels, labels)``.

    A truth or prediction outside ``labels`` raises rather than being dropped:
    a confusion matrix that silently loses rows is a wrong confusion matrix.
    """
    true_array = np.asarray(truth)
    predicted_array = np.asarray(predicted)
    if true_array.shape != predicted_array.shape:
        raise ValueError(
            f"truth has shape {true_array.shape} and predictions {predicted_array.shape}"
        )
    position = {label: index for index, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for actual, guess in zip(true_array, predicted_array, strict=True):
        if int(actual) not in position or int(guess) not in position:
            raise ValueError(
                f"label pair ({actual}, {guess}) is outside the declared label set "
                f"{labels}"
            )
        matrix[position[int(actual)], position[int(guess)]] += 1
    return matrix


def brier_score(probabilities: np.ndarray, truth_column: np.ndarray) -> float:
    """Multi-class Brier score, Brier 1950. Lower is better; 0 is perfect.

    ``probabilities`` has shape ``(cases, classes)`` and must sum to 1 along
    each row to within 1e-9; ``truth_column`` holds the index of the true class
    in that layout. A case whose true class is not among the columns -- which
    is exactly the situation inside the unidentifiable subspace for a
    classifier with no refusal class -- is given a one-hot target of all zeros,
    so the score charges the full squared probability mass. That is the honest
    treatment and it is why the score is reported separately for each case
    population.
    """
    matrix = np.atleast_2d(np.asarray(probabilities, dtype=float))
    column = np.asarray(truth_column, dtype=int)
    if column.shape != (matrix.shape[0],):
        raise ValueError(
            f"{matrix.shape[0]} probability rows but {column.shape} truth entries"
        )
    sums = matrix.sum(axis=1)
    if np.any(np.abs(sums - 1.0) > 1.0e-9):
        raise ValueError(
            "every probability row must sum to 1 to within 1e-9; worst row sums to "
            f"{float(sums[np.argmax(np.abs(sums - 1.0))])!r}"
        )
    target = np.zeros_like(matrix)
    inside = (column >= 0) & (column < matrix.shape[1])
    target[np.arange(matrix.shape[0])[inside], column[inside]] = 1.0
    return float(((matrix - target) ** 2).sum(axis=1).mean())


def calibration_table(
    confidence: np.ndarray, correct: np.ndarray, bins: tuple[float, ...] = CONFIDENCE_BINS
) -> tuple[tuple[float, float, int, float, float], ...]:
    """Reliability table: ``(bin low, bin high, count, mean confidence, accuracy)``.

    Empty bins are returned with a count of 0 and ``nan`` for both means, so
    the table always has one row per declared bin and a reader can see which
    bins the population never reached.
    """
    values = np.asarray(confidence, dtype=float)
    hits = np.asarray(correct, dtype=bool)
    if values.shape != hits.shape:
        raise ValueError(f"confidence shape {values.shape} against correct {hits.shape}")
    edges = (0.0, *bins)
    rows = []
    for index in range(len(edges) - 1):
        low, high = edges[index], edges[index + 1]
        inside = (values >= low) & (values < high)
        count = int(inside.sum())
        rows.append(
            (
                float(low),
                float(min(high, 1.0)),
                count,
                float(values[inside].mean()) if count else float("nan"),
                float(hits[inside].mean()) if count else float("nan"),
            )
        )
    return tuple(rows)


def confident_error_rate(
    confidence: np.ndarray,
    correct: np.ndarray,
    threshold: float = CONFIDENT_THRESHOLD,
    level: float = 0.95,
) -> Proportion:
    """Fraction of cases answered wrongly AND with confidence at or above ``threshold``.

    The denominator is every case in the population, not only the confident
    ones, so the number is directly comparable with an accuracy and with an
    analytic refusal rate.
    """
    values = np.asarray(confidence, dtype=float)
    hits = np.asarray(correct, dtype=bool)
    if values.shape != hits.shape:
        raise ValueError(f"confidence shape {values.shape} against correct {hits.shape}")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must lie in [0, 1], got {threshold!r}")
    return proportion(int(((values >= threshold) & ~hits).sum()), int(values.size), level)
