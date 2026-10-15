"""Calibration measurement: reliability curve, expected calibration error, Brier.

Definitions, with sources, because a calibration number quoted without its
binning rule is not reproducible:

- **Expected calibration error (ECE).** Partition the ``n`` predictions into
  ``m`` equal-width confidence bins ``B_1..B_m`` over ``[0, 1]`` and take
  ``ECE = sum_i (|B_i| / n) * |acc(B_i) - conf(B_i)|``, where ``acc`` is the
  fraction correct in the bin and ``conf`` the mean confidence in it. Binned
  ECE in this form is from Naeini, Cooper and Hauskrecht, "Obtaining Well
  Calibrated Probabilities Using Bayesian Binning", AAAI 2015, and is the
  top-label form used by Guo, Pleiss, Sun and Weinberger, "On Calibration of
  Modern Neural Networks", ICML 2017 (PMLR 70, pp. 1321-1330), equation 3.
  Dimensionless, in [0, 1], lower is better. ECE is a BIASED estimator whose
  value depends on ``m``; it is reported here together with the bin count and
  with the bin table, and ``validation/validate_calibration.py`` reports it at
  three bin counts so the dependence is visible rather than hidden.

- **Brier score**, multi-class form: the mean squared error between the
  one-hot label vector and the predicted probability vector, ``(1/n) sum_i
  sum_k (p_ik - y_ik)^2``. Brier, G. W., "Verification of Forecasts Expressed
  in Terms of Probability", Monthly Weather Review 78(1), pp. 1-3, 1950,
  generalised to ``K`` classes as in that paper's multi-category form. Range
  ``[0, 2]`` for the multi-class sum convention used here; lower is better.

- **Reliability curve.** The per-bin ``(conf, acc, count)`` triples that ECE
  aggregates. A perfectly calibrated classifier lies on ``acc = conf``.

Validity range: all three measures assume the predictions are independent and
identically distributed draws from the deployment distribution. They are
computed here on a held-out split of a synthetic generator, so they describe
that generator and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "CalibrationResult",
    "brier_score",
    "expected_calibration_error",
    "reliability_curve",
]


def _check_inputs(confidence: np.ndarray, correct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    confidence = np.asarray(confidence, dtype=float).ravel()
    correct = np.asarray(correct).ravel().astype(bool)
    if confidence.shape != correct.shape:
        raise ValueError(
            f"confidence has {confidence.shape[0]} entries and correct has {correct.shape[0]}"
        )
    if confidence.size == 0:
        raise ValueError("need at least one prediction to measure calibration")
    if np.any(~np.isfinite(confidence)):
        raise ValueError("confidence contains a non-finite value")
    if np.any(confidence < 0.0) or np.any(confidence > 1.0):
        raise ValueError("confidence must lie in [0, 1]")
    return confidence, correct


@dataclass(frozen=True)
class CalibrationResult:
    """Binned calibration of a set of top-label predictions.

    Attributes:
        n_bins: number of equal-width confidence bins.
        bin_edges: ``n_bins + 1`` edges over [0, 1].
        bin_counts: predictions per bin.
        bin_confidence: mean confidence per bin; NaN for an empty bin.
        bin_accuracy: fraction correct per bin; NaN for an empty bin.
        ece: expected calibration error, dimensionless in [0, 1].
        mce: maximum calibration error over non-empty bins, dimensionless.
        accuracy: overall fraction correct, dimensionless.
        mean_confidence: overall mean confidence, dimensionless.
    """

    n_bins: int
    bin_edges: np.ndarray
    bin_counts: np.ndarray
    bin_confidence: np.ndarray
    bin_accuracy: np.ndarray
    ece: float
    mce: float
    accuracy: float
    mean_confidence: float

    @property
    def overconfidence(self) -> float:
        """Mean confidence minus accuracy. Positive means overconfident."""
        return self.mean_confidence - self.accuracy

    def table(self) -> str:
        """Fixed-width reliability table, one row per non-empty bin."""
        lines = [
            f"{'bin':<14} {'count':>7} {'conf':>9} {'acc':>9} {'gap':>9}",
            "-" * 52,
        ]
        for index in range(self.n_bins):
            if self.bin_counts[index] == 0:
                continue
            low, high = self.bin_edges[index], self.bin_edges[index + 1]
            lines.append(
                f"[{low:.2f},{high:.2f})".ljust(14)
                + f"{int(self.bin_counts[index]):>7} "
                f"{self.bin_confidence[index]:>9.5f} "
                f"{self.bin_accuracy[index]:>9.5f} "
                f"{self.bin_accuracy[index] - self.bin_confidence[index]:>9.5f}"
            )
        lines.append(
            f"ECE {self.ece:.6f}  MCE {self.mce:.6f}  acc {self.accuracy:.6f}  "
            f"mean conf {self.mean_confidence:.6f}"
        )
        return "\n".join(lines)


def reliability_curve(
    confidence: np.ndarray, correct: np.ndarray, n_bins: int = 10
) -> CalibrationResult:
    """Bin top-label predictions and return the reliability curve and ECE.

    Args:
        confidence: top-label confidence per prediction, in [0, 1].
        correct: whether each prediction was right.
        n_bins: number of equal-width bins over [0, 1], >= 1.

    Returns:
        A :class:`CalibrationResult`. Bin membership is ``low <= c < high``,
        with the last bin closed on the right so a confidence of exactly 1
        falls in it.
    """
    confidence, correct = _check_inputs(confidence, correct)
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    index = np.clip(np.digitize(confidence, edges[1:-1], right=False), 0, n_bins - 1)
    counts = np.zeros(n_bins, dtype=int)
    conf_mean = np.full(n_bins, np.nan)
    acc_mean = np.full(n_bins, np.nan)
    for bin_index in range(n_bins):
        mask = index == bin_index
        counts[bin_index] = int(mask.sum())
        if counts[bin_index]:
            conf_mean[bin_index] = float(confidence[mask].mean())
            acc_mean[bin_index] = float(correct[mask].mean())
    non_empty = counts > 0
    gaps = np.abs(acc_mean[non_empty] - conf_mean[non_empty])
    weights = counts[non_empty] / confidence.size
    ece = float(np.sum(weights * gaps)) if gaps.size else 0.0
    mce = float(np.max(gaps)) if gaps.size else 0.0
    return CalibrationResult(
        n_bins=n_bins,
        bin_edges=edges,
        bin_counts=counts,
        bin_confidence=conf_mean,
        bin_accuracy=acc_mean,
        ece=ece,
        mce=mce,
        accuracy=float(correct.mean()),
        mean_confidence=float(confidence.mean()),
    )


def expected_calibration_error(
    confidence: np.ndarray, correct: np.ndarray, n_bins: int = 10
) -> float:
    """Top-label expected calibration error, dimensionless in [0, 1]."""
    return reliability_curve(confidence, correct, n_bins).ece


def brier_score(
    probabilities: np.ndarray, labels: np.ndarray, classes: tuple[str, ...]
) -> float:
    """Multi-class Brier score (Brier 1950), summed over classes.

    Args:
        probabilities: ``(n, K)`` predicted probabilities, rows summing to 1
            to within 1e-6.
        labels: ``(n,)`` true labels, each a member of ``classes``.
        classes: the ``K`` class labels in column order.

    Returns:
        Mean over samples of the squared error summed over classes,
        dimensionless in [0, 2].
    """
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2:
        raise ValueError(f"probabilities must be 2-D, got shape {probabilities.shape}")
    if probabilities.shape[1] != len(classes):
        raise ValueError(
            f"probabilities has {probabilities.shape[1]} columns and {len(classes)} "
            "classes were given"
        )
    labels = np.asarray(labels).ravel()
    if labels.shape[0] != probabilities.shape[0]:
        raise ValueError("probabilities and labels have different lengths")
    row_sums = probabilities.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise ValueError(
            f"probability rows must sum to 1 within 1e-6; worst row sum {row_sums.max():.9f}"
        )
    index = {name: position for position, name in enumerate(classes)}
    one_hot = np.zeros_like(probabilities)
    for row, label in enumerate(labels):
        try:
            one_hot[row, index[str(label)]] = 1.0
        except KeyError as error:
            raise ValueError(f"label {label!r} is not one of the given classes") from error
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
