"""Scoring: the rules, the learned model and a majority-class guess, on one split.

Everything here is measurement. No tolerance is tuned, no threshold is chosen
to improve a number, and a result unfavourable to the learned component is
reported in exactly the form it came out.

Populations
-----------
``all``
    Every ``(frame, channel)`` pair of the held-out record. This is where the
    claim that "the rules win on every class they cover" is tested.
``ambiguous``
    Only the pairs the strengthened rules left ``undecided``. This is the
    population the product specification assigns to the learned component, and
    it is scored against a majority-class guess over the same population so
    that a model with no skill cannot appear to have some.

Metrics
-------
Per class ``c``, with ``U`` the rows a predictor declined to classify:

    precision(c) = TP(c) / predicted(c)                                 (13)
    recall(c)    = TP(c) / support(c)                                   (14)
    undecided(c) = |{rows of true class c in U}| / support(c)           (15)
    surfaced(c)  = recall(c) + undecided(c)                             (15b)

An undecided row is **not** counted as a prediction, so a predictor that
abstains loses recall and keeps its precision. That asymmetry is deliberate,
and equation (15b) is declared here alongside the others rather than introduced
later, because the two measure different things and both are reported for every
predictor:

* **recall** is the share of a class that was correctly named.
* **surfaced** is the share that was either correctly named *or* explicitly
  flagged as undecided -- that is, the share a human reading the report would
  be told about at all. For a tool whose output a human acts on, this is the
  operational quantity; for a tool feeding an automated pipeline, recall is.
  A predictor that never abstains has ``surfaced == recall`` by construction,
  so the metric cannot flatter one and not the other.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .classify import AnomalyClass, IntegrityReport
from .learned import CLASS_NAMES
from .record import TelemetryRecord

#: Label code for a row no predictor would classify.
UNDECIDED = -1

#: Map from the classifier's verdict strings to value-level class codes. A gap
#: is a frame-level verdict and has no ``(frame, channel)`` row, so
#: ``missing`` does not appear here.
VERDICT_TO_CODE = {
    AnomalyClass.CORRUPT.value: 1,
    AnomalyClass.MISPARSED.value: 2,
    AnomalyClass.UNDECIDED.value: UNDECIDED,
}

#: Value-level observations that carry a ``(frame, channel)`` position.
VALUE_OBSERVATIONS = (
    "value_out_of_range",
    "value_local_outlier",
    "value_in_failed_frame",
)


@dataclass(frozen=True)
class ClassScore:
    """Equations (13) to (15) for one class."""

    name: str
    support: int
    predicted: int
    true_positive: int
    precision: float
    recall: float
    f1: float
    undecided: int
    undecided_rate: float

    @property
    def surfaced_rate(self) -> float:
        """Equation (15b): the share of the class named correctly or flagged undecided."""
        if self.support == 0:
            return float("nan")
        return float((self.true_positive + self.undecided) / self.support)


@dataclass
class ScoreTable:
    """Per-class scores plus the confusion matrix they were computed from."""

    label: str
    population: int
    scores: tuple[ClassScore, ...]
    confusion: np.ndarray
    accuracy_decided: float
    decided_fraction: float

    def __getitem__(self, name: str) -> ClassScore:
        for score in self.scores:
            if score.name == name:
                return score
        raise KeyError(f"no class named {name!r}; expected one of {CLASS_NAMES}")

    def recall(self, name: str) -> float:
        """Recall of one class, equation (14)."""
        return self[name].recall

    def precision(self, name: str) -> float:
        """Precision of one class, equation (13)."""
        return self[name].precision


def rule_labels(record: TelemetryRecord, report: IntegrityReport) -> np.ndarray:
    """Value-level rule predictions as a ``(n, n_channels)`` code array.

    ``0`` clean, ``1`` corrupt, ``2`` misparsed, :data:`UNDECIDED` where the
    rules declined. Several value-level observations can land on the same pair:
    a range violation, a local-residual violation and membership of a frame
    whose error control failed. A decided verdict wins over an undecided one,
    and among decided verdicts the range screen wins, because it rests on a
    declared limit while the residual screen rests on an estimated one.
    """
    labels = np.zeros((len(record), record.n_channels), dtype=np.int64)
    priority = np.full(labels.shape, -1, dtype=np.int64)
    rank = {"value_out_of_range": 3, "value_local_outlier": 2, "value_in_failed_frame": 1}
    for anomaly in report.anomalies:
        if anomaly.observation not in VALUE_OBSERVATIONS:
            continue
        code = VERDICT_TO_CODE.get(anomaly.verdict.value)
        if code is None:  # pragma: no cover - MISSING cannot reach a value row
            continue
        here = rank[anomaly.observation] + (10 if code != UNDECIDED else 0)
        if here > priority[anomaly.index, anomaly.channel]:
            priority[anomaly.index, anomaly.channel] = here
            labels[anomaly.index, anomaly.channel] = code
    return labels


def confusion_matrix(truth: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """``(3, 4)`` counts: rows true class, columns predicted class then undecided."""
    truth = np.asarray(truth, dtype=np.int64).reshape(-1)
    predicted = np.asarray(predicted, dtype=np.int64).reshape(-1)
    if truth.shape != predicted.shape:
        raise ValueError(f"truth {truth.shape} and predicted {predicted.shape} must match")
    matrix = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES) + 1), dtype=np.int64)
    for true_code in range(len(CLASS_NAMES)):
        rows = truth == true_code
        for predicted_code in range(len(CLASS_NAMES)):
            matrix[true_code, predicted_code] = int((rows & (predicted == predicted_code)).sum())
        matrix[true_code, len(CLASS_NAMES)] = int((rows & (predicted == UNDECIDED)).sum())
    return matrix


def score(truth: np.ndarray, predicted: np.ndarray, label: str) -> ScoreTable:
    """Equations (13) to (15) for every class, from one truth and prediction pair."""
    truth = np.asarray(truth, dtype=np.int64).reshape(-1)
    predicted = np.asarray(predicted, dtype=np.int64).reshape(-1)
    matrix = confusion_matrix(truth, predicted)
    scores: list[ClassScore] = []
    for code, name in enumerate(CLASS_NAMES):
        support = int((truth == code).sum())
        predicted_count = int((predicted == code).sum())
        true_positive = int(matrix[code, code])
        undecided = int(matrix[code, len(CLASS_NAMES)])
        precision = float(true_positive / predicted_count) if predicted_count else float("nan")
        recall = float(true_positive / support) if support else float("nan")
        if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0.0:
            f1 = float(2.0 * precision * recall / (precision + recall))
        else:
            f1 = float("nan")
        scores.append(
            ClassScore(
                name=name,
                support=support,
                predicted=predicted_count,
                true_positive=true_positive,
                precision=precision,
                recall=recall,
                f1=f1,
                undecided=undecided,
                undecided_rate=float(undecided / support) if support else float("nan"),
            )
        )
    decided = predicted != UNDECIDED
    if decided.any():
        accuracy = float((truth[decided] == predicted[decided]).mean())
    else:
        accuracy = float("nan")
    return ScoreTable(
        label=label,
        population=int(truth.size),
        scores=tuple(scores),
        confusion=matrix,
        accuracy_decided=accuracy,
        decided_fraction=float(decided.mean()) if decided.size else float("nan"),
    )


@dataclass
class BenchmarkResult:
    """Every score table of one benchmark run, plus the timings behind them."""

    tables: dict[str, ScoreTable] = field(default_factory=dict)
    timings_s: dict[str, float] = field(default_factory=dict)
    populations: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def add(self, key: str, table: ScoreTable) -> None:
        """Store one score table under ``key``."""
        self.tables[key] = table

    def verdict_lines(self) -> list[str]:
        """Per-class comparison of the strongest rule baseline against the learned model.

        Returns one line per class naming which predictor won on recall and on
        precision, with the margin. Written to say "the rules won" as plainly
        as it would say the reverse.
        """
        lines: list[str] = []
        rules = self.tables.get("all_rules_strengthened")
        learned = self.tables.get("all_learned")
        if rules is None or learned is None:
            return ["baseline-versus-learned verdict unavailable: a table is missing"]
        for name in CLASS_NAMES:
            rule_score, learned_score = rules[name], learned[name]
            for metric in ("recall", "precision", "surfaced_rate"):
                rule_value = getattr(rule_score, metric)
                learned_value = getattr(learned_score, metric)
                if not (np.isfinite(rule_value) and np.isfinite(learned_value)):
                    lines.append(f"{name:>10s} {metric:<9s}: not comparable (empty class)")
                    continue
                delta = learned_value - rule_value
                if abs(delta) < 1e-12:
                    winner = "tie"
                elif delta > 0.0:
                    winner = "learned"
                else:
                    winner = "rules"
                lines.append(
                    f"{name:>10s} {metric:<9s}: rules {rule_value:.4f}  learned "
                    f"{learned_value:.4f}  delta {delta:+.4f}  winner {winner}"
                )
        return lines
