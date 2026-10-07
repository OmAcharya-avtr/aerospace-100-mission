"""The learned switch predictor, and the benchmark that decides whether it earns
its place.

The AI component of this package is a classifier on the state that tries to say
whether the exact switching condition of :mod:`simplexguard.guard` will fire. It
is built **after** the exact condition and the exact multi-step predictor of
:mod:`simplexguard.reachability`, and it is reported against them on the same
held-out episodes.

The honest expectation, stated before any model was fitted
--------------------------------------------------------
The switching condition is a closed-form inequality in the state and the
proposed input. A classifier trained to reproduce it is approximating a function
that is already exactly computable in a few microseconds, so on the condition's
own criterion -- *does the guard fire at this step* -- the exact computation is
right by construction and the classifier can at best tie. The only way a learned
predictor can be worth anything here is

* by being **cheaper** than the exact computation, or
* by answering a question the exact computation answers only conservatively,
  namely *will the guard fire in the next L steps of this realisation*, where
  the exact worst-case reachability answer is "may fire" and over-predicts.

Both are measured. ``validation/validate_predictor.py`` reports the result
including the losses, and the README states them before it states anything else
about the model.

What the model is given
-----------------------
Raw measurable quantities only: the two state coordinates, the reference, the
tracking error and the unsaturated performance input. It is deliberately **not**
given the support-function margin of the switching condition. A model handed the
output of the exact computation is not an alternative to the exact computation,
and benchmarking one would be meaningless.

Calibration
-----------
The classifier's probability output is calibrated by isotonic regression on a
held-out calibration split (Zadrozny and Elkan, "Transforming classifier scores
into accurate multiclass probability estimates", *KDD*, 2002; Niculescu-Mizil and
Caruana, "Predicting good probabilities with supervised learning", *ICML*, 2005),
and reported with its Brier score (Brier, *Monthly Weather Review* 78(1), 1950)
and expected calibration error. The exact predictors have no probability output
at all, which is the one structural advantage the learned model has and is
reported as such.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .controllers import PerformanceController
from .guard import Mode, SimplexGuard
from .plant import Plant
from .reachability import ExactLeadPredictor
from .simulate import (
    CostWeights,
    disturbance_sequence,
    simulate_guarded,
    square_wave_reference,
)

__all__ = [
    "ClassificationScores",
    "SwitchDataset",
    "build_dataset",
    "exact_predictor_scores",
    "expected_calibration_error",
    "fit_switch_predictor",
    "guard_condition_scores",
    "lead_times",
    "measure_decision_cost",
    "score_binary",
]

FEATURE_NAMES: tuple[str, ...] = (
    "theta",
    "theta_dot",
    "reference",
    "tracking_error",
    "u_perf_unsaturated",
)


@dataclass(frozen=True)
class SwitchDataset:
    """Rows of one step each, from simulated guarded episodes.

    Attributes
    ----------
    features :
        Shape ``(N, 5)``, columns named by :data:`FEATURE_NAMES`.
    labels :
        The prediction target: ``True`` if the guard holds baseline authority at
        any step in ``(k, k + lead]`` of the realised episode. With
        ``lead = 0`` the target is "the guard fires at step ``k``", which is the
        exact condition's own criterion.
    fires_now :
        ``True`` if the guard holds baseline authority at step ``k``. Equal to
        ``labels`` when ``lead == 0``.
    saturated :
        ``True`` if the performance controller's input saturated at step ``k``.
        Reported because the exact multi-step predictor ignores saturation.
    """

    features: np.ndarray
    labels: np.ndarray
    states: np.ndarray
    references: np.ndarray
    fires_now: np.ndarray
    saturated: np.ndarray
    episode_ids: np.ndarray
    step_indices: np.ndarray
    lead: int
    feature_names: tuple[str, ...] = FEATURE_NAMES

    def __len__(self) -> int:
        return int(self.features.shape[0])

    @property
    def base_rate(self) -> float:
        """Fraction of positive labels."""
        return float(np.mean(self.labels)) if len(self) else 0.0

    @property
    def saturation_rate(self) -> float:
        """Fraction of steps at which the performance input saturated."""
        return float(np.mean(self.saturated)) if len(self) else 0.0

    def select_episodes(self, episodes: np.ndarray) -> SwitchDataset:
        """A subset containing only the listed episode ids. Splits are by episode.

        Splitting by episode and not by row is required: consecutive rows of one
        episode are a correlated trajectory, so a random row split would put
        near-duplicates of test rows into training and inflate every metric.
        """
        keep = np.isin(self.episode_ids, np.asarray(episodes))
        return SwitchDataset(
            features=self.features[keep],
            labels=self.labels[keep],
            states=self.states[keep],
            references=self.references[keep],
            fires_now=self.fires_now[keep],
            saturated=self.saturated[keep],
            episode_ids=self.episode_ids[keep],
            step_indices=self.step_indices[keep],
            lead=self.lead,
            feature_names=self.feature_names,
        )


def _features(
    states: np.ndarray, references: np.ndarray, proposed: np.ndarray
) -> np.ndarray:
    theta = states[:, 0]
    theta_dot = states[:, 1]
    ref = references[:, 0]
    return np.column_stack([theta, theta_dot, ref, theta - ref, proposed[:, 0]])


def build_dataset(
    plant: Plant,
    guard: SimplexGuard,
    performance: PerformanceController,
    n_episodes: int,
    n_steps: int,
    seed: int,
    lead: int,
    reference_amplitude: float = 0.18,
    reference_period: int = 80,
    disturbance_mode: str = "uniform",
    reference_factory: Callable[[int], Callable[[int], np.ndarray]] | None = None,
) -> SwitchDataset:
    """Simulate ``n_episodes`` guarded episodes and tabulate one row per step.

    Determinism: episode ``i`` uses ``default_rng(seed + 1000 * i)`` for its
    disturbance sequence and, by default, a reference amplitude jittered by
    ``+- 25 %`` from the same generator so that the episodes are not identical.
    Re-running with the same ``seed`` reproduces the dataset exactly.
    """
    if int(n_episodes) < 1:
        raise ValueError(f"n_episodes must be at least 1, got {n_episodes}")
    if int(n_steps) < 2:
        raise ValueError(f"n_steps must be at least 2, got {n_steps}")
    if int(lead) < 0:
        raise ValueError(f"lead must be non-negative, got {lead}")
    rows_f, rows_lab, rows_x, rows_r = [], [], [], []
    rows_now, rows_sat, rows_ep, rows_k = [], [], [], []
    for episode in range(int(n_episodes)):
        rng = np.random.default_rng(int(seed) + 1000 * episode)
        if reference_factory is None:
            amp = float(reference_amplitude) * float(rng.uniform(0.75, 1.25))
            reference = square_wave_reference(amp, int(reference_period), plant.n_states)
        else:
            reference = reference_factory(episode)
        w = disturbance_sequence(plant, int(n_steps), rng, disturbance_mode)
        ep = simulate_guarded(
            plant, guard, performance, int(n_steps), reference, w, weights=CostWeights()
        )
        fires = ep.modes == Mode.BASELINE.value
        n = ep.n_steps
        if int(lead) == 0:
            label = fires.copy()
        else:
            label = np.zeros(n, dtype=bool)
            for offset in range(1, int(lead) + 1):
                label[: n - offset] |= fires[offset:]
        sat = np.array(
            [
                performance.saturates_at(ep.states[k], ep.references[k])
                for k in range(n)
            ],
            dtype=bool,
        )
        proposed_unsat = np.array(
            [performance.unsaturated(ep.states[k], ep.references[k]) for k in range(n)]
        )
        usable = np.arange(n - int(lead)) if int(lead) > 0 else np.arange(n)
        rows_f.append(_features(ep.states[:-1], ep.references, proposed_unsat)[usable])
        rows_lab.append(label[usable])
        rows_x.append(ep.states[:-1][usable])
        rows_r.append(ep.references[usable])
        rows_now.append(fires[usable])
        rows_sat.append(sat[usable])
        rows_ep.append(np.full(usable.size, episode, dtype=int))
        rows_k.append(usable)
    return SwitchDataset(
        features=np.vstack(rows_f),
        labels=np.concatenate(rows_lab),
        states=np.vstack(rows_x),
        references=np.vstack(rows_r),
        fires_now=np.concatenate(rows_now),
        saturated=np.concatenate(rows_sat),
        episode_ids=np.concatenate(rows_ep),
        step_indices=np.concatenate(rows_k),
        lead=int(lead),
    )


@dataclass(frozen=True)
class ClassificationScores:
    """Metrics for one predictor on one split."""

    name: str
    n: int
    base_rate: float
    precision: float
    recall: float
    f1: float
    accuracy: float
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    brier: float = float("nan")
    ece: float = float("nan")
    roc_auc: float = float("nan")
    average_precision: float = float("nan")
    microseconds_per_decision: float = float("nan")
    notes: str = ""

    def row(self) -> str:
        """One fixed-width line, used by the validation scripts and the README."""
        return (
            f"{self.name:<34s} P={self.precision:7.5f} R={self.recall:7.5f} "
            f"F1={self.f1:7.5f} acc={self.accuracy:7.5f} "
            f"TP={self.true_positive:<6d} FP={self.false_positive:<6d} "
            f"FN={self.false_negative:<6d} "
            f"Brier={self.brier:9.6f} ECE={self.ece:8.5f} "
            f"AUC={self.roc_auc:7.5f} us/decision={self.microseconds_per_decision:8.2f}"
        )


def score_binary(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probability: np.ndarray | None = None,
    score: np.ndarray | None = None,
    microseconds_per_decision: float = float("nan"),
    notes: str = "",
) -> ClassificationScores:
    """Confusion-matrix metrics, plus Brier/ECE/AUC when a score is supplied."""
    yt = np.asarray(y_true, dtype=bool).ravel()
    yp = np.asarray(y_pred, dtype=bool).ravel()
    if yt.shape != yp.shape:
        raise ValueError(f"y_true has shape {yt.shape}, y_pred has shape {yp.shape}")
    if yt.size == 0:
        raise ValueError("cannot score an empty split")
    tp = int(np.sum(yt & yp))
    fp = int(np.sum(~yt & yp))
    tn = int(np.sum(~yt & ~yp))
    fn = int(np.sum(yt & ~yp))
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if np.isfinite(precision) and np.isfinite(recall) and precision + recall > 0
        else float("nan")
    )
    ranking = probability if probability is not None else score
    brier = ece = auc = ap = float("nan")
    if probability is not None:
        brier = float(brier_score_loss(yt.astype(int), np.asarray(probability, dtype=float)))
        ece = expected_calibration_error(yt, np.asarray(probability, dtype=float))
    if ranking is not None and 0 < yt.sum() < yt.size:
        auc = float(roc_auc_score(yt.astype(int), np.asarray(ranking, dtype=float)))
        ap = float(average_precision_score(yt.astype(int), np.asarray(ranking, dtype=float)))
    return ClassificationScores(
        name=name,
        n=int(yt.size),
        base_rate=float(np.mean(yt)),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        accuracy=float((tp + tn) / yt.size),
        true_positive=tp,
        false_positive=fp,
        true_negative=tn,
        false_negative=fn,
        brier=brier,
        ece=ece,
        roc_auc=auc,
        average_precision=ap,
        microseconds_per_decision=float(microseconds_per_decision),
        notes=notes,
    )


def expected_calibration_error(
    y_true: np.ndarray, probability: np.ndarray, n_bins: int = 10
) -> float:
    """Equal-width expected calibration error, ``sum_b (n_b/N) |acc_b - conf_b|``.

    Equal-width bins on ``[0, 1]``; empty bins contribute nothing. This is the
    standard definition (Naeini, Cooper and Hauskrecht, *AAAI*, 2015; Guo, Pleiss,
    Sun and Weinberger, "On calibration of modern neural networks", *ICML*, 2017)
    and is known to be biased by binning, which is why the Brier score is
    reported beside it rather than instead of it.
    """
    yt = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(probability, dtype=float).ravel()
    if yt.shape != p.shape:
        raise ValueError(f"y_true has shape {yt.shape}, probability has shape {p.shape}")
    if int(n_bins) < 1:
        raise ValueError(f"n_bins must be at least 1, got {n_bins}")
    if yt.size == 0:
        raise ValueError("cannot compute ECE on an empty split")
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, int(n_bins) - 1)
    total = 0.0
    for b in range(int(n_bins)):
        sel = idx == b
        if not np.any(sel):
            continue
        total += (np.sum(sel) / yt.size) * abs(float(yt[sel].mean() - p[sel].mean()))
    return float(total)


@dataclass
class _FittedModel:
    name: str
    estimator: object
    microseconds_per_decision: float = float("nan")
    extra: dict[str, float] = field(default_factory=dict)


def fit_switch_predictor(
    train: SwitchDataset,
    calibration: SwitchDataset,
    kind: str = "forest",
    n_estimators: int = 150,
    min_samples_leaf: int = 8,
    random_state: int = 5101,
    n_jobs: int = 2,
    inference_n_jobs: int = 1,
) -> object:
    """Fit a calibrated classifier on ``train`` and calibrate it on ``calibration``.

    Parameters
    ----------
    kind :
        ``"forest"`` for :class:`~sklearn.ensemble.RandomForestClassifier`,
        ``"logistic"`` for a standardised L2 logistic regression. The logistic
        model is included because the exact decision boundary is piecewise
        linear, so a single hyperplane is a measurable lower bound on what a
        learned model can do here.
    n_estimators, min_samples_leaf, random_state, n_jobs :
        Passed to the forest. ``n_jobs`` defaults to 2 for the 2-core budget.
    inference_n_jobs :
        The forest's ``n_jobs`` is set to this **after** fitting. It defaults to
        1 because joblib's per-call dispatch dominates single-row latency: a
        150-tree forest is several times slower per single-row ``predict_proba``
        at ``n_jobs=2`` than at ``n_jobs=1`` in this container, and a runtime
        guard decides one step at a time. The measured factor is reported by
        ``validation/validate_predictor.py`` check 7, which is the only place
        this repository states it, so the number cannot drift from the run that
        produced it. Fitting still uses ``n_jobs``.

    Returns
    -------
    A fitted :class:`~sklearn.calibration.CalibratedClassifierCV` wrapping a
    :class:`~sklearn.frozen.FrozenEstimator`, so the isotonic map is fitted on
    data the base estimator never saw.

    Notes
    -----
    On scikit-learn 1.9.1 (the version in this container)
    ``CalibratedClassifierCV(base, cv="prefit")`` raises
    ``InvalidParameterError``: the ``"prefit"`` value was removed in favour of
    wrapping the fitted estimator in :class:`sklearn.frozen.FrozenEstimator`.
    Code written against scikit-learn 1.5 or earlier will fail here, and this
    function uses the ``FrozenEstimator`` form.
    """
    if len(train) == 0 or len(calibration) == 0:
        raise ValueError("both splits must be non-empty")
    if len(np.unique(train.labels)) < 2:
        raise ValueError("the training split has only one class; widen the episode set")
    if len(np.unique(calibration.labels)) < 2:
        raise ValueError("the calibration split has only one class; widen the episode set")
    if kind == "forest":
        base: object = RandomForestClassifier(
            n_estimators=int(n_estimators),
            min_samples_leaf=int(min_samples_leaf),
            random_state=int(random_state),
            n_jobs=int(n_jobs),
        )
    elif kind == "logistic":
        base = Pipeline(
            [
                ("scale", StandardScaler()),
                ("logit", LogisticRegression(max_iter=2000, random_state=int(random_state))),
            ]
        )
    else:
        raise ValueError(f"unknown kind {kind!r}; expected forest or logistic")
    base.fit(train.features, train.labels.astype(int))  # type: ignore[attr-defined]
    if isinstance(base, RandomForestClassifier):
        base.n_jobs = int(inference_n_jobs)
    calibrated = CalibratedClassifierCV(FrozenEstimator(base), method="isotonic")
    calibrated.fit(calibration.features, calibration.labels.astype(int))
    return calibrated


def measure_decision_cost(
    call: Callable[[], object], n_calls: int = 2000, warmup: int = 50
) -> float:
    """Mean microseconds per call of ``call``, after ``warmup`` untimed calls.

    Single-row latency, not batch throughput, because a runtime guard decides one
    step at a time. The number is wall-clock on a contended 2-core container and
    is reported as such everywhere it appears.
    """
    if int(n_calls) < 1:
        raise ValueError(f"n_calls must be at least 1, got {n_calls}")
    for _ in range(int(warmup)):
        call()
    start = time.perf_counter()
    for _ in range(int(n_calls)):
        call()
    return (time.perf_counter() - start) * 1e6 / float(n_calls)


def lead_times(
    predictions: np.ndarray,
    fires_now: np.ndarray,
    max_lookback: int = 40,
) -> dict[str, float]:
    """Lead time of a predictor before each firing event, in steps.

    A *firing event* is a step at which baseline authority begins, that is
    ``fires_now[k]`` is true and ``fires_now[k-1]`` is false. The lead is the
    number of consecutive steps immediately before ``k`` at which the predictor
    was positive, capped at ``max_lookback``. A lead of 0 means the event was not
    anticipated at all.

    Returns the event count, the mean, median and maximum lead, and the fraction
    of events with zero lead. An empty event set returns NaNs rather than
    pretending to a number.
    """
    pred = np.asarray(predictions, dtype=bool).ravel()
    fires = np.asarray(fires_now, dtype=bool).ravel()
    if pred.shape != fires.shape:
        raise ValueError(f"predictions has shape {pred.shape}, fires_now has {fires.shape}")
    if int(max_lookback) < 1:
        raise ValueError(f"max_lookback must be at least 1, got {max_lookback}")
    events = np.flatnonzero(fires[1:] & ~fires[:-1]) + 1
    if events.size == 0:
        return {
            "n_events": 0.0,
            "mean_lead": float("nan"),
            "median_lead": float("nan"),
            "max_lead": float("nan"),
            "fraction_zero_lead": float("nan"),
        }
    leads = np.zeros(events.size, dtype=int)
    for i, k in enumerate(events):
        j = 0
        while j < int(max_lookback) and k - 1 - j >= 0 and pred[k - 1 - j]:
            j += 1
        leads[i] = j
    return {
        "n_events": float(events.size),
        "mean_lead": float(leads.mean()),
        "median_lead": float(np.median(leads)),
        "max_lead": float(leads.max()),
        "fraction_zero_lead": float(np.mean(leads == 0)),
    }


def exact_predictor_scores(
    dataset: SwitchDataset,
    guard: SimplexGuard,
    performance: PerformanceController,
    horizon: int,
    mode: str = "worst_case",
    name: str | None = None,
) -> ClassificationScores:
    """Score an :class:`ExactLeadPredictor` on a dataset split, with its latency."""
    predictor = ExactLeadPredictor(guard, performance, horizon, mode)
    pred = predictor.predict_many(dataset.states, dataset.references)
    score = predictor.score_many(dataset.states, dataset.references)
    x0, r0 = dataset.states[0], dataset.references[0]
    cost = measure_decision_cost(lambda: predictor.predict(x0, r0))
    label = name or f"exact {mode} L={horizon}"
    return score_binary(
        label,
        dataset.labels,
        pred,
        score=score,
        microseconds_per_decision=cost,
        notes="exact set computation; no probability output",
    )


def guard_condition_scores(
    dataset: SwitchDataset,
    guard: SimplexGuard,
    performance: PerformanceController,
) -> ClassificationScores:
    """Score the one-step guard condition itself, as the lead-0 analytic baseline.

    On the ``lead = 0`` target this is the criterion, so it is right by
    construction and its precision and recall are 1 up to the saturation of the
    performance input. It is scored anyway, because a benchmark that omits the
    exact answer is not a benchmark.
    """
    pred = np.array(
        [
            guard.condition_margin(x, performance(x, r))[0] < 0.0
            or not guard.invariant_set.contains(x)
            for x, r in zip(dataset.states, dataset.references, strict=True)
        ],
        dtype=bool,
    )
    x0, r0 = dataset.states[0], dataset.references[0]
    cost = measure_decision_cost(
        lambda: guard.condition_margin(x0, performance(x0, r0))[0] < 0.0
    )
    return score_binary(
        "exact one-step guard condition",
        dataset.labels,
        pred,
        score=pred.astype(float),
        microseconds_per_decision=cost,
        notes="the criterion itself; no probability output",
    )
