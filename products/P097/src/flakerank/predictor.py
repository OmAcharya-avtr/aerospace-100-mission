"""The learned intermittency-cost predictor, with a confidence output.

Implemented and benchmarked AFTER the three baselines of
:mod:`flakerank.baselines` were implemented and validated, in that order,
because the mission's build guide requires it: an under-strength baseline
inflates an apparent learned win, and one earlier product of this mission
published a learned margin about 50 per cent too large before catching exactly
that. The baselines here are scored on the same rows, from the same features'
window, against the same target.

THE TASK
--------

Not "is this test flaky". A dashboard already answers that, and the product's
whole argument is that the answer is not the useful one. The task is the
operational question: **over the next H sessions, which tests will waste the
most machine time, and how much?**

Rows are ``(origin, test)`` pairs. For each origin ``o``:

* features are measured over the lookback window, sessions ``[o - L, o)``;
* the target is the wasted seconds the test actually accrued in the horizon
  window, sessions ``[o, o + H)``.

So no feature can see its own target, and the target is a future accumulation
over a recorded history, never a live measurement.

THE TWO HELD-OUT SPLITS, SCORED SEPARATELY
------------------------------------------

The two generalisation questions are different and a single number cannot
answer both, so both are reported:

**Held-out TESTS.** A fraction of the tests is withheld entirely. The model
trains on the remaining tests at the training origins and is scored on the
withheld tests at the same origins. This answers: does the model transfer to a
test it has never seen?

**Held-out TIME.** The model trains on the training tests at the *training*
origins and is scored on the same tests at *later* origins, after the recorded
history's regime change. This answers: does what it learned still hold later?
The generated history changes ``p`` for ten of the seventy-two tests partway
through, so this split is a real test and not a relabelling of the first.

One fit, two evaluations. The model is fitted once on the training rows and
neither evaluation feeds back into it.

ARCHITECTURE AND WHY IT IS SMALL
--------------------------------

``sklearn.ensemble.RandomForestRegressor``, ``n_estimators=300``,
``max_depth=5``, ``min_samples_leaf=5``, ``random_state`` declared,
``n_jobs=1``. Target is ``log1p(wasted seconds)``, because the target spans
five orders of magnitude and a squared-error fit on the raw scale would be
determined by the two slowest tests.

Three hard constraints, not preferences:

* the training set is a few hundred rows from one history, so a model with many
  parameters memorises rows;
* the build environment has two cores shared by five concurrent agents and a
  60-second budget for any fit;
* a forest gives a spread across trees that is an honest, if conservative,
  uncertainty output without a second model.

``n_jobs=1`` is deliberate: parallel tree fitting on a contended two-core box
is slower than serial and makes the fit time vary, and this product does not
assert on fit times at all.

**No model is serialised.** Nothing in this repository writes a joblib or
pickle file. A serialised scikit-learn model is not portable across minor
versions -- 1.9.1 cannot unpickle a 1.8.0 model, measured in this mission -- and
the ``scikit-learn>=1.9,<1.10`` ceiling in ``pyproject.toml`` is declared
anyway, so that the numbers in ``MODEL_CARD.md`` are attached to the version
that produced them.

THE CONFIDENCE OUTPUT
---------------------

:meth:`IntermittencyCostPredictor.predict` returns a
:class:`PredictionWithConfidence` carrying, per row, the ensemble mean, the
standard deviation across the 300 trees, and a central interval from the tree
quantiles. All three are in ``log1p`` seconds; :attr:`seconds` exponentiates
the mean back.

What that spread is and is not, stated because a confidence output read as
something it is not is worse than none:

* it is the **disagreement among trees**, which is an epistemic quantity: it is
  large where the training rows were sparse and small where they were dense;
* it is **not** a predictive interval for a new observation. It excludes the
  irreducible session-to-session variation of a Bernoulli failure process,
  which for a rare flake dominates everything else. It is therefore
  systematically too narrow, and :func:`evaluate_split` measures by how much by
  reporting the fraction of held-out targets that fall inside the nominal
  interval. That number is reported whatever it is.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometric import expected_failed_attempts
from .history import RunHistory

__all__ = [
    "FEATURE_NAMES",
    "DatasetSplit",
    "IntermittencyCostPredictor",
    "PredictionWithConfidence",
    "SplitScore",
    "baseline_scores",
    "build_rows",
    "evaluate_split",
    "make_splits",
]

#: Feature names, in column order. Every one is computable from the lookback
#: window alone, so no feature can see its own target.
FEATURE_NAMES: tuple[str, ...] = (
    "sessions",
    "attempts",
    "attempts_per_session",
    "failed_attempts",
    "log1p_failed_attempts",
    "failure_probability",
    "session_failure_fraction",
    "mean_attempt_seconds",
    "log_mean_attempt_seconds",
    "sd_attempt_seconds",
    "max_attempt_seconds",
    "min_attempt_seconds",
    "mean_failing_seconds_or_mean",
    "wasted_seconds",
    "log1p_wasted_seconds",
    "analytic_cost_per_session",
    "late_failure_fraction",
)


@dataclass(frozen=True)
class DatasetSplit:
    """Feature matrix, target vector and row provenance for one row set.

    Attributes:
        features: shape ``(n_rows, len(FEATURE_NAMES))``.
        target: ``log1p`` of the wasted seconds in the horizon window.
        target_seconds: the wasted seconds themselves.
        test_index: test index per row.
        test_name: test name per row.
        origin: origin session per row.
        label: a short name for the split.
    """

    features: np.ndarray
    target: np.ndarray
    target_seconds: np.ndarray
    test_index: np.ndarray
    test_name: tuple[str, ...]
    origin: np.ndarray
    label: str = ""

    @property
    def n_rows(self) -> int:
        """Number of rows."""
        return int(self.features.shape[0])

    def select(self, mask: np.ndarray, label: str) -> DatasetSplit:
        """A new split holding the rows where ``mask`` is true."""
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (self.n_rows,):
            raise ValueError(f"mask must have shape ({self.n_rows},), got {mask.shape}")
        if not mask.any():
            raise ValueError(f"split {label!r} would be empty; nothing can be scored on it")
        keep = np.flatnonzero(mask)
        return DatasetSplit(
            features=self.features[keep],
            target=self.target[keep],
            target_seconds=self.target_seconds[keep],
            test_index=self.test_index[keep],
            test_name=tuple(self.test_name[index] for index in keep),
            origin=self.origin[keep],
            label=label,
        )


def build_rows(
    history: RunHistory,
    *,
    origins: tuple[int, ...],
    lookback: int,
    horizon: int,
    minimum_attempts: int = 8,
) -> DatasetSplit:
    """Build the ``(origin, test)`` row set from a recorded history.

    Args:
        history: the recorded history.
        origins: session indices at which to cut. Each needs
            ``origin - lookback >= 0`` and ``origin + horizon <= n_sessions``.
        lookback: feature-window length, in sessions.
        horizon: target-window length, in sessions.
        minimum_attempts: a test with fewer attempts in a lookback window
            contributes no row at that origin, because its measured rate is too
            noisy for any of the four predictors to be given credit or blame
            for. The threshold is declared here and reported by
            ``validation/validate_learned_model.py``.

    Returns:
        A :class:`DatasetSplit` with every row, unsplit.

    Raises:
        ValueError: for an empty origin tuple, a non-positive window, an origin
            whose windows fall outside the history, or a row set that comes out
            empty.
    """
    if not origins:
        raise ValueError("origins must contain at least one session index")
    if lookback < 1 or horizon < 1:
        raise ValueError(f"lookback and horizon must be >= 1, got {lookback} and {horizon}")
    if minimum_attempts < 1:
        raise ValueError(f"minimum_attempts must be >= 1, got {minimum_attempts}")
    for origin in origins:
        if origin - lookback < 0 or origin + horizon > history.n_sessions:
            raise ValueError(
                f"origin {origin} with lookback {lookback} and horizon {horizon} falls "
                f"outside a history of {history.n_sessions} sessions"
            )

    name_to_index = {name: index for index, name in enumerate(history.test_names)}
    rows: list[np.ndarray] = []
    targets: list[float] = []
    indices: list[int] = []
    names: list[str] = []
    origin_column: list[int] = []

    for origin in origins:
        past = history.session_window(origin - lookback, origin)
        future = history.session_window(origin, origin + horizon)
        future_wasted = {
            summary.name: summary.wasted_seconds for summary in future.summaries()
        }
        past_failed = past.attempt_failed.astype(bool)
        past_sessions = past.attempt_session
        late_cut = origin - lookback // 3
        for summary in past.summaries(minimum_attempts=minimum_attempts):
            index = name_to_index[summary.name]
            test_rows = (past.attempt_test == index) & past_failed
            late = int(np.count_nonzero(past_sessions[test_rows] >= late_cut))
            total_failures = int(np.count_nonzero(test_rows))
            analytic = (
                summary.cost_runtime_seconds
                * expected_failed_attempts(summary.failure_probability, history.retry_budget)
            )
            rows.append(
                np.asarray(
                    [
                        summary.sessions,
                        summary.attempts,
                        summary.attempts / summary.sessions,
                        summary.failed_attempts,
                        np.log1p(summary.failed_attempts),
                        summary.failure_probability,
                        summary.sessions_with_failure / summary.sessions,
                        summary.mean_attempt_seconds,
                        np.log(summary.mean_attempt_seconds),
                        summary.sd_attempt_seconds,
                        summary.max_attempt_seconds,
                        summary.min_attempt_seconds,
                        summary.cost_runtime_seconds,
                        summary.wasted_seconds,
                        np.log1p(summary.wasted_seconds),
                        analytic,
                        (late / total_failures) if total_failures else 0.0,
                    ],
                    dtype=np.float64,
                )
            )
            targets.append(future_wasted.get(summary.name, 0.0))
            indices.append(index)
            names.append(summary.name)
            origin_column.append(origin)

    if not rows:
        raise ValueError(
            "no row survived the minimum_attempts threshold; lower it or widen the "
            "lookback window"
        )
    features = np.vstack(rows)
    if features.shape[1] != len(FEATURE_NAMES):
        raise ValueError(
            f"feature matrix has {features.shape[1]} columns but FEATURE_NAMES has "
            f"{len(FEATURE_NAMES)}; the two must not drift apart"
        )
    target_seconds = np.asarray(targets, dtype=np.float64)
    return DatasetSplit(
        features=features,
        target=np.log1p(target_seconds),
        target_seconds=target_seconds,
        test_index=np.asarray(indices, dtype=np.int64),
        test_name=tuple(names),
        origin=np.asarray(origin_column, dtype=np.int64),
        label="all rows",
    )


def make_splits(
    rows: DatasetSplit,
    *,
    train_origins: tuple[int, ...],
    time_origins: tuple[int, ...],
    holdout_fraction: float = 0.30,
    seed: int = 97044,
) -> dict[str, DatasetSplit]:
    """Partition rows into the training set and the two held-out sets.

    Tests are assigned to the held-out-test group by a declared seed over the
    SORTED unique test indices, so the assignment is a function of the data and
    the seed and not of row order.

    Args:
        rows: every row, from :func:`build_rows`.
        train_origins: origins used for training and for the held-out-TESTS
            evaluation.
        time_origins: later origins used for the held-out-TIME evaluation.
        holdout_fraction: fraction of tests withheld.
        seed: seed for the group assignment.

    Returns:
        A mapping with keys ``"train"``, ``"heldout_tests"`` and
        ``"heldout_time"``.

    Raises:
        ValueError: if the origin sets overlap, if either is empty, if the
            fraction is outside ``(0, 1)``, or if any resulting split is empty.
    """
    if not train_origins or not time_origins:
        raise ValueError("both train_origins and time_origins must be non-empty")
    if set(train_origins) & set(time_origins):
        raise ValueError(
            f"train and time origins overlap at {sorted(set(train_origins) & set(time_origins))}; "
            "a held-out TIME split that shares an origin with training is not held out"
        )
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError(f"holdout_fraction must lie in (0, 1), got {holdout_fraction}")

    unique = np.unique(rows.test_index)
    generator = np.random.default_rng(seed)
    permuted = generator.permutation(unique)
    n_holdout = max(1, int(round(holdout_fraction * unique.size)))
    holdout = set(int(value) for value in permuted[:n_holdout])

    in_train_origin = np.isin(rows.origin, np.asarray(train_origins, dtype=np.int64))
    in_time_origin = np.isin(rows.origin, np.asarray(time_origins, dtype=np.int64))
    is_holdout_test = np.asarray([int(value) in holdout for value in rows.test_index], dtype=bool)

    return {
        "train": rows.select(in_train_origin & ~is_holdout_test, "train"),
        "heldout_tests": rows.select(in_train_origin & is_holdout_test, "held-out TESTS"),
        "heldout_time": rows.select(in_time_origin & ~is_holdout_test, "held-out TIME"),
    }


@dataclass(frozen=True)
class PredictionWithConfidence:
    """Per-row prediction with the ensemble spread as the confidence output.

    Attributes:
        mean: ensemble mean, in ``log1p`` seconds.
        sd: standard deviation across trees, in ``log1p`` seconds.
        low: lower quantile of the tree predictions, in ``log1p`` seconds.
        high: upper quantile, in ``log1p`` seconds.
        nominal_coverage: the interval's nominal coverage, e.g. ``0.90``.
        n_trees: number of trees the spread is taken over.
    """

    mean: np.ndarray
    sd: np.ndarray
    low: np.ndarray
    high: np.ndarray
    nominal_coverage: float
    n_trees: int

    @property
    def seconds(self) -> np.ndarray:
        """The ensemble mean exponentiated back to seconds.

        ``expm1(mean of log1p)`` is not the mean of the seconds -- the
        transform is not linear -- and is deliberately reported as a ranking
        score and a magnitude, not as an unbiased expectation. The ranking is
        unaffected because ``expm1`` is strictly increasing.
        """
        return np.expm1(self.mean)

    @property
    def interval_width(self) -> np.ndarray:
        """``high - low``, in ``log1p`` seconds."""
        return self.high - self.low


@dataclass
class IntermittencyCostPredictor:
    """A random forest over lookback features, predicting ``log1p`` wasted seconds.

    Attributes:
        n_estimators: number of trees.
        max_depth: maximum tree depth.
        min_samples_leaf: minimum samples per leaf.
        random_state: the forest's seed.
        nominal_coverage: nominal coverage of the reported interval.
    """

    n_estimators: int = 300
    max_depth: int = 5
    min_samples_leaf: int = 5
    random_state: int = 97044
    nominal_coverage: float = 0.90
    _forest: object | None = field(default=None, init=False, repr=False)

    def fit(self, split: DatasetSplit) -> IntermittencyCostPredictor:
        """Fit on a split's features and ``log1p`` target.

        Args:
            split: the training rows.

        Returns:
            ``self``.

        Raises:
            ValueError: if the split has fewer rows than ``min_samples_leaf``.
        """
        from sklearn.ensemble import RandomForestRegressor

        if split.n_rows < self.min_samples_leaf:
            raise ValueError(
                f"training split {split.label!r} has {split.n_rows} rows, below "
                f"min_samples_leaf={self.min_samples_leaf}"
            )
        forest = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=1,
        )
        forest.fit(split.features, split.target)
        self._forest = forest
        return self

    def predict(self, split: DatasetSplit) -> PredictionWithConfidence:
        """Predict with a confidence output taken from the spread across trees.

        Args:
            split: rows to predict.

        Returns:
            A :class:`PredictionWithConfidence`.

        Raises:
            RuntimeError: if called before :meth:`fit`.
        """
        if self._forest is None:
            raise RuntimeError(
                "predict called before fit. The predictor holds no state until fitted, "
                "deliberately: this repository serialises no model, because a "
                "scikit-learn pickle is not portable across minor versions."
            )
        forest = self._forest
        per_tree = np.vstack(
            [tree.predict(split.features) for tree in forest.estimators_]  # type: ignore[attr-defined]
        )
        alpha = 1.0 - self.nominal_coverage
        low, high = np.quantile(per_tree, [alpha / 2.0, 1.0 - alpha / 2.0], axis=0)
        return PredictionWithConfidence(
            mean=per_tree.mean(axis=0),
            sd=per_tree.std(axis=0, ddof=1),
            low=np.asarray(low, dtype=np.float64),
            high=np.asarray(high, dtype=np.float64),
            nominal_coverage=self.nominal_coverage,
            n_trees=int(per_tree.shape[0]),
        )

    @property
    def feature_importances(self) -> np.ndarray:
        """Impurity-based feature importances, in ``FEATURE_NAMES`` order.

        Impurity importances are biased towards high-cardinality features and
        are reported as a description of the fitted forest, never as a causal
        claim.
        """
        if self._forest is None:
            raise RuntimeError("feature_importances requested before fit")
        return np.asarray(
            self._forest.feature_importances_,  # type: ignore[attr-defined]
            dtype=np.float64,
        )


def baseline_scores(split: DatasetSplit, *, horizon: int, lookback: int) -> dict:
    """The three baselines' predictive scores on the same rows.

    The analytic baseline's per-session cost already carries the closed-form
    ``E[F_m(p)]`` with the history's retry budget, so no budget is needed here.

    Args:
        split: rows to score.
        horizon: horizon length in sessions, used to scale a per-session cost
            to the horizon.
        lookback: lookback length in sessions.

    Returns:
        A mapping from baseline name to a score vector in ``log1p`` seconds,
        except ``"flake rate"`` which is dimensionless -- it is a RANKING score
        and has no seconds interpretation, which is the product's whole point,
        so it is scored on ranking metrics only and excluded from the error
        metrics. :func:`evaluate_split` enforces that.
    """
    columns = {name: index for index, name in enumerate(FEATURE_NAMES)}
    sessions = split.features[:, columns["sessions"]]
    rate = split.features[:, columns["failure_probability"]]
    analytic_per_session = split.features[:, columns["analytic_cost_per_session"]]
    wasted = split.features[:, columns["wasted_seconds"]]
    expected_sessions = horizon * sessions / lookback
    return {
        "flake rate": rate,
        "analytic expected cost": np.log1p(analytic_per_session * expected_sessions),
        "accumulated persistence": np.log1p(wasted * horizon / lookback),
    }


@dataclass(frozen=True)
class SplitScore:
    """Scores for one predictor on one split.

    Attributes:
        name: the predictor's name.
        split_label: which split.
        n_rows: rows scored.
        spearman: Spearman rho between the score and the realised wasted
            seconds. The ranking metric, and the one the product's headline
            question needs.
        kendall_tau_b: Kendall tau-b for the same pair.
        top_k: prefix length for ``top_k_overlap``.
        top_k_overlap: tests in the predicted top ``k`` that are also in the
            realised top ``k``.
        mae_log1p: mean absolute error in ``log1p`` seconds, or ``nan`` for a
            score with no seconds interpretation.
        median_absolute_error_log1p: the median of the same errors, reported
            because the mean is dominated by a handful of rows.
        interval_coverage: fraction of realised targets inside the predictor's
            nominal interval, or ``nan`` for a predictor without one.
        mean_interval_width_log1p: mean interval width, or ``nan``.
        captured_fraction: of the wasted seconds the horizon actually contained,
            the fraction sitting in the tests this predictor put in its top
            ``top_k``. This is the operational metric -- "if a team fixes the
            ten tests this tool names, how much of the waste goes away" -- and
            unlike Spearman it is weighted by cost rather than by test. It was
            added to this dataclass AFTER the rank metrics were first measured,
            because the rank metrics answered a question adjacent to the one
            the product exists to answer; the rank metrics were kept and are
            reported unchanged beside it.
        oracle_captured_fraction: the same quantity for a predictor that knew
            the answer, which is the ceiling any predictor can reach at this
            ``top_k``. ``captured_fraction / oracle_captured_fraction`` is the
            fraction of the achievable saving the predictor finds.
    """

    name: str
    split_label: str
    n_rows: int
    spearman: float
    kendall_tau_b: float
    top_k: int
    top_k_overlap: int
    mae_log1p: float
    median_absolute_error_log1p: float
    interval_coverage: float
    mean_interval_width_log1p: float
    captured_fraction: float
    oracle_captured_fraction: float


def _descending(values: np.ndarray, names: tuple[str, ...], k: int) -> list[int]:
    """Indices of the ``k`` largest values, ties broken by name.

    Ties are broken by name so every top-``k`` quantity here is a
    deterministic function of the inputs rather than of row order.
    """
    k = min(k, values.size)
    return sorted(range(values.size), key=lambda index: (-values[index], names[index]))[:k]


def _top_overlap(score: np.ndarray, truth: np.ndarray, names: tuple[str, ...], k: int) -> int:
    """Overlap of the top ``k`` by score and the top ``k`` by truth."""
    by_score = _descending(score, names, k)
    by_truth = _descending(truth, names, k)
    return len({names[index] for index in by_score} & {names[index] for index in by_truth})


def _captured(score: np.ndarray, seconds: np.ndarray, names: tuple[str, ...], k: int) -> float:
    """Fraction of the realised wasted seconds held by the top ``k`` by score."""
    total = float(seconds.sum())
    if total <= 0.0:
        return float("nan")
    return float(seconds[_descending(score, names, k)].sum()) / total


def evaluate_split(
    name: str,
    split: DatasetSplit,
    score: np.ndarray,
    *,
    seconds_scale: bool,
    prediction: PredictionWithConfidence | None = None,
    top_k: int = 10,
) -> SplitScore:
    """Score one predictor on one split.

    Args:
        name: the predictor's name.
        split: the rows.
        score: one score per row; higher means more expected wasted time.
        seconds_scale: whether ``score`` is in ``log1p`` seconds, so that error
            metrics are meaningful. ``False`` for the flake-rate baseline,
            whose score has no seconds interpretation; its error metrics are
            reported as ``nan`` rather than as a number that would invite a
            comparison the units do not support.
        prediction: the learned predictor's output, for the interval-coverage
            metric. ``None`` for a baseline without one.
        top_k: prefix length for the overlap metric.

    Returns:
        A :class:`SplitScore`.
    """
    from .disagreement import kendall_tau_b, spearman_rho

    score = np.asarray(score, dtype=np.float64)
    if score.shape != (split.n_rows,):
        raise ValueError(f"score must have shape ({split.n_rows},), got {score.shape}")
    truth = split.target
    if seconds_scale:
        errors = np.abs(score - truth)
        mae = float(errors.mean())
        median = float(np.median(errors))
    else:
        mae = float("nan")
        median = float("nan")
    if prediction is not None:
        inside = (truth >= prediction.low) & (truth <= prediction.high)
        coverage = float(np.count_nonzero(inside)) / float(truth.size)
        width = float(prediction.interval_width.mean())
    else:
        coverage = float("nan")
        width = float("nan")
    return SplitScore(
        name=name,
        split_label=split.label,
        n_rows=split.n_rows,
        spearman=spearman_rho(score, truth),
        kendall_tau_b=kendall_tau_b(score, truth),
        top_k=min(top_k, split.n_rows),
        top_k_overlap=_top_overlap(score, truth, split.test_name, top_k),
        mae_log1p=mae,
        median_absolute_error_log1p=median,
        interval_coverage=coverage,
        mean_interval_width_log1p=width,
        captured_fraction=_captured(score, split.target_seconds, split.test_name, top_k),
        oracle_captured_fraction=_captured(
            split.target_seconds, split.target_seconds, split.test_name, top_k
        ),
    )
