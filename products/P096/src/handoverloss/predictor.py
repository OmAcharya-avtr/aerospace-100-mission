"""The learned component, and the scoring harness both it and the baselines run through.

WHAT IS PREDICTED, AND FROM WHAT
--------------------------------

From pre-handover state only — the columns of
:data:`~handoverloss.corpus.FEATURE_COLUMNS`, none of which is a realised
duration or a realised loss — predict the THREE per-mechanism losses of the
coming handover, in bits, and hence which mechanism will dominate it.

The headline metric is **dominance accuracy**: the fraction of held-out
handovers whose dominant mechanism is predicted correctly. That is the metric
because it is the decision an operator makes. Loss magnitude is reported beside
it as the mean absolute error of ``log10(1 + loss)``, which is the scale the
quantity spans: per-mechanism losses in this corpus range over several orders of
magnitude, and an absolute-bit error would be a report about the largest few
events only.

ARCHITECTURE
------------

:class:`sklearn.ensemble.RandomForestRegressor` with three outputs, trained on
``log1p`` of each mechanism's loss in bits. A forest rather than a linear model
because the mechanism structure is piecewise — each term is a ``max(0, ...)`` of
a different combination of the inputs — and a forest represents a piecewise
constant surface without being told where the breakpoints are. Breiman, L.,
"Random Forests", *Machine Learning*, volume 45, number 1, pages 5-32, 2001.

The log1p transform is applied because the targets are non-negative and
heavy-tailed; predictions are inverted with ``expm1`` and clipped at zero, so a
prediction is never negative.

THE UNCERTAINTY OUTPUT
----------------------

Not a point estimate. :class:`Prediction` carries, per mechanism:

* ``log_mean``: the forest's mean across trees, on the ``log1p`` scale;
* ``log_sd``: the standard deviation ACROSS TREES on that scale, which is the
  forest's own disagreement and is the uncertainty estimate used here. The
  across-tree spread is a lower bound on predictive uncertainty — it captures
  model uncertainty and not the irreducible noise of the handover itself — and
  that is stated rather than glossed. Wager, S., Hastie, T. and Efron, B.,
  "Confidence intervals for random forests: the jackknife and the infinitesimal
  jackknife", *Journal of Machine Learning Research*, volume 15, pages
  1625-1651, 2014, is the reference for why the raw across-tree spread
  understates and for the jackknife corrections that are NOT applied here;
* ``low_bits`` / ``high_bits``: the ``log_mean +/- z * log_sd`` band inverted to
  bits, with ``z`` from the requested nominal coverage.

and, for the dominance decision:

* ``dominant``: the mechanism with the largest predicted mean;
* ``dominance_confidence``: the fraction of trees whose own three-output
  prediction has the same argmax. This is a vote share over the ensemble, not a
  calibrated probability, and
  ``validation/validate_learned_vs_baselines.py`` reports its measured
  reliability rather than assuming it.

The achieved coverage of the ``low_bits`` / ``high_bits`` band is measured on
both held-out splits and reported. It is expected to fall below nominal, for the
reason above.

COMPUTE
-------

200 trees on roughly 1000 training rows and 18 features. Measured training time
is printed by ``validation/validate_learned_vs_baselines.py``; the declared
budget is 60 s of CPU and the measurement is far inside it. No model is
serialised anywhere in this repository: the forest is refitted from the
committed corpus in under a second, and a pickled estimator would bind the
repository to one scikit-learn build whose unpickling format is not part of that
library's public API.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor

from .baselines import fluid_decomposition_array
from .corpus import FEATURE_COLUMNS, Corpus, Split
from .sequence import MECHANISMS

#: Hyperparameters, chosen by three-fold GROUPED cross-validation over station
#: pairs on the TRAINING side of the station-pair split only. The test side was
#: never consulted. Six candidates were compared and their grouped-CV dominance
#: accuracies are reported by ``validation/validate_learned_vs_baselines.py``;
#: they span 0.343 to 0.360, so the choice is close to immaterial and the
#: limitation it exposes is structural rather than a matter of tuning. That
#: finding is in MODEL_CARD.md.
DEFAULT_N_ESTIMATORS = 300
DEFAULT_MAX_DEPTH = 20
DEFAULT_MIN_SAMPLES_LEAF = 2

#: Monte Carlo draws behind each EXACT acquisition-time quantile used by
#: :func:`baseline_predictions`. 60 000 keeps the sampling error of a 95th
#: percentile below about half a per cent while leaving the whole scoring pass
#: inside this package's 60 s compute budget; the measured error is reported by
#: ``validation/validate_acquisition_model.py``.
QUANTILE_DRAWS = 60_000

#: The candidate grid, kept so the selection is reproducible from the package
#: rather than from a transcript. ``(max_depth, min_samples_leaf, n_estimators)``.
HYPERPARAMETER_CANDIDATES: tuple[tuple[int | None, int, int], ...] = (
    (14, 3, 200),
    (None, 1, 200),
    (8, 5, 200),
    (20, 2, 300),
    (None, 3, 400),
    (6, 8, 200),
)


@dataclass(frozen=True)
class Prediction:
    """Per-mechanism prediction with an uncertainty output. Bits, plus log-scale.

    Attributes
    ----------
    loss_bits:
        ``(n, 3)`` point predictions, bits, non-negative.
    low_bits, high_bits:
        ``(n, 3)`` band at :attr:`confidence`, bits.
    log_mean, log_sd:
        ``(n, 3)`` across-tree mean and standard deviation on the ``log1p``
        scale. :attr:`log_sd` is the uncertainty output.
    dominant_index:
        ``(n,)`` argmax of :attr:`loss_bits`.
    dominance_confidence:
        ``(n,)`` fraction of trees agreeing with :attr:`dominant_index`.
    confidence:
        Nominal coverage of the band.
    """

    loss_bits: np.ndarray
    low_bits: np.ndarray
    high_bits: np.ndarray
    log_mean: np.ndarray
    log_sd: np.ndarray
    dominant_index: np.ndarray
    dominance_confidence: np.ndarray
    confidence: float

    def dominant_names(self) -> list[str]:
        """Predicted dominant mechanism per event, in event order."""
        return [MECHANISMS[int(i)] for i in self.dominant_index]


@dataclass(frozen=True)
class PredictorScore:
    """One predictor's score on one split. Dimensionless unless stated.

    Attributes
    ----------
    predictor:
        Name of the predictor scored.
    split:
        ``"station_pairs"`` or ``"time"``.
    n_test:
        Events scored.
    dominance_accuracy:
        Fraction of events whose dominant mechanism was predicted correctly.
        The headline metric.
    dominance_accuracy_low, dominance_accuracy_high:
        Event-level bootstrap bounds on it.
    log_mae_total:
        Mean absolute error of ``log10(1 + total loss in bits)``.
    log_mae_per_mechanism:
        The same, per mechanism, in :data:`MECHANISMS` order.
    band_coverage:
        Fraction of per-mechanism ground truths inside the predictor's band, or
        ``nan`` for a predictor with no band.
    """

    predictor: str
    split: str
    n_test: int
    dominance_accuracy: float
    dominance_accuracy_low: float
    dominance_accuracy_high: float
    log_mae_total: float
    log_mae_per_mechanism: tuple[float, float, float]
    band_coverage: float


class HandoverLossPredictor:
    """Random-forest predictor of the three per-mechanism losses, with uncertainty.

    Parameters
    ----------
    n_estimators:
        Number of trees, at least 10.
    max_depth:
        Maximum tree depth, or ``None`` for unlimited.
    min_samples_leaf:
        Minimum samples per leaf, at least 1.
    random_state:
        Seed for the forest. Fixed by default so a refit is reproducible.

    Raises
    ------
    ValueError
        On any out-of-range hyperparameter, or on a fit with fewer rows than
        trees' worth of signal (below 20 rows this class refuses, because a
        forest fitted on fewer rows reports an across-tree spread that is an
        artefact of the bootstrap sample size and not of the problem).
    """

    def __init__(
        self,
        n_estimators: int = DEFAULT_N_ESTIMATORS,
        max_depth: int | None = DEFAULT_MAX_DEPTH,
        min_samples_leaf: int = DEFAULT_MIN_SAMPLES_LEAF,
        random_state: int = 96_096,
    ) -> None:
        if n_estimators < 10:
            raise ValueError(f"n_estimators must be at least 10, got {n_estimators!r}")
        if max_depth is not None and max_depth < 1:
            raise ValueError(f"max_depth must be at least 1 or None, got {max_depth!r}")
        if min_samples_leaf < 1:
            raise ValueError(
                f"min_samples_leaf must be at least 1, got {min_samples_leaf!r}"
            )
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.min_samples_leaf = int(min_samples_leaf)
        self.random_state = int(random_state)
        self._forest: RandomForestRegressor | None = None
        self._feature_names: tuple[str, ...] = FEATURE_COLUMNS

    @property
    def fitted(self) -> bool:
        """Whether :meth:`fit` has been called."""
        return self._forest is not None

    def fit(self, features: np.ndarray, targets_bits: np.ndarray) -> HandoverLossPredictor:
        """Fit on ``log1p`` of the three per-mechanism losses.

        Parameters
        ----------
        features:
            ``(n, len(FEATURE_COLUMNS))`` pre-handover observables.
        targets_bits:
            ``(n, 3)`` ground-truth per-mechanism loss, bits, non-negative.
        """
        x = np.asarray(features, dtype=float)
        y = np.asarray(targets_bits, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(FEATURE_COLUMNS):
            raise ValueError(
                f"features must have shape (n, {len(FEATURE_COLUMNS)}), got {x.shape}"
            )
        if y.ndim != 2 or y.shape[1] != 3:
            raise ValueError(f"targets_bits must have shape (n, 3), got {y.shape}")
        if x.shape[0] != y.shape[0]:
            raise ValueError(
                f"features has {x.shape[0]} rows and targets_bits has {y.shape[0]}"
            )
        if x.shape[0] < 20:
            raise ValueError(
                f"fit needs at least 20 rows, got {x.shape[0]}. Below that the "
                "across-tree spread this class reports as its uncertainty is an "
                "artefact of the bootstrap sample size rather than a property of the "
                "problem."
            )
        if np.any(y < 0.0):
            raise ValueError("targets_bits must be non-negative everywhere")
        if not np.isfinite(x).all():
            raise ValueError("features contains a non-finite value")
        self._forest = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=1,
        ).fit(x, np.log1p(y))
        return self

    def predict(self, features: np.ndarray, confidence: float = 0.90) -> Prediction:
        """Predict the three losses with an uncertainty output.

        Parameters
        ----------
        features:
            ``(n, len(FEATURE_COLUMNS))`` pre-handover observables.
        confidence:
            Nominal coverage of the returned band, in ``(0, 1)``.

        Returns
        -------
        Prediction

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        ValueError
            On a wrong feature width or a ``confidence`` outside ``(0, 1)``.
        """
        if self._forest is None:
            raise RuntimeError(
                "HandoverLossPredictor.predict called before fit; there is no forest to "
                "predict with"
            )
        if not 0.0 < confidence < 1.0:
            raise ValueError(f"confidence must lie in (0, 1), got {confidence!r}")
        x = np.asarray(features, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(FEATURE_COLUMNS):
            raise ValueError(
                f"features must have shape (n, {len(FEATURE_COLUMNS)}), got {x.shape}"
            )
        # (trees, n, 3) on the log1p scale.
        per_tree = np.stack([tree.predict(x) for tree in self._forest.estimators_])
        log_mean = per_tree.mean(axis=0)
        log_sd = per_tree.std(axis=0, ddof=0)
        z = float(stats.norm.ppf(0.5 + 0.5 * confidence))
        loss = np.maximum(0.0, np.expm1(log_mean))
        low = np.maximum(0.0, np.expm1(log_mean - z * log_sd))
        high = np.maximum(0.0, np.expm1(log_mean + z * log_sd))
        dominant = np.argmax(log_mean, axis=1)
        votes = np.argmax(per_tree, axis=2)
        agreement = (votes == dominant[None, :]).mean(axis=0)
        return Prediction(
            loss_bits=loss,
            low_bits=low,
            high_bits=high,
            log_mean=log_mean,
            log_sd=log_sd,
            dominant_index=dominant,
            dominance_confidence=agreement,
            confidence=float(confidence),
        )

    def feature_importances(self) -> dict[str, float]:
        """Impurity-based importances keyed by feature name.

        Impurity importances are biased towards high-cardinality features and are
        reported here as a description of the fitted forest, not as evidence
        about the mechanism. That caveat is in ``MODEL_CARD.md``.
        """
        if self._forest is None:
            raise RuntimeError("feature_importances called before fit")
        return dict(
            zip(self._feature_names, self._forest.feature_importances_.tolist(), strict=True)
        )


def _dominance_bootstrap(
    correct: np.ndarray, resamples: int, confidence: float, seed: int
) -> tuple[float, float]:
    """Event-level percentile bootstrap bounds on a mean of 0/1 outcomes."""
    rng = np.random.default_rng(seed)
    count = correct.size
    draws = np.empty(resamples, dtype=float)
    for index in range(resamples):
        draws[index] = correct[rng.integers(0, count, size=count)].mean()
    alpha = 1.0 - confidence
    low, high = np.percentile(draws, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return float(low), float(high)


def _score(
    name: str,
    split_name: str,
    truth_bits: np.ndarray,
    predicted_bits: np.ndarray,
    *,
    low_bits: np.ndarray | None = None,
    high_bits: np.ndarray | None = None,
    resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 96_096,
) -> PredictorScore:
    """Score any predictor's ``(n, 3)`` prediction against the ground truth."""
    truth = np.asarray(truth_bits, dtype=float)
    predicted = np.asarray(predicted_bits, dtype=float)
    if truth.shape != predicted.shape or truth.ndim != 2 or truth.shape[1] != 3:
        raise ValueError(
            f"truth_bits and predicted_bits must both have shape (n, 3); got "
            f"{truth.shape} and {predicted.shape}"
        )
    correct = (np.argmax(predicted, axis=1) == np.argmax(truth, axis=1)).astype(float)
    accuracy = float(correct.mean())
    low, high = _dominance_bootstrap(correct, resamples, confidence, seed)
    log_truth = np.log10(1.0 + truth)
    log_predicted = np.log10(1.0 + np.maximum(predicted, 0.0))
    per_mechanism = np.abs(log_truth - log_predicted).mean(axis=0)
    total_truth = np.log10(1.0 + truth.sum(axis=1))
    total_predicted = np.log10(1.0 + np.maximum(predicted.sum(axis=1), 0.0))
    coverage = float("nan")
    if low_bits is not None and high_bits is not None:
        inside = (truth >= np.asarray(low_bits)) & (truth <= np.asarray(high_bits))
        coverage = float(inside.mean())
    return PredictorScore(
        predictor=name,
        split=split_name,
        n_test=int(truth.shape[0]),
        dominance_accuracy=accuracy,
        dominance_accuracy_low=low,
        dominance_accuracy_high=high,
        log_mae_total=float(np.abs(total_truth - total_predicted).mean()),
        log_mae_per_mechanism=tuple(float(v) for v in per_mechanism),  # type: ignore[arg-type]
        band_coverage=coverage,
    )


def _column(corpus: Corpus, name: str, index: np.ndarray) -> np.ndarray:
    return np.asarray(corpus.table[name], dtype=float)[index]


def baseline_predictions(
    corpus: Corpus, index: np.ndarray, *, acquisition_quantile: float | None = None
) -> dict[str, np.ndarray]:
    """``(n, 3)`` predictions of both analytic baselines on ``corpus[index]``.

    Both are evaluated from PRE-HANDOVER state alone, exactly as the learned
    component is, so the comparison is on the same information. In particular
    neither sees the realised acquisition time: both use the acquisition model's
    declared mean, or its ``acquisition_quantile`` upper quantile when one is
    supplied, which is the conservative engineer's version of baseline 1.

    Returns
    -------
    dict
        ``{"acquisition_only": (n, 3), "fluid": (n, 3)}``. Baseline 1 places its
        whole prediction in the acquisition column, because that is what the
        assumption says: the other two mechanisms do not exist.
    """
    from .acquisition import AcquisitionModel

    source = _column(corpus, "source_rate_bps", index)
    downlink = _column(corpus, "downlink_rate_bps", index)
    capacity = _column(corpus, "buffer_capacity_bits", index)
    occupancy = _column(corpus, "initial_occupancy_bits", index)
    ramp = _column(corpus, "declared_ramp_s", index)
    restart = _column(corpus, "declared_restart_s", index)
    horizon = _column(corpus, "horizon_s", index)
    median = _column(corpus, "acquisition_median_s", index)
    log_sigma = _column(corpus, "acquisition_log_sigma", index)
    abort = _column(corpus, "acquisition_abort_probability", index)
    window = _column(corpus, "acquisition_attempt_window_s", index)

    # Memoised over the DISTINCT acquisition parameter tuples. A corpus of 1536
    # events carries only 128 of them -- one per station pair and epoch -- so the
    # exact Monte Carlo quantile is affordable here and no approximation is
    # needed.
    #
    # An earlier version used a normal approximation, mean + z * sd, to avoid a
    # Monte Carlo quantile inside a per-event loop.
    # ``validation/validate_acquisition_model.py`` measured that approximation
    # against 200 000 draws and found it OVERSTATES the 95th percentile by a
    # factor of 2.032448 at (median 6 s, log-sigma 0.25, abort 0.01, window
    # 70 s) and understates it by up to 14.2 per cent at high abort
    # probability. The cause is structural, not a tuning matter: a geometric
    # retry tail inflates the variance through the rare long attempt while
    # leaving the 95th percentile inside the body, so a symmetric
    # two-parameter approximation cannot track both. The measurement and the
    # replacement are recorded in that script's output and in README.md under
    # Limitations.
    memo: dict[tuple[float, float, float, float], tuple[float, float]] = {}
    expected = np.empty(index.size, dtype=float)
    for row in range(index.size):
        key = (
            float(median[row]),
            float(log_sigma[row]),
            float(abort[row]),
            float(window[row]),
        )
        if key not in memo:
            model = AcquisitionModel(
                median_s=key[0],
                log_sigma=key[1],
                abort_probability=key[2],
                attempt_window_s=key[3],
            )
            mean = model.moments()[0]
            upper = (
                mean
                if acquisition_quantile is None
                else model.quantile(acquisition_quantile, size=QUANTILE_DRAWS)
            )
            memo[key] = (mean, upper)
        expected[row] = memo[key][0 if acquisition_quantile is None else 1]

    headroom = capacity - occupancy
    acquisition_only = np.zeros((index.size, 3), dtype=float)
    acquisition_only[:, 0] = np.maximum(0.0, source * expected - headroom)

    # The fluid model needs a horizon that contains its own excursion; the
    # expected acquisition time can exceed the corpus horizon minus the other
    # phases for a pair with a long tail, so the horizon handed to the closed
    # form is extended for the prediction and the extension is reported by
    # validate_learned_vs_baselines.py.
    needed = expected + restart + ramp
    fluid = fluid_decomposition_array(
        source, downlink, capacity, occupancy, expected, restart, ramp,
        np.maximum(horizon, needed),
    )
    return {"acquisition_only": acquisition_only, "fluid": fluid}


def score_baselines(
    corpus: Corpus,
    split: Split,
    *,
    resamples: int = 2000,
    confidence: float = 0.95,
    acquisition_quantile: float | None = None,
) -> list[PredictorScore]:
    """Score baseline 1 and baseline 2 on ``split``'s test side.

    Called BEFORE the learned component anywhere this package benchmarks, which
    is the order the mission's build guide requires: an under-strength baseline
    inflates an apparent learned win, and the only way to know a baseline is at
    full strength is to validate it against the ground truth first.
    """
    truth = corpus.targets[split.test]
    predictions = baseline_predictions(
        corpus, split.test, acquisition_quantile=acquisition_quantile
    )
    suffix = "" if acquisition_quantile is None else f"@q{acquisition_quantile:g}"
    return [
        _score(
            f"baseline1_acquisition_only{suffix}",
            split.name,
            truth,
            predictions["acquisition_only"],
            resamples=resamples,
            confidence=confidence,
        ),
        _score(
            f"baseline2_analytic_fluid{suffix}",
            split.name,
            truth,
            predictions["fluid"],
            resamples=resamples,
            confidence=confidence,
        ),
    ]


def score_learned(
    corpus: Corpus,
    split: Split,
    predictor: HandoverLossPredictor | None = None,
    *,
    resamples: int = 2000,
    confidence: float = 0.95,
    band_confidence: float = 0.90,
) -> tuple[PredictorScore, Prediction, HandoverLossPredictor]:
    """Fit on ``split``'s train side and score on its test side.

    Returns the score, the raw :class:`Prediction` (so a caller can inspect the
    uncertainty output) and the fitted predictor.
    """
    model = HandoverLossPredictor() if predictor is None else predictor
    model.fit(corpus.features[split.train], corpus.targets[split.train])
    prediction = model.predict(corpus.features[split.test], confidence=band_confidence)
    score = _score(
        "learned_random_forest",
        split.name,
        corpus.targets[split.test],
        prediction.loss_bits,
        low_bits=prediction.low_bits,
        high_bits=prediction.high_bits,
        resamples=resamples,
        confidence=confidence,
    )
    return score, prediction, model
