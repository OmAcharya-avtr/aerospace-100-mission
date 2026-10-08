"""Learned drift classifier on windowed residual features, with a confidence output.

The learned component is a random forest over the nine features of
:mod:`twininvalidate.features`, wrapped in an isotonic probability calibration
so that its output is a usable confidence rather than an uncalibrated score.
It is deliberately small: the whole training run must finish in well under a
minute on two cores, and the compute budget is stated in the README.

How it is compared with the analytic baselines
---------------------------------------------
The calibrated probability path is treated as just another statistic path, so
its alarm threshold is set by exactly the same procedure as the baselines':
bisection to a declared in-control ARL0 target on in-control streams only, with
no out-of-control stream and no delay visible to the calibration. That is the
only comparison that means anything.

Two structural handicaps are stated rather than engineered away:

1. **Window-fill latency.** No decision exists until ``window`` samples have
   arrived, so the learned monitor's detection delay cannot be below
   ``window``. The recursive baselines have no such floor.
2. **Decision granularity.** The classifier is evaluated on every sliding
   window, i.e. once per sample once the window is full, so there is no extra
   penalty from striding; the cost of that is per-sample inference latency,
   which is measured in ``validation/validate_classifier.py``.

Expected outcome
----------------
On a mean shift in a Gaussian, unit-variance, white residual, the CUSUM is the
optimal sequential test and the windowed GLR is the standard test for an
unknown-onset shift. A classifier on window summaries cannot beat a sufficient
statistic on the model that generated the data; it can only approach it. The
honest expectation is therefore that the learned model **loses on
``parameter_step`` and may win on ``noise_variance``**, where the baselines are
mis-specified. Whatever is measured is what is published.

Test-split strategy
-------------------
Splits are **by run, never by window.** Consecutive windows of one run overlap
in ``window - 1`` of their samples, so a random window split would put almost
the same data on both sides and inflate every score. Training, probability
calibration, threshold calibration and evaluation each use a separate
``StreamSpec`` seed, so the four sets are independent noise realisations and
not merely disjoint slices of one.

Dataset limitations
-------------------
The training data is simulated from the declared twin with three injected
change kinds at declared magnitudes. There is no flight data here, and nothing
measured on this dataset transfers to an asset whose residual is not white,
unit-variance Gaussian in control. See ``DATASET_CARD.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator

from .features import DEFAULT_WINDOW, FEATURE_NAMES, N_FEATURES, window_features

DEFAULT_N_ESTIMATORS = 200
"""Declared forest size. Chosen for the compute budget, not tuned on a score."""

DEFAULT_MAX_DEPTH = 12
"""Declared maximum tree depth, which caps both fit time and inference time."""

DEFAULT_SEED = 53007
"""Declared seed for the forest and the calibrator."""


@dataclass
class DriftClassifier:
    """Isotonic-calibrated random forest over windowed residual features.

    Parameters
    ----------
    window:
        Feature window length in samples. Must match the window used at
        inference; the fitted object records it so a mismatch cannot happen
        silently.
    n_estimators:
        Trees in the forest.
    max_depth:
        Maximum tree depth.
    seed:
        Random seed for the forest.

    Notes
    -----
    ``CalibratedClassifierCV(base, cv="prefit")`` raises
    ``InvalidParameterError`` on scikit-learn 1.9.1, so the already-fitted
    forest is wrapped in :class:`sklearn.frozen.FrozenEstimator` instead, which
    is the supported route on this version. Recorded here because it is a tool
    defect, not a modelling choice.

    After fitting, the forest's ``n_jobs`` is forced to 1. On this container a
    forest with ``n_jobs > 1`` is several times *slower* at single-row
    inference than with ``n_jobs = 1``, because the thread dispatch dominates
    the tree traversal, and single-row inference is what a streaming monitor
    does.
    """

    window: int = DEFAULT_WINDOW
    n_estimators: int = DEFAULT_N_ESTIMATORS
    max_depth: int = DEFAULT_MAX_DEPTH
    seed: int = DEFAULT_SEED
    model: CalibratedClassifierCV | None = field(default=None, repr=False)
    forest: RandomForestClassifier | None = field(default=None, repr=False)
    n_train_windows: int = 0
    n_calibration_windows: int = 0

    @property
    def is_fitted(self) -> bool:
        """True once :meth:`fit` has run."""
        return self.model is not None

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_calibration: np.ndarray,
        y_calibration: np.ndarray,
    ) -> DriftClassifier:
        """Fit the forest on the training set and calibrate on a disjoint set.

        Parameters
        ----------
        x_train, x_calibration:
            Feature matrices, shape ``(n, 9)``, dimensionless.
        y_train, y_calibration:
            Labels, 0 for in-control and 1 for changed.

        Returns
        -------
        ``self``, fitted.
        """
        xt = _check_features(x_train, "x_train")
        xc = _check_features(x_calibration, "x_calibration")
        yt = _check_labels(y_train, xt.shape[0], "y_train")
        yc = _check_labels(y_calibration, xc.shape[0], "y_calibration")
        if len(np.unique(yt)) < 2:
            raise ValueError("y_train must contain both classes")
        if len(np.unique(yc)) < 2:
            raise ValueError("y_calibration must contain both classes")
        forest = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=5,
            random_state=self.seed,
            n_jobs=2,
        )
        forest.fit(xt, yt)
        forest.n_jobs = 1
        calibrated = CalibratedClassifierCV(FrozenEstimator(forest), method="isotonic")
        calibrated.fit(xc, yc)
        self.forest = forest
        self.model = calibrated
        self.n_train_windows = int(xt.shape[0])
        self.n_calibration_windows = int(xc.shape[0])
        return self

    def confidence(self, x: np.ndarray) -> np.ndarray:
        """Calibrated P(changed) for each feature row, in ``[0, 1]``.

        This is the model's uncertainty output: a per-window posterior
        probability that the twin no longer describes the asset, calibrated by
        isotonic regression on a held-out set. Its reliability is measured, not
        assumed -- see ``validation/validate_classifier.py`` for the reliability
        diagram and Brier score.
        """
        if self.model is None:
            raise RuntimeError("DriftClassifier is not fitted; call fit() first")
        xx = _check_features(x, "x")
        return np.asarray(self.model.predict_proba(xx)[:, 1], dtype=float)

    def statistic(self, z: np.ndarray) -> np.ndarray:
        """Confidence path aligned to the residual stream, for the ARL machinery.

        Returns an array of shape ``(n_runs, n_samples)`` in which the first
        ``window - 1`` entries are 0 -- no window has been filled yet, so no
        alarm is possible -- and entry ``k >= window - 1`` is the calibrated
        confidence of the window ending at sample ``k``. Feeding this to
        :func:`twininvalidate.arl.arl1_estimate` therefore charges the model for
        its window-fill latency instead of concealing it.
        """
        if self.model is None:
            raise RuntimeError("DriftClassifier is not fitted; call fit() first")
        arr = np.asarray(z, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        feats = window_features(arr, self.window)
        n_runs, n_windows, _ = feats.shape
        conf = self.confidence(feats.reshape(-1, N_FEATURES)).reshape(n_runs, n_windows)
        out = np.zeros((n_runs, arr.shape[1]))
        out[:, self.window - 1 :] = conf
        return out

    def feature_importance(self) -> dict[str, float]:
        """Forest impurity-based importances by feature name, summing to 1.

        Impurity importance is biased towards high-cardinality features and is
        reported only as a description of the fitted forest, not as evidence
        about the physics.
        """
        if self.forest is None:
            raise RuntimeError("DriftClassifier is not fitted; call fit() first")
        values = (float(v) for v in self.forest.feature_importances_)
        return dict(zip(FEATURE_NAMES, values, strict=True))


def _check_features(x: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != N_FEATURES:
        raise ValueError(f"{name} must have shape (n, {N_FEATURES}), got {arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} is empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite; found NaN or inf")
    return arr


def _check_labels(y: np.ndarray, n: int, name: str) -> np.ndarray:
    arr = np.asarray(y)
    if arr.ndim != 1 or arr.shape[0] != n:
        raise ValueError(f"{name} must have shape ({n},), got {arr.shape}")
    uniq = set(np.unique(arr).tolist())
    if not uniq <= {0, 1}:
        raise ValueError(f"{name} must contain only 0 and 1, got values {sorted(uniq)}")
    return arr.astype(int)


def brier_score(confidence: np.ndarray, label: np.ndarray) -> float:
    """Mean squared error of a probabilistic forecast.

    ``mean((p_i - y_i)^2)``. Lower is better; 0.25 is the score of the
    uninformative constant forecast 0.5 on a balanced set. Source: Brier,
    G. W. (1950), "Verification of forecasts expressed in terms of
    probability", *Monthly Weather Review* 78(1). No page number is quoted
    because none was verified in this environment.
    """
    p = np.asarray(confidence, dtype=float)
    y = np.asarray(label, dtype=float)
    if p.shape != y.shape:
        raise ValueError(f"shapes must match, got {p.shape} and {y.shape}")
    if p.size == 0:
        raise ValueError("cannot score an empty forecast")
    return float(np.mean((p - y) ** 2))


@dataclass(frozen=True)
class ReliabilityBin:
    """One bin of a reliability diagram."""

    lower: float
    upper: float
    count: int
    mean_confidence: float
    observed_frequency: float


def reliability_diagram(
    confidence: np.ndarray, label: np.ndarray, n_bins: int = 10
) -> list[ReliabilityBin]:
    """Equal-width reliability bins of a probabilistic forecast.

    A perfectly calibrated forecast has ``observed_frequency ==
    mean_confidence`` in every populated bin. Empty bins are omitted rather
    than reported as zeros.
    """
    p = np.asarray(confidence, dtype=float)
    y = np.asarray(label, dtype=float)
    if p.shape != y.shape:
        raise ValueError(f"shapes must match, got {p.shape} and {y.shape}")
    if n_bins < 2:
        raise ValueError(f"n_bins must be at least 2, got {n_bins}")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[ReliabilityBin] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        count = int(mask.sum())
        if count == 0:
            continue
        bins.append(
            ReliabilityBin(
                lower=float(lo),
                upper=float(hi),
                count=count,
                mean_confidence=float(p[mask].mean()),
                observed_frequency=float(y[mask].mean()),
            )
        )
    return bins


def expected_calibration_error(bins: list[ReliabilityBin]) -> float:
    """Count-weighted mean absolute gap between confidence and frequency.

    ``sum_b (n_b / N) |conf_b - freq_b|``. Dimensionless, in ``[0, 1]``.
    """
    if not bins:
        raise ValueError("no populated bins")
    total = sum(b.count for b in bins)
    return float(
        sum(b.count * abs(b.mean_confidence - b.observed_frequency) for b in bins) / total
    )
