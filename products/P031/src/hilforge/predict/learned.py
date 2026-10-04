"""The learned deadline-overrun predictor, with a calibrated probability output.

Architecture
------------
``sklearn.ensemble.HistGradientBoostingClassifier`` on the 18 causal features
of :mod:`hilforge.predict.features`. Histogram-based gradient boosting is
chosen because it is the strongest tabular learner available in this
environment — PyTorch is not installed and a small MLP on 18 features would
not add anything a boosted ensemble cannot do — and because it trains in a few
seconds on one CPU core, which the compute budget requires.

Calibration and the uncertainty output
--------------------------------------
The training part is split **chronologically** into a fit slice and a
calibration slice; a random split would leak the autocorrelation the model is
supposed to exploit. The classifier is fitted on the fit slice, then an
isotonic regression maps its raw probability onto a calibrated one using the
calibration slice (Zadrozny & Elkan, "Transforming Classifier Scores into
Accurate Multiclass Probability Estimates", *KDD '02*, pp. 694-699, which
introduced isotonic calibration for this purpose; the underlying monotone
regression is the pool-adjacent-violators algorithm of Ayer et al., "An
Empirical Distribution Function for Sampling with Incomplete Information",
*Annals of Mathematical Statistics* 26(4):641-647, 1955).

The calibrated probability **is** the uncertainty output. It is reported with
a Brier score and a reliability table on held-out data, so a reader can see
whether it means anything rather than taking the word "probability" on trust.

The decision cut-off is chosen on the calibration slice by maximising F1, the
same rule used for both baselines, so the three predictors are compared at
operating points chosen the same way.
"""

from __future__ import annotations

import itertools

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression

from ..errors import ConfigurationError
from .features import FEATURE_NAMES
from .metrics import cutoff_for_flag_rate

__all__ = ["LearnedOverrunPredictor"]


def _f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = float(np.sum(y_true & y_pred))
    fp = float(np.sum(~y_true & y_pred))
    fn = float(np.sum(y_true & ~y_pred))
    denom = 2.0 * tp + fp + fn
    return 2.0 * tp / denom if denom > 0.0 else 0.0


class LearnedOverrunPredictor:
    """Boosted-tree overrun predictor with an isotonically calibrated output.

    Parameters
    ----------
    max_iter:
        Boosting iterations, >= 1. Default 160, chosen for the compute budget.
    max_depth:
        Tree depth, >= 1.
    learning_rate:
        Shrinkage, > 0.
    min_samples_leaf:
        Minimum samples in a leaf, >= 1.
    l2_regularization:
        L2 penalty on leaf values, >= 0.
    calibration_fraction:
        Trailing fraction of the training set reserved for calibration and
        cut-off selection, in ``(0, 0.5]``.
    seed:
        ``random_state`` of the classifier.

    Attributes
    ----------
    cutoff_:
        Decision threshold on the calibrated probability [-].
    train_time_s:
        Wall-clock fit time [s], recorded for the compute budget.
    """

    name = "learned_hgb"

    def __init__(
        self,
        *,
        max_iter: int = 160,
        max_depth: int = 4,
        learning_rate: float = 0.08,
        min_samples_leaf: int = 40,
        l2_regularization: float = 1.0,
        calibration_fraction: float = 0.25,
        seed: int = 20261004,
    ) -> None:
        for name, value in (
            ("max_iter", max_iter),
            ("max_depth", max_depth),
            ("min_samples_leaf", min_samples_leaf),
        ):
            if value < 1:
                raise ConfigurationError(f"{name} must be >= 1, got {value!r}")
        if not (learning_rate > 0.0):
            raise ConfigurationError(f"learning_rate must be > 0, got {learning_rate!r}")
        if l2_regularization < 0.0:
            raise ConfigurationError(
                f"l2_regularization must be >= 0, got {l2_regularization!r}"
            )
        if not (0.0 < calibration_fraction <= 0.5):
            raise ConfigurationError(
                f"calibration_fraction must be in (0, 0.5], got {calibration_fraction!r}"
            )
        self.params = {
            "max_iter": int(max_iter),
            "max_depth": int(max_depth),
            "learning_rate": float(learning_rate),
            "min_samples_leaf": int(min_samples_leaf),
            "l2_regularization": float(l2_regularization),
            "random_state": int(seed),
            "early_stopping": False,
        }
        self.calibration_fraction = float(calibration_fraction)
        self.seed = int(seed)
        self._clf: HistGradientBoostingClassifier | None = None
        self._iso: IsotonicRegression | None = None
        self.cutoff_ = 0.5
        self.train_time_s = float("nan")
        self.n_fit_ = 0
        self.n_calib_ = 0
        self.fitted_ = False

    def fit(self, x: np.ndarray, y: np.ndarray) -> LearnedOverrunPredictor:
        """Fit on a chronologically ordered training set.

        The last ``calibration_fraction`` of the rows are held back for
        isotonic calibration and cut-off selection. ``x`` must be ordered in
        time; shuffling it before calling this invalidates the split.
        """
        import time

        xs = np.asarray(x, dtype=np.float64)
        ys = np.asarray(y, dtype=bool).ravel()
        if xs.ndim != 2 or xs.shape[1] != len(FEATURE_NAMES):
            raise ConfigurationError(
                f"x must have shape (m, {len(FEATURE_NAMES)}), got {xs.shape}"
            )
        if xs.shape[0] != ys.size:
            raise ConfigurationError(f"x {xs.shape} and y {ys.shape} do not line up")
        cut = round(xs.shape[0] * (1.0 - self.calibration_fraction))
        if cut < 20 or xs.shape[0] - cut < 20:
            raise ConfigurationError(
                f"training set of {xs.shape[0]} rows is too small to split into "
                f"{cut} fit and {xs.shape[0] - cut} calibration rows"
            )
        if ys[:cut].all() or not ys[:cut].any():
            raise ConfigurationError(
                "the fit slice contains a single class; the trace has no usable "
                "overrun variation"
            )
        t0 = time.perf_counter()
        clf = HistGradientBoostingClassifier(**self.params)
        clf.fit(xs[:cut], ys[:cut])
        raw = clf.predict_proba(xs[cut:])[:, 1]
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(raw, ys[cut:].astype(np.float64))
        cal = np.clip(iso.predict(raw), 0.0, 1.0)
        cands = np.unique(np.quantile(cal, np.linspace(0.001, 0.999, 199)))
        best = (-1.0, 0.5)
        for c in cands:
            f1 = _f1(ys[cut:], cal > c)
            if f1 > best[0]:
                best = (f1, float(c))
        self.train_time_s = time.perf_counter() - t0
        self._clf = clf
        self._iso = iso
        self.cutoff_ = best[1]
        self.n_fit_ = cut
        self.n_calib_ = int(xs.shape[0] - cut)
        self.fitted_ = True
        return self

    def _require_fit(self) -> None:
        if not self.fitted_ or self._clf is None or self._iso is None:
            raise RuntimeError("LearnedOverrunPredictor.fit must be called first")

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Calibrated ``P(overrun within horizon)`` per row [-]."""
        self._require_fit()
        assert self._clf is not None and self._iso is not None
        raw = self._clf.predict_proba(np.asarray(x, dtype=np.float64))[:, 1]
        return np.clip(self._iso.predict(raw), 0.0, 1.0)

    def score(self, x: np.ndarray) -> np.ndarray:
        """Raw classifier probability, used as the ranking score [-].

        The raw score is used rather than the calibrated one because isotonic
        calibration is a step function and ties large groups of rows, which
        depresses AUC for a reason that has nothing to do with ranking
        quality.
        """
        self._require_fit()
        assert self._clf is not None
        return self._clf.predict_proba(np.asarray(x, dtype=np.float64))[:, 1]

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Boolean flag per row, at the fitted cut-off."""
        return self.predict_proba(x) > self.cutoff_

    def calibrate_flag_rate(self, x: np.ndarray, target_rate: float) -> float:
        """Move the cut-off to hit ``target_rate`` on ``x``. Returns it [-].

        See :meth:`hilforge.predict.baselines.FixedThresholdPredictor.calibrate_flag_rate`
        for why a matched flag rate is the operating point worth comparing at.
        """
        if not (0.0 < target_rate < 1.0):
            raise ConfigurationError(
                f"target_rate must be in (0, 1), got {target_rate!r}"
            )
        self.cutoff_ = cutoff_for_flag_rate(self.predict_proba(x), target_rate)
        return self.cutoff_

    def reliability_table(
        self, x: np.ndarray, y: np.ndarray, *, n_bins: int = 10
    ) -> np.ndarray:
        """Calibration table on held-out data.

        Returns an ``(n_bins, 4)`` array of
        ``[bin_lower, bin_upper, mean_predicted, observed_frequency]``, with
        ``nan`` in empty bins. A well-calibrated output has the third and
        fourth columns close; the gap is the thing that falsifies the
        probability claim.
        """
        self._require_fit()
        if n_bins < 2:
            raise ConfigurationError(f"n_bins must be >= 2, got {n_bins!r}")
        p = self.predict_proba(x)
        yt = np.asarray(y, dtype=bool).ravel()
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        rows = []
        for lo, hi in itertools.pairwise(edges):
            mask = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= hi)
            if np.any(mask):
                rows.append([lo, hi, float(p[mask].mean()), float(yt[mask].mean())])
            else:
                rows.append([lo, hi, np.nan, np.nan])
        return np.asarray(rows, dtype=np.float64)

    def feature_importance_permutation(
        self, x: np.ndarray, y: np.ndarray, *, n_repeats: int = 3, seed: int = 0
    ) -> np.ndarray:
        """Permutation importance of each feature, as a Brier-score increase.

        Returns an array of length 18 aligned with
        :data:`hilforge.predict.features.FEATURE_NAMES`. Permutation
        importance is used rather than a split-count importance because split
        counts are biased toward high-cardinality features (Breiman,
        *Random Forests*, Machine Learning 45(1):5-32, 2001, §10, where the
        permutation measure is introduced).
        """
        self._require_fit()
        if n_repeats < 1:
            raise ConfigurationError(f"n_repeats must be >= 1, got {n_repeats!r}")
        xs = np.asarray(x, dtype=np.float64)
        yt = np.asarray(y, dtype=bool).ravel().astype(np.float64)
        base = float(np.mean((self.predict_proba(xs) - yt) ** 2))
        rng = np.random.Generator(np.random.PCG64(int(seed)))
        out = np.zeros(xs.shape[1], dtype=np.float64)
        for j in range(xs.shape[1]):
            acc = 0.0
            for _ in range(n_repeats):
                shuffled = xs.copy()
                rng.shuffle(shuffled[:, j])
                acc += float(np.mean((self.predict_proba(shuffled) - yt) ** 2))
            out[j] = acc / n_repeats - base
        return out

    def describe(self) -> str:
        """Multi-line description of the fitted model."""
        self._require_fit()
        return (
            f"{self.name}: HistGradientBoostingClassifier"
            f"(max_iter={self.params['max_iter']}, max_depth={self.params['max_depth']}, "
            f"learning_rate={self.params['learning_rate']}, "
            f"min_samples_leaf={self.params['min_samples_leaf']}, "
            f"l2={self.params['l2_regularization']}, random_state={self.seed})\n"
            f"  isotonic calibration on the trailing "
            f"{self.calibration_fraction:.2f} of training rows\n"
            f"  fit rows {self.n_fit_}, calibration rows {self.n_calib_}, "
            f"cut-off {self.cutoff_:.4f}, fit time {self.train_time_s:.3f} s"
        )
