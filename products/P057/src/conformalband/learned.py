"""The learned components. Both exist only because a baseline was measured first.

Two of them, each answering a question the analytic baseline cannot:

1. :class:`LearnedRegressor` is a gradient-boosted regression tree on the five
   covariates. It is free of the propulsive-efficiency misspecification that
   :class:`conformalband.baseline.PhysicsRegressor` carries, and it has no
   physics at all, so it has nothing to extrapolate with. Which of those two
   facts dominates is an empirical question and is answered in
   ``validation/validate_baseline_vs_learned.py``.

2. :class:`LearnedWeightEstimator` estimates the likelihood ratio
   ``w(x) = dP_test/dP_cal`` from unlabelled calibration and test covariates
   by logistic regression, the classifier reduction of density-ratio
   estimation (Sugiyama, M., Suzuki, T. and Kanamori, T., *Density Ratio
   Estimation in Machine Learning*, Cambridge University Press, 2012,
   chapter 4). It exists because the exact ratio in
   :mod:`conformalband.shift` is available only because this package declared
   the shift; a real deployment has to estimate it, and the audit measures
   what that estimate costs.

Neither has a native uncertainty output. Both are used only wrapped in a
conformal or parametric band, which is where the uncertainty comes from; see
``MODEL_CARD.md``.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

DEFAULT_N_ESTIMATORS = 200
DEFAULT_MAX_DEPTH = 3
DEFAULT_LEARNING_RATE = 0.05


class LearnedRegressor:
    """Gradient-boosted trees on the five covariates.

    Parameters
    ----------
    n_estimators:
        Number of boosting stages.
    max_depth:
        Maximum tree depth.
    learning_rate:
        Shrinkage applied to each stage.
    random_state:
        Seed for the estimator; the fit is deterministic given the data.
    """

    def __init__(
        self,
        *,
        n_estimators: int = DEFAULT_N_ESTIMATORS,
        max_depth: int = DEFAULT_MAX_DEPTH,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        random_state: int = 0,
    ) -> None:
        if n_estimators <= 0:
            raise ValueError(f"n_estimators must be > 0, got {n_estimators}")
        if max_depth <= 0:
            raise ValueError(f"max_depth must be > 0, got {max_depth}")
        if learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be > 0, got {learning_rate}")
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
        )
        self._fitted = False

    def fit(self, features: np.ndarray, energy: np.ndarray) -> LearnedRegressor:
        """Fit on ``(n, 5)`` features and ``(n,)`` energy [Wh]."""
        x = np.asarray(features, dtype=float)
        y = np.asarray(energy, dtype=float).ravel()
        if x.ndim != 2 or x.shape[1] != 5:
            raise ValueError(f"features must have shape (n, 5), got {x.shape}")
        if y.shape != (x.shape[0],):
            raise ValueError("energy must have shape (n,) matching features")
        self.model.fit(x, y)
        self._fitted = True
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predicted energy per leg [Wh]."""
        if not self._fitted:
            raise RuntimeError("call fit(...) before predict(...)")
        x = np.asarray(features, dtype=float)
        if x.ndim != 2 or x.shape[1] != 5:
            raise ValueError(f"features must have shape (n, 5), got {x.shape}")
        return np.asarray(self.model.predict(x), dtype=float).ravel()

    @property
    def feature_importances(self) -> np.ndarray:
        """Impurity-based importances, one per covariate [-], summing to 1."""
        if not self._fitted:
            raise RuntimeError("call fit(...) first")
        return np.asarray(self.model.feature_importances_, dtype=float)


class LearnedWeightEstimator:
    """Classifier-based estimate of the covariate-shift likelihood ratio.

    Fits a logistic regression to separate calibration covariates (label 0)
    from test covariates (label 1) on standardised inputs, then converts the
    posterior odds into a density ratio,

        w(x) = [p(x) / (1 - p(x))] * (n_cal / n_test),

    which is the classifier reduction of density-ratio estimation. The
    ``n_cal / n_test`` factor removes the class prior, so a constant shift of
    zero returns weights near 1 rather than near ``n_test / n_cal``.

    Parameters
    ----------
    columns:
        Indices of the covariates the shift is declared on. Restricting the
        classifier to these is the honest version of "we know which variables
        moved"; passing ``None`` uses all columns and is measurably worse at
        small calibration sizes.
    regularisation:
        Inverse L2 strength ``C`` of the logistic regression.
    clip:
        Weights are clipped to ``[1 / clip, clip]`` so one point cannot take
        all the mass. The clip is reported, not hidden: see
        :attr:`clipped_fraction`.
    """

    def __init__(
        self,
        *,
        columns: tuple[int, ...] | None = (1, 3),
        regularisation: float = 1.0,
        clip: float = 100.0,
    ) -> None:
        if regularisation <= 0.0:
            raise ValueError(f"regularisation must be > 0, got {regularisation}")
        if clip <= 1.0:
            raise ValueError(f"clip must be > 1, got {clip}")
        self.columns = columns
        self.regularisation = float(regularisation)
        self.clip = float(clip)
        self._scaler: StandardScaler | None = None
        self._model: LogisticRegression | None = None
        self._ratio: float | None = None
        self._auc: float | None = None
        self._clipped_fraction = 0.0

    def _select(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=float)
        if x.ndim != 2:
            raise ValueError(f"features must be 2-D, got shape {x.shape}")
        return x if self.columns is None else x[:, list(self.columns)]

    def fit(
        self, calibration_features: np.ndarray, test_features: np.ndarray
    ) -> LearnedWeightEstimator:
        """Fit on unlabelled covariates from both samples. No targets are used."""
        a = self._select(calibration_features)
        b = self._select(test_features)
        if a.shape[1] != b.shape[1]:
            raise ValueError("calibration and test features must have the same width")
        if a.shape[0] < 2 or b.shape[0] < 2:
            raise ValueError("need at least 2 rows in each sample")
        x = np.vstack([a, b])
        y = np.concatenate([np.zeros(a.shape[0], dtype=int), np.ones(b.shape[0], dtype=int)])
        self._scaler = StandardScaler().fit(x)
        self._model = LogisticRegression(C=self.regularisation, max_iter=2000).fit(
            self._scaler.transform(x), y
        )
        self._ratio = a.shape[0] / b.shape[0]
        posterior = self._model.predict_proba(self._scaler.transform(x))[:, 1]
        self._auc = float(roc_auc_score(y, posterior))
        return self

    @property
    def auc(self) -> float:
        """In-sample ROC AUC of the calibration-versus-test classifier [-].

        0.5 means the two samples are indistinguishable, which is what an
        unshifted pair should look like. It is the confidence output of this
        model: a value near 0.5 says the learned weights carry no information,
        not that the shift is absent.
        """
        if self._auc is None:
            raise RuntimeError("call fit(...) first")
        return self._auc

    @property
    def clipped_fraction(self) -> float:
        """Fraction of the most recent :meth:`weights` call that hit the clip [-]."""
        return self._clipped_fraction

    def weights(self, features: np.ndarray) -> np.ndarray:
        """Estimated likelihood ratio ``w(x) > 0`` [-]."""
        if self._model is None or self._scaler is None or self._ratio is None:
            raise RuntimeError("call fit(...) first")
        z = self._scaler.transform(self._select(features))
        probability = self._model.predict_proba(z)[:, 1]
        probability = np.clip(probability, 1e-12, 1.0 - 1e-12)
        raw = probability / (1.0 - probability) * self._ratio
        clipped = np.clip(raw, 1.0 / self.clip, self.clip)
        self._clipped_fraction = float(np.mean(clipped != raw))
        return clipped
