"""The learned estimator, and the trivial baseline it must beat to be worth anything.

The learned component maps the twenty-four motion features of
:mod:`aligncal.features` to an alignment correction, for the **short-record**
case where the unbiased batch solution of :mod:`aligncal.batch` has a
Cramer-Rao standard deviation comparable to the spread of the misalignment
population itself. In that regime a biased estimator that shrinks toward the
population it was trained on can have a smaller mean squared error than the
unbiased solution, and whether it actually does is measured, not assumed, in
``validation/validate_learned.py``.

Every configuration reports an uncertainty, not only a point estimate: the
per-member standard deviation of the ensemble, per output component and in any
requested direction. That is what makes the headline measurement of this
product possible. An estimator with no uncertainty output cannot be shown to be
*confidently* wrong.

Declared configurations
-----------------------
``forest``
    ``RandomForestRegressor``, 200 trees, ``min_samples_leaf=2``,
    ``max_features=0.5``. Uncertainty from the spread of the 200 tree
    predictions.
``forest_deep``
    The same with ``min_samples_leaf=1`` and all features at every split, so
    the two differ only in how much they are allowed to overfit.
``bagged_ridge``
    Forty ridge regressions on bootstrap resamples, uncertainty from the spread
    of the forty. Linear in the features, so it can represent the shrinkage map
    exactly, which makes it the strongest learned configuration a priori and
    the one to beat.

Three configurations, no hyperparameter search. Stated rather than implied: the
numbers in this repository are not the best achievable by a learned estimator
on this problem, and the comparison is against the configurations declared
here.

``TrivialEstimator``
    Predicts the training-set mean misalignment for every record, with the
    training-set standard deviation as its uncertainty. This is the regression
    analogue of a majority-class guess. Because the generated misalignment
    direction is uniform on the sphere, the training mean is within rounding of
    zero, so this estimator is approximately "no misalignment at all". A
    learned estimator that does not beat it has learned nothing, and reporting
    scores without it would let that pass unnoticed.

Reference: Pedregosa, F. et al. (2011), Scikit-learn: machine learning in
Python, *Journal of Machine Learning Research* 12, 2825-2830, for
``RandomForestRegressor`` and ``Ridge``. The shrinkage argument is the
James-Stein / ridge bias-variance trade-off: James, W. and Stein, C. (1961),
Estimation with quadratic loss, *Proceedings of the Fourth Berkeley Symposium
on Mathematical Statistics and Probability* 1, 361-379.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

__all__ = [
    "CONFIGURATIONS",
    "LearnedEstimator",
    "Prediction",
    "TrivialEstimator",
]

#: The declared configurations. No search was run over them.
CONFIGURATIONS: dict[str, dict[str, Any]] = {
    "forest": {
        "kind": "forest",
        "n_estimators": 200,
        "min_samples_leaf": 2,
        "max_features": 0.5,
    },
    "forest_deep": {
        "kind": "forest",
        "n_estimators": 200,
        "min_samples_leaf": 1,
        "max_features": None,
    },
    "bagged_ridge": {
        "kind": "bagged_ridge",
        "n_estimators": 40,
        "alpha": 1.0e-3,
    },
}


@dataclass(frozen=True)
class Prediction:
    """A learned estimate with its uncertainty.

    Attributes
    ----------
    theta_rad:
        Shape ``(m, 3)``, radians.
    sigma_rad:
        Shape ``(m, 3)``, radians: the per-component standard deviation across
        ensemble members. This is a **model-spread** uncertainty, not a
        calibrated posterior standard deviation, and it is labelled as such
        everywhere it is reported.
    members:
        Shape ``(k, m, 3)``, the individual member predictions in radians, kept
        so that an uncertainty can be formed in any direction after the fact.
        This is what makes the unobservable-direction measurement possible.
    """

    theta_rad: np.ndarray
    sigma_rad: np.ndarray
    members: np.ndarray

    def sigma_in_direction(self, direction: np.ndarray) -> np.ndarray:
        """Ensemble standard deviation along ``direction``, radians, shape ``(m,)``.

        ``direction`` is a single ``(3,)`` unit vector or an ``(m, 3)`` block of
        them, one per row.
        """
        unit = np.atleast_2d(np.asarray(direction, dtype=float))
        if unit.shape[-1] != 3:
            raise ValueError(f"sigma_in_direction() needs 3 components, got {unit.shape}")
        norms = np.linalg.norm(unit, axis=1, keepdims=True)
        if np.any(norms == 0.0):
            raise ValueError("sigma_in_direction() needs non-zero directions")
        unit = unit / norms
        if unit.shape[0] == 1:
            unit = np.repeat(unit, self.theta_rad.shape[0], axis=0)
        if unit.shape[0] != self.theta_rad.shape[0]:
            raise ValueError(
                f"sigma_in_direction() got {unit.shape[0]} directions for "
                f"{self.theta_rad.shape[0]} predictions"
            )
        projected = np.einsum("kmc,mc->km", self.members, unit)
        return np.std(projected, axis=0, ddof=0)

    def component_in_direction(self, direction: np.ndarray) -> np.ndarray:
        """Point estimate along ``direction``, radians, shape ``(m,)``."""
        unit = np.atleast_2d(np.asarray(direction, dtype=float))
        norms = np.linalg.norm(unit, axis=1, keepdims=True)
        if np.any(norms == 0.0):
            raise ValueError("component_in_direction() needs non-zero directions")
        unit = unit / norms
        if unit.shape[0] == 1:
            unit = np.repeat(unit, self.theta_rad.shape[0], axis=0)
        return np.einsum("mc,mc->m", self.theta_rad, unit)


class LearnedEstimator:
    """An ensemble regressor from motion features to an alignment correction.

    Parameters
    ----------
    kind:
        ``"forest"`` or ``"bagged_ridge"``.
    seed:
        Integer seed. Fixed by the caller so every reported number reproduces.
    **parameters:
        Passed to the underlying estimator. See :data:`CONFIGURATIONS`.
    """

    def __init__(self, *, kind: str = "forest", seed: int = 20261029, **parameters: Any) -> None:
        if kind not in {"forest", "bagged_ridge"}:
            raise ValueError(f"unknown kind {kind!r}; use 'forest' or 'bagged_ridge'")
        self.kind = kind
        self.seed = int(seed)
        self.parameters = dict(parameters)
        self._members: list[Any] = []
        self._n_features = 0

    def fit(self, features: np.ndarray, targets: np.ndarray) -> LearnedEstimator:
        """Fit on ``features`` ``(m, 24)`` and ``targets`` ``(m, 3)`` in radians."""
        matrix = np.asarray(features, dtype=float)
        truth = np.asarray(targets, dtype=float)
        if matrix.ndim != 2 or truth.ndim != 2 or truth.shape[1] != 3:
            raise ValueError(
                f"fit() needs features (m, k) and targets (m, 3), got {matrix.shape} "
                f"and {truth.shape}"
            )
        if matrix.shape[0] != truth.shape[0]:
            raise ValueError(
                f"fit() got {matrix.shape[0]} feature rows and {truth.shape[0]} targets"
            )
        if matrix.shape[0] < 4:
            raise ValueError(
                f"fit() needs at least 4 records; an ensemble spread over {matrix.shape[0]} "
                "is not an uncertainty"
            )
        if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(truth)):
            raise ValueError("fit() needs finite features and targets")
        self._n_features = int(matrix.shape[1])
        self._members = []
        if self.kind == "forest":
            forest = RandomForestRegressor(
                random_state=self.seed,
                n_jobs=1,
                **{k: v for k, v in self.parameters.items() if k != "kind"},
            )
            forest.fit(matrix, truth)
            self._members = list(forest.estimators_)
            self._forest = forest
        else:
            count = int(self.parameters.get("n_estimators", 40))
            alpha = float(self.parameters.get("alpha", 1.0e-3))
            rng = np.random.default_rng(self.seed)
            rows = matrix.shape[0]
            for _ in range(count):
                index = rng.integers(0, rows, size=rows)
                model = Ridge(alpha=alpha)
                model.fit(matrix[index], truth[index])
                self._members.append(model)
        return self

    def predict(self, features: np.ndarray) -> Prediction:
        """Predict alignment corrections and their ensemble spread.

        Returns a :class:`Prediction` whose ``theta_rad`` is the ensemble mean
        in radians. The mean of the members is used rather than the forest's
        own ``predict`` so that the point estimate and the spread describe the
        same ensemble; for a ``RandomForestRegressor`` the two agree exactly in
        exact arithmetic and to rounding in practice.
        """
        if not self._members:
            raise RuntimeError("predict() called before fit()")
        matrix = np.atleast_2d(np.asarray(features, dtype=float))
        if matrix.shape[1] != self._n_features:
            raise ValueError(
                f"predict() got {matrix.shape[1]} features, fitted on {self._n_features}"
            )
        if not np.all(np.isfinite(matrix)):
            raise ValueError("predict() needs finite features")
        members = np.stack([np.asarray(m.predict(matrix), dtype=float) for m in self._members])
        if members.ndim == 2:  # pragma: no cover - a single-output member
            members = members[:, :, None]
        return Prediction(
            theta_rad=members.mean(axis=0),
            sigma_rad=members.std(axis=0, ddof=0),
            members=members,
        )


class TrivialEstimator:
    """Predicts the training mean for every record. The baseline a model must beat.

    The regression analogue of a majority-class guess. Its uncertainty is the
    training-set standard deviation, which is the honest spread of a predictor
    that uses no input at all. Because the generated misalignment direction is
    uniform on the sphere, the training mean is within rounding of zero, so
    this estimator says "the declared mounting is right" on every record.
    """

    def __init__(self) -> None:
        self._mean = np.zeros(3)
        self._sigma = np.zeros(3)
        self._samples = np.zeros((0, 3))
        self._fitted = False

    def fit(self, features: np.ndarray, targets: np.ndarray) -> TrivialEstimator:
        """Record the training mean and standard deviation. ``features`` is ignored."""
        truth = np.asarray(targets, dtype=float)
        if truth.ndim != 2 or truth.shape[1] != 3:
            raise ValueError(f"fit() needs targets (m, 3), got {truth.shape}")
        if truth.shape[0] < 2:
            raise ValueError("fit() needs at least 2 targets to form a standard deviation")
        self._mean = truth.mean(axis=0)
        self._sigma = truth.std(axis=0, ddof=1)
        self._samples = truth
        self._fitted = True
        return self

    @property
    def mean_rad(self) -> np.ndarray:
        """The training mean misalignment, radians."""
        return self._mean.copy()

    def predict(self, features: np.ndarray) -> Prediction:
        """Return the training mean for every row, with the training spread as sigma.

        The ``members`` block is the training targets themselves, recentred on
        the mean and broadcast, so that a spread in an arbitrary direction is
        the training spread in that direction. That is the correct uncertainty
        for a predictor that ignores its input.
        """
        if not self._fitted:
            raise RuntimeError("predict() called before fit()")
        matrix = np.atleast_2d(np.asarray(features, dtype=float))
        rows = matrix.shape[0]
        theta = np.repeat(self._mean[None, :], rows, axis=0)
        sigma = np.repeat(self._sigma[None, :], rows, axis=0)
        members = np.repeat(self._samples[:, None, :], rows, axis=1)
        return Prediction(theta_rad=theta, sigma_rad=sigma, members=members)
