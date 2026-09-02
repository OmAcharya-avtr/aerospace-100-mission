"""Learned warm start for the constrained slew planner.

This is the AI element of the package. It exists to be measured against the
cold-started optimiser, not to replace it: the planner in
:mod:`slewforge.planner` was written, tested and validated first, and the
model's training labels are its output.

What it predicts
----------------
The via parameter vector ``p`` (3 numbers, radians, in the problem's canonical
frame) that the cold multi-start planner arrived at. Feeding that vector back
in as the single SLSQP starting point is the "warm start"; the alternative is
the deterministic sweep of seven starting points that the cold planner runs.
A warm start can therefore win on **solve time** (one start instead of seven)
and lose on **solution quality** (one local optimum instead of the best of
seven). ``validation/validate_warm_start.py`` measures both and
``MODEL_CARD.md`` records the answer, including where it is a loss.

Architecture
------------
A single ``sklearn.ensemble.ExtraTreesRegressor`` with multi-output targets.
Extremely randomised trees were chosen over a neural network because the
labelled dataset is small -- a few hundred problems, because every label costs
a full cold solve -- and because the per-tree predictions give an ensemble
spread for free. PyTorch is not available in the target environment and would
not have helped at this dataset size.

Confidence
----------
:meth:`LearnedWarmStart.predict` returns, besides the parameter vector:

* ``spread``: the standard deviation across trees, per component [rad];
* ``confidence``: ``exp(-mean spread / reference_spread)`` in ``[0, 1]``,
  where ``reference_spread`` is the mean out-of-bag-style spread on the
  training set, so 1 means the trees agree as well as they did in training;
* ``extrapolating``: ``True`` when any feature falls outside the training
  range, which is a statement about the input, not about the prediction.

The confidence is an ensemble-agreement statistic, **not** a calibrated
probability that the warm start will work. ``MODEL_CARD.md`` sec. 8 reports
what it does and does not correlate with.

The planner never acts on the confidence. It is returned to the caller and
recorded on the result, so a badly-calibrated confidence cannot change what
the planner does.

This model is not certified for operational flight use.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.ensemble import ExtraTreesRegressor

from .dataset import problem_features
from .planner import SlewProblem

__all__ = ["LearnedWarmStart", "WarmStartPrediction"]


@dataclass(frozen=True)
class WarmStartPrediction:
    """Batch output of :meth:`LearnedWarmStart.predict`.

    Attributes
    ----------
    params : ndarray
        ``(n, 3)`` predicted via parameter vectors [rad], canonical frame.
    spread : ndarray
        ``(n, 3)`` per-component standard deviation across the trees [rad].
    confidence : ndarray
        ``(n,)`` in ``[0, 1]``; see the module docstring.
    extrapolating : ndarray
        ``(n,)`` booleans: at least one feature outside the training range.
    """

    params: NDArray[np.float64]
    spread: NDArray[np.float64]
    confidence: NDArray[np.float64]
    extrapolating: NDArray[np.bool_]

    def __len__(self) -> int:
        return int(self.params.shape[0])


class LearnedWarmStart:
    """Extremely randomised trees predicting a via parameterisation.

    Parameters
    ----------
    n_estimators : int
        Trees, ``>= 2`` so a spread exists.
    max_depth : int or None
        Passed to scikit-learn.
    min_samples_leaf : int
        Passed to scikit-learn; the default 2 is deliberately conservative for
        a few-hundred-sample dataset.
    random_state : int
        Seed; the fit is deterministic given the data and this seed.

    Notes
    -----
    The model is tied to the feature definition in
    :func:`slewforge.dataset.problem_features` and, through the training
    labels, to the planner settings used to produce them (one via point,
    ``start_magnitudes = (0.45,)``, ``maxiter = 40``). Applying it to problems
    generated under a different distribution is extrapolation and is flagged,
    not prevented.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int | None = None,
        min_samples_leaf: int = 2,
        random_state: int = 0,
    ) -> None:
        if n_estimators < 2:
            raise ValueError(f"n_estimators must be >= 2 for a spread, got {n_estimators}")
        if min_samples_leaf < 1:
            raise ValueError(f"min_samples_leaf must be >= 1, got {min_samples_leaf}")
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.min_samples_leaf = int(min_samples_leaf)
        self.random_state = int(random_state)
        self._model: ExtraTreesRegressor | None = None
        self._lo: NDArray[np.float64] | None = None
        self._hi: NDArray[np.float64] | None = None
        self._reference_spread: float = 1.0

    @property
    def fitted(self) -> bool:
        """``True`` once :meth:`fit` has run."""
        return self._model is not None

    def fit(self, features: ArrayLike, targets: ArrayLike) -> LearnedWarmStart:
        """Train on ``(n, 28)`` features and ``(n, 3)`` via parameters.

        Returns ``self``. Raises ``ValueError`` below 20 samples: the spread
        statistic is meaningless on fewer and a silent fit would be worse than
        a refusal.
        """
        x = np.atleast_2d(np.asarray(features, dtype=float))
        y = np.atleast_2d(np.asarray(targets, dtype=float))
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"features has {x.shape[0]} rows, targets has {y.shape[0]}")
        if x.shape[0] < 20:
            raise ValueError(f"need at least 20 training samples, got {x.shape[0]}")
        if y.shape[1] != 3:
            raise ValueError(f"targets must have 3 columns, got {y.shape[1]}")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError("features and targets must be finite")
        model = ExtraTreesRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=1,
        )
        model.fit(x, y)
        self._model = model
        self._lo = np.min(x, axis=0)
        self._hi = np.max(x, axis=0)
        spread = self._tree_spread(x)
        self._reference_spread = float(np.mean(spread)) or 1.0
        return self

    def _tree_spread(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        assert self._model is not None
        preds = np.stack([t.predict(x) for t in self._model.estimators_])
        return np.std(preds, axis=0)

    def predict(self, features: ArrayLike) -> WarmStartPrediction:
        """Predict via parameters for ``(n, 28)`` features.

        Raises ``RuntimeError`` if unfitted -- the model refuses to guess.
        """
        if self._model is None:
            raise RuntimeError("LearnedWarmStart is not fitted; call fit() first")
        x = np.atleast_2d(np.asarray(features, dtype=float))
        if x.shape[1] != self._lo.size:
            raise ValueError(f"expected {self._lo.size} features, got {x.shape[1]}")
        params = np.atleast_2d(self._model.predict(x))
        spread = self._tree_spread(x)
        conf = np.exp(-np.mean(spread, axis=1) / self._reference_spread)
        outside = np.any((x < self._lo) | (x > self._hi), axis=1)
        return WarmStartPrediction(params, spread, np.clip(conf, 0.0, 1.0), outside)

    def predict_problem(self, problem: SlewProblem) -> tuple[NDArray[np.float64], float, bool]:
        """Convenience wrapper: ``(params (3,), confidence, extrapolating)``."""
        if not isinstance(problem, SlewProblem):
            raise TypeError(f"expected a SlewProblem, got {type(problem).__name__}")
        out = self.predict(problem_features(problem)[None, :])
        return out.params[0], float(out.confidence[0]), bool(out.extrapolating[0])
