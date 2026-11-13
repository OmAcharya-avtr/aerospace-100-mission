"""The learned contention-aware miss-rate predictor, and what it is scored against.

The learned component is a random forest regressor from
:mod:`sklearn.ensemble` mapping the window descriptors of
:mod:`deadlinerisk.windows` to that window's counted miss rate. A forest rather
than anything larger for three reasons that are all measured constraints and not
preferences: the dataset is a few hundred windows from one campaign, the build
environment has two cores and a three-minute budget for any fit, and a forest
gives a spread across trees that is an honest -- if conservative -- uncertainty
output without a second model.

**Uncertainty output.** :meth:`ContentionMissPredictor.predict` returns a
point estimate and a per-sample standard deviation across the ensemble's trees.
That is a statement about the ensemble's internal disagreement, which is a lower
bound on the predictive uncertainty: it does not include the noise in the
target, which for a window's counted rate over ``N`` releases is at least the
binomial ``sqrt(p(1-p)/N)``. :func:`target_noise_sd` computes that, and
:meth:`ContentionMissPredictor.predict` adds the two in quadrature when the
release counts are supplied, which is the honest total.

Baselines it is scored against, on the SAME splits:

- :class:`ConstantRatePredictor` -- the training mean. A learned model that
  cannot beat this has learned nothing from the descriptors.
- the classical speed-scaling predictor in :mod:`deadlinerisk.speed_scaling`,
  which is fitted to nothing and uses only the worker count plus the idle-run
  execution-time distribution.

Compute budget: the fit is measured in ``validation/validate_measured_learned.py``
and reported there. Nothing in this module trains for longer than a few seconds
on two cores.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestRegressor

#: Forest size. Large enough for a stable across-tree spread, small enough that
#: a fit on a few hundred rows is a fraction of a second on two cores.
N_ESTIMATORS = 300

#: Depth cap. The dataset has a few hundred rows and ten descriptors; an
#: unbounded tree memorises single windows.
MAX_DEPTH = 6

#: Declared seed for every fit in this product.
RANDOM_STATE = 920092


@dataclass(frozen=True)
class Prediction:
    """A predicted miss rate with its uncertainty.

    Attributes
    ----------
    rate
        Point estimate, dimensionless, clipped to ``[0, 1]``.
    sigma
        Standard deviation, dimensionless. Across-tree spread alone, or that
        combined in quadrature with the binomial target noise when release
        counts were supplied.
    sigma_model
        The across-tree component on its own.
    sigma_target
        The binomial target-noise component, zero when no release counts were
        supplied.
    """

    rate: np.ndarray
    sigma: np.ndarray
    sigma_model: np.ndarray
    sigma_target: np.ndarray


def target_noise_sd(rate: np.ndarray, releases: np.ndarray) -> np.ndarray:
    """Binomial standard deviation of a counted rate, ``sqrt(p(1-p)/N)``.

    Dimensionless. This is the irreducible noise in the TARGET: a window's
    counted miss rate over ``N`` releases is itself an estimate.
    """
    p = np.clip(np.asarray(rate, dtype=np.float64), 0.0, 1.0)
    n = np.asarray(releases, dtype=np.float64)
    if np.any(n <= 0.0):
        raise ValueError("releases must be positive")
    return np.sqrt(p * (1.0 - p) / n)


class ConstantRatePredictor:
    """Predict the training mean rate for every window. The null model."""

    def __init__(self) -> None:
        self.rate_: float | None = None
        self.sd_: float | None = None

    def fit(self, features: np.ndarray, targets: np.ndarray) -> ConstantRatePredictor:
        """Record the training mean and standard deviation of ``targets``."""
        values = np.asarray(targets, dtype=np.float64)
        if values.size == 0:
            raise ValueError("cannot fit a constant rate to an empty target vector")
        self.rate_ = float(values.mean())
        self.sd_ = float(values.std(ddof=1)) if values.size > 1 else 0.0
        return self

    def predict(self, features: np.ndarray) -> Prediction:
        """Return the training mean, with the training spread as uncertainty."""
        if self.rate_ is None:
            raise RuntimeError("ConstantRatePredictor.predict called before fit")
        n = int(np.asarray(features).shape[0])
        zeros = np.zeros(n, dtype=np.float64)
        sigma = np.full(n, float(self.sd_ or 0.0), dtype=np.float64)
        return Prediction(
            rate=np.full(n, self.rate_, dtype=np.float64),
            sigma=sigma,
            sigma_model=sigma,
            sigma_target=zeros,
        )


class ContentionMissPredictor:
    """Random-forest miss-rate predictor with an across-tree uncertainty output."""

    def __init__(
        self,
        n_estimators: int = N_ESTIMATORS,
        max_depth: int = MAX_DEPTH,
        random_state: int = RANDOM_STATE,
    ) -> None:
        if n_estimators < 2:
            raise ValueError(
                f"n_estimators must be >= 2 for an across-tree spread, got {n_estimators}"
            )
        if max_depth < 1:
            raise ValueError(f"max_depth must be >= 1, got {max_depth}")
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.random_state = int(random_state)
        self.forest_: RandomForestRegressor | None = None
        self.feature_names_: tuple[str, ...] = ()

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        feature_names: tuple[str, ...] = (),
    ) -> ContentionMissPredictor:
        """Fit the forest.

        Raises
        ------
        ValueError
            On mismatched shapes, an empty training set, or a target outside
            ``[0, 1]`` -- the target is a rate and a value outside the unit
            interval means the caller built the dataset wrongly.
        """
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError(f"features must be two-dimensional, got shape {x.shape}")
        if y.ndim != 1 or y.size != x.shape[0]:
            raise ValueError(
                f"targets shape {y.shape} does not match features shape {x.shape}"
            )
        if x.shape[0] < 2:
            raise ValueError(f"need at least 2 training windows, got {x.shape[0]}")
        if np.any(y < 0.0) or np.any(y > 1.0):
            raise ValueError("targets must be miss RATES in [0, 1]")
        self.forest_ = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=1,
        ).fit(x, y)
        self.feature_names_ = tuple(feature_names)
        return self

    def predict(
        self, features: np.ndarray, releases: np.ndarray | None = None
    ) -> Prediction:
        """Predict miss rates with uncertainty.

        Parameters
        ----------
        features
            ``(n, n_features)`` descriptor matrix.
        releases
            Optional per-window release counts. When supplied, the binomial
            target noise at the predicted rate is combined in quadrature with
            the across-tree spread.
        """
        if self.forest_ is None:
            raise RuntimeError("ContentionMissPredictor.predict called before fit")
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError(f"features must be two-dimensional, got shape {x.shape}")
        per_tree = np.stack(
            [tree.predict(x) for tree in self.forest_.estimators_], axis=0
        )
        rate = np.clip(per_tree.mean(axis=0), 0.0, 1.0)
        sigma_model = per_tree.std(axis=0, ddof=0)
        if releases is None:
            sigma_target = np.zeros_like(rate)
        else:
            sigma_target = target_noise_sd(rate, releases)
        return Prediction(
            rate=rate,
            sigma=np.sqrt(sigma_model**2 + sigma_target**2),
            sigma_model=sigma_model,
            sigma_target=sigma_target,
        )

    def importances(self) -> dict[str, float]:
        """Impurity-based feature importances, by name where names were given."""
        if self.forest_ is None:
            raise RuntimeError("ContentionMissPredictor.importances called before fit")
        values = self.forest_.feature_importances_
        names = self.feature_names_ or tuple(f"f{i}" for i in range(values.size))
        return {name: float(value) for name, value in zip(names, values, strict=True)}


@dataclass(frozen=True)
class Score:
    """Scores of one predictor on one split. All dimensionless.

    ``mae`` and ``rmse`` are over the window miss rates. ``mae_ratio`` is the
    predictor's MAE divided by the constant-rate baseline's MAE on the same
    split, so a value below 1 is a win over the null model and a value above 1
    is a loss. ``coverage`` is the fraction of test windows whose true rate lies
    within one reported sigma of the prediction, which is the only check that
    the uncertainty output means anything; it is ``nan`` when no sigma was
    reported.
    """

    label: str
    n_test: int
    mae: float
    rmse: float
    bias: float
    coverage: float


def score(
    label: str,
    truth: np.ndarray,
    prediction: Prediction | np.ndarray,
    sigma: np.ndarray | None = None,
) -> Score:
    """Score a prediction against measured window rates."""
    y = np.asarray(truth, dtype=np.float64)
    if isinstance(prediction, Prediction):
        yhat = np.asarray(prediction.rate, dtype=np.float64)
        s = np.asarray(prediction.sigma, dtype=np.float64)
    else:
        yhat = np.asarray(prediction, dtype=np.float64)
        s = None if sigma is None else np.asarray(sigma, dtype=np.float64)
    if y.shape != yhat.shape:
        raise ValueError(f"truth shape {y.shape} does not match prediction {yhat.shape}")
    if y.size == 0:
        raise ValueError("cannot score an empty split")
    residual = yhat - y
    if s is None or not np.any(s > 0.0):
        coverage = float("nan")
    else:
        coverage = float(np.mean(np.abs(residual) <= s))
    return Score(
        label=label,
        n_test=int(y.size),
        mae=float(np.mean(np.abs(residual))),
        rmse=float(np.sqrt(np.mean(residual**2))),
        bias=float(np.mean(residual)),
        coverage=coverage,
    )
