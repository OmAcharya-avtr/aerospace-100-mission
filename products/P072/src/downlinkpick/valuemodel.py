"""The learned value model, and the two baselines it has to beat.

What this is for. A selection is only as good as the values it is given. In a
real recorder inventory some products carry a declared value an operator or
science planner wrote down, and some carry nothing: an autonomously triggered
acquisition, a product from an instrument whose team has not been consulted, a
declaration that was made at acquisition and has not been revisited. The exact
solver has nothing to say about a missing value. This module estimates one.

What it is benchmarked against, and the prediction. The batch specification
predicts in writing that *the exact solver on declared values beats everything
and the learned value model helps only where declarations are absent*. That
prediction is the expected result, not a hedge, and
``validation/validate_valuemodel.py`` is written to measure it rather than to
confirm it. Three estimators are scored on the same held-out products and
through the same solver:

* :func:`declared_values` -- the passthrough. Uses the declaration where it
  exists and a fallback where it does not.
* :func:`size_age_values` -- a fitted size/age heuristic, three coefficients by
  ordinary least squares on ``log10(bytes)`` and age. This is the "obvious thing
  to do" baseline and it is fitted rather than guessed so that the comparison is
  not against a straw man.
* :class:`ValueModel` -- a random-forest regressor on the ten features of
  :data:`downlinkpick.dataset.VALUE_FEATURES`.

Uncertainty. :meth:`ValueModel.predict_with_uncertainty` returns the spread of
the individual tree predictions alongside the mean. This is a **dispersion
measure over the ensemble, not a calibrated confidence interval**: it describes
how much the trees disagree, and a random forest's tree spread is known to
understate predictive uncertainty because the trees share training data
(Wager, Hastie and Efron 2014). MODEL_CARD.md reports the measured empirical
coverage of the plus-or-minus-two-sigma band so that a reader can see by how
much. The dispersion is still the operationally useful number: a product whose
value the ensemble cannot agree on is a product to look at by hand before
spending a pass on it.

References
----------
Breiman, L. (2001), Random forests, Machine Learning 45(1), 5-32. The estimator.

Wager, S., Hastie, T. and Efron, B. (2014), Confidence intervals for random
forests: the jackknife and the infinitesimal jackknife, Journal of Machine
Learning Research 15, 1625-1651. Why the per-tree spread reported here is a
dispersion measure and not a confidence interval.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .dataset import VALUE_FEATURES, value_feature_row
from .inventory import Inventory

#: Default random-forest size. 300 trees of unrestricted depth on a table of a
#: few thousand rows and ten features trains in under two seconds on the two
#: cores this product was measured on; the compute budget is recorded in
#: MODEL_CARD.md.
DEFAULT_N_ESTIMATORS = 300


@dataclass(frozen=True)
class ValuePrediction:
    """Per-product value prediction with its ensemble dispersion.

    Attributes
    ----------
    names:
        Product names, in prediction order.
    value:
        Ensemble mean prediction in value units.
    dispersion:
        Standard deviation of the per-tree predictions, in value units. A
        dispersion measure over the ensemble, not a confidence interval.
    lower, upper:
        ``value -+ 2 * dispersion``, clipped at zero below. The measured
        empirical coverage of this band is reported in MODEL_CARD.md and is not
        the nominal 95 per cent.
    """

    names: tuple[str, ...]
    value: np.ndarray
    dispersion: np.ndarray
    lower: np.ndarray
    upper: np.ndarray

    def as_mapping(self) -> dict[str, float]:
        """Predictions as a ``{name: value}`` mapping for a selection problem."""
        return {name: float(v) for name, v in zip(self.names, self.value, strict=True)}

    def dispersion_mapping(self) -> dict[str, float]:
        """Dispersions as a ``{name: sigma}`` mapping."""
        return {name: float(s) for name, s in zip(self.names, self.dispersion, strict=True)}


class ValueModel:
    """Random-forest regressor for realised product value.

    Parameters
    ----------
    n_estimators:
        Number of trees.
    max_depth:
        Maximum tree depth, ``None`` for unrestricted.
    min_samples_leaf:
        Minimum samples in a leaf. Raising this smooths the prediction and
        widens the ensemble dispersion.
    seed:
        Seed for the forest's bootstrap and feature subsampling. Fixed by
        default so that two fits on the same table are identical.
    """

    def __init__(
        self,
        *,
        n_estimators: int = DEFAULT_N_ESTIMATORS,
        max_depth: int | None = None,
        min_samples_leaf: int = 2,
        seed: int = 20261021,
    ) -> None:
        if n_estimators < 2:
            raise ValueError(f"n_estimators must be >= 2, got {n_estimators!r}")
        if min_samples_leaf < 1:
            raise ValueError(f"min_samples_leaf must be >= 1, got {min_samples_leaf!r}")
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.min_samples_leaf = int(min_samples_leaf)
        self.seed = int(seed)
        self._forest: object | None = None

    @property
    def fitted(self) -> bool:
        """True once :meth:`fit` has run."""
        return self._forest is not None

    def fit(self, features: np.ndarray, target: np.ndarray) -> ValueModel:
        """Fit on a feature matrix in :data:`VALUE_FEATURES` column order.

        Parameters
        ----------
        features:
            ``(n, len(VALUE_FEATURES))`` array.
        target:
            ``(n,)`` realised value in value units, non-negative.
        """
        from sklearn.ensemble import RandomForestRegressor

        x = np.asarray(features, dtype=float)
        y = np.asarray(target, dtype=float).ravel()
        if x.ndim != 2 or x.shape[1] != len(VALUE_FEATURES):
            raise ValueError(
                f"features must have shape (n, {len(VALUE_FEATURES)}), got {x.shape}"
            )
        if y.shape[0] != x.shape[0]:
            raise ValueError(f"target has {y.shape[0]} rows against {x.shape[0]} features")
        if np.any(y < 0.0):
            raise ValueError("target values must be non-negative")
        forest = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.seed,
            n_jobs=1,
        )
        forest.fit(x, y)
        self._forest = forest
        return self

    def _require_forest(self) -> object:
        if self._forest is None:
            raise RuntimeError("ValueModel.fit must be called before prediction")
        return self._forest

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Ensemble mean prediction in value units."""
        forest = self._require_forest()
        return np.clip(forest.predict(np.asarray(features, dtype=float)), 0.0, None)  # type: ignore[attr-defined]

    def predict_with_dispersion(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(mean, per-tree standard deviation)`` in value units."""
        forest = self._require_forest()
        x = np.asarray(features, dtype=float)
        per_tree = np.vstack([tree.predict(x) for tree in forest.estimators_])  # type: ignore[attr-defined]
        return np.clip(per_tree.mean(axis=0), 0.0, None), per_tree.std(axis=0, ddof=1)

    def predict_inventory(self, inventory: Inventory) -> ValuePrediction:
        """Predict every product in ``inventory``, with dispersion."""
        features = np.array(
            [value_feature_row(product, inventory) for product in inventory], dtype=float
        )
        mean, dispersion = self.predict_with_dispersion(features)
        return ValuePrediction(
            names=inventory.names,
            value=mean,
            dispersion=dispersion,
            lower=np.clip(mean - 2.0 * dispersion, 0.0, None),
            upper=mean + 2.0 * dispersion,
        )

    def feature_importances(self) -> dict[str, float]:
        """Impurity-based feature importances keyed by feature name.

        Impurity importances are biased towards high-cardinality features and
        are reported in MODEL_CARD.md with that caveat attached.
        """
        forest = self._require_forest()
        return {
            name: float(weight)
            for name, weight in zip(
                VALUE_FEATURES, forest.feature_importances_, strict=True  # type: ignore[attr-defined]
            )
        }


def declared_values(
    inventory: Inventory,
    *,
    fallback: str = "median",
) -> dict[str, float]:
    """Declared-value passthrough with an explicit fallback for missing ones.

    Parameters
    ----------
    inventory:
        The inventory.
    fallback:
        What to use where ``declared_value`` is ``None``. ``"median"`` uses the
        median of the declarations that exist, ``"mean"`` their mean, and
        ``"zero"`` uses zero, which is the behaviour of a planning tool that
        ignores anything nobody declared. All three are measured in
        ``validation/validate_valuemodel.py``; the choice matters a great deal
        when declarations are absent and not at all when they are complete.

    Returns
    -------
    dict
        Value per product name, in value units.
    """
    if fallback not in {"median", "mean", "zero"}:
        raise ValueError(f"unknown fallback {fallback!r}; expected median, mean or zero")
    declared = [p.declared_value for p in inventory if p.declared_value is not None]
    if fallback == "zero" or not declared:
        default = 0.0
    elif fallback == "median":
        default = float(np.median(declared))
    else:
        default = float(np.mean(declared))
    return {
        p.name: float(p.declared_value) if p.declared_value is not None else default
        for p in inventory
    }


@dataclass(frozen=True)
class SizeAgeHeuristic:
    """Fitted three-coefficient size/age value heuristic.

    The model is ``value = a + b * log10(raw_bytes) + c * age_s``, fitted by
    ordinary least squares. It is the simplest thing a planner with no value
    declarations would write down, and it is *fitted* rather than guessed so
    that the learned model is compared against the best version of the simple
    idea rather than against a badly chosen one.
    """

    intercept: float
    log_bytes_coefficient: float
    age_coefficient: float

    def predict(self, log10_bytes: np.ndarray, age_s: np.ndarray) -> np.ndarray:
        """Predicted value in value units, clipped at zero below."""
        return np.clip(
            self.intercept
            + self.log_bytes_coefficient * np.asarray(log10_bytes, dtype=float)
            + self.age_coefficient * np.asarray(age_s, dtype=float),
            0.0,
            None,
        )

    def predict_inventory(self, inventory: Inventory) -> dict[str, float]:
        """Predicted value per product name."""
        log_bytes = np.array([np.log10(p.raw_bytes) for p in inventory])
        ages = np.array([p.age_s for p in inventory])
        values = self.predict(log_bytes, ages)
        return {name: float(v) for name, v in zip(inventory.names, values, strict=True)}


def fit_size_age_heuristic(features: np.ndarray, target: np.ndarray) -> SizeAgeHeuristic:
    """Least-squares fit of :class:`SizeAgeHeuristic` on a value training table.

    ``features`` is the :data:`VALUE_FEATURES` matrix; columns 0 and 1 are
    ``log10_raw_bytes`` and ``age_s``.
    """
    x = np.asarray(features, dtype=float)
    y = np.asarray(target, dtype=float).ravel()
    if x.ndim != 2 or x.shape[1] != len(VALUE_FEATURES):
        raise ValueError(f"features must have shape (n, {len(VALUE_FEATURES)}), got {x.shape}")
    design = np.column_stack([np.ones(x.shape[0]), x[:, 0], x[:, 1]])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    return SizeAgeHeuristic(
        intercept=float(coefficients[0]),
        log_bytes_coefficient=float(coefficients[1]),
        age_coefficient=float(coefficients[2]),
    )


def size_age_values(heuristic: SizeAgeHeuristic, inventory: Inventory) -> dict[str, float]:
    """Value per product from the fitted size/age heuristic."""
    return heuristic.predict_inventory(inventory)


def hybrid_values(
    inventory: Inventory, learned: Mapping[str, float]
) -> dict[str, float]:
    """Declared value where it exists, learned prediction where it does not.

    This is the estimator the specification predicts will be the useful one: it
    never overrides a declaration and only fills gaps. Keeping it separate from
    the pure learned estimator is what makes the two claims -- "the learned
    model does not beat a declaration" and "the learned model helps where there
    is none" -- separately measurable.
    """
    missing = [name for name in inventory.names if name not in learned]
    if missing:
        raise ValueError("no learned prediction for: " + ", ".join(sorted(missing)))
    return {
        p.name: float(p.declared_value) if p.declared_value is not None else float(learned[p.name])
        for p in inventory
    }


def spearman(a: Sequence[float] | np.ndarray, b: Sequence[float] | np.ndarray) -> float:
    """Spearman rank correlation between two sequences, dimensionless.

    Reported alongside the error metrics because a knapsack cares about the
    *order* of the values far more than their level: an estimator that is
    uniformly half the true value selects exactly the same products.

    A constant input has no ranks, so SciPy returns ``nan`` and emits a
    ``ConstantInputWarning``. That is passed through rather than suppressed: an
    estimator that emits one value for every product -- the declared-value
    passthrough when nothing at all is declared -- has no ranking information,
    and ``nan`` is the honest report of that.
    """
    from scipy.stats import spearmanr

    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {x.shape} against {y.shape}")
    if x.size < 2:
        raise ValueError("spearman needs at least two points")
    result = spearmanr(x, y)
    return float(result.statistic)
