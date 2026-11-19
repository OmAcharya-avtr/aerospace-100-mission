"""The learned per-pass yield predictor, with an uncertainty output.

WHAT IT PREDICTS AND FROM WHAT
------------------------------
Target: the **realised fraction** :math:`Y_{real}/Y_{cap}` of one pass,
dimensionless in ``[0, 1]``. Normalising by the window capacity rather than by
the booked volume divides the geometry out of the target, which is what makes
the target transferable to a station the model has never seen: the geometry
and the link budget are already known before the pass, so a model that
predicts the fraction and is handed the capacity predicts the yield.

Features: **pre-pass state and geometry only**. Every one of them is available
to a scheduler before the contact starts, which is the condition for the thing
to be a predictor rather than a measurement. The station enters through its
own figures -- figure of merit, zenith attenuation, mask, latitude -- and
**never through its identity**, because a model with a station-identity
feature cannot be scored on a held-out station at all. See
:data:`FEATURE_NAMES`.

The quantities the product is about -- acquisition time, outage count, outage
duration, idle time -- are **not** features. They are what the pass did.

ARCHITECTURE, AND WHY IT IS SMALL
---------------------------------
``sklearn.ensemble.RandomForestRegressor``, 300 trees, ``max_depth=10``,
``n_jobs=1``, ``oob_score=True``. Three hard constraints, not preferences:

- the dataset is roughly two thousand passes from one committed campaign, so
  anything with many parameters memorises passes;
- the build environment has two cores shared with four other builds and a
  60 s budget for any fit;
- a forest supplies two different uncertainty outputs for free, and the
  difference between them turns out to be the most useful thing the model
  says. See below.

TWO UNCERTAINTY OUTPUTS, AND THE ONE THAT IS CALIBRATED
-------------------------------------------------------
:meth:`PassYieldPredictor.predict_ensemble` takes a ``mode``:

- ``"tree-spread"`` -- the 300 per-tree predictions. This is the uncertainty
  output people reach for first and it is **structurally over-confident**: the
  spread across trees measures disagreement about the conditional mean, not
  the dispersion of the outcome around it, so it omits the irreducible noise
  entirely. It is implemented and reported because reporting it is the honest
  way to show the size of the effect.
- ``"residual"`` -- the point prediction plus the empirical distribution of
  the **out-of-bag** training residuals. Out-of-bag rather than in-sample,
  because in-sample forest residuals are shrunk by the fit and would reproduce
  the over-confidence in a subtler form. This is the product's declared
  uncertainty output.

Both are scored by CRPS, PIT calibration error and interval coverage in
``validation/validate_learned.py``, and both numbers are published.

COMPUTE BUDGET
--------------
Measured on the build machine and reported in ``README.md``: the fit of 300
trees on the committed campaign runs in a couple of seconds on one core, far
inside the 60 s budget. Nothing here is serialised to disk, so there is no
pickle whose scikit-learn version has to be matched -- the model is refitted
from the committed data by every script that needs it, and the
``scikit-learn`` ceiling in ``pyproject.toml`` is there because the fitted
numbers depend on the library version, not because a file format does.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestRegressor

#: The feature columns, in order. Every one is knowable before the pass.
FEATURE_NAMES: tuple[str, ...] = (
    "window_duration_s",
    "max_elevation_deg",
    "min_slant_range_km",
    "capacity_mbit",
    "mean_rate_mbit_s",
    "station_gain_over_temperature_db",
    "station_zenith_attenuation_db",
    "station_mask_deg",
    "station_abs_latitude_deg",
    "slew_angle_deg",
    "backlog_mbit",
    "hours_since_last_contact",
    "prior_pass_outage_count",
    "local_hour_sin",
    "local_hour_cos",
    "rise_azimuth_sin",
    "rise_azimuth_cos",
)

#: Uncertainty modes :meth:`PassYieldPredictor.predict_ensemble` accepts.
UNCERTAINTY_MODES: tuple[str, ...] = ("tree-spread", "residual")

#: The product's declared uncertainty output. See the module docstring.
DECLARED_UNCERTAINTY_MODE = "residual"


@dataclass(frozen=True)
class PredictorSettings:
    """Hyper-parameters, in one place, all fixed before any measurement."""

    n_estimators: int = 300
    max_depth: int = 10
    min_samples_leaf: int = 4
    random_state: int = 99099
    residual_members: int = 256


class PassYieldPredictor:
    """A random forest over pre-pass features, with an uncertainty output.

    The public surface is :meth:`fit`, :meth:`predict_fraction`,
    :meth:`predict_ensemble` and :meth:`predict_shortfall_ensemble`. The last
    is the one the product's comparisons use, because the quantity the
    product reports is a shortfall in Mbit.
    """

    def __init__(self, settings: PredictorSettings | None = None) -> None:
        self.settings = settings if settings is not None else PredictorSettings()
        self._forest: RandomForestRegressor | None = None
        self._oob_residuals: np.ndarray | None = None

    @property
    def fitted(self) -> bool:
        """Whether :meth:`fit` has been called."""
        return self._forest is not None

    @property
    def out_of_bag_residuals(self) -> np.ndarray:
        """The out-of-bag training residuals of the realised fraction.

        Dimensionless. The empirical law of these is the product's uncertainty
        output; see the module docstring for why they are out-of-bag.
        """
        if self._oob_residuals is None:
            raise ValueError("the predictor is not fitted; call fit() first")
        return self._oob_residuals

    def fit(self, features: np.ndarray, realised_fraction: np.ndarray) -> PassYieldPredictor:
        """Fit the forest. ``features`` is ``(n, len(FEATURE_NAMES))``."""
        matrix = np.asarray(features, dtype=float)
        target = np.asarray(realised_fraction, dtype=float).ravel()
        if matrix.ndim != 2 or matrix.shape[1] != len(FEATURE_NAMES):
            raise ValueError(
                f"features must have shape (n, {len(FEATURE_NAMES)}) to match FEATURE_NAMES, "
                f"got {matrix.shape}"
            )
        if matrix.shape[0] != target.size:
            raise ValueError(
                f"features has {matrix.shape[0]} rows but there are {target.size} targets"
            )
        if matrix.shape[0] < 2 * self.settings.min_samples_leaf:
            raise ValueError(
                f"at least {2 * self.settings.min_samples_leaf} passes are needed to fit, "
                f"got {matrix.shape[0]}"
            )
        if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(target)):
            raise ValueError("features and realised_fraction must be finite")
        forest = RandomForestRegressor(
            n_estimators=self.settings.n_estimators,
            max_depth=self.settings.max_depth,
            min_samples_leaf=self.settings.min_samples_leaf,
            random_state=self.settings.random_state,
            oob_score=True,
            bootstrap=True,
            n_jobs=1,
        )
        forest.fit(matrix, target)
        out_of_bag = np.asarray(forest.oob_prediction_, dtype=float)
        finite = np.isfinite(out_of_bag)
        self._oob_residuals = target[finite] - out_of_bag[finite]
        self._forest = forest
        return self

    def _checked(self, features: np.ndarray) -> tuple[RandomForestRegressor, np.ndarray]:
        if self._forest is None:
            raise ValueError("the predictor is not fitted; call fit() first")
        matrix = np.asarray(features, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != len(FEATURE_NAMES):
            raise ValueError(
                f"features must have shape (n, {len(FEATURE_NAMES)}), got {matrix.shape}"
            )
        return self._forest, matrix

    def predict_fraction(self, features: np.ndarray) -> np.ndarray:
        """Point prediction of the realised fraction, dimensionless."""
        forest, matrix = self._checked(features)
        return np.asarray(forest.predict(matrix), dtype=float)

    def tree_predictions(self, features: np.ndarray) -> np.ndarray:
        """Per-tree predictions of the realised fraction, ``(n, n_estimators)``."""
        forest, matrix = self._checked(features)
        return np.stack(
            [np.asarray(tree.predict(matrix), dtype=float) for tree in forest.estimators_], axis=1
        )

    def predict_ensemble(
        self, features: np.ndarray, mode: str = DECLARED_UNCERTAINTY_MODE
    ) -> np.ndarray:
        """Predictive ensemble of the realised fraction, ``(n, members)``.

        See the module docstring for what each ``mode`` is and which one is
        calibrated. The ``"residual"`` ensemble is clipped into ``[0, 1]``,
        because a realised fraction outside that interval is not a possible
        outcome and a forecast that places mass there is wrong in a way the
        caller should not have to correct.
        """
        if mode not in UNCERTAINTY_MODES:
            raise ValueError(f"mode must be one of {UNCERTAINTY_MODES}, got {mode!r}")
        if mode == "tree-spread":
            return np.clip(self.tree_predictions(features), 0.0, 1.0)
        point = self.predict_fraction(features)
        residuals = self.out_of_bag_residuals
        members = min(self.settings.residual_members, residuals.size)
        if members < 2:
            raise ValueError(
                "too few out-of-bag residuals to form a predictive ensemble; the training set "
                "is too small for this forest's bootstrap to leave samples out"
            )
        # A deterministic quantile ensemble of the residual law: the
        # mid-points of `members` equal-probability intervals. No random
        # number generator, so two calls agree exactly.
        levels = (np.arange(members) + 0.5) / members
        quantiles = np.quantile(residuals, levels)
        return np.clip(point[:, None] + quantiles[None, :], 0.0, 1.0)

    def predict_shortfall_ensemble(
        self,
        features: np.ndarray,
        nominal_mbit: np.ndarray,
        capacity_mbit: np.ndarray,
        mode: str = DECLARED_UNCERTAINTY_MODE,
    ) -> np.ndarray:
        """Predictive ensemble of the **shortfall** in Mbit, ``(n, members)``.

        ``shortfall = nominal - fraction * capacity``, applied member by
        member, so the returned ensemble is a predictive distribution of the
        product's headline quantity and is directly comparable with
        baseline 2's convolution and baseline 1's point.
        """
        fraction = self.predict_ensemble(features, mode)
        nominal = np.asarray(nominal_mbit, dtype=float).ravel()
        capacity = np.asarray(capacity_mbit, dtype=float).ravel()
        if nominal.size != fraction.shape[0] or capacity.size != fraction.shape[0]:
            raise ValueError(
                f"nominal_mbit and capacity_mbit must have {fraction.shape[0]} entries, got "
                f"{nominal.size} and {capacity.size}"
            )
        return nominal[:, None] - fraction * capacity[:, None]

    def feature_importance(self) -> dict[str, float]:
        """Impurity-based feature importances, keyed by name, summing to 1.

        Impurity importances are biased toward high-cardinality features
        (Strobl, C. et al. (2007), "Bias in random forest variable importance
        measures", BMC Bioinformatics 8, 25). They are reported as a
        description of the fitted trees, not as evidence of causation, and
        nothing in this product is selected by them.
        """
        if self._forest is None:
            raise ValueError("the predictor is not fitted; call fit() first")
        return dict(
            zip(FEATURE_NAMES, (float(v) for v in self._forest.feature_importances_), strict=True)
        )
