"""Population-level drift predictors: the analytic baselines and the learned one.

Every predictor here maps the same design matrix -- the ten static features of
:mod:`orbitdrift.features` plus epoch age in days -- to a median drift in km
and an interval around it. They see identical columns and identical rows, which
is the only arrangement in which the benchmark between them says anything.

The order of implementation is the order of the list, and it is not
cosmetic. The analytic forms came first and were validated against the
simulation (``validation/validate_growth_baseline.py``) before the learned
model was fitted at all, because a baseline that is weaker than it could be
manufactures a learned win. In this mission a published learned win was once
inflated by about half by exactly that.

The predictors
--------------
:class:`RuleOfThumbPredictor`
    One number of kilometres per day, times the age, with a fixed
    multiplicative band. This is what the product is positioned against: the
    operator's rule of thumb. It is in the benchmark because a baseline nobody
    would defend is still the thing being replaced, and its measured coverage
    is the honest statement of what guessing costs.

:class:`AnalyticGrowthPredictor`
    ``log(drift) = beta . x + g(t)`` where ``g`` is the declared time form.
    ``"linear"`` fixes the exponent at 1, ``"quadratic"`` at 2, ``"power"``
    fits one global exponent, ``"power_altitude"`` lets the exponent vary
    linearly with the logarithm of perigee altitude -- which is the physical
    expectation, since the dominant error term changes with altitude -- and
    ``"drag_exponential"`` uses ``log(exp(t/tau) - 1)`` with ``tau`` fitted.
    Ordinary least squares in log space, closed form, no iteration except the
    one-dimensional search for ``tau``.

:class:`LearnedQuantilePredictor`
    Three gradient-boosted regressors with the quantile loss, at the lower
    quantile, the median and the upper quantile, on the logarithm of drift.
    scikit-learn's ``GradientBoostingRegressor``; PyTorch is not installed in
    the environment these numbers were measured in and nothing here uses it.

:class:`ConformalPredictor`
    Wraps any of the above. Fits the base predictor on a training split, then
    takes the empirical quantile of the absolute log residual on a disjoint
    calibration split and uses it as a symmetric multiplicative half-width.
    Split conformal prediction (Papadopoulos et al. 2002; Lei et al. 2018)
    gives marginal coverage at least the nominal level for *exchangeable*
    data.

    That exchangeability assumption is violated here and the violation is the
    point rather than a footnote. The rows are grouped: twenty-eight ages of
    the same object share its features and its physics, so a calibration set
    drawn by object is exchangeable at the level of objects and not at the
    level of rows. Marginal coverage therefore comes out close to nominal
    while coverage *conditional on altitude or on epoch age* does not, and
    that is exactly what the coverage table in
    ``validation/validate_coverage_table.py`` shows. Conformal prediction
    promises marginal coverage and this package does not claim it promises
    more.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from .errors import InsufficientDataError
from .features import ALL_FEATURE_NAMES, FEATURE_NAMES

_LOG_FLOOR_KM = 1e-9
_AGE_FLOOR_DAYS = 1e-6
_PERIGEE_COLUMN = FEATURE_NAMES.index("perigee_altitude_km")
_AGE_COLUMN = len(FEATURE_NAMES)

ANALYTIC_FORMS: tuple[str, ...] = (
    "linear",
    "quadratic",
    "power",
    "power_altitude",
    "drag_exponential",
)


def _split_design(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.atleast_2d(np.asarray(x, dtype=float))
    if matrix.shape[1] != len(ALL_FEATURE_NAMES):
        raise ValueError(
            f"design matrix must have {len(ALL_FEATURE_NAMES)} columns "
            f"{ALL_FEATURE_NAMES}; got {matrix.shape[1]}"
        )
    return matrix[:, :_AGE_COLUMN], np.maximum(matrix[:, _AGE_COLUMN], _AGE_FLOOR_DAYS)


@dataclass
class RuleOfThumbPredictor:
    """``drift = rate * age``, with a fixed multiplicative band.

    The defaults are deliberately the kind of figure that circulates as
    practical advice -- one kilometre per day, a factor of three either way --
    and they are not fitted to anything. :meth:`fit` ignores its arguments
    entirely, which is the honest implementation of a rule of thumb.
    """

    rate_km_per_day: float = 1.0
    band_factor: float = 3.0
    name: str = "rule_of_thumb"

    def fit(self, x: np.ndarray, drift_km: np.ndarray) -> RuleOfThumbPredictor:
        """Return ``self``. A rule of thumb does not learn, by definition."""
        del x, drift_km
        return self

    def predict_median_km(self, x: np.ndarray) -> np.ndarray:
        _, ages = _split_design(x)
        return self.rate_km_per_day * ages

    def predict_interval_km(
        self, x: np.ndarray, coverage: float = 0.90
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fixed band. ``coverage`` is accepted and ignored, and that is the point."""
        del coverage
        centre = self.predict_median_km(x)
        return centre / self.band_factor, centre * self.band_factor


@dataclass
class AnalyticGrowthPredictor:
    """Log-linear-in-features analytic growth model. ``form`` is in :data:`ANALYTIC_FORMS`."""

    form: str = "power_altitude"
    timescale_bounds_days: tuple[float, float] = (0.2, 2000.0)
    name: str = field(init=False, default="")
    coefficients_: np.ndarray | None = field(init=False, default=None)
    timescale_days_: float = field(init=False, default=float("nan"))
    log_residual_std_: float = field(init=False, default=float("nan"))
    feature_mean_: np.ndarray | None = field(init=False, default=None)
    feature_scale_: np.ndarray | None = field(init=False, default=None)
    feature_precision_: np.ndarray | None = field(init=False, default=None)
    training_leverage_max_: float = field(init=False, default=float("nan"))

    def __post_init__(self) -> None:
        if self.form not in ANALYTIC_FORMS:
            raise ValueError(f"form must be one of {ANALYTIC_FORMS}; got {self.form!r}")
        self.name = f"analytic_{self.form}"

    def _time_basis(
        self, static: np.ndarray, ages: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        log_t = np.log(ages)
        if self.form == "linear":
            return np.zeros((ages.size, 0)), log_t
        if self.form == "quadratic":
            return np.zeros((ages.size, 0)), 2.0 * log_t
        if self.form == "power":
            return log_t[:, None], np.zeros_like(log_t)
        if self.form == "power_altitude":
            altitude = np.log10(np.maximum(static[:, _PERIGEE_COLUMN], 1.0))
            return np.column_stack([log_t, log_t * altitude]), np.zeros_like(log_t)
        basis = np.log(np.expm1(ages / self.timescale_days_))
        return basis[:, None], np.zeros_like(log_t)

    def _design(self, static: np.ndarray, ages: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert self.feature_mean_ is not None and self.feature_scale_ is not None
        scaled = (static - self.feature_mean_) / self.feature_scale_
        free, offset = self._time_basis(static, ages)
        return np.hstack([np.ones((ages.size, 1)), scaled, free]), offset

    def fit(self, x: np.ndarray, drift_km: np.ndarray) -> AnalyticGrowthPredictor:
        """Ordinary least squares on ``log(drift)``.

        Raises
        ------
        InsufficientDataError
            With fewer rows than parameters, or when every usable drift is at
            or below the logarithm floor.
        """
        static, ages = _split_design(x)
        y = np.asarray(drift_km, dtype=float).ravel()
        keep = np.isfinite(y) & (y > _LOG_FLOOR_KM)
        static, ages, y = static[keep], ages[keep], y[keep]
        if y.size < len(FEATURE_NAMES) + 4:
            raise InsufficientDataError(
                f"{y.size} usable rows is too few to fit {len(FEATURE_NAMES) + 2} "
                "parameters with any residual degrees of freedom"
            )
        self.feature_mean_ = static.mean(axis=0)
        scale = static.std(axis=0)
        # A feature with no spread in the training split would divide by zero.
        # Replacing its scale by one leaves the column at exactly zero after
        # centring, so the coefficient is unidentified and the least-squares
        # solution puts it at zero -- which is the correct handling of a
        # constant feature and is never a silent NaN.
        self.feature_scale_ = np.where(scale > 0.0, scale, 1.0)

        self._fit_manifold(static)
        if self.form == "drag_exponential":
            self.timescale_days_ = self._fit_timescale(static, ages, y)
        design, offset = self._design(static, ages)
        target = np.log(y) - offset
        solution, *_ = np.linalg.lstsq(design, target, rcond=None)
        self.coefficients_ = solution
        residual = target - design @ solution
        dof = max(1, residual.size - solution.size)
        self.log_residual_std_ = float(np.sqrt(np.sum(residual**2) / dof))
        return self

    def _fit_manifold(self, static: np.ndarray, *, rcond: float = 1e-10) -> None:
        """Record how far the training features reach, so a query can be compared.

        The ten features are strongly collinear by construction: perigee
        altitude, apogee altitude, period, the logarithm of the perigee density
        and the logarithm of the drag parameter are all functions of the same
        orbit and move together. A log-linear model fitted on such data carries
        large coefficients in nearly opposite directions along the collinear
        combinations. On the manifold the training data occupy that is
        harmless; off it, it is catastrophic.

        That is not a theoretical worry. A hand-assembled feature vector with a
        perigee of 1300 km, an apogee 25 km above it and a period of 105.6
        minutes is off-manifold -- a 1300 by 1325 km orbit has a period of 112
        minutes, not 106 -- and this model predicted a divergence of 2.5e8 km
        for it, against the 17.7 km measured for a real object at that
        altitude. The model was not wrong; it was asked about a point no orbit
        occupies.

        :meth:`leverage` is the Mahalanobis distance of a query from the
        training features in their standardised, pseudo-inverted covariance,
        and :attr:`training_leverage_max_` is how far the training data
        themselves reach. Measured on the committed population: the training
        maximum is about 7.5, the held-out objects reach about 8.5, and the
        hand-assembled vectors above score 24 to 259.

        The practical rule, which README.md states: build the feature vector
        with :func:`orbitdrift.features.features_from_satrec` from an actual
        element set. Do not assemble one by hand.
        """
        assert self.feature_mean_ is not None and self.feature_scale_ is not None
        scaled = (static - self.feature_mean_) / self.feature_scale_
        covariance = scaled.T @ scaled / scaled.shape[0]
        self.feature_precision_ = np.linalg.pinv(covariance, rcond=rcond)
        quadratic = np.einsum("ij,jk,ik->i", scaled, self.feature_precision_, scaled)
        self.training_leverage_max_ = float(np.sqrt(np.max(np.maximum(quadratic, 0.0))))

    def leverage(self, x: np.ndarray) -> np.ndarray:
        """Mahalanobis distance of each row's features from the training features.

        Compare against :attr:`training_leverage_max_`. A value several times
        larger means the query is off the manifold the model was fitted on and
        its prediction there is an extrapolation of a collinear fit, not a
        prediction. See :meth:`_fit_manifold` for the measured numbers.
        """
        if self.feature_precision_ is None or self.feature_mean_ is None:
            raise InsufficientDataError("predictor is not fitted")
        assert self.feature_scale_ is not None
        static, _ = _split_design(x)
        scaled = (static - self.feature_mean_) / self.feature_scale_
        quadratic = np.einsum(
            "ij,jk,ik->i", scaled, self.feature_precision_, scaled
        )
        return np.sqrt(np.maximum(quadratic, 0.0))

    def _fit_timescale(self, static: np.ndarray, ages: np.ndarray, y: np.ndarray) -> float:
        low, high = self.timescale_bounds_days
        grid = np.geomspace(low, high, 40)
        best_tau, best_score = high, np.inf
        for tau in grid:
            self.timescale_days_ = float(tau)
            design, offset = self._design(static, ages)
            target = np.log(y) - offset
            solution, *_ = np.linalg.lstsq(design, target, rcond=None)
            score = float(np.var(target - design @ solution))
            if score < best_score:
                best_score, best_tau = score, float(tau)
        return best_tau

    def predict_median_km(self, x: np.ndarray) -> np.ndarray:
        if self.coefficients_ is None:
            raise InsufficientDataError("predictor is not fitted")
        static, ages = _split_design(x)
        design, offset = self._design(static, ages)
        return np.exp(design @ self.coefficients_ + offset)

    def predict_interval_km(
        self, x: np.ndarray, coverage: float = 0.90
    ) -> tuple[np.ndarray, np.ndarray]:
        """Gaussian interval in log space from the training residual spread.

        Nominal, not measured: see :meth:`orbitdrift.growth.GrowthFit.interval_km`
        for the same caveat. The measured coverage of this interval is a row of
        the coverage table.
        """
        from scipy.stats import norm

        z = float(norm.ppf(0.5 + 0.5 * coverage))
        centre = self.predict_median_km(x)
        spread = float(np.exp(z * self.log_residual_std_))
        return centre / spread, centre * spread

    @property
    def effective_exponent(self) -> float:
        """The fitted global time exponent, or ``nan`` where there is not one.

        For ``"power_altitude"`` this is the exponent at the mean of
        ``log10(perigee altitude)`` over the training set, which is a summary
        and not the whole model; :meth:`exponent_at_altitude` gives the rest.
        """
        if self.coefficients_ is None:
            return float("nan")
        if self.form == "linear":
            return 1.0
        if self.form == "quadratic":
            return 2.0
        if self.form == "power":
            return float(self.coefficients_[-1])
        if self.form == "power_altitude":
            assert self.feature_mean_ is not None
            mean_altitude = float(
                np.log10(max(self.feature_mean_[_PERIGEE_COLUMN], 1.0))
            )
            return float(self.coefficients_[-2] + self.coefficients_[-1] * mean_altitude)
        return float("nan")

    def exponent_at_altitude(self, perigee_altitude_km: float) -> float:
        """Time exponent at a given perigee altitude, for ``"power_altitude"``."""
        if self.form != "power_altitude" or self.coefficients_ is None:
            return self.effective_exponent
        altitude = float(np.log10(max(perigee_altitude_km, 1.0)))
        return float(self.coefficients_[-2] + self.coefficients_[-1] * altitude)


@dataclass
class LearnedQuantilePredictor:
    """Gradient-boosted quantile regression on ``log(drift)``.

    Three separate fits -- lower quantile, median, upper quantile -- because
    the quantile loss fits one quantile per model. The hyperparameters are
    small on purpose: the compute budget is two shared cores and the dataset is
    a few thousand rows, and an unbounded search would make the comparison
    against a closed-form least-squares baseline a comparison of search
    budgets. The values used are the defaults below and nothing was tuned on
    the test split.
    """

    lower_quantile: float = 0.05
    upper_quantile: float = 0.95
    n_estimators: int = 300
    max_depth: int = 3
    learning_rate: float = 0.05
    min_samples_leaf: int = 20
    subsample: float = 0.9
    random_state: int = 20261025
    name: str = "learned_gbr_quantile"
    models_: dict[str, GradientBoostingRegressor] = field(default_factory=dict)

    def _make(self, alpha: float | None) -> GradientBoostingRegressor:
        return GradientBoostingRegressor(
            loss="quantile" if alpha is not None else "squared_error",
            alpha=0.5 if alpha is None else alpha,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            min_samples_leaf=self.min_samples_leaf,
            subsample=self.subsample,
            random_state=self.random_state,
        )

    def fit(self, x: np.ndarray, drift_km: np.ndarray) -> LearnedQuantilePredictor:
        matrix = np.atleast_2d(np.asarray(x, dtype=float))
        y = np.asarray(drift_km, dtype=float).ravel()
        keep = np.isfinite(y) & (y > _LOG_FLOOR_KM)
        matrix, y = matrix[keep], np.log(y[keep])
        if y.size < 50:
            raise InsufficientDataError(
                f"{y.size} usable rows is too few for a 300-tree gradient-boosted fit"
            )
        self.models_ = {
            "lower": self._make(self.lower_quantile).fit(matrix, y),
            "median": self._make(0.5).fit(matrix, y),
            "upper": self._make(self.upper_quantile).fit(matrix, y),
        }
        return self

    def predict_median_km(self, x: np.ndarray) -> np.ndarray:
        if not self.models_:
            raise InsufficientDataError("predictor is not fitted")
        return np.exp(self.models_["median"].predict(np.atleast_2d(np.asarray(x, float))))

    def predict_interval_km(
        self, x: np.ndarray, coverage: float = 0.90
    ) -> tuple[np.ndarray, np.ndarray]:
        """The two fitted quantiles, exponentiated.

        ``coverage`` must match the quantile pair the model was fitted with;
        asking for a different coverage raises rather than silently returning
        the interval for another level. The quantile crossing that boundless
        quantile regression permits -- a lower prediction above the upper one
        -- is **not** repaired by sorting. It is reported: a crossing is a
        statement that the fit is not supported at that point, and sorting
        would hide it. :func:`interval_crossing_fraction` counts them and the
        count appears in the coverage table.
        """
        nominal = self.upper_quantile - self.lower_quantile
        if abs(coverage - nominal) > 1e-9:
            raise ValueError(
                f"this predictor was fitted for coverage {nominal:.4f} "
                f"({self.lower_quantile}, {self.upper_quantile}); asked for {coverage}. "
                "Refit with the quantile pair you want rather than reusing these trees."
            )
        if not self.models_:
            raise InsufficientDataError("predictor is not fitted")
        matrix = np.atleast_2d(np.asarray(x, dtype=float))
        return (
            np.exp(self.models_["lower"].predict(matrix)),
            np.exp(self.models_["upper"].predict(matrix)),
        )


@dataclass
class ConformalPredictor:
    """Split-conformal multiplicative interval around any base predictor's median.

    ``base`` is fitted on the first part of the data passed to :meth:`fit` and
    the absolute log residuals of the second part give the half-width. The
    finite-sample quantile index is ``ceil((n+1)(1-alpha))/n``, which is the
    correction that makes the marginal coverage guarantee hold at finite ``n``
    rather than only asymptotically.
    """

    base: object
    coverage: float = 0.90
    calibration_fraction: float = 0.35
    random_state: int = 20261025
    name: str = field(init=False, default="")
    log_half_width_: float = field(init=False, default=float("nan"))
    n_calibration_: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if not 0.0 < self.calibration_fraction < 1.0:
            raise ValueError(
                f"calibration_fraction must be in (0, 1); got {self.calibration_fraction}"
            )
        self.name = f"conformal[{getattr(self.base, 'name', type(self.base).__name__)}]"

    def fit(
        self, x: np.ndarray, drift_km: np.ndarray, groups: np.ndarray | None = None
    ) -> ConformalPredictor:
        """Fit the base predictor and calibrate the half-width.

        ``groups``, when given, is a per-row object identifier and the split is
        made **by group**, so no object appears in both the fitting and the
        calibration part. Splitting by row instead would leak an object's own
        physics into its calibration residual and shrink the interval; that is
        a real effect and ``validation/validate_predictors.py`` reports the
        half-width under both splits so the size of the leak is visible.
        """
        matrix = np.atleast_2d(np.asarray(x, dtype=float))
        y = np.asarray(drift_km, dtype=float).ravel()
        rng = np.random.default_rng(self.random_state)
        if groups is None:
            order = rng.permutation(y.size)
            cut = int(round((1.0 - self.calibration_fraction) * y.size))
            fit_index, calibrate_index = order[:cut], order[cut:]
        else:
            group_values = np.asarray(groups).ravel()
            unique = np.unique(group_values)
            shuffled = rng.permutation(unique)
            cut = int(round((1.0 - self.calibration_fraction) * unique.size))
            fit_groups = set(shuffled[:cut].tolist())
            mask = np.array([value in fit_groups for value in group_values])
            fit_index = np.flatnonzero(mask)
            calibrate_index = np.flatnonzero(~mask)
        if fit_index.size == 0 or calibrate_index.size == 0:
            raise InsufficientDataError(
                f"the conformal split left {fit_index.size} fitting and "
                f"{calibrate_index.size} calibration rows; both must be non-empty"
            )
        self.base.fit(matrix[fit_index], y[fit_index])
        predicted = self.base.predict_median_km(matrix[calibrate_index])
        actual = y[calibrate_index]
        usable = (
            np.isfinite(predicted)
            & (predicted > _LOG_FLOOR_KM)
            & np.isfinite(actual)
            & (actual > _LOG_FLOOR_KM)
        )
        residual = np.abs(np.log(actual[usable]) - np.log(predicted[usable]))
        n = residual.size
        if n == 0:
            raise InsufficientDataError("no usable calibration residuals")
        index = min(n, int(np.ceil((n + 1) * self.coverage)))
        self.log_half_width_ = float(np.sort(residual)[index - 1])
        self.n_calibration_ = int(n)
        return self

    def predict_median_km(self, x: np.ndarray) -> np.ndarray:
        return self.base.predict_median_km(x)

    def predict_interval_km(
        self, x: np.ndarray, coverage: float | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        if coverage is not None and abs(coverage - self.coverage) > 1e-9:
            raise ValueError(
                f"calibrated for coverage {self.coverage}; asked for {coverage}. "
                "Refit rather than rescaling a conformal half-width."
            )
        centre = self.predict_median_km(x)
        factor = float(np.exp(self.log_half_width_))
        return centre / factor, centre * factor


def interval_crossing_fraction(lower: np.ndarray, upper: np.ndarray) -> float:
    """Fraction of rows where ``lower > upper``, which quantile regression permits."""
    low = np.asarray(lower, dtype=float)
    high = np.asarray(upper, dtype=float)
    return float(np.mean(low > high)) if low.size else 0.0
