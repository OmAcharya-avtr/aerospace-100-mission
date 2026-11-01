"""The learned discrepancy model, and the conformal repair of its coverage.

What is learned and what is not
-------------------------------
The analytic baselines of :mod:`transfergap.bounds` and
:mod:`transfergap.resample` are **marginal**: they bound the discrepancy over
the operating-condition population as a whole. The thing they cannot do is say
that the twin is fine at Mach 0.6 and poor at Mach 0.9. That is what a learned
model is for here: a map from the operating condition ``x`` to the conditional
discrepancy distribution.

The model is a gradient-boosted regression tree ensemble fitted with the
**pinball (quantile) loss**, so that it estimates a conditional quantile
directly rather than a mean with an assumed error law. Scikit-learn's
``GradientBoostingRegressor(loss="quantile", alpha=q)`` minimises

    L_q(y, f) = q (y - f)_+ + (1 - q) (f - y)_+,                         (23)

whose population minimiser is the conditional ``q``-quantile of ``y`` given
``x``. Koenker, R. and Bassett, G. (1978), "Regression quantiles",
*Econometrica* 46(1), 33-50, equation (1.2); the boosting algorithm is
Friedman, J. H. (2001), "Greedy function approximation: a gradient boosting
machine", *Annals of Statistics* 29(5), 1189-1232.

The uncertainty output
----------------------
:meth:`DiscrepancyModel.predict_with_uncertainty` returns, for every operating
condition, a point estimate (the conditional median), a lower and an upper
conditional quantile, and the width between them. That interval **is** the
model's uncertainty statement; the model is never used as a point predictor
anywhere in this package. Three separate ensembles are fitted, one per
quantile, which is why the upper quantile can in principle fall below the
median at a condition where the training data is thin -- quantile crossing.
The crossing rate is measured and reported rather than repaired by sorting,
because sorting hides how thin the data was.

The conformal repair
--------------------
A conditional quantile estimated from tens of points is not a conditional
quantile; it is a fit. Conformalised quantile regression repairs the coverage
by calibrating the model's own output on a held-out split: with
``s_i = y_i - upper(x_i)`` as the nonconformity score and ``qhat`` the
conformal quantile of equation (12), the band ``upper(x) + qhat`` has the
finite-sample marginal guarantee of equation (13) whatever the model does.
Romano, Y., Patterson, E. and Candes, E. J. (2019), "Conformalized quantile
regression", *Advances in Neural Information Processing Systems* 32.

The cost is that the repair is marginal: it adds one constant to every
condition, so it fixes the average coverage and not the conditional coverage.
What it buys is a guarantee; what it does not buy is the thing the model was
fitted for. Both are reported.

Compute
-------
Every model here is fitted from scratch by the script that reports it, from a
declared seed. **No model artifact is serialised or committed**, so there is
no pickle whose format has to be bounded to a library minor version. A fit on
60 training points with the default settings takes a few tens of
milliseconds on the two cores this package was measured on; the Monte Carlo
budget is stated in ``MODEL_CARD.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from .bounds import Bound
from .discrepancy import GapSample, validate_confidence
from .resample import split_conformal_quantile

__all__ = [
    "ConditionalPrediction",
    "ConformalisedDiscrepancyModel",
    "DEFAULT_HYPERPARAMETERS",
    "DiscrepancyModel",
    "gap_sample_split",
    "marginal_bound_from_model",
    "plugin_mean_upper",
]

#: The declared hyperparameters. Small trees and a slow learning rate because
#: the training sets this product is about have tens of rows; the values are
#: declared rather than tuned, and no tuning loop exists in this repository.
DEFAULT_HYPERPARAMETERS: dict[str, float | int] = {
    "n_estimators": 80,
    "learning_rate": 0.12,
    "max_depth": 2,
    "min_samples_leaf": 3,
    "subsample": 1.0,
}


@dataclass(frozen=True)
class ConditionalPrediction:
    """The model's output at a set of operating conditions. Units: kN.m.

    Attributes
    ----------
    point:
        Conditional median discrepancy.
    lower, upper:
        Conditional lower and upper quantiles at the declared levels.
    width:
        ``upper - lower``.
    crossing:
        Boolean mask, ``True`` where ``upper < point`` or ``point < lower``.
        Quantile crossing is a symptom of thin training data at that
        condition. It is reported, not sorted away.
    lower_level, upper_level:
        The quantile levels the two ensembles were fitted at.
    """

    point: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    width: np.ndarray
    crossing: np.ndarray
    lower_level: float
    upper_level: float

    @property
    def crossing_rate(self) -> float:
        """Fraction of conditions at which the fitted quantiles cross."""
        return float(np.mean(self.crossing))


class DiscrepancyModel:
    """Conditional discrepancy quantiles from operating conditions.

    Parameters
    ----------
    upper_level:
        The upper conditional quantile to fit, e.g. ``0.95``.
    lower_level:
        The lower conditional quantile to fit, e.g. ``0.05``.
    seed:
        Seed passed to every ensemble's ``random_state``.
    hyperparameters:
        Overrides for :data:`DEFAULT_HYPERPARAMETERS`.
    fit_full:
        ``True`` fits all three quantile ensembles and
        :meth:`predict_with_uncertainty` works. ``False`` fits only the upper
        ensemble, which is three times cheaper and is what the repeated-draw
        coverage study in ``validation/`` uses, since only the upper band is
        scored. With ``fit_full=False`` the median and lower predictors raise
        rather than returning the upper band under another name.
    """

    def __init__(
        self,
        upper_level: float = 0.95,
        lower_level: float = 0.05,
        seed: int = 0,
        hyperparameters: dict[str, float | int] | None = None,
        fit_full: bool = True,
    ) -> None:
        self.upper_level = validate_confidence(upper_level, "upper_level")
        self.lower_level = validate_confidence(lower_level, "lower_level")
        if self.lower_level >= self.upper_level:
            raise ValueError(
                f"lower_level must be below upper_level; got {lower_level} and "
                f"{upper_level}"
            )
        self.seed = int(seed)
        self.hyperparameters = {**DEFAULT_HYPERPARAMETERS, **(hyperparameters or {})}
        self.fit_full = bool(fit_full)
        self._models: dict[str, GradientBoostingRegressor] = {}
        self._n_features = 0
        self._n_train = 0

    @property
    def fitted(self) -> bool:
        """Whether :meth:`fit` has been called."""
        return bool(self._models)

    @property
    def n_train(self) -> int:
        """Number of training rows used by the last :meth:`fit`."""
        return self._n_train

    def _ensemble(self, level: float) -> GradientBoostingRegressor:
        return GradientBoostingRegressor(
            loss="quantile",
            alpha=float(level),
            random_state=self.seed,
            **self.hyperparameters,
        )

    def fit(self, features: np.ndarray, gaps: np.ndarray) -> DiscrepancyModel:
        """Fit three quantile ensembles: lower, median and upper.

        Raises
        ------
        ValueError
            If there are fewer than 5 training rows. Three gradient-boosted
            ensembles cannot be fitted to four points in any sense worth
            reporting, and a model that returns a number anyway is the defect
            this package exists to measure.
        """
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(gaps, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError(f"features must be two-dimensional; got shape {x.shape}")
        if y.ndim != 1 or y.size != x.shape[0]:
            raise ValueError(
                f"gaps must be one-dimensional with one entry per feature row; got "
                f"{y.shape} against {x.shape}"
            )
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError("features and gaps must both be finite")
        if y.size < 5:
            raise ValueError(
                f"the learned model needs at least 5 training rows; got {y.size}. Use "
                f"the distribution-free bounds of transfergap.bounds at this sample size."
            )
        self._n_features = int(x.shape[1])
        self._n_train = int(y.size)
        wanted = (
            (("lower", self.lower_level), ("median", 0.5), ("upper", self.upper_level))
            if self.fit_full
            else (("upper", self.upper_level),)
        )
        for key, level in wanted:
            self._models[key] = self._ensemble(level).fit(x, y)
        return self

    def _check_ready(self, features: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("the model is not fitted; call fit() first")
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self._n_features:
            raise ValueError(
                f"features must have shape (m, {self._n_features}); got {x.shape}"
            )
        if not np.all(np.isfinite(x)):
            raise ValueError("features must be finite")
        return x

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Conditional median discrepancy at each row, kN.m."""
        x = self._check_ready(features)
        if "median" not in self._models:
            raise RuntimeError(
                "this model was fitted with fit_full=False, so only predict_upper is "
                "available; refit with fit_full=True for the median and the interval"
            )
        return np.asarray(self._models["median"].predict(x))

    def predict_upper(self, features: np.ndarray) -> np.ndarray:
        """Conditional upper quantile at :attr:`upper_level`, kN.m.

        This is a model output, not a bound. Its achieved coverage at field
        sample sizes is measured in
        ``validation/validate_learned_model_output.txt`` and is the number
        this product is scored on.
        """
        x = self._check_ready(features)
        return np.asarray(self._models["upper"].predict(x))

    def predict_with_uncertainty(self, features: np.ndarray) -> ConditionalPrediction:
        """Point estimate, conditional interval and crossing mask. Units kN.m."""
        x = self._check_ready(features)
        if "median" not in self._models or "lower" not in self._models:
            raise RuntimeError(
                "this model was fitted with fit_full=False; refit with fit_full=True "
                "for the uncertainty output"
            )
        point = np.asarray(self._models["median"].predict(x))
        lower = np.asarray(self._models["lower"].predict(x))
        upper = np.asarray(self._models["upper"].predict(x))
        return ConditionalPrediction(
            point=point,
            lower=lower,
            upper=upper,
            width=upper - lower,
            crossing=(upper < point) | (point < lower),
            lower_level=self.lower_level,
            upper_level=self.upper_level,
        )


class ConformalisedDiscrepancyModel:
    """A :class:`DiscrepancyModel` whose upper band is calibrated by split conformal.

    The band is ``model.predict_upper(x) + qhat`` with ``qhat`` the conformal
    quantile of equation (12) over the calibration scores
    ``s_i = y_i - model.predict_upper(x_i)``. Under exchangeability of the
    calibration and test points the band has the marginal coverage guarantee
    of equation (13) whatever the underlying model does, including when the
    underlying model is useless.

    Parameters
    ----------
    confidence:
        ``1 - alpha``, the marginal coverage to guarantee.
    calibration_fraction:
        Fraction of the training sample held out for calibration. The
        trade-off is explicit and is reported: a larger calibration split
        makes ``qhat`` less noisy and the fitted model worse.
    seed:
        Seed for the split and for the underlying ensembles.
    """

    def __init__(
        self,
        confidence: float = 0.95,
        calibration_fraction: float = 0.5,
        seed: int = 0,
        hyperparameters: dict[str, float | int] | None = None,
        fit_full: bool = True,
    ) -> None:
        self.confidence = validate_confidence(confidence)
        self.calibration_fraction = validate_confidence(
            calibration_fraction, "calibration_fraction"
        )
        self.seed = int(seed)
        self.hyperparameters = hyperparameters
        self.model = DiscrepancyModel(
            upper_level=self.confidence, lower_level=1.0 - self.confidence,
            seed=self.seed, hyperparameters=hyperparameters, fit_full=fit_full,
        )
        self.offset = float("nan")
        self.n_calibration = 0
        self.n_fit = 0

    def fit(self, features: np.ndarray, gaps: np.ndarray) -> ConformalisedDiscrepancyModel:
        """Split, fit the model on one part and calibrate the offset on the other."""
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(gaps, dtype=np.float64)
        if y.ndim != 1 or x.ndim != 2 or y.size != x.shape[0]:
            raise ValueError(
                f"features {x.shape} and gaps {y.shape} do not describe the same rows"
            )
        n = y.size
        n_calibration = int(round(self.calibration_fraction * n))
        if n_calibration < 1 or n - n_calibration < 5:
            raise ValueError(
                f"a conformalised fit needs at least 5 rows to fit and 1 to calibrate; "
                f"n = {n} with calibration_fraction = {self.calibration_fraction} gives "
                f"{n - n_calibration} and {n_calibration}"
            )
        rng = np.random.default_rng(self.seed)
        order = rng.permutation(n)
        calibration_index = order[:n_calibration]
        fit_index = order[n_calibration:]
        self.model.fit(x[fit_index], y[fit_index])
        scores = y[calibration_index] - self.model.predict_upper(x[calibration_index])
        self.offset = split_conformal_quantile(scores, 1.0 - self.confidence)
        self.n_calibration = int(n_calibration)
        self.n_fit = int(n - n_calibration)
        return self

    def predict_upper(self, features: np.ndarray) -> np.ndarray:
        """The calibrated conditional upper band, kN.m."""
        if not np.isfinite(self.offset):
            raise RuntimeError("the conformalised model is not fitted; call fit() first")
        return self.model.predict_upper(features) + self.offset


def marginal_bound_from_model(
    model: DiscrepancyModel,
    conditions: np.ndarray,
    proportion: float,
    confidence: float,
    unit: str = "",
) -> Bound:
    """The model's marginal upper bound: the ``proportion`` quantile of its band.

    The learned model is conditional and the analytic baselines are marginal,
    so a comparison needs one of them moved. This moves the model: it is
    evaluated over a large pool of operating conditions drawn from the same
    envelope and the ``proportion`` quantile of its conditional upper band is
    taken as a marginal limit.

    The returned :attr:`Bound.confidence` is the level the model's quantile
    was FITTED at. It is not a confidence statement -- nothing guarantees it --
    and :attr:`Bound.distribution_free` is ``False``. The field exists so
    that the model appears in the same table as the bounds that do carry a
    guarantee, with the difference visible in the row rather than in a
    footnote.
    """
    band = model.predict_upper(conditions)
    limit = float(np.quantile(band, validate_confidence(proportion, "proportion")))
    return Bound(
        method="learned_quantile",
        target="quantile",
        estimate=float(np.median(band)),
        upper=limit,
        slack=limit - float(np.median(band)),
        confidence=validate_confidence(confidence),
        proportion=float(proportion),
        n=int(model.n_train),
        unit=unit,
        distribution_free=False,
    )


def gap_sample_split(
    sample: GapSample, calibration_fraction: float, seed: int
) -> tuple[GapSample, GapSample]:
    """Split a :class:`GapSample` into a fit part and a calibration part."""
    fraction = validate_confidence(calibration_fraction, "calibration_fraction")
    n = sample.size
    n_calibration = int(round(fraction * n))
    if n_calibration < 1 or n - n_calibration < 1:
        raise ValueError(
            f"a split of {n} rows at fraction {fraction} leaves "
            f"{n - n_calibration} and {n_calibration}"
        )
    order = np.random.default_rng(int(seed)).permutation(n)
    return sample.subset(order[n_calibration:]), sample.subset(order[:n_calibration])


def plugin_mean_upper(
    features: np.ndarray,
    gaps: np.ndarray,
    conditions: np.ndarray,
    confidence: float = 0.95,
    seed: int = 0,
    unit: str = "",
    hyperparameters: dict[str, float | int] | None = None,
) -> Bound:
    """A learned upper bound on the MEAN gap, for comparison with equations (3) and (5).

    The construction is the obvious one and the one a programme would reach
    for: fit a conditional-mean model of the discrepancy, average its
    prediction over the operating-condition population, and attach a normal
    interval built from the model's own residual spread,

        upper = mean_x[ f(x) ] + z_{1-delta} * s_resid / sqrt(n).          (29)

    Every step of equation (29) is defensible and the result has no guarantee
    whatever. ``s_resid`` is an IN-SAMPLE residual standard deviation, so it
    understates the error of a model fitted to tens of points; the normal
    multiplier assumes a sampling distribution nothing has established; and
    the average over the pool ignores the model's own bias entirely. It is
    included because its measured coverage is the point: see
    ``validation/validate_learned_model_output.txt``.

    Parameters
    ----------
    features, gaps:
        The field sample: ``(n, p)`` operating conditions and ``(n,)``
        discrepancies in ``unit``.
    conditions:
        ``(m, p)`` operating conditions drawn from the population the mean is
        taken over.
    confidence:
        ``1 - delta``.
    seed:
        ``random_state`` for the ensemble.
    unit:
        Unit string carried into the result.
    """
    from scipy import stats

    level = validate_confidence(confidence)
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(gaps, dtype=np.float64)
    if y.ndim != 1 or x.ndim != 2 or y.size != x.shape[0]:
        raise ValueError(f"features {x.shape} and gaps {y.shape} do not describe the same rows")
    if y.size < 5:
        raise ValueError(f"the plug-in mean model needs at least 5 rows; got {y.size}")
    model = GradientBoostingRegressor(
        loss="squared_error",
        random_state=int(seed),
        **{**DEFAULT_HYPERPARAMETERS, **(hyperparameters or {})},
    ).fit(x, y)
    residual_sd = float(np.std(y - model.predict(x), ddof=1))
    estimate = float(np.mean(model.predict(np.asarray(conditions, dtype=np.float64))))
    slack = float(stats.norm.ppf(level)) * residual_sd / math.sqrt(y.size)
    return Bound(
        method="learned_mean_plugin",
        target="mean",
        estimate=estimate,
        upper=estimate + slack,
        slack=slack,
        confidence=level,
        proportion=None,
        n=int(y.size),
        unit=unit,
        distribution_free=False,
    )
