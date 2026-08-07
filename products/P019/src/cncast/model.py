"""Learned Cn^2 profile model with prediction intervals, and its comparators.

The learned model maps surface meteorology + altitude to log10 Cn^2:

    (T_surface, wind_surface, RH, hour-of-day, day-of-year, log10 h)  ->  log10 Cn^2

Three ``sklearn.ensemble.GradientBoostingRegressor`` instances are fitted:

* ``median``: quantile loss, alpha = 0.50 (the point prediction),
* ``lower`` : quantile loss, alpha = (1 - coverage)/2,
* ``upper`` : quantile loss, alpha = 1 - (1 - coverage)/2,

so the interval is a genuine conditional-quantile interval and not a Gaussian
assumption bolted onto a point estimate.  Quantile crossing (lower > upper at
some query point) is possible with independently fitted quantile models; it is
detected and repaired by sorting, and the repair rate is reported by
:meth:`CnCastModel.fit_report`.

The comparators in this module exist so that the learned model is never scored
alone:

* :class:`Hv57Baseline`     - the published HV 5/7 climatology (the mandated baseline),
* :class:`SlcBaseline`      - published SLC-Day/SLC-Night, switched on hour of day,
* :class:`ClimatologyBaseline` - the mean training profile, a data-driven but
  non-learned reference that is deliberately hard to beat.

THE MODEL IS TRAINED ON SYNTHETIC DATA (``DATASET_CARD.md``).  It is not
certified for operational flight use.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.ensemble import GradientBoostingRegressor

from .baselines import hv57, slc_day, slc_night
from .dataset import (
    ALTITUDE_MAX_M,
    ALTITUDE_MIN_M,
    Scenario,
    build_table,
    generate_scenarios,
    met_features,
    split_scenarios,
)

__all__ = [
    "ClimatologyBaseline",
    "CnCastModel",
    "Hv57Baseline",
    "ProfilePrediction",
    "SlcBaseline",
    "TRAINING_DOMAIN",
    "interval_coverage",
    "train_default_model",
]

TRAINING_DOMAIN: dict[str, tuple[float, float]] = {
    "surface_temp_c": (-10.0, 38.0),
    "surface_wind_m_s": (0.0, 14.0),
    "relative_humidity_pct": (10.0, 95.0),
    "altitude_m": (ALTITUDE_MIN_M, ALTITUDE_MAX_M),
}
"""Domain covered by the default synthetic training set.  Queries outside it are
flagged ``extrapolating``; hour-of-day and day-of-year are cyclic and fully
covered, so they are not listed."""


@dataclass(frozen=True)
class ProfilePrediction:
    """A predicted Cn^2 profile with a prediction interval.

    Attributes
    ----------
    altitude_m : ndarray
        Query altitudes, m.
    cn2 : ndarray
        Median prediction, m^-2/3.
    cn2_lower, cn2_upper : ndarray
        Interval bounds, m^-2/3, at nominal ``coverage``.
    coverage : float
        Nominal central coverage (e.g. 0.90).  Empirical coverage measured on
        held-out data is reported in ``validation/VALIDATION.md`` §4 - it is
        NOT guaranteed to equal this number.
    extrapolating : bool
        True if any input fell outside :data:`TRAINING_DOMAIN`.
    """

    altitude_m: NDArray[np.float64]
    cn2: NDArray[np.float64]
    cn2_lower: NDArray[np.float64]
    cn2_upper: NDArray[np.float64]
    coverage: float
    extrapolating: bool

    @property
    def interval_width_dex(self) -> NDArray[np.float64]:
        """Interval width in decades (log10 upper - log10 lower)."""
        return np.log10(self.cn2_upper) - np.log10(self.cn2_lower)


class Hv57Baseline:
    """Hufnagel-Valley 5/7 climatology as a predictor (ignores all inputs).

    This is the mandated analytic baseline.  It has no meteorological
    dependence whatsoever: HV 5/7 is a single fixed curve, so its prediction is
    the same at noon in July and at midnight in January.  That is precisely the
    gap the learned model is asked to fill, and it is also why beating HV 5/7 on
    this dataset is a weak result (see ``MODEL_CARD.md`` §7).
    """

    name = "HV 5/7"

    def predict_log10_cn2(self, x: ArrayLike) -> NDArray[np.float64]:
        """Predict log10 Cn^2 for a feature matrix in ``FEATURE_NAMES`` order."""
        xa = np.atleast_2d(np.asarray(x, dtype=float))
        return np.log10(hv57(10.0 ** xa[:, 7]))


class SlcBaseline:
    """SLC-Day / SLC-Night climatology, switched on hour of day.

    Day is taken as 07:00-17:00 local solar time (SLC-Day), otherwise SLC-Night.
    The hour is recovered from the cyclic (sin, cos) features.  Unlike HV 5/7
    this baseline does react to one input, so it is a fairer comparator for the
    diurnal part of the problem.  It is still a site-specific climatology
    (AMOS/Haleakala) applied to a generic site.
    """

    name = "SLC day/night"

    def predict_log10_cn2(self, x: ArrayLike) -> NDArray[np.float64]:
        """Predict log10 Cn^2 for a feature matrix in ``FEATURE_NAMES`` order."""
        xa = np.atleast_2d(np.asarray(x, dtype=float))
        hour = np.mod(np.arctan2(xa[:, 3], xa[:, 4]) * 24.0 / (2.0 * np.pi), 24.0)
        h = 10.0 ** xa[:, 7]
        is_day = (hour >= 7.0) & (hour < 17.0)
        cn2 = np.where(is_day, slc_day(h), slc_night(h))
        # SLC profiles are exactly 0 above their validity ceiling; floor them so
        # the log-domain metric stays finite. The floor is stated, not hidden.
        return np.log10(np.maximum(cn2, 1e-21))


class ClimatologyBaseline:
    """Mean log10 Cn^2 profile of the training set, interpolated in log-altitude.

    Non-learned in the sense that it uses no input features other than altitude,
    but it is fitted to the training data, so it encodes the dataset's own
    climatology.  It is the honest "do nothing clever" reference: any learned
    model that cannot beat it has learned nothing about meteorology.
    """

    name = "training climatology"

    def __init__(self, n_bins: int = 40) -> None:
        if int(n_bins) < 2:
            raise ValueError("n_bins must be >= 2.")
        self.n_bins = int(n_bins)
        self._edges: NDArray[np.float64] | None = None
        self._values: NDArray[np.float64] | None = None

    def fit(self, x: ArrayLike, y: ArrayLike) -> ClimatologyBaseline:
        """Fit the altitude-binned mean of ``y`` (log10 Cn^2)."""
        xa = np.atleast_2d(np.asarray(x, dtype=float))
        ya = np.asarray(y, dtype=float)
        if xa.shape[0] != ya.size:
            raise ValueError("x and y must have matching first dimension.")
        logh = xa[:, 7]
        edges = np.linspace(np.log10(ALTITUDE_MIN_M), np.log10(ALTITUDE_MAX_M), self.n_bins + 1)
        idx = np.clip(np.digitize(logh, edges) - 1, 0, self.n_bins - 1)
        means = np.array(
            [ya[idx == b].mean() if np.any(idx == b) else np.nan for b in range(self.n_bins)]
        )
        good = np.isfinite(means)
        if not np.any(good):
            raise ValueError("No usable training rows for the climatology baseline.")
        centres = 0.5 * (edges[:-1] + edges[1:])
        self._edges = centres[good]
        self._values = means[good]
        return self

    def predict_log10_cn2(self, x: ArrayLike) -> NDArray[np.float64]:
        """Predict log10 Cn^2 by linear interpolation of the binned means."""
        if self._edges is None or self._values is None:
            raise RuntimeError("ClimatologyBaseline.fit must be called before predict.")
        xa = np.atleast_2d(np.asarray(x, dtype=float))
        return np.interp(xa[:, 7], self._edges, self._values)


class CnCastModel:
    """Quantile gradient-boosting Cn^2 profile predictor.

    Parameters
    ----------
    coverage : float
        Nominal central coverage of the prediction interval, in (0, 1).
        Default 0.90 -> the 5th and 95th conditional percentiles.
    n_estimators, max_depth, learning_rate, min_samples_leaf :
        ``GradientBoostingRegressor`` hyperparameters, shared by all three
        sub-models.  Defaults chosen for a <2 min fit on 2 CPU cores.
    random_state : int
        Seed for all three regressors; fits are deterministic.

    Notes
    -----
    Targets are ``log10 Cn^2``.  Cn^2 spans roughly 1e-21 to 1e-13 m^-2/3 over
    0-20 km, so any loss in linear units would be dominated entirely by the
    surface layer.  Because ``10**x`` is monotone, quantiles in log space map
    exactly to quantiles in linear space - no Jensen correction is needed for
    the interval bounds (it *would* be needed to turn the median into a mean).
    """

    def __init__(
        self,
        coverage: float = 0.90,
        n_estimators: int = 300,
        max_depth: int = 4,
        learning_rate: float = 0.06,
        min_samples_leaf: int = 20,
        random_state: int = 7,
    ) -> None:
        c = float(coverage)
        if not 0.0 < c < 1.0:
            raise ValueError(f"coverage must be in (0, 1) (got {coverage!r}).")
        self.coverage = c
        alpha_lo = 0.5 * (1.0 - c)
        common = {
            "n_estimators": int(n_estimators),
            "max_depth": int(max_depth),
            "learning_rate": float(learning_rate),
            "min_samples_leaf": int(min_samples_leaf),
            "random_state": int(random_state),
        }
        self.median_model = GradientBoostingRegressor(loss="quantile", alpha=0.5, **common)
        self.lower_model = GradientBoostingRegressor(loss="quantile", alpha=alpha_lo, **common)
        self.upper_model = GradientBoostingRegressor(
            loss="quantile", alpha=1.0 - alpha_lo, **common
        )
        self.random_state = int(random_state)
        self._fitted = False
        self._report: dict[str, float] = {}
        self._conformal_delta: float = 0.0

    # -- fitting -----------------------------------------------------------
    def fit(self, x: ArrayLike, y: ArrayLike) -> CnCastModel:
        """Fit the three quantile models on features ``x`` and target ``y``.

        Parameters
        ----------
        x : array_like, shape (n, 8)
            Feature matrix in :data:`cncast.dataset.FEATURE_NAMES` order.
        y : array_like, shape (n,)
            Target ``log10 Cn^2``.
        """
        xa = np.atleast_2d(np.asarray(x, dtype=float))
        ya = np.asarray(y, dtype=float)
        if xa.ndim != 2 or xa.shape[1] != 8:
            raise ValueError(f"x must have 8 feature columns (got shape {xa.shape}).")
        if ya.ndim != 1 or ya.size != xa.shape[0]:
            raise ValueError("y must be 1-D with one entry per row of x.")
        if not np.all(np.isfinite(xa)) or not np.all(np.isfinite(ya)):
            raise ValueError("x and y must be finite.")
        self.median_model.fit(xa, ya)
        self.lower_model.fit(xa, ya)
        self.upper_model.fit(xa, ya)
        self._fitted = True
        lo = self.lower_model.predict(xa)
        hi = self.upper_model.predict(xa)
        self._report = {
            "n_rows": float(xa.shape[0]),
            "quantile_crossing_fraction": float(np.mean(lo > hi)),
        }
        return self

    def fit_report(self) -> dict[str, float]:
        """Diagnostics from the last fit (rows used, quantile-crossing rate)."""
        if not self._fitted:
            raise RuntimeError("Model is not fitted.")
        return dict(self._report)

    def calibrate(self, x_cal: ArrayLike, y_cal: ArrayLike) -> float:
        r"""Conformalise the interval on a held-out calibration set (CQR).

        Split conformal calibration for quantile regression, Romano, Patterson &
        Candes (2019), "Conformalized Quantile Regression", *NeurIPS 32*:

        .. math::

            E_i = \max(\hat q_{lo}(x_i) - y_i,\; y_i - \hat q_{hi}(x_i)), \qquad
            \delta = \text{the } \lceil (n{+}1)(1-\alpha)\rceil / n
            \text{ empirical quantile of } \{E_i\}

        and the reported interval becomes ``[q_lo - delta, q_hi + delta]``.
        A positive ``delta`` widens (the raw quantile models under-cover on
        unseen data); a negative ``delta`` narrows.

        The calibration data MUST be disjoint from both the fitting data and the
        test data, and - because rows within one profile are strongly dependent -
        must be split by *scenario*.  :func:`train_default_model` does this.

        Parameters
        ----------
        x_cal, y_cal : array_like
            Calibration features (n, 8) and targets (n,), in log10 Cn^2.

        Returns
        -------
        float
            The applied ``delta`` in dex.

        Notes
        -----
        The finite-sample coverage guarantee of split conformal assumes
        exchangeable calibration and test points.  Here both are drawn from the
        same synthetic generator, so the assumption holds *by construction* -
        which is exactly why the resulting coverage guarantee does not transfer
        to the real atmosphere.
        """
        if not self._fitted:
            raise RuntimeError("CnCastModel.fit must be called before calibrate.")
        xa = np.atleast_2d(np.asarray(x_cal, dtype=float))
        ya = np.asarray(y_cal, dtype=float)
        if xa.shape[0] != ya.size or ya.ndim != 1:
            raise ValueError("x_cal and y_cal must have matching first dimension.")
        if ya.size < 20:
            raise ValueError("Need at least 20 calibration points for a usable quantile.")
        self._conformal_delta = 0.0
        lo, _, hi = self._three(xa)
        scores = np.maximum(lo - ya, ya - hi)
        n = ya.size
        level = min(1.0, np.ceil((n + 1) * self.coverage) / n)
        self._conformal_delta = float(np.quantile(scores, level, method="higher"))
        self._report["conformal_delta_dex"] = self._conformal_delta
        return self._conformal_delta

    @property
    def conformal_delta_dex(self) -> float:
        """Half-width added to each interval bound by :meth:`calibrate`, in dex."""
        return self._conformal_delta

    # -- prediction --------------------------------------------------------
    def predict_log10_cn2(self, x: ArrayLike) -> NDArray[np.float64]:
        """Median prediction of ``log10 Cn^2`` for a raw feature matrix."""
        return self._three(x)[1]

    def _three(self, x: ArrayLike) -> tuple[NDArray, NDArray, NDArray]:
        if not self._fitted:
            raise RuntimeError("CnCastModel.fit must be called before predict.")
        xa = np.atleast_2d(np.asarray(x, dtype=float))
        if xa.shape[1] != 8:
            raise ValueError(f"x must have 8 feature columns (got shape {xa.shape}).")
        lo = self.lower_model.predict(xa) - self._conformal_delta
        mid = self.median_model.predict(xa)
        hi = self.upper_model.predict(xa) + self._conformal_delta
        # Repair quantile crossing by sorting the triple pointwise.
        stacked = np.sort(np.vstack([lo, mid, hi]), axis=0)
        return stacked[0], stacked[1], stacked[2]

    def predict(
        self,
        surface_temp_c: float,
        surface_wind_m_s: float,
        relative_humidity_pct: float,
        hour_of_day: float,
        day_of_year: int,
        altitude_m: ArrayLike,
    ) -> ProfilePrediction:
        """Predict a Cn^2 profile with prediction interval from surface weather.

        Parameters
        ----------
        surface_temp_c : float
            Surface air temperature, degrees Celsius.
        surface_wind_m_s : float
            Surface wind speed, m/s.
        relative_humidity_pct : float
            Relative humidity, per cent.
        hour_of_day : float
            Local solar time, hours in [0, 24).
        day_of_year : int
            1-365.
        altitude_m : array_like
            Altitudes above the site, metres (> 0).

        Returns
        -------
        ProfilePrediction

        Raises
        ------
        ValueError
            On physically invalid inputs (see :func:`cncast.dataset.met_features`).
        RuntimeError
            If the model has not been fitted.
        """
        h = np.atleast_1d(np.asarray(altitude_m, dtype=float))
        x = met_features(
            surface_temp_c,
            surface_wind_m_s,
            relative_humidity_pct,
            hour_of_day,
            day_of_year,
            h,
        )
        lo, mid, hi = self._three(x)
        dom = TRAINING_DOMAIN
        outside = [
            not dom["surface_temp_c"][0] <= float(surface_temp_c) <= dom["surface_temp_c"][1],
            not dom["surface_wind_m_s"][0] <= float(surface_wind_m_s) <= dom["surface_wind_m_s"][1],
            not dom["relative_humidity_pct"][0]
            <= float(relative_humidity_pct)
            <= dom["relative_humidity_pct"][1],
            bool(np.any(h < dom["altitude_m"][0])),
            bool(np.any(h > dom["altitude_m"][1])),
        ]
        extrapolating = any(outside)
        return ProfilePrediction(
            altitude_m=h,
            cn2=10.0**mid,
            cn2_lower=10.0**lo,
            cn2_upper=10.0**hi,
            coverage=self.coverage,
            extrapolating=extrapolating,
        )

    def predict_scenario(self, scenario: Scenario, altitude_m: ArrayLike) -> ProfilePrediction:
        """Convenience wrapper: predict from a :class:`~cncast.dataset.Scenario`.

        Only the five observable meteorological fields are used; the latent
        generator state of the scenario is ignored, as it must be.
        """
        return self.predict(
            scenario.surface_temp_c,
            scenario.surface_wind_m_s,
            scenario.relative_humidity_pct,
            scenario.hour_of_day,
            scenario.day_of_year,
            altitude_m,
        )


def interval_coverage(
    y_true: ArrayLike, lower: ArrayLike, upper: ArrayLike
) -> tuple[float, float]:
    """Empirical coverage and mean interval width of a prediction interval.

    Parameters
    ----------
    y_true, lower, upper : array_like
        Same length; all in the same units (log10 Cn^2 here).

    Returns
    -------
    (coverage, mean_width) : tuple of float
        Fraction of ``y_true`` inside ``[lower, upper]`` and the mean width.
    """
    yt = np.asarray(y_true, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    if not (yt.shape == lo.shape == hi.shape):
        raise ValueError("y_true, lower and upper must have the same shape.")
    if yt.size == 0:
        raise ValueError("Cannot compute coverage from empty arrays.")
    inside = (yt >= lo) & (yt <= hi)
    return float(np.mean(inside)), float(np.mean(hi - lo))


def train_default_model(
    n_scenarios: int = 700,
    n_altitudes: int = 28,
    data_seed: int = 20260807,
    altitude_seed: int = 99,
    split_seed: int = 4242,
    test_fraction: float = 0.25,
    calibration_fraction: float = 0.25,
    coverage: float = 0.90,
    random_state: int = 7,
    calibrate: bool = True,
) -> tuple[CnCastModel, dict[str, object]]:
    """Generate data, split by scenario, fit and conformally calibrate the model.

    This is the single entry point used by the validation scripts, the examples
    and the CLI, so every reported number comes from the same recipe.

    Split strategy (all at *scenario* level, never row level):
    ``test_fraction`` of scenarios are held out for evaluation; of the
    remainder, ``calibration_fraction`` are held out again for the conformal
    calibration of the interval, and the rest are used to fit the three quantile
    regressors.  With the defaults that is 525/700 non-test scenarios ->
    394 fit + 131 calibration, and 175 test.

    Parameters
    ----------
    calibrate : bool
        If False, skip conformal calibration (raw quantile-model intervals).

    Returns
    -------
    (model, artefacts)
        ``artefacts`` holds the scenario lists and the ``x``/``y`` tables for
        the fit, calibration and test splits (test rows on the shared
        :func:`cncast.dataset.default_altitude_grid`) plus the seeds used.
    """
    scenarios = generate_scenarios(n_scenarios, seed=data_seed)
    train_idx, test_idx = split_scenarios(n_scenarios, test_fraction, seed=split_seed)
    train_sc = [scenarios[i] for i in train_idx]
    test_sc = [scenarios[i] for i in test_idx]

    fit_rel, cal_rel = split_scenarios(len(train_sc), calibration_fraction, seed=split_seed + 1)
    fit_sc = [train_sc[i] for i in fit_rel]
    cal_sc = [train_sc[i] for i in cal_rel]

    x_fit, y_fit, _ = build_table(fit_sc, n_altitudes=n_altitudes, seed=altitude_seed)
    x_cal, y_cal, _ = build_table(cal_sc, n_altitudes=n_altitudes, seed=altitude_seed + 1)
    x_test, y_test, groups_test = build_table(test_sc, n_altitudes=n_altitudes)

    model = CnCastModel(coverage=coverage, random_state=random_state).fit(x_fit, y_fit)
    if calibrate:
        model.calibrate(x_cal, y_cal)
    artefacts: dict[str, object] = {
        "train_scenarios": train_sc,
        "fit_scenarios": fit_sc,
        "calibration_scenarios": cal_sc,
        "test_scenarios": test_sc,
        "x_train": x_fit,
        "y_train": y_fit,
        "x_cal": x_cal,
        "y_cal": y_cal,
        "x_test": x_test,
        "y_test": y_test,
        "groups_test": groups_test,
        "n_altitudes": n_altitudes,
        "seeds": {
            "data_seed": data_seed,
            "altitude_seed": altitude_seed,
            "split_seed": split_seed,
            "model_random_state": random_state,
        },
    }
    return model, artefacts
