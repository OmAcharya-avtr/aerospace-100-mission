"""Learned multi-sensor Cn2_path regressor with prediction intervals, and its
comparators.

The learned model maps four observable quantities

    (log10 sigma_I^2 [scintillometer], log10 var_long [DIMM],
     log10 var_trans [DIMM], log10 path_length_m)  ->  log10 Cn2_path

using three ``sklearn.ensemble.GradientBoostingRegressor`` quantile models
(median / lower / upper), the same architecture used in this mission's other
turbulence products (P019 CnCast, P020 AtmoProfile) for consistency, plus
split-conformal calibration (Romano, Patterson & Candes 2019,
"Conformalized Quantile Regression", NeurIPS 32) so the reported interval
coverage is measured, not assumed.

The comparators in this module exist so the learned model is never scored
alone:

* :class:`ScintillometerWeakBaseline` -- the classical closed-form
  single-sensor inversion (:func:`turbscope.scintillometer.invert_cn2_weak`).
  **This is the mission-mandated baseline** the learned model must beat, or
  the loss must be reported plainly (``MODEL_CARD.md``).
* :class:`DimmOnlyBaseline` -- the DIMM-only closed-form inversion, a second
  honest single-sensor comparator that does not share the scintillometer's
  saturation failure mode.
* :class:`MeanTrainingBaseline` -- the mean log10 Cn2 of the training set,
  ignoring every input; the "learned nothing" floor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.ensemble import GradientBoostingRegressor

from .dataset import FEATURE_NAMES, build_table, generate_default_scenarios, grouped_split
from .dimm import invert_cn2_from_variance
from .scintillometer import invert_cn2_weak
from .synthetic import (
    APERTURE_DIAM_M,
    DIMM_WAVELENGTH_M,
    PATH_LENGTH_RANGE_M,
    SCINT_WAVELENGTH_M,
    SEPARATION_M,
    WAVE_TYPE,
)

__all__ = [
    "TRAINING_DOMAIN",
    "DimmOnlyBaseline",
    "MeanTrainingBaseline",
    "PathCn2Prediction",
    "ScintillometerWeakBaseline",
    "TurbScopeModel",
    "interval_coverage",
    "train_default_model",
]

# Feature-domain bounds implied by turbscope.synthetic's draw ranges, used
# only to flag extrapolation -- not enforced as hard limits on prediction.
TRAINING_DOMAIN: dict[str, tuple[float, float]] = {
    "log10_sigma_i2_scint": (-6.0, 2.5),
    "log10_var_long_dimm": (-10.0, 1.0),
    "log10_var_trans_dimm": (-10.0, 1.0),
    "log10_path_length_m": (
        float(np.log10(PATH_LENGTH_RANGE_M[0])),
        float(np.log10(PATH_LENGTH_RANGE_M[1])),
    ),
}
"""Approximate domain covered by the default synthetic training set (for the
``extrapolating`` flag on predictions). Generous, hand-set margins around the
values actually produced by :mod:`turbscope.synthetic`'s draw ranges."""


@dataclass(frozen=True)
class PathCn2Prediction:
    """A predicted path-averaged Cn2 with a prediction interval.

    Attributes
    ----------
    cn2_path : float
        Median prediction, m^-2/3.
    cn2_lower, cn2_upper : float
        Interval bounds, m^-2/3, at nominal ``coverage``.
    coverage : float
        Nominal central coverage (e.g. 0.90). Empirical coverage measured on
        held-out data is in ``validation/VALIDATION.md`` -- not guaranteed to
        equal this number.
    extrapolating : bool
        True if any input feature fell outside :data:`TRAINING_DOMAIN`.
    """

    cn2_path: float
    cn2_lower: float
    cn2_upper: float
    coverage: float
    extrapolating: bool

    @property
    def interval_width_dex(self) -> float:
        """Interval width in decades (log10 upper - log10 lower)."""
        return float(np.log10(self.cn2_upper) - np.log10(self.cn2_lower))


class ScintillometerWeakBaseline:
    """Classical closed-form single-sensor inversion (the mandated baseline).

    Ignores every DIMM feature; inverts the weak-fluctuation formula on the
    scintillometer channel alone, regardless of whether the true regime is
    weak or saturated. See :func:`turbscope.scintillometer.invert_cn2_weak`.
    """

    name = "scintillometer weak inversion"

    def __init__(
        self, path_wavelength_m: float = SCINT_WAVELENGTH_M, wave_type: str = WAVE_TYPE
    ) -> None:
        self.wavelength_m = float(path_wavelength_m)
        self.wave_type = wave_type

    def predict_log10_cn2(self, x: ArrayLike) -> NDArray[np.float64]:
        """Predict log10 Cn2 for a feature matrix in :data:`FEATURE_NAMES` order."""
        xa = np.atleast_2d(np.asarray(x, dtype=float))
        sigma_i2 = 10.0 ** xa[:, 0]
        length = 10.0 ** xa[:, 3]
        out = np.array(
            [
                invert_cn2_weak(float(s), float(length_i), self.wavelength_m, self.wave_type)
                for s, length_i in zip(sigma_i2, length, strict=True)
            ]
        )
        return np.log10(np.maximum(out, 1e-30))


class DimmOnlyBaseline:
    """Classical closed-form single-sensor inversion using DIMM alone
    (average of the longitudinal and transverse channel inversions)."""

    name = "DIMM-only inversion"

    def __init__(
        self,
        dimm_wavelength_m: float = DIMM_WAVELENGTH_M,
        aperture_diam_m: float = APERTURE_DIAM_M,
        separation_m: float = SEPARATION_M,
    ) -> None:
        self.wavelength_m = float(dimm_wavelength_m)
        self.aperture_diam_m = float(aperture_diam_m)
        self.separation_m = float(separation_m)

    def predict_log10_cn2(self, x: ArrayLike) -> NDArray[np.float64]:
        """Predict log10 Cn2 for a feature matrix in :data:`FEATURE_NAMES` order."""
        xa = np.atleast_2d(np.asarray(x, dtype=float))
        var_long = 10.0 ** xa[:, 1]
        var_trans = 10.0 ** xa[:, 2]
        length = 10.0 ** xa[:, 3]
        out = []
        for vl, vt, length_i in zip(var_long, var_trans, length, strict=True):
            cn2_l = invert_cn2_from_variance(
                float(vl), float(length_i), self.wavelength_m, self.aperture_diam_m,
                self.separation_m, "longitudinal",
            )
            cn2_t = invert_cn2_from_variance(
                float(vt), float(length_i), self.wavelength_m, self.aperture_diam_m,
                self.separation_m, "transverse",
            )
            out.append(0.5 * (cn2_l + cn2_t))
        return np.log10(np.maximum(np.asarray(out), 1e-30))


class MeanTrainingBaseline:
    """Mean log10 Cn2 of the training set, ignoring every input.

    The honest "learned nothing" floor: any model that cannot beat this has
    not extracted usable signal from the features.
    """

    name = "training mean"

    def __init__(self) -> None:
        self._mean: float | None = None

    def fit(self, x: ArrayLike, y: ArrayLike) -> MeanTrainingBaseline:
        ya = np.asarray(y, dtype=float)
        if ya.size == 0:
            raise ValueError("y must be non-empty.")
        self._mean = float(np.mean(ya))
        return self

    def predict_log10_cn2(self, x: ArrayLike) -> NDArray[np.float64]:
        if self._mean is None:
            raise RuntimeError("MeanTrainingBaseline.fit must be called before predict.")
        xa = np.atleast_2d(np.asarray(x, dtype=float))
        return np.full(xa.shape[0], self._mean)


class TurbScopeModel:
    """Quantile gradient-boosting Cn2_path predictor with a conformalised
    prediction interval.

    Parameters
    ----------
    coverage : float
        Nominal central coverage, in (0, 1). Default 0.90.
    n_estimators, max_depth, learning_rate, min_samples_leaf :
        Shared ``GradientBoostingRegressor`` hyperparameters, sized for a
        well under 2-minute fit on 2 CPU cores.
    random_state : int
        Seed shared by all three regressors.

    Notes
    -----
    Targets are ``log10 Cn2_path``; Cn2_path spans several decades in the
    training set (:data:`turbscope.synthetic.LOG10_RYTOV_RANGE`), so a
    linear-space loss would be dominated by the largest values.
    """

    def __init__(
        self,
        coverage: float = 0.90,
        n_estimators: int = 250,
        max_depth: int = 3,
        learning_rate: float = 0.08,
        min_samples_leaf: int = 15,
        random_state: int = 11,
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
        self._fitted = False
        self._report: dict[str, float] = {}
        self._conformal_delta = 0.0

    def fit(self, x: ArrayLike, y: ArrayLike) -> TurbScopeModel:
        """Fit the three quantile models.

        Parameters
        ----------
        x : array_like, shape (n, 4)
            Feature matrix, :data:`turbscope.dataset.FEATURE_NAMES` order.
        y : array_like, shape (n,)
            Target ``log10 Cn2_path``.
        """
        xa = np.atleast_2d(np.asarray(x, dtype=float))
        ya = np.asarray(y, dtype=float)
        if xa.ndim != 2 or xa.shape[1] != len(FEATURE_NAMES):
            raise ValueError(f"x must have {len(FEATURE_NAMES)} feature columns (got {xa.shape}).")
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
        """Split-conformal calibration of the interval (CQR); returns delta (dex).

        See :meth:`turbscope.model.TurbScopeModel.calibrate` docstring parity
        with the CnCast/AtmoProfile pattern used elsewhere in this mission for
        the derivation. Calibration data must be disjoint from both fit and
        test data.
        """
        if not self._fitted:
            raise RuntimeError("TurbScopeModel.fit must be called before calibrate.")
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
        """Half-width added to each bound by :meth:`calibrate`, in dex."""
        return self._conformal_delta

    def predict_log10_cn2(self, x: ArrayLike) -> NDArray[np.float64]:
        """Median prediction of log10 Cn2_path for a raw feature matrix."""
        return self._three(x)[1]

    def _three(self, x: ArrayLike) -> tuple[NDArray, NDArray, NDArray]:
        if not self._fitted:
            raise RuntimeError("TurbScopeModel.fit must be called before predict.")
        xa = np.atleast_2d(np.asarray(x, dtype=float))
        if xa.shape[1] != len(FEATURE_NAMES):
            raise ValueError(f"x must have {len(FEATURE_NAMES)} feature columns (got {xa.shape}).")
        lo = self.lower_model.predict(xa) - self._conformal_delta
        mid = self.median_model.predict(xa)
        hi = self.upper_model.predict(xa) + self._conformal_delta
        stacked = np.sort(np.vstack([lo, mid, hi]), axis=0)
        return stacked[0], stacked[1], stacked[2]

    def predict(
        self,
        sigma_i2_scint: float,
        var_long_dimm: float,
        var_trans_dimm: float,
        path_length_m: float,
    ) -> PathCn2Prediction:
        """Predict Cn2_path with a prediction interval from raw sensor readings.

        Parameters
        ----------
        sigma_i2_scint : float
            Measured scintillometer scintillation index (> 0).
        var_long_dimm, var_trans_dimm : float
            Measured DIMM differential variances, rad^2 (> 0).
        path_length_m : float
            Known path length, m (> 0).

        Returns
        -------
        PathCn2Prediction

        Raises
        ------
        ValueError
            On non-positive/non-finite inputs.
        RuntimeError
            If the model has not been fitted.
        """
        for name, val in (
            ("sigma_i2_scint", sigma_i2_scint),
            ("var_long_dimm", var_long_dimm),
            ("var_trans_dimm", var_trans_dimm),
            ("path_length_m", path_length_m),
        ):
            if not np.isfinite(val) or val <= 0.0:
                raise ValueError(f"{name} must be finite and > 0 (got {val!r}).")
        x = np.array(
            [[
                np.log10(sigma_i2_scint),
                np.log10(var_long_dimm),
                np.log10(var_trans_dimm),
                np.log10(path_length_m),
            ]]
        )
        lo, mid, hi = self._three(x)
        dom = TRAINING_DOMAIN
        cols = list(FEATURE_NAMES)
        outside = [
            not dom[cols[i]][0] <= float(x[0, i]) <= dom[cols[i]][1] for i in range(len(cols))
        ]
        return PathCn2Prediction(
            cn2_path=float(10.0 ** mid[0]),
            cn2_lower=float(10.0 ** lo[0]),
            cn2_upper=float(10.0 ** hi[0]),
            coverage=self.coverage,
            extrapolating=any(outside),
        )


def interval_coverage(
    y_true: ArrayLike, lower: ArrayLike, upper: ArrayLike
) -> tuple[float, float]:
    """Empirical coverage and mean interval width of a prediction interval.

    Parameters
    ----------
    y_true, lower, upper : array_like
        Same length; consistent units (log10 Cn2 here).

    Returns
    -------
    (coverage, mean_width) : tuple of float
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
    n_scenarios: int = 900,
    n_realisations: int = 3,
    data_seed: int = 20260829,
    noise_seed: int = 99,
    split_seed: int = 4242,
    test_fraction: float = 0.25,
    calibration_fraction: float = 0.25,
    coverage: float = 0.90,
    random_state: int = 11,
    calibrate: bool = True,
) -> tuple[TurbScopeModel, dict[str, object]]:
    """Generate data, split by scenario, fit and conformally calibrate the model.

    Split strategy, all at *scenario* level (never row level, since
    ``n_realisations`` noisy rows share the same ground truth):
    ``test_fraction`` of scenarios held out for evaluation; of the remainder,
    ``calibration_fraction`` held out again for conformal calibration; the
    rest fit the three quantile regressors.

    Returns
    -------
    (model, artefacts)
        ``artefacts`` holds the scenario lists, the ``x``/``y``/``groups``
        tables for each split, and the seeds used.
    """
    scenarios = generate_default_scenarios(n_scenarios, seed=data_seed)
    train_idx, test_idx = grouped_split(n_scenarios, test_fraction, seed=split_seed)
    train_sc = [scenarios[i] for i in train_idx]
    test_sc = [scenarios[i] for i in test_idx]

    fit_rel, cal_rel = grouped_split(len(train_sc), calibration_fraction, seed=split_seed + 1)
    fit_sc = [train_sc[i] for i in fit_rel]
    cal_sc = [train_sc[i] for i in cal_rel]

    x_fit, y_fit, _ = build_table(fit_sc, n_realisations=n_realisations, seed=noise_seed)
    x_cal, y_cal, _ = build_table(cal_sc, n_realisations=n_realisations, seed=noise_seed + 1)
    x_test, y_test, groups_test = build_table(
        test_sc, n_realisations=n_realisations, seed=noise_seed + 2
    )

    model = TurbScopeModel(coverage=coverage, random_state=random_state).fit(x_fit, y_fit)
    if calibrate:
        model.calibrate(x_cal, y_cal)
    artefacts: dict[str, object] = {
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
        "seeds": {
            "data_seed": data_seed,
            "noise_seed": noise_seed,
            "split_seed": split_seed,
            "model_random_state": random_state,
        },
    }
    return model, artefacts
