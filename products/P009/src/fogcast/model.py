"""Gradient-boosting attenuation model with 90 % prediction intervals.

The model predicts log10(specific attenuation, dB/km) from
(visibility_km, wavelength_nm, rh_percent) using three scikit-learn
``GradientBoostingRegressor`` instances:

- point model  : squared-error loss (conditional-mean estimate in log space),
- lower model  : quantile loss, alpha = 0.05,
- upper model  : quantile loss, alpha = 0.95,

giving a nominal 90 % prediction interval. Predictions are transformed back with
10**y; because 10**y is monotone, quantiles in log space map to quantiles in dB/km.

Features: [log10(V_km), lambda_um, RH_percent]. Targets and predictions in dB/km.
All seeds fixed; training on the default 6000-sample dataset takes well under one
minute on 2 CPU cores.

The model is trained on SYNTHETIC data (see DATASET_CARD.md); it is not certified
for operational flight use.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.ensemble import GradientBoostingRegressor

from .baselines import _validate_inputs
from .dataset import generate_dataset, split_indices


def _validate_rh(rh_percent: ArrayLike) -> NDArray[np.float64]:
    rh = np.asarray(rh_percent, dtype=float)
    if np.any(~np.isfinite(rh)):
        raise ValueError("rh_percent must be finite.")
    if np.any(rh < 0.0) or np.any(rh > 100.0):
        raise ValueError("rh_percent must be within [0, 100] %.")
    return rh


def _features(
    visibility_km: ArrayLike, wavelength_nm: ArrayLike, rh_percent: ArrayLike
) -> NDArray[np.float64]:
    v, lam = _validate_inputs(visibility_km, wavelength_nm)
    rh = _validate_rh(rh_percent)
    v, lam, rh = np.broadcast_arrays(v, lam, rh)
    return np.column_stack(
        [np.log10(np.ravel(v)), np.ravel(lam) / 1000.0, np.ravel(rh)]
    )


class FogCastModel:
    """Gradient-boosting fog-attenuation regressor with 90 % prediction intervals.

    Parameters
    ----------
    n_estimators, max_depth, learning_rate : GradientBoostingRegressor hyperparameters
        (shared by the point and the two quantile models).
    random_state : seed for all three regressors (reproducible fits).
    """

    #: Nominal central prediction-interval coverage.
    INTERVAL_COVERAGE = 0.90

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 3,
        learning_rate: float = 0.05,
        random_state: int = 42,
    ) -> None:
        common = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
        )
        self._point = GradientBoostingRegressor(loss="squared_error", **common)
        self._lower = GradientBoostingRegressor(loss="quantile", alpha=0.05, **common)
        self._upper = GradientBoostingRegressor(loss="quantile", alpha=0.95, **common)
        self._fitted = False

    def fit(
        self,
        visibility_km: ArrayLike,
        wavelength_nm: ArrayLike,
        rh_percent: ArrayLike,
        attenuation_db_km: ArrayLike,
    ) -> "FogCastModel":
        """Fit on attenuation targets in dB/km (must be strictly positive)."""
        x = _features(visibility_km, wavelength_nm, rh_percent)
        y = np.asarray(attenuation_db_km, dtype=float).ravel()
        if y.shape[0] != x.shape[0]:
            raise ValueError("attenuation_db_km length must match the feature arrays.")
        if np.any(y <= 0.0) or np.any(~np.isfinite(y)):
            raise ValueError("attenuation_db_km must be finite and > 0 (dB/km).")
        y_log = np.log10(y)
        self._point.fit(x, y_log)
        self._lower.fit(x, y_log)
        self._upper.fit(x, y_log)
        self._fitted = True
        return self

    def predict(
        self,
        visibility_km: ArrayLike,
        wavelength_nm: ArrayLike,
        rh_percent: ArrayLike,
        return_interval: bool = True,
    ) -> NDArray[np.float64] | float | tuple:
        """Predict specific attenuation (dB/km).

        Parameters
        ----------
        visibility_km : V in [0.05, 100] km.
        wavelength_nm : lambda in [500, 2000] nm.
        rh_percent : relative humidity in [0, 100] %.
        return_interval : if True (default) return ``(point, lower, upper)`` where
            (lower, upper) is the nominal 90 % prediction interval; else return the
            point estimate only.

        Returns
        -------
        Scalars if all inputs are scalars, else float64 arrays. Intervals are
        clipped so that lower <= point <= upper always holds.
        """
        if not self._fitted:
            raise RuntimeError("Model is not fitted; call fit() or FogCastModel.train_default().")
        scalar = (
            np.isscalar(visibility_km) and np.isscalar(wavelength_nm) and np.isscalar(rh_percent)
        )
        x = _features(visibility_km, wavelength_nm, rh_percent)
        point = 10.0 ** self._point.predict(x)
        if not return_interval:
            return float(point[0]) if scalar else point
        lower = 10.0 ** self._lower.predict(x)
        upper = 10.0 ** self._upper.predict(x)
        # Guard against quantile crossing (possible with independently fit models).
        lower = np.minimum(lower, point)
        upper = np.maximum(upper, point)
        if scalar:
            return float(point[0]), float(lower[0]), float(upper[0])
        return point, lower, upper

    @classmethod
    def train_default(cls, n_samples: int = 6000, seed: int = 42) -> "FogCastModel":
        """Generate the seeded synthetic dataset and fit on its training split.

        Deterministic: the same ``seed`` yields bit-identical predictions.
        """
        data = generate_dataset(n_samples=n_samples, seed=seed)
        idx_train, _, _ = split_indices(n_samples, seed=seed)
        model = cls(random_state=seed)
        model.fit(
            data["visibility_km"][idx_train],
            data["wavelength_nm"][idx_train],
            data["rh_percent"][idx_train],
            data["attenuation_db_km"][idx_train],
        )
        return model


_DEFAULT_MODEL: FogCastModel | None = None


def predict(
    visibility_km: ArrayLike,
    wavelength_nm: ArrayLike,
    rh_percent: ArrayLike,
    return_interval: bool = True,
) -> NDArray[np.float64] | float | tuple:
    """Predict fog/aerosol specific attenuation (dB/km) with the default model.

    Lazily trains a cached :class:`FogCastModel` on the seeded synthetic dataset on
    first call (a few seconds), then reuses it. See :meth:`FogCastModel.predict` for
    parameter units and the ``return_interval`` output contract (nominal 90 % PI).
    """
    global _DEFAULT_MODEL
    if _DEFAULT_MODEL is None:
        _DEFAULT_MODEL = FogCastModel.train_default()
    return _DEFAULT_MODEL.predict(
        visibility_km, wavelength_nm, rh_percent, return_interval=return_interval
    )
