"""MLP-ensemble surrogate for the scintillation index sigma_I^2.

The surrogate maps link parameters (Cn^2, path length L, wavelength lambda,
aperture diameter D) to the aperture-averaged scintillation index measured
from split-step simulation. An ensemble of small scikit-learn MLPRegressor
models (different random initialisations) provides an uncertainty estimate
via the member spread (deep-ensemble style; Lakshminarayanan et al. 2017).

Feature transform: [log10 Cn^2, log10 L, log10 lambda, D] with
standardisation; target is log10(sigma_I^2) for dynamic range. Predictions
are returned in linear sigma_I^2 space.

The analytic baseline for benchmarking is
``rytov_baseline`` = Andrews (1992) aperture-averaged weak-fluctuation
Rytov index (see :mod:`scintinet.rytov`).
"""

from __future__ import annotations

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .rytov import scintillation_index_weak

__all__ = ["Surrogate", "rytov_baseline"]

_EPS = 1e-12


def _validate_x(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != 4:
        raise ValueError(
            f"X must have shape (n_samples, 4) with columns "
            f"[cn2, path_length_m, wavelength_m, aperture_d_m]; got shape {x.shape}"
        )
    if not np.all(np.isfinite(x)):
        raise ValueError("X contains non-finite values")
    if np.any(x[:, :3] <= 0.0):
        raise ValueError("cn2, path_length and wavelength must all be > 0")
    if np.any(x[:, 3] < 0.0):
        raise ValueError("aperture diameter must be >= 0")
    return x


def rytov_baseline(x: np.ndarray) -> np.ndarray:
    """Analytic baseline: aperture-averaged weak-fluctuation Rytov index.

    Parameters
    ----------
    x : numpy.ndarray
        (n, 4) array with columns [Cn^2 (m^-2/3), L (m), lambda (m), D (m)].
        D = 0 is treated as a point receiver.

    Returns
    -------
    numpy.ndarray
        (n,) predicted sigma_I^2 [dimensionless].
    """
    x = _validate_x(x)
    out = np.empty(x.shape[0])
    for i, (cn2, ell, lam, dia) in enumerate(x):
        out[i] = scintillation_index_weak(
            cn2, lam, ell, aperture_diameter=(dia if dia > 0.0 else None)
        )
    return out


class Surrogate:
    """Ensemble MLP surrogate: sigma_I^2 = f(Cn^2, L, lambda, D).

    Parameters
    ----------
    n_members : int
        Ensemble size (>= 2 required for a meaningful std output).
    hidden_layer_sizes : tuple of int
        Architecture of each MLPRegressor member.
    max_iter : int
        Training iterations per member (L-BFGS).
    random_state : int
        Base seed; member i uses random_state + i.
    """

    def __init__(
        self,
        n_members: int = 5,
        hidden_layer_sizes: tuple[int, ...] = (32, 32),
        max_iter: int = 2000,
        random_state: int = 0,
    ) -> None:
        if n_members < 2:
            raise ValueError(f"n_members must be >= 2 for uncertainty output, got {n_members}")
        self.n_members = n_members
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.random_state = random_state
        self._members: list[Pipeline] = []

    @staticmethod
    def _features(x: np.ndarray) -> np.ndarray:
        return np.column_stack(
            [np.log10(x[:, 0]), np.log10(x[:, 1]), np.log10(x[:, 2]), x[:, 3]]
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> "Surrogate":
        """Fit the ensemble.

        Parameters
        ----------
        x : numpy.ndarray
            (n, 4) inputs [Cn^2 (m^-2/3), L (m), lambda (m), D (m)].
        y : numpy.ndarray
            (n,) simulated scintillation index sigma_I^2 (> 0).
        """
        x = _validate_x(x)
        y = np.asarray(y, dtype=float).ravel()
        if y.shape[0] != x.shape[0]:
            raise ValueError(f"X has {x.shape[0]} rows but y has {y.shape[0]}")
        if np.any(~np.isfinite(y)) or np.any(y <= 0.0):
            raise ValueError("y (sigma_I^2) must be finite and > 0 for log-space training")
        feats = self._features(x)
        target = np.log10(y + _EPS)
        self._members = []
        for i in range(self.n_members):
            model = Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "mlp",
                        MLPRegressor(
                            hidden_layer_sizes=self.hidden_layer_sizes,
                            solver="lbfgs",
                            max_iter=self.max_iter,
                            random_state=self.random_state + i,
                        ),
                    ),
                ]
            )
            model.fit(feats, target)
            self._members.append(model)
        return self

    def predict(
        self, x: np.ndarray, return_std: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Predict sigma_I^2, optionally with ensemble-spread uncertainty.

        Parameters
        ----------
        x : numpy.ndarray
            (n, 4) inputs [Cn^2 (m^-2/3), L (m), lambda (m), D (m)].
        return_std : bool
            If True, also return the standard deviation of the ensemble
            member predictions (in linear sigma_I^2 space). This measures
            model (epistemic) disagreement only; it is not a calibrated
            predictive interval.

        Returns
        -------
        numpy.ndarray or (numpy.ndarray, numpy.ndarray)
            Mean prediction [dimensionless], and std if requested.
        """
        if not self._members:
            raise RuntimeError("Surrogate is not fitted; call fit(X, y) first")
        x = _validate_x(x)
        feats = self._features(x)
        preds = np.stack([10.0 ** m.predict(feats) for m in self._members])
        mean = preds.mean(axis=0)
        if return_std:
            return mean, preds.std(axis=0)
        return mean
