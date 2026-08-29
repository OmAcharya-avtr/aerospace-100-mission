"""Learned slope-to-Zernike reconstructor with deep-ensemble uncertainty.

The regularised least-squares baselines in :mod:`wavelab.reconstruct` were
implemented, validated and tuned **before** this model, and the model is scored
against them on identical held-out samples (`validation/VALIDATION.md`, README
"Benchmark results").

Architecture
------------
An ensemble of independently seeded `sklearn.neural_network.MLPRegressor`
networks, each mapping the measurement feature vector to all estimated Zernike
coefficients at once. The ensemble mean is the point estimate; the per-mode
standard deviation across members is the uncertainty output. Deep ensembles as
an uncertainty proxy: B. Lakshminarayanan, A. Pritzel and C. Blundell, "Simple
and Scalable Predictive Uncertainty Estimation using Deep Ensembles",
*Advances in Neural Information Processing Systems 30* (2017). The spread
measures **disagreement between members**, which is not the same thing as a
calibrated 1-sigma error bar; the measured calibration ratio is reported in
`MODEL_CARD.md` and is not close to one everywhere.

PyTorch is not available in this build environment, so a graph- or
convolution-structured network over the subaperture lattice -- the natural
choice for this problem -- is not an option. A fully connected ensemble is used
instead and the absence of any spatial inductive bias is a documented
limitation.

Features
--------
For a sample with measured scaled slopes ``u`` [rad] (dropped subapertures set
to exactly zero), subaperture availability ``a`` and photon count ``N``:

1. ``u`` itself -- ``2 n_sub`` values, with dropped subapertures at exactly
   zero. The internal `StandardScaler` puts each component on a common scale.
   An earlier revision divided by the photon-noise scale ``sigma_u(N)``; that
   was measurably worse, because it makes the input magnitude swing by a factor
   of 30 across the flux sweep and the standardiser then compresses every
   low-flux sample toward zero.
2. ``a`` as 0/1 -- ``n_sub`` values. Without this the network cannot tell a
   genuinely zero slope from a missing one.
3. ``log10(N)`` -- one value, telling the network which noise regime it is in
   so it can choose how hard to shrink.

Feature 3 assumes the photon count is known per measurement, which a real
system does know from its own subaperture flux. Feature 1 makes the model
**not** gain invariant: slopes in different units invalidate it.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

__all__ = ["EnsembleZernikeReconstructor", "build_features"]


def build_features(
    u_meas: NDArray[np.float64],
    available: NDArray[np.bool_],
    n_photons: NDArray[np.float64] | float,
) -> NDArray[np.float64]:
    """Assemble the model's feature matrix.

    Parameters
    ----------
    u_meas : ndarray, shape (n, 2 * n_sub)
        Measured scaled slopes [rad]; dropped subapertures must be 0.0.
    available : ndarray of bool, shape (n, n_sub)
        Subaperture availability.
    n_photons : float or ndarray, shape (n,)
        Photons per subaperture [-], > 0.

    Returns
    -------
    ndarray, shape (n, 3 * n_sub + 1)
        Columns: scaled slopes, availability, ``log10(n_photons)``.
    """
    u = np.asarray(u_meas, dtype=float)
    if u.ndim != 2 or u.shape[1] % 2 != 0:
        raise ValueError(f"u_meas must be 2-D with an even width, got shape {u.shape}")
    n, two_ns = u.shape
    n_sub = two_ns // 2
    a = np.asarray(available, dtype=bool)
    if a.shape != (n, n_sub):
        raise ValueError(f"available must have shape ({n}, {n_sub}), got {a.shape}")
    nph = np.broadcast_to(np.asarray(n_photons, dtype=float), (n,))
    if np.any(~np.isfinite(nph)) or np.any(nph <= 0.0):
        raise ValueError("n_photons must be finite and > 0")
    return np.hstack([u, a.astype(float), np.log10(nph)[:, None]])


class EnsembleZernikeReconstructor:
    """Ensemble MLP mapping Shack-Hartmann slopes to Zernike coefficients.

    Parameters
    ----------
    n_modes : int
        Number of Zernike coefficients predicted [-], >= 1.
    n_estimators : int
        Ensemble size [-], >= 2 (a spread needs at least two members).
    hidden_layer_sizes : tuple[int, ...]
        MLP hidden widths, passed to `MLPRegressor`.
    alpha : float
        L2 penalty of each member.
    max_iter : int
        Maximum Adam epochs per member.
    batch_size : int
        Adam mini-batch size. Larger batches are markedly faster on the 2-core
        build machine at the cost of fewer parameter updates per epoch.
    random_state : int
        Base seed; member ``k`` uses ``random_state + 1000 * k`` and a
        bootstrap resample drawn from the same stream, so training is
        reproducible.
    bootstrap : bool
        If True each member is trained on an independent bootstrap resample,
        which decorrelates the members and widens the ensemble spread.

    Notes
    -----
    Inputs are the features of :func:`build_features`; targets are Zernike
    coefficients in radians. Both are standardised internally.
    """

    def __init__(
        self,
        n_modes: int,
        n_estimators: int = 5,
        hidden_layer_sizes: tuple[int, ...] = (192, 96),
        alpha: float = 1e-4,
        max_iter: int = 120,
        batch_size: int = 512,
        random_state: int = 0,
        bootstrap: bool = True,
    ) -> None:
        if isinstance(n_modes, bool) or not isinstance(n_modes, (int, np.integer)):
            raise TypeError(f"n_modes must be an integer, got {n_modes!r}")
        if n_modes < 1:
            raise ValueError(f"n_modes must be >= 1, got {n_modes}")
        if isinstance(n_estimators, bool) or not isinstance(n_estimators, (int, np.integer)):
            raise TypeError(f"n_estimators must be an integer, got {n_estimators!r}")
        if n_estimators < 2:
            raise ValueError(f"n_estimators must be >= 2 to define a spread, got {n_estimators}")
        if float(alpha) < 0.0:
            raise ValueError(f"alpha must be >= 0, got {alpha!r}")
        if int(max_iter) < 1:
            raise ValueError(f"max_iter must be >= 1, got {max_iter!r}")
        if int(batch_size) < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size!r}")
        self.n_modes = int(n_modes)
        self.n_estimators = int(n_estimators)
        self.hidden_layer_sizes = tuple(int(h) for h in hidden_layer_sizes)
        self.alpha = float(alpha)
        self.max_iter = int(max_iter)
        self.batch_size = int(batch_size)
        self.random_state = int(random_state)
        self.bootstrap = bool(bootstrap)
        self.fitted_ = False
        self._members: list[MLPRegressor] = []
        self._x_scaler: StandardScaler | None = None
        self._y_scaler: StandardScaler | None = None

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> EnsembleZernikeReconstructor:
        """Train every ensemble member.

        Parameters
        ----------
        x : ndarray, shape (n, n_features)
            Features from :func:`build_features`.
        y : ndarray, shape (n, n_modes)
            Zernike coefficient targets [rad].
        """
        xa = np.asarray(x, dtype=float)
        ya = np.asarray(y, dtype=float)
        if xa.ndim != 2:
            raise ValueError(f"x must be 2-D, got shape {xa.shape}")
        if ya.ndim != 2 or ya.shape[1] != self.n_modes:
            raise ValueError(f"y must have shape (n, {self.n_modes}), got {ya.shape}")
        if xa.shape[0] != ya.shape[0]:
            raise ValueError(f"x and y disagree on sample count: {xa.shape[0]} vs {ya.shape[0]}")
        if xa.shape[0] < 2:
            raise ValueError("at least 2 training samples are required")
        if not np.all(np.isfinite(xa)) or not np.all(np.isfinite(ya)):
            raise ValueError("x and y must be finite")

        self._x_scaler = StandardScaler().fit(xa)
        self._y_scaler = StandardScaler().fit(ya)
        xs = self._x_scaler.transform(xa)
        ys = self._y_scaler.transform(ya)

        self._members = []
        n = xs.shape[0]
        for k in range(self.n_estimators):
            seed = self.random_state + 1000 * k
            if self.bootstrap:
                rng = np.random.default_rng(seed)
                sel = rng.integers(0, n, size=n)
            else:
                sel = np.arange(n)
            net = MLPRegressor(
                hidden_layer_sizes=self.hidden_layer_sizes,
                activation="relu",
                solver="adam",
                alpha=self.alpha,
                max_iter=self.max_iter,
                batch_size=min(self.batch_size, xs.shape[0]),
                random_state=seed,
                early_stopping=False,
                tol=1e-6,
                n_iter_no_change=15,
            )
            net.fit(xs[sel], ys[sel])
            self._members.append(net)
        self.fitted_ = True
        return self

    def _check_fitted(self) -> None:
        if not self.fitted_ or self._x_scaler is None or self._y_scaler is None:
            raise RuntimeError("model is not fitted; call fit() first")

    def predict(
        self, x: NDArray[np.float64], return_std: bool = False
    ) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Predict Zernike coefficients [rad], optionally with the ensemble spread.

        Returns
        -------
        mean : ndarray, shape (n, n_modes)
            Ensemble mean [rad].
        std : ndarray, shape (n, n_modes)
            Population standard deviation across members [rad], returned only
            when ``return_std``. This is a *disagreement* measure, not a
            calibrated error bar -- see `MODEL_CARD.md` for its measured
            calibration.
        """
        self._check_fitted()
        assert self._x_scaler is not None and self._y_scaler is not None
        xa = np.asarray(x, dtype=float)
        if xa.ndim != 2:
            raise ValueError(f"x must be 2-D, got shape {xa.shape}")
        if xa.shape[1] != self._x_scaler.n_features_in_:
            raise ValueError(
                f"x must have {self._x_scaler.n_features_in_} features, got {xa.shape[1]}"
            )
        xs = self._x_scaler.transform(xa)
        preds = np.stack([self._y_scaler.inverse_transform(m.predict(xs)) for m in self._members])
        mean = preds.mean(axis=0)
        if not return_std:
            return mean
        return mean, preds.std(axis=0)

    def predict_measurements(
        self,
        u_meas: NDArray[np.float64],
        available: NDArray[np.bool_],
        n_photons: NDArray[np.float64] | float,
        return_std: bool = False,
    ) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Convenience wrapper: build features from raw measurements, then predict."""
        return self.predict(build_features(u_meas, available, n_photons), return_std=return_std)
