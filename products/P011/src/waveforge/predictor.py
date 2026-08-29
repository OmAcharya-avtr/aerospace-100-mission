"""Wavefront predictors: pure delay, linear auto-regression, and a learned ensemble.

The three predictors share one interface so the closed loop can swap them:

```
predict(history) -> (mean, std)
```

``history`` is ``(n_history, n_slopes)``, most-recent-last, in radians of phase
per metre; ``mean`` is the forecast of the slope vector ``horizon`` frames
ahead and ``std`` is a per-component 1-sigma uncertainty in the same units.

Why prediction can help
-----------------------
Under Taylor's frozen-flow hypothesis (Taylor 1938) the wavefront at frame
``k + L`` is the wavefront at frame ``k`` translated by ``v L dt``. That is a
deterministic, linear, time-invariant map, so a linear auto-regressive filter
on past measurements is the natural estimator, and the minimum-variance linear
predictor is the classical answer (Dessenne, C., Madec, P.-Y. & Rousset, G.
1998, "Modal prediction for closed-loop adaptive optics", *Optics Letters*
**22**, 1535-1537, and *Applied Optics* **37**, 4623-4633, 1998; Poyneer, L.,
Macintosh, B. & Veran, J.-P. 2007, "Fourier transform wavefront control with
adaptive prediction of the atmosphere", *JOSA A* **24**, 2645-2660). What a
non-linear model can add over the linear predictor is exactly what this product
measures; the answer is reported as found, in ``validation/VALIDATION.md`` and
``MODEL_CARD.md``.

Dimensionality reduction
------------------------
All learned predictors here work in a truncated principal-component basis of
the training slope vectors: with ``n_slopes`` typically 100-200 and only a few
thousand training frames available inside the compute budget, regressing
``n_slopes`` outputs on ``n_history * n_slopes`` inputs directly over-fits. The
PCA basis is computed once from the training data (SVD of the centred slope
matrix) and shared by the linear and the learned predictor, so the comparison
between them is like for like.

Uncertainty output
------------------
:class:`EnsemblePredictor` returns the **ensemble standard deviation** of its
members combined in quadrature with the residual scatter measured on a
held-out split:

```
std^2 = var_ensemble + var_residual
```

This is the "deep ensembles" recipe (Lakshminarayanan, B., Pritzel, A. &
Blundell, C. 2017, "Simple and scalable predictive uncertainty estimation using
deep ensembles", *NeurIPS 2017*, arXiv:1612.01474) without the learned variance
head, which scikit-learn's `MLPRegressor` cannot express. It is a *proxy*, and
its calibration is measured rather than assumed -- see
``validation/validate_predictor.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.neural_network import MLPRegressor

__all__ = [
    "PersistencePredictor",
    "LinearPredictor",
    "EnsemblePredictor",
    "build_windows",
]


def _as_sequences(sequences) -> list[NDArray[np.float64]]:
    if isinstance(sequences, np.ndarray) and sequences.ndim == 2:
        sequences = [sequences]
    out = []
    for s in sequences:
        arr = np.asarray(s, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"each sequence must be 2-D (frames, slopes), got {arr.shape}")
        out.append(arr)
    if not out:
        raise ValueError("at least one training sequence is required")
    widths = {a.shape[1] for a in out}
    if len(widths) != 1:
        raise ValueError(f"all sequences must have the same slope count, got {sorted(widths)}")
    return out


def build_windows(
    sequences, n_history: int, horizon: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Cut ``(inputs, targets)`` windows out of slope time series.

    Parameters
    ----------
    sequences:
        One ``(n_frames, n_slopes)`` array, or an iterable of them. Windows
        never straddle a sequence boundary.
    n_history:
        Number of past frames per input window [-], >= 1.
    horizon:
        Forecast horizon in frames [-], >= 1.

    Returns
    -------
    inputs:
        ``(n_windows, n_history, n_slopes)``.
    targets:
        ``(n_windows, n_slopes)`` -- the slope vector ``horizon`` frames after
        the last input frame.
    """
    n_history = int(n_history)
    horizon = int(horizon)
    if n_history < 1:
        raise ValueError(f"n_history must be >= 1, got {n_history}")
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    xs: list[NDArray[np.float64]] = []
    ys: list[NDArray[np.float64]] = []
    for arr in _as_sequences(sequences):
        n = arr.shape[0]
        last = n - horizon
        if last <= n_history - 1:
            continue
        for k in range(n_history - 1, last):
            xs.append(arr[k - n_history + 1 : k + 1])
            ys.append(arr[k + horizon])
    if not xs:
        raise ValueError("no windows could be built: sequences are too short")
    return np.stack(xs), np.stack(ys)


@dataclass
class PersistencePredictor:
    """Pure-delay baseline: the forecast is the most recent measurement.

    This is the assumption a classical integrator makes implicitly. Its
    uncertainty output is the RMS frame-to-frame change measured during
    :meth:`fit`, which is the correct 1-sigma for this estimator if the
    increments are stationary and zero-mean.

    Parameters
    ----------
    horizon:
        Forecast horizon in frames [-], >= 1.
    n_history:
        Kept at 1; present so the loop can treat every predictor alike.
    """

    horizon: int = 1
    n_history: int = 1
    sigma: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        if int(self.horizon) < 1:
            raise ValueError(f"horizon must be >= 1, got {self.horizon!r}")
        self.horizon = int(self.horizon)
        self.n_history = 1

    def fit(self, sequences) -> PersistencePredictor:
        """Measure the persistence residual scatter, per slope component."""
        x, y = build_windows(sequences, 1, self.horizon)
        self.sigma = np.sqrt(np.mean((y - x[:, -1, :]) ** 2, axis=0))
        return self

    def predict(self, history) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Forecast and 1-sigma uncertainty [rad/m]."""
        h = np.asarray(history, dtype=np.float64)
        if h.ndim != 2 or h.shape[0] < 1:
            raise ValueError(f"history must be (n_history, n_slopes), got {h.shape}")
        mean = h[-1].copy()
        std = (
            np.zeros_like(mean)
            if self.sigma is None
            else np.broadcast_to(self.sigma, mean.shape).copy()
        )
        return mean, std

    def predict_many(self, histories) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Vectorised forecast for ``(n_windows, n_history, n_slopes)`` input."""
        h = np.asarray(histories, dtype=np.float64)
        if h.ndim != 3:
            raise ValueError(f"histories must be 3-D, got {h.shape}")
        mean = h[:, -1, :].copy()
        std = (
            np.zeros_like(mean)
            if self.sigma is None
            else np.broadcast_to(self.sigma, mean.shape).copy()
        )
        return mean, std


class _PcaMixin:
    """Shared truncated-PCA reduction of the slope vectors."""

    def _fit_pca(self, y: NDArray[np.float64], n_components: int) -> None:
        self._mean = y.mean(axis=0)
        centred = y - self._mean
        _, s, vt = np.linalg.svd(centred, full_matrices=False)
        k = int(min(n_components, vt.shape[0]))
        self._components = vt[:k]  # (k, n_slopes), orthonormal rows
        self._explained = float(np.sum(s[:k] ** 2) / max(np.sum(s**2), 1e-300))
        self.n_components = k

    def _encode(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return (x - self._mean) @ self._components.T

    def _decode(self, c: NDArray[np.float64]) -> NDArray[np.float64]:
        return c @ self._components + self._mean

    def _decode_delta(self, c: NDArray[np.float64]) -> NDArray[np.float64]:
        """Decode an *increment* -- no mean offset."""
        return c @ self._components

    @property
    def explained_variance_ratio(self) -> float:
        """Fraction of training slope variance retained by the PCA basis [-]."""
        return self._explained


class LinearPredictor(_PcaMixin):
    """Ridge-regularised linear auto-regressive predictor in the PCA basis.

    This is the classical minimum-variance linear predictor of Dessenne et al.
    (1998), implemented as a single multi-output ridge regression rather than a
    per-mode filter, so that spatial coupling between modes (which frozen flow
    produces) is captured.

    Parameters
    ----------
    n_history:
        Past frames used [-], >= 1.
    horizon:
        Forecast horizon in frames [-], >= 1.
    n_components:
        PCA components retained [-], >= 1.
    alpha:
        Ridge regularisation [-], > 0. Applied to the standardised design
        matrix.
    """

    def __init__(
        self,
        n_history: int = 4,
        horizon: int = 1,
        n_components: int = 40,
        alpha: float = 1.0e-3,
    ) -> None:
        for name, v in (("n_history", n_history), ("horizon", horizon),
                        ("n_components", n_components)):
            if int(v) < 1:
                raise ValueError(f"{name} must be >= 1, got {v!r}")
        if not (np.isfinite(alpha) and alpha > 0):
            raise ValueError(f"alpha must be > 0, got {alpha!r}")
        self.n_history = int(n_history)
        self.horizon = int(horizon)
        self.n_components_requested = int(n_components)
        self.alpha = float(alpha)
        self._weights: NDArray[np.float64] | None = None
        self.residual_sigma: NDArray[np.float64] | None = None

    def fit(self, sequences) -> LinearPredictor:
        """Fit the AR weights. ``sequences``: one or many ``(frames, slopes)`` arrays."""
        x, y = build_windows(sequences, self.n_history, self.horizon)
        return self.fit_windows(x, y)

    def fit_windows(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> LinearPredictor:
        """Fit from pre-cut windows -- used when a caller has already split the data."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if x.ndim != 3 or y.ndim != 2 or x.shape[0] != y.shape[0]:
            raise ValueError(f"incompatible window shapes {x.shape} and {y.shape}")
        flat = x.reshape(-1, x.shape[2])
        self._fit_pca(np.concatenate([flat, y]), self.n_components_requested)
        feats = self._encode(flat).reshape(x.shape[0], -1)
        # Like the learned model, the regression targets the *increment* on top
        # of persistence, so both predictors inherit the full-rank persistence
        # output and differ only in the increment they add. Regressing the
        # absolute slope vector instead would cap both at the PCA span and make
        # the pure-delay baseline artificially hard to beat.
        targets = self._encode(y) - self._encode(x[:, -1, :])
        design = np.hstack([feats, np.ones((feats.shape[0], 1))])
        gram = design.T @ design
        reg = self.alpha * np.trace(gram) / gram.shape[0] * np.eye(gram.shape[0])
        reg[-1, -1] = 0.0
        self._weights = np.linalg.solve(gram + reg, design.T @ targets)
        pred = x[:, -1, :] + self._decode_delta(design @ self._weights)
        self.residual_sigma = np.sqrt(np.mean((y - pred) ** 2, axis=0))
        return self

    def predict(self, history) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Forecast and 1-sigma uncertainty [rad/m]."""
        if self._weights is None:
            raise ValueError("LinearPredictor.fit must be called before predict")
        h = np.asarray(history, dtype=np.float64)
        if h.ndim != 2 or h.shape[0] != self.n_history:
            raise ValueError(
                f"history must be ({self.n_history}, n_slopes), got {h.shape}"
            )
        feats = self._encode(h).reshape(1, -1)
        design = np.hstack([feats, np.ones((1, 1))])
        mean = h[-1] + self._decode_delta(design @ self._weights)[0]
        std = (
            np.zeros_like(mean)
            if self.residual_sigma is None
            else self.residual_sigma.copy()
        )
        return mean, std

    def predict_many(self, histories) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Vectorised forecast for ``(n_windows, n_history, n_slopes)`` input."""
        if self._weights is None:
            raise ValueError("LinearPredictor.fit must be called before predict")
        h = np.asarray(histories, dtype=np.float64)
        if h.ndim != 3 or h.shape[1] != self.n_history:
            raise ValueError(
                f"histories must be (n, {self.n_history}, n_slopes), got {h.shape}"
            )
        flat = h.reshape(-1, h.shape[2])
        feats = self._encode(flat).reshape(h.shape[0], -1)
        design = np.hstack([feats, np.ones((feats.shape[0], 1))])
        mean = h[:, -1, :] + self._decode_delta(design @ self._weights)
        std = (
            np.zeros_like(mean)
            if self.residual_sigma is None
            else np.broadcast_to(self.residual_sigma, mean.shape).copy()
        )
        return mean, std


class EnsemblePredictor(_PcaMixin):
    """Learned non-linear predictor: an ensemble of MLPs with an uncertainty output.

    Parameters
    ----------
    n_history, horizon, n_components:
        As for :class:`LinearPredictor`.
    base:
        ``"linear"`` (default) or ``"persistence"``. The ensemble always
        predicts a **correction** on top of this base predictor, so a member
        that outputs zero reproduces the base exactly. ``"persistence"`` makes
        the network learn the whole temporal map from scratch; ``"linear"``
        gives it the ridge auto-regressive forecast to correct, which is the
        standard way to ask "does a non-linear model add anything to the
        classical linear predictor?". Both are benchmarked in
        ``validation/validate_predictor.py``.
    n_members:
        Ensemble size [-], >= 2. Members differ in initialisation and in a
        bootstrap resample of the training windows.
    hidden_layer_sizes:
        Passed to ``sklearn.neural_network.MLPRegressor``.
    max_iter, alpha, learning_rate_init:
        Passed to ``MLPRegressor``.
    random_state:
        Base seed; member ``m`` uses ``random_state + m``.

    Notes
    -----
    PyTorch is not available in the build environment, so this is a
    scikit-learn MLP ensemble rather than the recurrent or convolutional model
    that would be the natural architecture for a spatio-temporal forecast. That
    deviation is recorded in ``MODEL_CARD.md``.
    """

    def __init__(
        self,
        n_history: int = 4,
        horizon: int = 1,
        n_components: int = 32,
        base: str = "linear",
        n_members: int = 4,
        hidden_layer_sizes: tuple[int, ...] = (48, 24),
        max_iter: int = 60,
        alpha: float = 1.0e-4,
        learning_rate_init: float = 3.0e-3,
        random_state: int = 0,
    ) -> None:
        for name, v in (("n_history", n_history), ("horizon", horizon),
                        ("n_components", n_components)):
            if int(v) < 1:
                raise ValueError(f"{name} must be >= 1, got {v!r}")
        if int(n_members) < 2:
            raise ValueError(f"n_members must be >= 2, got {n_members!r}")
        if base not in ("linear", "persistence"):
            raise ValueError(f"base must be 'linear' or 'persistence', got {base!r}")
        self.n_history = int(n_history)
        self.horizon = int(horizon)
        self.n_components_requested = int(n_components)
        self.base = base
        self.n_members = int(n_members)
        self.hidden_layer_sizes = tuple(hidden_layer_sizes)
        self.max_iter = int(max_iter)
        self.alpha = float(alpha)
        self.learning_rate_init = float(learning_rate_init)
        self.random_state = int(random_state)
        self.members: list[MLPRegressor] = []
        self.residual_sigma: NDArray[np.float64] | None = None
        self._feat_scale: NDArray[np.float64] | None = None
        self._targ_scale: NDArray[np.float64] | None = None
        self._base: LinearPredictor | None = None

    # ------------------------------------------------------------------- fitting
    def fit(self, sequences, validation_fraction: float = 0.2) -> EnsemblePredictor:
        """Train the ensemble and calibrate the residual term of the uncertainty.

        The last ``validation_fraction`` of the windows is held out to measure
        the residual scatter; the members are trained on the rest.
        """
        vf = float(validation_fraction)
        if not (0.0 < vf < 0.9):
            raise ValueError(f"validation_fraction must be in (0, 0.9), got {vf!r}")
        x, y = build_windows(sequences, self.n_history, self.horizon)
        n = x.shape[0]
        n_val = max(1, int(round(vf * n)))
        n_train = n - n_val
        if n_train < 10:
            raise ValueError(f"not enough training windows: {n_train}")

        flat = x.reshape(-1, x.shape[2])
        self._fit_pca(np.concatenate([flat, y]), self.n_components_requested)
        feats_all = self._encode(flat).reshape(n, -1)
        if self.base == "linear":
            # Fitted on the training windows only, so the held-out split that
            # calibrates the uncertainty stays genuinely held out.
            self._base = LinearPredictor(
                n_history=self.n_history,
                horizon=self.horizon,
                n_components=self.n_components_requested,
            ).fit_windows(x[:n_train], y[:n_train])
            base_all = self._base.predict_many(x)[0]
        else:
            self._base = None
            base_all = x[:, -1, :]
        # The members learn the *correction* to the base predictor, so a member
        # that outputs zero reproduces the base exactly.
        targ_all = self._encode(y) - self._encode(base_all)
        self._feat_scale = np.maximum(feats_all[:n_train].std(axis=0), 1e-30)
        self._targ_scale = np.maximum(targ_all[:n_train].std(axis=0), 1e-30)

        xt = feats_all[:n_train] / self._feat_scale
        yt = targ_all[:n_train] / self._targ_scale
        rng = np.random.default_rng(self.random_state)
        self.members = []
        for m in range(self.n_members):
            idx = rng.integers(0, n_train, size=n_train)
            net = MLPRegressor(
                hidden_layer_sizes=self.hidden_layer_sizes,
                activation="relu",
                solver="adam",
                alpha=self.alpha,
                learning_rate_init=self.learning_rate_init,
                max_iter=self.max_iter,
                early_stopping=True,
                n_iter_no_change=12,
                validation_fraction=0.1,
                random_state=self.random_state + m,
            )
            net.fit(xt[idx], yt[idx])
            self.members.append(net)

        # Residual scatter on the held-out split, in slope units.
        xv = feats_all[n_train:] / self._feat_scale
        pred = np.mean([net.predict(xv) for net in self.members], axis=0)
        pred_slopes = base_all[n_train:] + self._decode_delta(pred * self._targ_scale)
        self.residual_sigma = np.sqrt(np.mean((y[n_train:] - pred_slopes) ** 2, axis=0))
        return self

    # ---------------------------------------------------------------- prediction
    def _base_prediction(self, histories: NDArray[np.float64]) -> NDArray[np.float64]:
        """Forecast of the base predictor the ensemble corrects."""
        if self._base is None:
            return histories[:, -1, :]
        return self._base.predict_many(histories)[0]

    def _predict_batch(self, feats: NDArray[np.float64], last: NDArray[np.float64]):
        preds = np.stack(
            [
                last + self._decode_delta(net.predict(feats) * self._targ_scale)
                for net in self.members
            ]
        )
        mean = preds.mean(axis=0)
        var_ens = preds.var(axis=0, ddof=1)
        var_res = 0.0 if self.residual_sigma is None else self.residual_sigma**2
        return mean, np.sqrt(var_ens + var_res)

    def predict(self, history) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Forecast and 1-sigma uncertainty [rad/m] for one history window."""
        if not self.members:
            raise ValueError("EnsemblePredictor.fit must be called before predict")
        h = np.asarray(history, dtype=np.float64)
        if h.ndim != 2 or h.shape[0] != self.n_history:
            raise ValueError(
                f"history must be ({self.n_history}, n_slopes), got {h.shape}"
            )
        feats = (self._encode(h).reshape(1, -1)) / self._feat_scale
        mean, std = self._predict_batch(feats, self._base_prediction(h[None, ...]))
        return mean[0], std[0]

    def predict_many(self, histories) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Vectorised forecast for ``(n_windows, n_history, n_slopes)`` input."""
        if not self.members:
            raise ValueError("EnsemblePredictor.fit must be called before predict")
        h = np.asarray(histories, dtype=np.float64)
        if h.ndim != 3 or h.shape[1] != self.n_history:
            raise ValueError(
                f"histories must be (n, {self.n_history}, n_slopes), got {h.shape}"
            )
        flat = h.reshape(-1, h.shape[2])
        feats = self._encode(flat).reshape(h.shape[0], -1) / self._feat_scale
        return self._predict_batch(feats, self._base_prediction(h))
