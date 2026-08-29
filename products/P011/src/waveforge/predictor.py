"""Predictive wavefront controllers: pure-delay baseline and a learned forecaster.

Why prediction
--------------
A closed AO loop applies a correction that was measured ``d`` frames earlier.
Under Taylor frozen flow the wavefront at ``t + d T`` is a rigid translation of
the wavefront at ``t``, so it is in principle *predictable* from a history of
measurements; the residual temporal error ``(d T / tau_0)^(5/3)`` is therefore
not a hard floor.  Linear prediction of this kind is long established:
M. B. Jorgenson and G. J. M. Aitken, "Prediction of atmospherically induced
wave-front degradations", *Opt. Lett.* **17**, 466-468 (1992); C. Dessenne,
P.-Y. Madec and G. Rousset, "Modal prediction for closed-loop adaptive
optics", *Opt. Lett.* **22**, 1535-1537 (1997).

What is implemented
-------------------
* :class:`PureDelayPredictor` — the honest "no prediction" baseline: the
  forecast is the newest available pseudo-open-loop slope vector.
* :class:`LinearSlopePredictor` — a bagged ensemble of ridge regressions (or,
  optionally, of small scikit-learn MLPs) mapping the last ``n_history``
  pseudo-open-loop slope vectors to the slope vector ``horizon`` frames ahead,
  with a per-slope predictive standard deviation.

The uncertainty is a *deep-ensemble* style decomposition (B. Lakshminarayanan,
A. Pritzel and C. Blundell, NeurIPS 2017): the epistemic part is the spread of
the ensemble members, the aleatoric part is the out-of-bag residual variance
measured per output during fitting, and the reported one sigma is the square
root of their sum.  Interval coverage is measured on held-out data in
``validation/``, not assumed.

Units: slopes in rad/m throughout; the models are fitted and evaluated in those
units and are therefore tied to the geometry they were trained on.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

__all__ = ["LinearSlopePredictor", "PureDelayPredictor", "build_lagged_dataset"]


def build_lagged_dataset(
    sequences: Sequence[np.ndarray],
    n_history: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble a supervised dataset of lagged slope vectors.

    Parameters
    ----------
    sequences:
        One or more ``(T, n_slopes)`` slope time series.  Each is treated as an
        independent realisation and no sample ever straddles two sequences.
    n_history:
        Number of consecutive past frames used as input, ``>= 1``.
    horizon:
        Forecast distance in frames, ``>= 1``.

    Returns
    -------
    (X, Y):
        ``X`` has shape ``(n_samples, n_history * n_slopes)`` with the oldest
        frame first; ``Y`` has shape ``(n_samples, n_slopes)``.
    """
    if int(n_history) != n_history or n_history < 1:
        raise ValueError(f"n_history must be an integer >= 1, got {n_history!r}")
    if int(horizon) != horizon or horizon < 1:
        raise ValueError(f"horizon must be an integer >= 1, got {horizon!r}")
    if len(sequences) == 0:
        raise ValueError("at least one sequence is required")
    n_history, horizon = int(n_history), int(horizon)
    xs, ys = [], []
    n_slopes = None
    for seq in sequences:
        arr = np.asarray(seq, dtype=float)
        if arr.ndim != 2:
            raise ValueError(f"each sequence must be 2-D (T, n_slopes), got shape {arr.shape}")
        if n_slopes is None:
            n_slopes = arr.shape[1]
        elif arr.shape[1] != n_slopes:
            raise ValueError("all sequences must have the same number of slopes")
        n_samples = arr.shape[0] - n_history - horizon + 1
        if n_samples <= 0:
            raise ValueError(
                f"sequence of length {arr.shape[0]} is too short for "
                f"n_history={n_history} and horizon={horizon}"
            )
        idx = np.arange(n_samples)[:, None] + np.arange(n_history)[None, :]
        xs.append(arr[idx].reshape(n_samples, n_history * arr.shape[1]))
        ys.append(arr[np.arange(n_samples) + n_history + horizon - 1])
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


@dataclass(frozen=True)
class PureDelayPredictor:
    """Baseline forecaster: return the newest available measurement unchanged.

    This is what an AO loop does implicitly — it corrects with information that
    is ``d`` frames stale.  Running it through the same predictive control path
    as the learned model isolates the effect of *prediction* from the effect of
    the pseudo-open-loop control formulation.

    ``horizon`` is ``None`` because no forecasting is attempted; the loop
    accepts that as "any latency".
    """

    n_history: int = 1
    horizon: int | None = None

    def __post_init__(self) -> None:
        if int(self.n_history) != self.n_history or self.n_history < 1:
            raise ValueError(f"n_history must be an integer >= 1, got {self.n_history!r}")

    def predict(self, history: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """Return the last row of ``history`` and no uncertainty."""
        hist = np.asarray(history, dtype=float)
        if hist.ndim != 2 or hist.shape[0] < 1:
            raise ValueError(f"history must be (n_history, n_slopes), got shape {hist.shape}")
        return hist[-1].copy(), None


@dataclass
class LinearSlopePredictor:
    """Bagged scikit-learn forecaster of pseudo-open-loop slopes.

    Parameters
    ----------
    n_history:
        Past frames used as input, ``>= 1``.
    horizon:
        Frames ahead to forecast, ``>= 1``.  Must equal the loop latency.
    model:
        ``"ridge"`` (default) or ``"mlp"``.
    alpha:
        Ridge regularisation.  ``"auto"`` selects from ``alpha_grid`` on the
        last ``validation_fraction`` of the *training* data only.
    alpha_grid:
        Candidate values used when ``alpha="auto"``.
    n_members:
        Bootstrap ensemble size, ``>= 1``.  Uncertainty needs ``>= 2``.
    hidden_layer_sizes, max_iter:
        Passed to :class:`sklearn.neural_network.MLPRegressor` when
        ``model="mlp"``.
    validation_fraction:
        Fraction of the training set held back for the ``alpha`` search.
    random_state:
        Seed for the bootstrap resampling and for the MLP initialisation.

    Notes
    -----
    Features are standardised with the training mean and standard deviation
    (stored on the instance), so the model is tied to the slope units and
    geometry it was trained on.  Feeding slopes from a different subaperture
    layout is a silent error; :meth:`predict` checks only the vector length.
    """

    n_history: int = 4
    horizon: int = 2
    model: str = "ridge"
    alpha: float | str = "auto"
    alpha_grid: tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
    n_members: int = 8
    hidden_layer_sizes: tuple[int, ...] = (128,)
    max_iter: int = 300
    validation_fraction: float = 0.2
    random_state: int = 0
    _members: list = field(init=False, default_factory=list, repr=False)
    _mean: np.ndarray | None = field(init=False, default=None, repr=False)
    _scale: np.ndarray | None = field(init=False, default=None, repr=False)
    _aleatoric_var: np.ndarray | None = field(init=False, default=None, repr=False)
    _n_slopes: int = field(init=False, default=0, repr=False)
    _chosen_alpha: float = field(init=False, default=float("nan"), repr=False)

    def __post_init__(self) -> None:
        if int(self.n_history) != self.n_history or self.n_history < 1:
            raise ValueError(f"n_history must be an integer >= 1, got {self.n_history!r}")
        if int(self.horizon) != self.horizon or self.horizon < 1:
            raise ValueError(f"horizon must be an integer >= 1, got {self.horizon!r}")
        if self.model not in ("ridge", "mlp"):
            raise ValueError(f"model must be 'ridge' or 'mlp', got {self.model!r}")
        if int(self.n_members) != self.n_members or self.n_members < 1:
            raise ValueError(f"n_members must be an integer >= 1, got {self.n_members!r}")
        if not (0.0 < self.validation_fraction < 1.0):
            raise ValueError(
                f"validation_fraction must lie in (0, 1), got {self.validation_fraction!r}"
            )
        if isinstance(self.alpha, str) and self.alpha != "auto":
            raise ValueError(f"alpha must be a positive float or 'auto', got {self.alpha!r}")
        if not isinstance(self.alpha, str) and (
            not np.isfinite(self.alpha) or float(self.alpha) <= 0.0
        ):
            raise ValueError(f"alpha must be a positive float or 'auto', got {self.alpha!r}")

    # -- properties -------------------------------------------------------
    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` has been called."""
        return bool(self._members)

    @property
    def n_slopes(self) -> int:
        """Slope-vector length the model was fitted on."""
        return self._n_slopes

    @property
    def chosen_alpha(self) -> float:
        """Ridge alpha actually used (``nan`` before fitting or for the MLP)."""
        return self._chosen_alpha

    @property
    def n_parameters(self) -> int:
        """Total number of fitted coefficients across the ensemble."""
        if not self.is_fitted:
            return 0
        total = 0
        for member in self._members:
            if hasattr(member, "coef_"):
                total += int(np.size(member.coef_)) + int(np.size(member.intercept_))
            else:  # MLP
                total += sum(int(np.size(w)) for w in member.coefs_)
                total += sum(int(np.size(b)) for b in member.intercepts_)
        return total

    # -- fitting ----------------------------------------------------------
    def _make_estimator(self, alpha: float, seed: int):
        if self.model == "ridge":
            return Ridge(alpha=alpha, fit_intercept=True)
        return MLPRegressor(
            hidden_layer_sizes=tuple(self.hidden_layer_sizes),
            alpha=alpha,
            max_iter=int(self.max_iter),
            early_stopping=True,
            n_iter_no_change=10,
            validation_fraction=0.1,
            random_state=int(seed),
        )

    def _standardise(self, x: np.ndarray) -> np.ndarray:
        return (x - self._mean) / self._scale

    def fit(self, sequences: Sequence[np.ndarray]) -> LinearSlopePredictor:
        """Fit the ensemble on independent slope sequences.

        ``sequences`` are ``(T, n_slopes)`` arrays; supply *different screens*
        for training and testing, never different slices of the same screen —
        a frozen-flow sequence is strongly autocorrelated and a random split
        would leak.
        """
        x, y = build_lagged_dataset(sequences, self.n_history, self.horizon)
        if x.shape[0] < 4:
            raise ValueError(f"need at least 4 training samples, got {x.shape[0]}")
        self._n_slopes = y.shape[1]
        self._mean = x.mean(axis=0)
        scale = x.std(axis=0)
        self._scale = np.where(scale > 0.0, scale, 1.0)
        xs = self._standardise(x)

        alpha = float(self.alpha) if not isinstance(self.alpha, str) else self._search_alpha(xs, y)
        self._chosen_alpha = alpha if self.model == "ridge" else float("nan")

        rng = np.random.default_rng(self.random_state)
        n = xs.shape[0]
        self._members = []
        oob_sq = np.zeros((0, self._n_slopes))
        for k in range(int(self.n_members)):
            idx = np.arange(n) if self.n_members == 1 else rng.integers(0, n, size=n)
            estimator = self._make_estimator(alpha, self.random_state + k)
            estimator.fit(xs[idx], y[idx])
            self._members.append(estimator)
            oob = np.setdiff1d(np.arange(n), np.unique(idx))
            if oob.size:
                oob_sq = np.vstack([oob_sq, (estimator.predict(xs[oob]) - y[oob]) ** 2])
        if oob_sq.shape[0] == 0:  # single member, no bootstrap
            residual = self._members[0].predict(xs) - y
            oob_sq = residual**2
        self._aleatoric_var = oob_sq.mean(axis=0)
        return self

    def _search_alpha(self, xs: np.ndarray, y: np.ndarray) -> float:
        n = xs.shape[0]
        n_val = max(1, int(round(self.validation_fraction * n)))
        if n_val >= n:  # pragma: no cover - guarded by the sample-count check
            return float(self.alpha_grid[0])
        x_tr, y_tr = xs[: n - n_val], y[: n - n_val]
        x_va, y_va = xs[n - n_val :], y[n - n_val :]
        best_alpha, best_score = float(self.alpha_grid[0]), np.inf
        for candidate in self.alpha_grid:
            estimator = self._make_estimator(float(candidate), self.random_state)
            estimator.fit(x_tr, y_tr)
            score = float(np.mean((estimator.predict(x_va) - y_va) ** 2))
            if score < best_score:
                best_alpha, best_score = float(candidate), score
        return best_alpha

    # -- prediction -------------------------------------------------------
    def predict(self, history: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forecast the slope vector ``horizon`` frames after the last history row.

        Returns ``(mean, sigma)`` — both length ``n_slopes``, in rad/m.
        ``sigma`` is ``sqrt(ensemble variance + out-of-bag residual variance)``.
        """
        if not self.is_fitted:
            raise RuntimeError("predictor is not fitted; call fit() first")
        hist = np.asarray(history, dtype=float)
        if hist.ndim != 2 or hist.shape != (self.n_history, self._n_slopes):
            raise ValueError(
                f"history must have shape ({self.n_history}, {self._n_slopes}), "
                f"got {hist.shape}"
            )
        x = self._standardise(hist.reshape(1, -1))
        preds = np.stack([m.predict(x)[0] for m in self._members])
        mean = preds.mean(axis=0)
        epistemic = preds.var(axis=0, ddof=1) if len(self._members) > 1 else np.zeros_like(mean)
        return mean, np.sqrt(epistemic + self._aleatoric_var)

    def predict_batch(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Vectorised prediction on a lagged design matrix from
        :func:`build_lagged_dataset`.

        ``x`` has shape ``(n_samples, n_history * n_slopes)``; returns
        ``(mean, sigma)`` each of shape ``(n_samples, n_slopes)``.
        """
        if not self.is_fitted:
            raise RuntimeError("predictor is not fitted; call fit() first")
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.n_history * self._n_slopes:
            raise ValueError(
                f"x must have shape (n, {self.n_history * self._n_slopes}), got {x.shape}"
            )
        xs = self._standardise(x)
        preds = np.stack([m.predict(xs) for m in self._members])
        mean = preds.mean(axis=0)
        epistemic = (
            preds.var(axis=0, ddof=1) if len(self._members) > 1 else np.zeros_like(mean)
        )
        return mean, np.sqrt(epistemic + self._aleatoric_var[None, :])
