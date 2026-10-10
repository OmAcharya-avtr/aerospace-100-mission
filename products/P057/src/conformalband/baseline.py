"""The classical baselines. Both are implemented and benchmarked before any
learned component, per the mission rule, and the parametric interval below is
the baseline the conformal bands have to beat.

Two separate baselines, for two separate jobs:

1. :class:`PhysicsRegressor` is the **point-prediction** baseline. It fits the
   four unknown coefficients of the closed-form energy model in
   :mod:`conformalband.physics` by nonlinear least squares. It is
   deliberately misspecified: it assumes a constant propulsive efficiency,
   while the data-generating model has a quadratic fall-off away from the
   design airspeed. That is a realistic amount of wrong, not a straw man.

2. :class:`GaussianResidualInterval` is the **interval** baseline, and the one
   the spec names: a symmetric Gaussian interval built from the model's own
   residual variance,

       yhat +/- z_{1 - alpha/2} * sigma_hat,   sigma_hat^2 = RSS / (n - p)

   It assumes homoscedastic Gaussian residuals and an unbiased mean model.
   Both assumptions are false on this data, which is exactly why the
   comparison is worth publishing: the interval is nonetheless tighter than
   every conformal band in distribution.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize, stats

from .physics import DEFAULT_AIRFRAME, GRAVITY, SECONDS_PER_HOUR, Airframe

PARAMETER_NAMES: tuple[str, ...] = ("cd0", "oswald", "eta_prop", "avionics_power")
"""Fitted coefficients, in the order :attr:`PhysicsRegressor.parameters` uses."""

PARAMETER_UNITS: tuple[str, ...] = ("-", "-", "-", "W")

_LOWER_BOUNDS = (1e-4, 0.05, 0.05, 0.0)
_UPPER_BOUNDS = (0.5, 1.0, 1.0, 200.0)
_INITIAL_GUESS = (0.03, 0.8, 0.6, 10.0)


def _energy_from_parameters(features: np.ndarray, *parameters: float) -> np.ndarray:
    """Closed-form energy [Wh] with a constant propulsive efficiency.

    This is the model form the baseline fits: equations as in
    :mod:`conformalband.physics` with ``eta(V) = eta_prop`` for all ``V``.
    """
    cd0, oswald, eta_prop, avionics = parameters
    airspeed = features[:, 0]
    mass = features[:, 1]
    density = features[:, 2]
    headwind = features[:, 3]
    distance = features[:, 4]
    airframe = DEFAULT_AIRFRAME
    parasite = 0.5 * density * airspeed**3 * airframe.wing_area * cd0
    weight = mass * GRAVITY
    induced = 2.0 * weight**2 / (density * airspeed * np.pi * airframe.wingspan**2 * oswald)
    power = (parasite + induced) / eta_prop + avionics
    ground_speed = airspeed - headwind
    return power * (distance / ground_speed) / SECONDS_PER_HOUR


class PhysicsRegressor:
    """Analytic baseline: least-squares fit of the closed-form energy model.

    Four free coefficients (``cd0``, ``oswald``, ``eta_prop``,
    ``avionics_power``); the airframe geometry is taken as known. The
    propulsive-efficiency curvature of the true model is **not** in the
    hypothesis class, so the fit carries an airspeed-dependent bias that no
    amount of data removes.

    Parameters
    ----------
    airframe:
        Airframe whose geometry (wing area, wingspan) is treated as known.
    max_iterations:
        Maximum least-squares iterations.
    """

    n_parameters = len(PARAMETER_NAMES)

    def __init__(
        self, airframe: Airframe = DEFAULT_AIRFRAME, *, max_iterations: int = 20000
    ) -> None:
        self.airframe = airframe
        self.max_iterations = int(max_iterations)
        self._parameters: np.ndarray | None = None

    def fit(self, features: np.ndarray, energy: np.ndarray) -> PhysicsRegressor:
        """Fit the four coefficients by bounded nonlinear least squares."""
        x = np.asarray(features, dtype=float)
        y = np.asarray(energy, dtype=float).ravel()
        if x.ndim != 2 or x.shape[1] != 5:
            raise ValueError(f"features must have shape (n, 5), got {x.shape}")
        if y.shape != (x.shape[0],):
            raise ValueError("energy must have shape (n,) matching features")
        if y.size <= self.n_parameters:
            raise ValueError(
                f"need more than {self.n_parameters} samples to fit "
                f"{self.n_parameters} coefficients, got {y.size}"
            )
        popt, _ = optimize.curve_fit(
            _energy_from_parameters,
            x,
            y,
            p0=_INITIAL_GUESS,
            bounds=(_LOWER_BOUNDS, _UPPER_BOUNDS),
            maxfev=self.max_iterations,
        )
        self._parameters = np.asarray(popt, dtype=float)
        return self

    @property
    def parameters(self) -> np.ndarray:
        """Fitted coefficients in the order of :data:`PARAMETER_NAMES`."""
        if self._parameters is None:
            raise RuntimeError("call fit(...) before reading parameters")
        return self._parameters.copy()

    def parameter_table(self) -> dict[str, float]:
        """Fitted coefficients keyed by name, with units in :data:`PARAMETER_UNITS`."""
        return dict(zip(PARAMETER_NAMES, self.parameters, strict=True))

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predicted energy per leg [Wh]."""
        if self._parameters is None:
            raise RuntimeError("call fit(...) before predict(...)")
        x = np.asarray(features, dtype=float)
        if x.ndim != 2 or x.shape[1] != 5:
            raise ValueError(f"features must have shape (n, 5), got {x.shape}")
        return _energy_from_parameters(x, *self._parameters)


class GaussianResidualInterval:
    """Parametric baseline interval from the model's own residual variance.

    ``sigma_hat^2 = sum(residual^2) / (n - p)`` and the interval is
    ``yhat +/- z_{1 - alpha/2} sigma_hat``, with ``z`` the standard normal
    quantile. Optionally ``t`` with ``n - p`` degrees of freedom, which is the
    textbook small-sample form; at the sample sizes used here the two differ
    in the fourth decimal.

    Parameters
    ----------
    alpha:
        Miscoverage level in (0, 1).
    n_parameters:
        Degrees of freedom consumed by the mean model, ``p``.
    use_t:
        Use the Student-t quantile instead of the normal one.
    """

    def __init__(self, alpha: float = 0.1, *, n_parameters: int = 0, use_t: bool = False) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if n_parameters < 0:
            raise ValueError(f"n_parameters must be >= 0, got {n_parameters}")
        self.alpha = float(alpha)
        self.n_parameters = int(n_parameters)
        self.use_t = bool(use_t)
        self._sigma: float | None = None
        self._dof: int | None = None

    def fit(self, y_true: np.ndarray, y_pred: np.ndarray) -> GaussianResidualInterval:
        """Estimate ``sigma_hat`` [Wh] from the residuals of a fitted model."""
        y = np.asarray(y_true, dtype=float).ravel()
        p = np.asarray(y_pred, dtype=float).ravel()
        if y.shape != p.shape:
            raise ValueError(f"y_true and y_pred must agree, got {y.shape} and {p.shape}")
        dof = y.size - self.n_parameters
        if dof <= 0:
            raise ValueError(
                f"residual degrees of freedom must be > 0: {y.size} samples minus "
                f"{self.n_parameters} parameters"
            )
        residual = y - p
        self._sigma = float(np.sqrt(np.sum(residual**2) / dof))
        self._dof = int(dof)
        return self

    @property
    def sigma(self) -> float:
        """Residual standard deviation estimate [Wh]."""
        if self._sigma is None:
            raise RuntimeError("call fit(...) first")
        return self._sigma

    @property
    def half_width(self) -> float:
        """``z_{1 - alpha/2} sigma_hat`` [Wh], the constant interval half-width."""
        if self._dof is None:
            raise RuntimeError("call fit(...) first")
        level = 1.0 - self.alpha / 2.0
        q = stats.t.ppf(level, self._dof) if self.use_t else stats.norm.ppf(level)
        return float(q) * self.sigma

    def interval(self, y_pred: np.ndarray):
        """Symmetric interval ``yhat +/- z sigma_hat`` [Wh]."""
        from .conformal import Interval

        p = np.asarray(y_pred, dtype=float).ravel()
        h = self.half_width
        return Interval(lower=p - h, point=p, upper=p + h)
