"""estimkit -- a compact, dependency-light Kalman filter family for aerospace estimation.

Contents
--------
- :class:`~estimkit.linear.KalmanFilter` -- discrete linear Kalman filter,
  Joseph-form covariance update, per-step NIS diagnostic.
- :func:`~estimkit.linear.steady_state` -- fixed-point solution of the
  filtering algebraic Riccati equation.
- :class:`~estimkit.ekf.ExtendedKalmanFilter` -- EKF taking user-supplied
  Jacobians, with :func:`~estimkit.ekf.numerical_jacobian` as a documented
  fallback.
- :class:`~estimkit.ukf.UnscentedKalmanFilter`,
  :class:`~estimkit.ukf.MerweSigmaPoints`,
  :func:`~estimkit.ukf.unscented_transform` -- scaled unscented transform
  with configurable ``alpha``/``beta``/``kappa``.
- :func:`~estimkit.smoother.rts_smooth` -- Rauch-Tung-Striebel
  fixed-interval smoother.
- :mod:`estimkit.covariance` -- Joseph update and numerical-health
  diagnostics.
- :mod:`estimkit.models` -- standard constant-velocity and random-walk
  models used as test cases.

Only NumPy is required. Units are the caller's; every public function
documents what it expects.

This is educational software. It is not flight-qualified, not certified,
and not approved for operational aerospace use.
"""

from __future__ import annotations

from .covariance import (
    covariance_health,
    is_positive_semidefinite,
    is_symmetric,
    joseph_update,
    min_eigenvalue,
    simple_update,
    symmetrize,
)
from .ekf import ExtendedKalmanFilter, numerical_jacobian
from .linear import FilterResult, KalmanFilter, UpdateResult, steady_state
from .models import constant_velocity_cwna, constant_velocity_dwna, random_walk
from .smoother import SmootherResult, rts_smooth
from .ukf import (
    MerweSigmaPoints,
    SigmaPoints,
    UnscentedKalmanFilter,
    unscented_transform,
)

__version__ = "0.1.0"

__all__ = [
    "ExtendedKalmanFilter",
    "FilterResult",
    "KalmanFilter",
    "MerweSigmaPoints",
    "SigmaPoints",
    "SmootherResult",
    "UnscentedKalmanFilter",
    "UpdateResult",
    "__version__",
    "constant_velocity_cwna",
    "constant_velocity_dwna",
    "covariance_health",
    "is_positive_semidefinite",
    "is_symmetric",
    "joseph_update",
    "min_eigenvalue",
    "numerical_jacobian",
    "random_walk",
    "rts_smooth",
    "simple_update",
    "steady_state",
    "symmetrize",
    "unscented_transform",
]
