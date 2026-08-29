"""navbench — a navigation-filter comparison bench with consistency diagnostics.

Public surface
--------------
* :mod:`navbench.quaternion` — attitude algebra (scalar-first Hamilton, active).
* :mod:`navbench.truth` — rigid-body attitude and position truth generators.
* :mod:`navbench.sensors` — gyro, star tracker, vector sensor, accelerometer, GPS.
* :mod:`navbench.models` — constant-velocity, range/bearing, univariate growth.
* :mod:`navbench.kf`, :mod:`navbench.ekf`, :mod:`navbench.ukf`,
  :mod:`navbench.mekf` — the four estimators.
* :mod:`navbench.consistency` — NEES, NIS, chi-squared bounds, whiteness.
* :mod:`navbench.adaptive` — classical process-noise adaptation (the baselines).
* :mod:`navbench.ai` — learned process-noise tuning with a calibrated interval.
* :mod:`navbench.bench` — Monte Carlo harness and scoring.

Research-grade software. Not flight-qualified, not certified, not approved for
operational aerospace use.
"""

from __future__ import annotations

__version__ = "0.1.0"

from . import adaptive, ai, bench, consistency, ekf, kf, linalg, mekf, models, quaternion
from . import sensors, truth, ukf
from .adaptive import CovarianceMatching, FixedQ, IaeScaleAdapter, QAdapter
from .ai import LearnedQAdapter, QScaleEnsemble, QScalePrediction
from .bench import BenchResult, run_linear_mc, tune_fixed_scale
from .consistency import ConsistencyReport, assess, chi2_average_bounds, nees_series, nis_series
from .ekf import ExtendedKalmanFilter
from .kf import KalmanFilter, UpdateInfo, steady_state_riccati
from .mekf import MekfConfig, MultiplicativeEKF
from .models import ConstantVelocity, RangeBearing, UnivariateGrowth
from .ukf import SigmaPointSpec, UnscentedKalmanFilter

__all__ = [
    "__version__",
    "adaptive",
    "ai",
    "bench",
    "consistency",
    "ekf",
    "kf",
    "linalg",
    "mekf",
    "models",
    "quaternion",
    "sensors",
    "truth",
    "ukf",
    "QAdapter",
    "FixedQ",
    "CovarianceMatching",
    "IaeScaleAdapter",
    "LearnedQAdapter",
    "QScaleEnsemble",
    "QScalePrediction",
    "BenchResult",
    "run_linear_mc",
    "tune_fixed_scale",
    "ConsistencyReport",
    "assess",
    "chi2_average_bounds",
    "nees_series",
    "nis_series",
    "ExtendedKalmanFilter",
    "KalmanFilter",
    "UpdateInfo",
    "steady_state_riccati",
    "MekfConfig",
    "MultiplicativeEKF",
    "ConstantVelocity",
    "RangeBearing",
    "UnivariateGrowth",
    "SigmaPointSpec",
    "UnscentedKalmanFilter",
]
