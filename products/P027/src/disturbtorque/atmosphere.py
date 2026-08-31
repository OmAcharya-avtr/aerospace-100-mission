"""Piecewise-exponential neutral atmosphere density, for aerodynamic torque sizing.

Model
-----
rho(h) = rho0_k * exp(-(h - h0_k) / H_k)  for h in band k,

with (h0, rho0, H) tabulated per band. This is the standard exponential atmosphere
table reproduced in Vallado, *Fundamentals of Astrodynamics and Applications*, whose
band values derive from the US Standard Atmosphere 1976 below 86 km and from CIRA-72
above it.

Units: ``h`` geodetic altitude in metres, ``rho`` in kg m^-3, ``H`` scale height in
metres.

Assumptions and validity
------------------------
* Spherically symmetric, static, non-rotating atmosphere: no diurnal bulge, no
  latitude or longitude dependence, no seasonal variation.
* **No solar-activity dependence.** This is the dominant limitation. Thermospheric
  density above 400 km varies by more than an order of magnitude between solar minimum
  and solar maximum; the table is a single mean profile. Aerodynamic torque scales
  linearly in density, so the aerodynamic column of any budget produced here carries at
  least a factor-of-several uncertainty above 400 km, and that is a property of the
  input model, not of the torque expression.
* Validity range 0 to 1000 km. Above 1000 km the last band is extrapolated and the
  function raises unless ``allow_extrapolation`` is set.
* Free-molecular flow, required by the aerodynamic torque model, holds above roughly
  150 km.

The table's internal consistency is checked in
``validation/atmosphere_table_continuity.py``: each band's base density equals the
previous band's density evaluated at the shared boundary, which is how a transcription
error in any of the 84 numbers would show up.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import _validate as _v

__all__ = ["EXPONENTIAL_TABLE", "MODEL_MIN_ALTITUDE_M", "MODEL_MAX_ALTITUDE_M", "density"]

MODEL_MIN_ALTITUDE_M: float = 0.0
MODEL_MAX_ALTITUDE_M: float = 1_000_000.0

# (base altitude [km], base density [kg m^-3], scale height [km])
EXPONENTIAL_TABLE: tuple[tuple[float, float, float], ...] = (
    (0.0, 1.225, 7.249),
    (25.0, 3.899e-2, 6.349),
    (30.0, 1.774e-2, 6.682),
    (40.0, 3.972e-3, 7.554),
    (50.0, 1.057e-3, 8.382),
    (60.0, 3.206e-4, 7.714),
    (70.0, 8.770e-5, 6.549),
    (80.0, 1.905e-5, 5.799),
    (90.0, 3.396e-6, 5.382),
    (100.0, 5.297e-7, 5.877),
    (110.0, 9.661e-8, 7.263),
    (120.0, 2.438e-8, 9.473),
    (130.0, 8.484e-9, 12.636),
    (140.0, 3.845e-9, 16.149),
    (150.0, 2.070e-9, 22.523),
    (180.0, 5.464e-10, 29.740),
    (200.0, 2.789e-10, 37.105),
    (250.0, 7.248e-11, 45.546),
    (300.0, 2.418e-11, 53.628),
    (350.0, 9.518e-12, 53.298),
    (400.0, 3.725e-12, 58.515),
    (450.0, 1.585e-12, 60.828),
    (500.0, 6.967e-13, 63.822),
    (600.0, 1.454e-13, 71.835),
    (700.0, 3.614e-14, 88.667),
    (800.0, 1.170e-14, 124.64),
    (900.0, 5.245e-15, 181.05),
    (1000.0, 3.019e-15, 268.00),
)

_BASE_ALT_M = np.array([row[0] for row in EXPONENTIAL_TABLE]) * 1000.0
_BASE_RHO = np.array([row[1] for row in EXPONENTIAL_TABLE])
_SCALE_H_M = np.array([row[2] for row in EXPONENTIAL_TABLE]) * 1000.0


def density(altitude_m: ArrayLike, allow_extrapolation: bool = False) -> NDArray[np.float64]:
    """Neutral atmospheric mass density [kg m^-3] at geodetic altitude ``altitude_m`` [m].

    Parameters
    ----------
    altitude_m : array_like
        Geodetic altitude above the reference ellipsoid [m]. Scalar or array.
    allow_extrapolation : bool
        If True, altitudes above 1000 km use the last band's exponential extrapolation
        instead of raising.

    Returns
    -------
    ndarray
        Density [kg m^-3], same shape as the input (0-d for a scalar input).

    Raises
    ------
    ValueError
        On a negative or non-finite altitude, or above 1000 km when
        ``allow_extrapolation`` is False.

    Notes
    -----
    Model, sources, assumptions and validity: see the module docstring. Accuracy against
    a solar-activity-aware model such as NRLMSISE-00 is not claimed and not tested here.
    """
    h = np.asarray(altitude_m, dtype=float)
    if not np.all(np.isfinite(h)):
        raise ValueError("altitude_m must be finite")
    if np.any(h < MODEL_MIN_ALTITUDE_M):
        raise ValueError(
            f"altitude_m must be >= {MODEL_MIN_ALTITUDE_M} m; got a minimum of {float(h.min())} m"
        )
    if not allow_extrapolation and np.any(h > MODEL_MAX_ALTITUDE_M):
        raise ValueError(
            f"altitude_m must be <= {MODEL_MAX_ALTITUDE_M} m for this model "
            f"(got a maximum of {float(h.max())} m); pass allow_extrapolation=True to "
            "extrapolate the top band, accepting that it is unvalidated"
        )
    idx = np.clip(np.searchsorted(_BASE_ALT_M, h, side="right") - 1, 0, len(_BASE_ALT_M) - 1)
    rho = _BASE_RHO[idx] * np.exp(-(h - _BASE_ALT_M[idx]) / _SCALE_H_M[idx])
    return rho


def scale_height(altitude_m: float) -> float:
    """Scale height [m] of the band containing ``altitude_m`` [m]."""
    h = _v.non_negative(altitude_m, "altitude_m")
    idx = int(np.clip(np.searchsorted(_BASE_ALT_M, h, side="right") - 1, 0, len(_BASE_ALT_M) - 1))
    return float(_SCALE_H_M[idx])
