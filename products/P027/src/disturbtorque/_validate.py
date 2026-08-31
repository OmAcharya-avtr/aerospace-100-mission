"""Input validation helpers.

Every public entry point routes its arguments through these so that a bad input raises
a ``ValueError``/``TypeError`` naming the offending argument and the accepted range,
rather than producing a plausible-looking wrong number.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray


def as_finite_float(value: object, name: str) -> float:
    """Return ``value`` as a finite float or raise.

    Parameters
    ----------
    value : object
        Candidate scalar.
    name : str
        Argument name used in the error message.
    """
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real scalar, got {value!r}") from exc
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {out!r}")
    return out


def positive(value: object, name: str) -> float:
    """Return ``value`` as a strictly positive finite float or raise."""
    out = as_finite_float(value, name)
    if out <= 0.0:
        raise ValueError(f"{name} must be > 0, got {out!r}")
    return out


def non_negative(value: object, name: str) -> float:
    """Return ``value`` as a non-negative finite float or raise."""
    out = as_finite_float(value, name)
    if out < 0.0:
        raise ValueError(f"{name} must be >= 0, got {out!r}")
    return out


def in_range(value: object, name: str, lo: float, hi: float) -> float:
    """Return ``value`` as a float inside the closed interval ``[lo, hi]`` or raise."""
    out = as_finite_float(value, name)
    if not (lo <= out <= hi):
        raise ValueError(f"{name} must lie in [{lo}, {hi}], got {out!r}")
    return out


def as_vector3(value: ArrayLike, name: str) -> NDArray[np.float64]:
    """Return ``value`` as a finite float64 array of shape (3,) or raise."""
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite, got {arr!r}")
    return arr


def as_unit_vector(value: ArrayLike, name: str, tol: float = 1e-6) -> NDArray[np.float64]:
    """Return ``value`` as a unit 3-vector, normalising if it is within ``tol`` of unity.

    A vector whose norm differs from 1 by more than ``tol`` is rejected rather than
    silently normalised, because a non-unit direction is almost always a units bug.
    """
    arr = as_vector3(value, name)
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        raise ValueError(f"{name} must be a non-zero direction, got the zero vector")
    if abs(norm - 1.0) > tol:
        raise ValueError(
            f"{name} must be a unit vector (|v| within {tol} of 1), got |v| = {norm!r}"
        )
    return arr / norm


def as_inertia_matrix(value: ArrayLike, name: str = "inertia") -> NDArray[np.float64]:
    """Return ``value`` as a valid inertia tensor [kg m^2] of shape (3, 3), or raise.

    Checks symmetry (to 1e-9 relative), positive definiteness, and the triangle
    inequality on the principal moments (I_i + I_j >= I_k), which any physical rigid
    body must satisfy.
    """
    arr = np.asarray(value, dtype=float)
    if arr.shape == (3,):
        arr = np.diag(arr)
    if arr.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3) or (3,), got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    scale = float(np.max(np.abs(arr)))
    if scale == 0.0:
        raise ValueError(f"{name} must be non-zero")
    if not np.allclose(arr, arr.T, rtol=0.0, atol=1e-9 * scale):
        raise ValueError(f"{name} must be symmetric")
    eig = np.linalg.eigvalsh(arr)
    if float(np.min(eig)) <= 0.0:
        raise ValueError(
            f"{name} must be positive definite; principal moments computed as {eig.tolist()}"
        )
    a, b, c = sorted(float(e) for e in eig)
    if a + b < c * (1.0 - 1e-9):
        raise ValueError(
            f"{name} violates the triangle inequality for a rigid body: principal moments "
            f"{[a, b, c]} have I1 + I2 < I3"
        )
    return arr


def altitude_in_model_range(alt_m: float, lo_m: float, hi_m: float, model: str) -> float:
    """Validate a geodetic altitude [m] against a model's stated validity range."""
    out = as_finite_float(alt_m, "altitude_m")
    if not (lo_m <= out <= hi_m):
        raise ValueError(
            f"altitude_m = {out!r} m is outside the stated validity range of the {model} "
            f"model, [{lo_m}, {hi_m}] m"
        )
    return out
