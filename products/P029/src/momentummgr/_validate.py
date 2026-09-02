"""Input validation helpers with actionable error messages.

No physics lives here. Every public entry point in the package funnels its arguments
through these functions so that an invalid inertia tensor or a non-unit direction fails
at the call site rather than three modules deeper.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "as_finite_float",
    "positive",
    "non_negative",
    "in_range",
    "as_vector3",
    "as_unit_vector",
    "as_inertia_matrix",
    "as_int_at_least",
]


def as_finite_float(value: object, name: str) -> float:
    """Return ``value`` as a finite float or raise ``TypeError``/``ValueError``."""
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number, got {value!r}") from exc
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite, got {out!r}")
    return out


def positive(value: object, name: str) -> float:
    """Return ``value`` as a strictly positive finite float."""
    out = as_finite_float(value, name)
    if out <= 0.0:
        raise ValueError(f"{name} must be > 0, got {out!r}")
    return out


def non_negative(value: object, name: str) -> float:
    """Return ``value`` as a non-negative finite float."""
    out = as_finite_float(value, name)
    if out < 0.0:
        raise ValueError(f"{name} must be >= 0, got {out!r}")
    return out


def in_range(value: object, name: str, low: float, high: float) -> float:
    """Return ``value`` as a finite float inside the closed interval ``[low, high]``."""
    out = as_finite_float(value, name)
    if not (low <= out <= high):
        raise ValueError(f"{name} must lie in [{low}, {high}], got {out!r}")
    return out


def as_int_at_least(value: object, name: str, minimum: int) -> int:
    """Return ``value`` as an int no smaller than ``minimum``."""
    try:
        out = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be an integer, got {value!r}") from exc
    if out < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {out!r}")
    return out


def as_vector3(value: ArrayLike, name: str) -> NDArray[np.float64]:
    """Return ``value`` as a finite float array of shape ``(3,)``."""
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite, got {arr!r}")
    return arr


def as_unit_vector(value: ArrayLike, name: str, tol: float = 1e-8) -> NDArray[np.float64]:
    """Return ``value`` as a shape-``(3,)`` unit vector, rejecting anything else.

    The norm must already be 1 to within ``tol``; this function does not silently
    normalise, because a caller passing a non-unit direction has usually made a frame
    error rather than a scaling error.
    """
    arr = as_vector3(value, name)
    norm = float(np.linalg.norm(arr))
    if abs(norm - 1.0) > tol:
        raise ValueError(f"{name} must be a unit vector (|v| = 1 +/- {tol}), got |v| = {norm!r}")
    return arr


def as_inertia_matrix(value: ArrayLike, name: str = "inertia") -> NDArray[np.float64]:
    """Return a valid rigid-body inertia tensor [kg m^2] as a ``(3, 3)`` array.

    Accepts a length-3 sequence of principal moments or a full ``(3, 3)`` tensor. The
    tensor must be symmetric and positive definite, and its eigenvalues must satisfy the
    rigid-body triangle inequality ``I_i + I_j >= I_k`` for every permutation, which is a
    necessary condition for a physically realisable mass distribution (Hughes,
    *Spacecraft Attitude Dynamics*).
    """
    arr = np.asarray(value, dtype=float)
    if arr.shape == (3,):
        arr = np.diag(arr)
    if arr.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3) or (3,), got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    if not np.allclose(arr, arr.T, rtol=0.0, atol=1e-12 * max(1.0, float(np.abs(arr).max()))):
        raise ValueError(f"{name} must be symmetric; got a maximum asymmetry of "
                         f"{float(np.abs(arr - arr.T).max())}")
    eig = np.linalg.eigvalsh(arr)
    if np.any(eig <= 0.0):
        raise ValueError(f"{name} must be positive definite; eigenvalues are {eig.tolist()}")
    a, b, c = float(eig[0]), float(eig[1]), float(eig[2])
    if a + b < c * (1.0 - 1e-12):
        raise ValueError(
            f"{name} violates the rigid-body triangle inequality I1 + I2 >= I3: "
            f"principal moments {eig.tolist()}"
        )
    return arr
