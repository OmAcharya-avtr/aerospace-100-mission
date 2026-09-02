"""Null-motion gimbal reconfiguration for SGCMG arrays.

Any gimbal rate in ``null(A)`` moves the gimbals without changing the array
momentum, so it delivers no torque and is free to be used for reconfiguration.
For a four-CMG array away from a singularity ``null(A)`` is one-dimensional,
so the whole null motion is a single signed scalar times the unit null vector.

Three classical policies are provided:

* **none** — no null motion, the steering law's particular solution only.
* **gradient** — climb the singularity measure,
  ``ddelta_null = k P grad(m)``, with ``P = N N^T`` the orthogonal projector
  onto ``null(A)``.  This is the manipulability-gradient null motion of
  Yoshikawa (1985) as applied to CMG steering by Bedrossian et al. (1990).
* **preferred** — drive the gimbals toward a stored preferred set,
  ``ddelta_null = -k P (delta - delta_pref)`` (Vadali, Walker & Oh 1990).

Sign convention for the unit null vector: for a one-dimensional null space the
sign is fixed so that ``n_hat . grad(m) >= 0``, i.e. positive coefficients
increase the singularity measure.  When the gradient is orthogonal to the null
space (a measure-zero set) the sign falls back to making the first non-zero
component positive.  Everything is in [rad/s]; ``grad(m)`` is in
[(N*m*s/rad)^3 / rad].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .arrays import CMGArray
from .singularity import manipulability_gradient, null_space_basis

__all__ = [
    "GradientNullMotion",
    "NoNullMotion",
    "NullMotionPolicy",
    "PreferredAngleNullMotion",
    "null_motion_from_coefficients",
    "null_projector",
    "unit_null_vector",
]


def null_projector(array: CMGArray, deltas: ArrayLike) -> NDArray[np.float64]:
    """Orthogonal projector ``P = N N^T`` onto ``null(A(delta))``, shape ``(n_free, n_free)``."""
    basis = null_space_basis(array.jacobian(deltas))
    if basis.shape[1] == 0:
        n = array.n_free
        return np.zeros((n, n))
    return basis @ basis.T


def unit_null_vector(
    array: CMGArray, deltas: ArrayLike, align_with_gradient: bool = True
) -> NDArray[np.float64]:
    """Unit vector spanning a one-dimensional ``null(A)``, with a fixed sign.

    Raises
    ------
    ValueError
        If the null space is not one-dimensional at this configuration (which
        happens for a rank-deficient Jacobian, or for an array with more than
        four free gimbals).
    """
    d = np.asarray(deltas, dtype=float).reshape(-1)
    basis = null_space_basis(array.jacobian(d))
    if basis.shape[1] != 1:
        raise ValueError(
            f"null space has dimension {basis.shape[1]}, not 1; a scalar null-motion "
            "coefficient is not enough to describe it"
        )
    vec = basis[:, 0]
    if align_with_gradient:
        grad = manipulability_gradient(array, d)
        dot = float(vec @ grad)
        scale = float(np.linalg.norm(grad)) * float(np.linalg.norm(vec))
        if abs(dot) > 1e-12 * max(scale, 1e-30):
            return vec if dot > 0.0 else -vec
    nz = np.flatnonzero(np.abs(vec) > 1e-12)
    return vec if (nz.size == 0 or vec[nz[0]] > 0.0) else -vec


def null_motion_from_coefficients(
    array: CMGArray, deltas: ArrayLike, coefficients: ArrayLike, scale: float = 1.0
) -> NDArray[np.float64]:
    """Null-motion gimbal rates from coefficients in the null-space basis.

    Parameters
    ----------
    coefficients
        Length ``dim(null(A))`` coefficients.  For a four-CMG array away from a
        singularity this is a single number.
    scale
        Multiplies the coefficients; use it to carry the maximum null rate
        [rad/s] so that coefficients stay in ``[-1, 1]``.
    """
    d = np.asarray(deltas, dtype=float).reshape(-1)
    basis = null_space_basis(array.jacobian(d))
    c = np.atleast_1d(np.asarray(coefficients, dtype=float)).reshape(-1)
    if c.shape[0] != basis.shape[1]:
        raise ValueError(
            f"coefficients must have length {basis.shape[1]} (the null-space dimension), "
            f"got {c.shape[0]}"
        )
    if basis.shape[1] == 1:
        vec = unit_null_vector(array, d)
        return float(scale) * float(c[0]) * vec
    return float(scale) * (basis @ c)


class NullMotionPolicy:
    """Interface every null-motion policy implements.

    A policy maps the current configuration and torque command to gimbal rates
    that lie in ``null(A)``.  ``name`` labels it in benchmark tables.
    """

    name: str = "policy"

    def rates(
        self,
        array: CMGArray,
        deltas: ArrayLike,
        torque: ArrayLike,
        time: float = 0.0,
    ) -> NDArray[np.float64]:
        """Null-motion gimbal rates [rad/s], length ``n_free``."""
        raise NotImplementedError

    def reset(self) -> None:
        """Clear any per-run internal state.  The stateless policies do nothing."""


@dataclass
class NoNullMotion(NullMotionPolicy):
    """Zero null motion: the steering law's particular solution alone."""

    name: str = "none"

    def rates(
        self,
        array: CMGArray,
        deltas: ArrayLike,
        torque: ArrayLike,
        time: float = 0.0,
    ) -> NDArray[np.float64]:
        del deltas, torque, time
        return np.zeros(array.n_free)


@dataclass
class GradientNullMotion(NullMotionPolicy):
    """Manipulability-gradient null motion, ``ddelta_null = k P grad(m)``.

    Parameters
    ----------
    gain
        ``k``, in [rad/s per unit of ``grad(m)``].  Must be finite.
    max_rate
        Optional cap on ``|ddelta_null|_inf`` [rad/s]; the vector is scaled,
        not clipped, so it stays inside ``null(A)``.
    """

    gain: float = 1.0
    max_rate: float | None = None
    name: str = "gradient"

    def rates(
        self,
        array: CMGArray,
        deltas: ArrayLike,
        torque: ArrayLike,
        time: float = 0.0,
    ) -> NDArray[np.float64]:
        del torque, time
        d = np.asarray(deltas, dtype=float).reshape(-1)
        grad = manipulability_gradient(array, d)
        out = self.gain * (null_projector(array, d) @ grad)
        return _cap(out, self.max_rate)


@dataclass
class PreferredAngleNullMotion(NullMotionPolicy):
    """Null motion toward a preferred gimbal set, ``ddelta_null = -k P (delta - delta_pref)``.

    Parameters
    ----------
    preferred
        Length ``n_free`` preferred gimbal angles [rad].
    gain
        ``k`` [1/s].
    max_rate
        Optional cap on ``|ddelta_null|_inf`` [rad/s], applied by scaling.
    """

    preferred: NDArray[np.float64]
    gain: float = 1.0
    max_rate: float | None = None
    name: str = "preferred"

    def rates(
        self,
        array: CMGArray,
        deltas: ArrayLike,
        torque: ArrayLike,
        time: float = 0.0,
    ) -> NDArray[np.float64]:
        del torque, time
        d = np.asarray(deltas, dtype=float).reshape(-1)
        pref = np.asarray(self.preferred, dtype=float).reshape(-1)
        if pref.shape[0] != array.n_free:
            raise ValueError(
                f"preferred must have length {array.n_free}, got {pref.shape[0]}"
            )
        err = np.arctan2(
            np.sin(d[array.free_indices] - pref), np.cos(d[array.free_indices] - pref)
        )
        out = -self.gain * (null_projector(array, d) @ err)
        return _cap(out, self.max_rate)


def _cap(vec: NDArray[np.float64], max_rate: float | None) -> NDArray[np.float64]:
    if max_rate is None:
        return vec
    if max_rate <= 0.0:
        raise ValueError(f"max_rate must be positive [rad/s], got {max_rate}")
    peak = float(np.max(np.abs(vec))) if vec.size else 0.0
    return vec if peak <= max_rate else vec * (max_rate / peak)
