"""Controllers: the conservative baseline and the performance controller.

The Simplex (runtime-assurance) architecture pairs a **baseline** controller
whose safety can be argued with a **performance** controller whose safety cannot
(Seto, Krogh, Sha and Chutinan, "The Simplex architecture for safe online
control system upgrades", *Proceedings of the American Control Conference*,
1998; Sha, "Using simplicity to control complexity", *IEEE Software* 18(4),
2001).

This module provides both as plain callables of the state, plus the discrete
infinite-horizon LQR gain used to construct them.

    u_baseline(x)    = sat_U( -K_b x )
    u_performance(x) = sat_U( -K_p (x - x_ref) )

The saturation ``sat_U`` is a projection onto the declared input box. For the
baseline the saturation never binds inside the robust invariant set, because the
set is constructed with the row ``-K_b x in U`` included (see
:mod:`simplexguard.invariant`); for the performance controller it binds often,
which is one of the two reasons the guard fires.

Discrete LQR
------------
``K = (R + B' P B)^{-1} B' P A`` with ``P`` the stabilising solution of the
discrete algebraic Riccati equation, from :func:`scipy.linalg.solve_discrete_are`
(Anderson and Moore, *Optimal Control: Linear Quadratic Methods*,
Prentice-Hall, 1990, chapter 3). Units follow the plant: ``Q`` in inverse
squared state units, ``R`` in inverse squared input units.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_discrete_are

from .plant import Plant
from .polytope import Box

__all__ = [
    "BaselineController",
    "PerformanceController",
    "closed_loop_matrix",
    "dlqr_gain",
    "reference_controllers",
    "saturate",
]


def dlqr_gain(plant: Plant, q_diag: np.ndarray, r_diag: np.ndarray) -> np.ndarray:
    """Discrete infinite-horizon LQR gain ``K``, shape ``(m, n)``.

    Parameters
    ----------
    plant :
        Supplies ``A`` and ``B``.
    q_diag :
        Diagonal of the state weight, length ``n``, strictly positive.
    r_diag :
        Diagonal of the input weight, length ``m``, strictly positive. A larger
        ``r_diag`` gives a more conservative, lower-authority controller.
    """
    q = np.asarray(q_diag, dtype=float).ravel()
    r = np.asarray(r_diag, dtype=float).ravel()
    if q.shape[0] != plant.n_states:
        raise ValueError(f"q_diag has length {q.shape[0]}, expected {plant.n_states}")
    if r.shape[0] != plant.n_inputs:
        raise ValueError(f"r_diag has length {r.shape[0]}, expected {plant.n_inputs}")
    if np.any(q <= 0.0) or np.any(r <= 0.0):
        raise ValueError("LQR weights must be strictly positive")
    p = solve_discrete_are(plant.A, plant.B, np.diag(q), np.diag(r))
    gain = np.linalg.solve(np.diag(r) + plant.B.T @ p @ plant.B, plant.B.T @ p @ plant.A)
    return np.atleast_2d(gain)


def closed_loop_matrix(plant: Plant, gain: np.ndarray) -> np.ndarray:
    """``A - B K``, shape ``(n, n)``."""
    k = np.atleast_2d(np.asarray(gain, dtype=float))
    if k.shape != (plant.n_inputs, plant.n_states):
        raise ValueError(
            f"gain has shape {k.shape}, expected {(plant.n_inputs, plant.n_states)}"
        )
    return plant.A - plant.B @ k


def saturate(u: np.ndarray, input_set: Box) -> np.ndarray:
    """Clip ``u`` coordinate-wise into the declared input box."""
    if not isinstance(input_set, Box):
        raise TypeError("saturate requires a Box input-constraint set")
    uv = np.atleast_1d(np.asarray(u, dtype=float)).ravel()
    if uv.shape[0] != input_set.dim:
        raise ValueError(f"u has length {uv.shape[0]}, expected {input_set.dim}")
    return np.clip(uv, input_set.lower, input_set.upper)


@dataclass(frozen=True)
class BaselineController:
    """``u = sat_U(-K_b x)``: the conservative controller the guard falls back to.

    The gain is expected to be low-authority enough that a non-empty robust
    invariant set exists; :func:`simplexguard.invariant.robust_invariant_set`
    raises if it does not, which is the honest failure mode rather than a
    silently empty certificate.
    """

    gain: np.ndarray
    input_set: Box

    def __post_init__(self) -> None:
        k = np.atleast_2d(np.asarray(self.gain, dtype=float))
        if not np.all(np.isfinite(k)):
            raise ValueError("gain contains non-finite entries")
        if k.shape[0] != self.input_set.dim:
            raise ValueError(
                f"gain has {k.shape[0]} rows, input set has dim {self.input_set.dim}"
            )
        object.__setattr__(self, "gain", k)

    def unsaturated(self, x: np.ndarray) -> np.ndarray:
        """``-K_b x`` before saturation."""
        xv = np.asarray(x, dtype=float).ravel()
        if xv.shape[0] != self.gain.shape[1]:
            raise ValueError(f"x has length {xv.shape[0]}, expected {self.gain.shape[1]}")
        return -(self.gain @ xv)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return saturate(self.unsaturated(x), self.input_set)


@dataclass(frozen=True)
class PerformanceController:
    """``u = sat_U(-K_p (x - x_ref))``: the controller whose safety is not argued.

    ``reference`` is a callable of the step index returning the reference state;
    :func:`square_wave_reference` is the shipped one. The controller is linear
    in ``x`` except for the saturation, and
    :mod:`simplexguard.reachability` uses the *unsaturated* linear part for its
    exact multi-step prediction, which is a stated and measured approximation.
    """

    gain: np.ndarray
    input_set: Box

    def __post_init__(self) -> None:
        k = np.atleast_2d(np.asarray(self.gain, dtype=float))
        if not np.all(np.isfinite(k)):
            raise ValueError("gain contains non-finite entries")
        if k.shape[0] != self.input_set.dim:
            raise ValueError(
                f"gain has {k.shape[0]} rows, input set has dim {self.input_set.dim}"
            )
        object.__setattr__(self, "gain", k)

    def unsaturated(self, x: np.ndarray, reference: np.ndarray | None = None) -> np.ndarray:
        """``-K_p (x - x_ref)`` before saturation."""
        xv = np.asarray(x, dtype=float).ravel()
        if xv.shape[0] != self.gain.shape[1]:
            raise ValueError(f"x has length {xv.shape[0]}, expected {self.gain.shape[1]}")
        if reference is None:
            err = xv
        else:
            rv = np.asarray(reference, dtype=float).ravel()
            if rv.shape[0] != xv.shape[0]:
                raise ValueError(f"reference has length {rv.shape[0]}, expected {xv.shape[0]}")
            err = xv - rv
        return -(self.gain @ err)

    def __call__(self, x: np.ndarray, reference: np.ndarray | None = None) -> np.ndarray:
        return saturate(self.unsaturated(x, reference), self.input_set)

    def saturates_at(self, x: np.ndarray, reference: np.ndarray | None = None) -> bool:
        """True if the saturation binds at this state, to any coordinate."""
        raw = self.unsaturated(x, reference)
        return bool(np.any(raw < self.input_set.lower - 1e-12)) or bool(
            np.any(raw > self.input_set.upper + 1e-12)
        )


def reference_controllers(
    plant: Plant,
    baseline_r: float = 1.0,
    performance_q_angle: float = 100.0,
    performance_r: float = 0.1,
) -> tuple[BaselineController, PerformanceController]:
    """The shipped illustrative controller pair for :func:`reference_plant`.

    The baseline is LQR with ``Q = I`` and ``R = baseline_r I``; raising
    ``baseline_r`` makes it more conservative and eventually empties the robust
    invariant set, which ``validation/validate_invariant_set.py`` measures.
    The performance controller is LQR with a heavy angle weight and a light
    input weight, so it tracks fast and wants more authority than the certified
    envelope allows.
    """
    if not isinstance(plant.input_constraints, Box):
        raise TypeError("reference_controllers requires box input constraints")
    q_perf = np.ones(plant.n_states)
    q_perf[0] = performance_q_angle
    baseline = BaselineController(
        gain=dlqr_gain(plant, np.ones(plant.n_states), np.full(plant.n_inputs, baseline_r)),
        input_set=plant.input_constraints,
    )
    performance = PerformanceController(
        gain=dlqr_gain(plant, q_perf, np.full(plant.n_inputs, performance_r)),
        input_set=plant.input_constraints,
    )
    return baseline, performance
