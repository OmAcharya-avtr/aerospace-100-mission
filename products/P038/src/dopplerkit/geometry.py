"""Range, range-rate and range-acceleration between two states.

THE SIGN CONVENTION, stated once and never deviated from
--------------------------------------------------------
Range-rate is the time derivative of the scalar range:

    rho_dot = d|rho|/dt,   rho = r_B - r_A

so **rho_dot is POSITIVE when the two endpoints are SEPARATING (receding)**
and negative when they are closing (approaching).  Zero range-rate is the
instant of closest approach.

This is the plain calculus derivative of the range observable, and it is the
convention used for the range-rate observation equation in Vallado,
*Fundamentals of Astrodynamics and Applications* (4th ed., observation /
orbit-determination chapters).  The Doppler sign that follows from it is fixed
in :mod:`dopplerkit.doppler`: approaching (rho_dot < 0) gives a POSITIVE
(up-shifted) Doppler shift.

DIRECTION INVARIANCE, which is the other half of the usual mistake
------------------------------------------------------------------
Range-rate is *invariant* under swapping the two endpoints::

    rho_dot(A -> B) == rho_dot(B -> A)

exactly, because rho and the relative velocity both change sign and the
quotient rho . v_rel / |rho| does not.  So reversing the link direction
(uplink vs downlink) does **not** flip the sign of the range-rate, and does
not flip the sign of the first-order Doppler shift either.  What changes
between uplink and downlink is only *which carrier frequency* the shift is
proportional to.  See :func:`range_rate_mps` and the tests in
``tests/test_signs.py``, which fail if this is ever broken.

Equations
---------
With rho = r_B - r_A, v_rel = v_B - v_A, a_rel = a_B - a_A (all in one common
inertial frame, SI units):

    rho       = |rho|                                             [m]
    rho_dot   = (rho . v_rel) / rho                               [m/s]
    rho_ddot  = (|v_rel|^2 + rho . a_rel - rho_dot^2) / rho       [m/s^2]

The range-acceleration form is the exact second derivative of |rho|; it is
obtained by differentiating rho_dot and is standard (Vallado, 4th ed.,
range-rate observable derivation).

Assumptions and validity
------------------------
* Both states are expressed in the **same inertial (non-rotating) frame** at
  the **same epoch**.  Mixing an Earth-fixed station position with an
  inertial satellite position is the single largest error source available
  here; use :mod:`dopplerkit.frames` to put the station into TEME with its
  rotation velocity included.
* Geometry only: no light-time correction (that is
  :mod:`dopplerkit.lighttime`), no aberration, no atmospheric delay, no
  relativity (see :mod:`dopplerkit.doppler` for the order kept and dropped).
* Valid for any separation except exactly zero, which raises.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

RANGE_RATE_CONVENTION = (
    "range-rate positive when separating (receding); "
    "rho = r_target - r_observer; rho_dot = d|rho|/dt"
)
"""One-line statement of the sign convention, carried on every result object."""


def _as_vec3(value: object, name: str) -> np.ndarray:
    """Coerce to a finite float array of shape (3,) or raise ``ValueError``."""
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite, got {arr!r}")
    return arr


@dataclass(frozen=True)
class State:
    """Inertial position and velocity of one endpoint of a link.

    Parameters
    ----------
    position_m
        Position in a common inertial frame [m], shape (3,).
    velocity_mps
        Velocity in the same frame [m/s], shape (3,).
    acceleration_mps2
        Optional acceleration in the same frame [m/s^2], shape (3,).  Defaults
        to zero, in which case :func:`range_acceleration_mps2` returns the
        kinematic part only and you must supply accelerations yourself if you
        need an exact Doppler rate.
    label
        Free-text identifier used in error messages and profile output.

    Notes
    -----
    Frozen and validated on construction; raises ``ValueError`` naming the
    offending field.  No frame is attached to the object -- the caller is
    responsible for both states being in the same inertial frame at the same
    epoch, and :mod:`dopplerkit.frames` exists to make that easy for a ground
    station.
    """

    position_m: np.ndarray
    velocity_mps: np.ndarray
    acceleration_mps2: np.ndarray | None = None
    label: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "position_m", _as_vec3(self.position_m, "position_m"))
        object.__setattr__(self, "velocity_mps", _as_vec3(self.velocity_mps, "velocity_mps"))
        if self.acceleration_mps2 is None:
            object.__setattr__(self, "acceleration_mps2", np.zeros(3))
        else:
            object.__setattr__(
                self, "acceleration_mps2", _as_vec3(self.acceleration_mps2, "acceleration_mps2")
            )
        if not isinstance(self.label, str):
            raise TypeError(f"label must be str, got {type(self.label).__name__}")

    @property
    def speed_mps(self) -> float:
        """Inertial speed |v| [m/s]."""
        return float(np.linalg.norm(self.velocity_mps))


def relative_position_m(observer: State, target: State) -> np.ndarray:
    """Line-of-sight vector rho = r_target - r_observer [m], shape (3,).

    Points **from the observer to the target**.  Reversing the arguments
    negates the result; range and range-rate are unaffected (see module
    docstring).
    """
    return target.position_m - observer.position_m


def slant_range_m(observer: State, target: State) -> float:
    """Scalar range |rho| [m] between two states in a common inertial frame.

    Symmetric in its arguments.  Raises ``ValueError`` if the two positions
    coincide, because range-rate and Doppler are undefined there.
    """
    rng = float(np.linalg.norm(relative_position_m(observer, target)))
    if rng == 0.0:
        raise ValueError(
            "observer and target positions coincide; range-rate and Doppler are undefined"
        )
    return rng


def range_rate_mps(observer: State, target: State) -> float:
    """Analytic range-rate rho_dot = (rho . v_rel)/|rho| [m/s].

    Sign convention: **positive when the observer and target are separating
    (receding)**, negative when approaching, zero at closest approach.  This
    is the time derivative of the scalar range, per the range-rate observation
    equation in Vallado, *Fundamentals of Astrodynamics and Applications*,
    4th ed.

    Direction: **invariant** under swapping ``observer`` and ``target`` --
    range-rate is a property of the pair, not of a link direction.  Uplink and
    downlink share one range-rate at a single epoch.

    Units: both states in metres and metres per second in one common inertial
    frame; the return value is metres per second.

    Validity: exact for the given states; no light-time, aberration or
    relativistic correction is applied.

    Raises
    ------
    ValueError
        If the two positions coincide.
    """
    rho = relative_position_m(observer, target)
    rng = slant_range_m(observer, target)
    v_rel = target.velocity_mps - observer.velocity_mps
    return float(rho @ v_rel / rng)


def range_acceleration_mps2(observer: State, target: State) -> float:
    """Analytic range-acceleration rho_ddot [m/s^2], the second derivative of |rho|.

    ``rho_ddot = (|v_rel|^2 + rho . a_rel - rho_dot^2) / |rho|``, the exact
    second derivative of the scalar range (Vallado, 4th ed., range-rate
    observable derivation).

    Sign convention: inherits the range-rate convention -- positive
    rho_ddot means the recession rate is increasing.  Feeds
    :func:`dopplerkit.doppler.doppler_rate_hz_per_s`, where it acquires a
    minus sign, so a positive rho_ddot gives a **negative** Doppler rate.

    Units: [m/s^2].  Uses ``State.acceleration_mps2``, which defaults to zero;
    with zero accelerations the result is the purely kinematic
    ``(|v_rel|^2 - rho_dot^2)/rho`` term and is *not* the true range
    acceleration of an orbiting body.  Supply accelerations, or use the
    closed-form model in :mod:`dopplerkit.analytic`, when the Doppler rate
    matters.
    """
    rho = relative_position_m(observer, target)
    rng = slant_range_m(observer, target)
    v_rel = target.velocity_mps - observer.velocity_mps
    a_rel = target.acceleration_mps2 - observer.acceleration_mps2
    rho_dot = float(rho @ v_rel / rng)
    return float((v_rel @ v_rel + rho @ a_rel - rho_dot**2) / rng)


def range_rate_finite_difference_mps(
    range_fn: Callable[[float], float],
    t_s: float,
    step_s: float,
) -> float:
    """Central-difference range-rate from a scalar range function [m/s].

    ``(range_fn(t + h/2) - range_fn(t - h/2)) / h`` -- the second-order
    accurate central difference, truncation error ``(h^2/24) * rho'''(t)``
    (standard result, e.g. Press et al., *Numerical Recipes*, finite-difference
    section).  Same sign convention as :func:`range_rate_mps`: positive when
    the range is increasing.

    Parameters
    ----------
    range_fn
        Callable mapping time [s] to scalar range [m].
    t_s
        Epoch at which the derivative is wanted [s].
    step_s
        Total differencing interval h [s], must be > 0.  The samples are taken
        at ``t +/- h/2``.

    Notes
    -----
    This function exists to *check* :func:`range_rate_mps`, not to replace it.
    Observed convergence order and the round-off floor for a representative
    LEO pass are measured in ``validation/validate_range_rate.py``.
    """
    if not np.isfinite(step_s) or step_s <= 0.0:
        raise ValueError(f"step_s must be a positive finite number, got {step_s}")
    half = 0.5 * step_s
    return float((range_fn(t_s + half) - range_fn(t_s - half)) / step_s)


__all__ = [
    "RANGE_RATE_CONVENTION",
    "State",
    "range_acceleration_mps2",
    "range_rate_finite_difference_mps",
    "range_rate_mps",
    "relative_position_m",
    "slant_range_m",
]
