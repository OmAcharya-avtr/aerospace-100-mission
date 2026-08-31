"""Magnetorquer detumbling control laws.

Two laws are implemented, both textbook:

**B-dot** (Stickler & Alfriend 1976).  The commanded dipole opposes the
measured rate of change of the body-frame magnetic field,

    m = -k_bdot * dB/dt                                     [A m^2]

with ``k_bdot`` in ``A m^2 s T^-1``.  A magnitude-normalised variant,
``m = -k * (dB/dt) / |B|``, keeps the commanded dipole scale roughly constant
around the orbit; both are provided.

**Cross-product** (Wertz 1978 sec. 19.4; Markley & Crassidis 2014 sec. 7.5).
Given a desired torque ``L_des``, the dipole that produces the closest
achievable torque in a least-squares sense is

    m = (B x L_des) / |B|^2

Choosing the rate-damping law ``L_des = -k_w omega`` gives

    m = -k_w (B x omega) / |B|^2                            [A m^2]

with ``k_w`` in ``N m s`` (i.e. kg m^2 s^-1).

The controllability gap
-----------------------
The realised torque of *any* magnetorquer is ``L = m x B``, which is
identically perpendicular to ``B``.  For the cross-product law the realised
torque is

    L = m x B = -k_w [(B x omega) x B] / |B|^2
      = -k_w [omega - (omega . B_hat) B_hat]
      = -k_w omega_perp

so the component of the body rate along the instantaneous field direction is
**not damped at that instant**.  For the ideal B-dot law the same result
follows: if the field is slowly varying in inertial space compared with the
body rotation, ``dB/dt|_body ~= -omega x B``, so

    m = k_bdot (omega x B)  and  L = m x B = -k_bdot |B|^2 omega_perp

i.e. B-dot is the cross-product law with an effective, field-strength-weighted
gain ``k_w_eff = k_bdot |B|^2``.  ``detumblesim.controllability`` quantifies
what this costs over a whole orbit.

References
----------
Stickler, A. C. and Alfriend, K. T., "Elementary Magnetic Attitude Control
    System", Journal of Spacecraft and Rockets, vol. 13, no. 5, 1976,
    pp. 282-287.  doi:10.2514/3.57089
Wertz, J. R. (ed.), "Spacecraft Attitude Determination and Control", D. Reidel,
    1978, sec. 19.4 (magnetic control).
Markley, F. L. and Crassidis, J. L., "Fundamentals of Spacecraft Attitude
    Determination and Control", Springer, 2014, sec. 7.5.
Avanzini, G. and Giulietti, F., "Magnetic Detumbling of a Rigid Spacecraft",
    Journal of Guidance, Control, and Dynamics, vol. 35, no. 4, 2012,
    pp. 1326-1334.  doi:10.2514/1.53074
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

#: Smallest field magnitude [T] treated as usable by the cross-product law.
#: Below this the 1/|B|^2 normalisation is numerically meaningless; the LEO
#: field is ~2e-5 T so this bound is never reached in a physical run.
MIN_FIELD_T: float = 1e-12


def _check_gain(gain: float, name: str) -> float:
    g = float(gain)
    if not np.isfinite(g) or g <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {gain}")
    return g


@dataclass(frozen=True)
class BDotController:
    """Classic B-dot law ``m = -k dB/dt``.

    Parameters
    ----------
    gain : float
        ``k_bdot`` [A m^2 s T^-1].  Must be positive.
    normalise_by_field : bool
        If True, use ``m = -k (dB/dt) / |B|`` instead; ``gain`` then carries
        units of ``A m^2 s`` per unit of ``(T/s)/T``, i.e. ``A m^2 s``.
    """

    gain: float
    normalise_by_field: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "gain", _check_gain(self.gain, "gain"))

    def command(
        self, b_body_t: ArrayLike, b_dot_body_t_s: ArrayLike, omega_body: ArrayLike | None = None
    ) -> NDArray[np.float64]:
        """Commanded (unsaturated) dipole [A m^2]."""
        bd = np.asarray(b_dot_body_t_s, dtype=float)
        if bd.shape != (3,):
            raise ValueError(f"b_dot must have shape (3,), got {bd.shape}")
        if not self.normalise_by_field:
            return -self.gain * bd
        b = np.asarray(b_body_t, dtype=float)
        bn = float(np.linalg.norm(b))
        if bn < MIN_FIELD_T:
            return np.zeros(3)
        return -self.gain * bd / bn


@dataclass(frozen=True)
class CrossProductController:
    """Cross-product rate damping ``m = -k (B x omega) / |B|^2``.

    Parameters
    ----------
    gain : float
        ``k_w`` [N m s]; the realised torque is ``-k_w omega_perp``.

    Notes
    -----
    Unlike B-dot this law needs a rate estimate, so it is not a
    magnetometer-only law.  It is included because it makes the
    controllability gap explicit: the realised torque is exactly
    ``-k_w omega_perp``.
    """

    gain: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "gain", _check_gain(self.gain, "gain"))

    def command(
        self, b_body_t: ArrayLike, b_dot_body_t_s: ArrayLike | None = None,
        omega_body: ArrayLike | None = None
    ) -> NDArray[np.float64]:
        """Commanded (unsaturated) dipole [A m^2]."""
        if omega_body is None:
            raise ValueError("CrossProductController requires omega_body")
        b = np.asarray(b_body_t, dtype=float)
        w = np.asarray(omega_body, dtype=float)
        if b.shape != (3,) or w.shape != (3,):
            raise ValueError("b_body_t and omega_body must have shape (3,)")
        b2 = float(b @ b)
        if b2 < MIN_FIELD_T**2:
            return np.zeros(3)
        return -self.gain * np.cross(b, w) / b2


def magnetic_torque(dipole_am2: ArrayLike, b_body_t: ArrayLike) -> NDArray[np.float64]:
    """Torque ``L = m x B`` [N m] from a dipole [A m^2] in a field [T]."""
    m = np.asarray(dipole_am2, dtype=float)
    b = np.asarray(b_body_t, dtype=float)
    if m.shape != (3,) or b.shape != (3,):
        raise ValueError("dipole and field must have shape (3,)")
    return np.cross(m, b)


def ideal_bdot_torque(
    omega_body: ArrayLike, b_body_t: ArrayLike, gain: float
) -> NDArray[np.float64]:
    """Torque of the *ideal* B-dot law, ``L = -k |B|^2 omega_perp`` [N m].

    Uses the slowly-varying-field approximation ``dB/dt|_body = -omega x B``
    and no saturation.  This is the closed form used by the property tests and
    by ``analytic.py``; the simulator uses the finite-difference law instead.
    """
    w = np.asarray(omega_body, dtype=float)
    b = np.asarray(b_body_t, dtype=float)
    k = _check_gain(gain, "gain")
    b2 = float(b @ b)
    if b2 <= 0.0:
        return np.zeros(3)
    w_par = (float(w @ b) / b2) * b
    return -k * b2 * (w - w_par)


def perpendicular_component(
    vector: ArrayLike, direction: ArrayLike
) -> NDArray[np.float64]:
    """Component of ``vector`` perpendicular to ``direction`` (same units)."""
    v = np.asarray(vector, dtype=float)
    d = np.asarray(direction, dtype=float)
    d2 = float(d @ d)
    if d2 <= 0.0:
        return v.copy()
    return v - (float(v @ d) / d2) * d
