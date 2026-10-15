"""Attitude kinematics. Imports quat and vecops."""

from sampleproj import quat, vecops


def propagate(q, omega, dt):
    """Advance attitude quaternion q by body rate omega (rad/s) over dt (s).

    First-order quaternion kinematics: q_dot = 0.5 * q (x) [0, omega].
    """
    if dt < 0.0:
        raise ValueError("dt must be non-negative")
    wq = [0.0, omega[0], omega[1], omega[2]]
    rate = quat.multiply(q, wq)
    advanced = [q[i] + 0.5 * dt * rate[i] for i in range(4)]
    return vecops.normalize(advanced)


def angular_momentum(inertia_diag, omega):
    """Body-frame angular momentum (kg m^2 / s) for a diagonal inertia."""
    if len(inertia_diag) != 3 or len(omega) != 3:
        raise ValueError("angular_momentum requires 3-element inputs")
    return [i * w for i, w in zip(inertia_diag, omega)]
