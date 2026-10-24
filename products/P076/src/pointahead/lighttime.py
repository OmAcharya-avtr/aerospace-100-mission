"""Light-time solutions for an optical crosslink: iterated and closed form.

Two light times matter at one event at the local terminal.

*Receive* light time ``tau_rx``: the photon arriving now left the remote
terminal when the remote terminal was where it was ``tau_rx`` ago. It solves

    tau = |r_remote(t - tau) - r_local(t)| / c.

*Transmit* light time ``tau_tx``: the photon leaving now will arrive when the
remote terminal has moved on by ``tau_tx``. It solves

    tau = |r_remote(t + tau) - r_local(t)| / c.

Both are solved by fixed-point iteration from ``tau_0 = range / c``, the
standard light-time iteration used for observable computation in deep-space
navigation (Moyer, *Formulation for Observed and Computed Values of Deep Space
Network Data Types for Navigation*, JPL Deep-Space Communications and
Navigation Series, Monograph 2, Wiley 2003, chapter 8). The iteration contracts
with factor ``|v_radial| / c``, which is below 1e-4 for any orbital crosslink,
so it converges in two or three iterations. That convergence is cheap and the
residual is small, which is exactly why the budget line for it has to be
*measured* rather than assumed: a product that iterates once and stops has a
real error and it should appear as its own line.

For a constant-velocity remote trajectory the equation is a quadratic in
``tau`` with a closed-form root, implemented in
:func:`linear_light_time_closed_form`. That gives the iterated solver an exact
reference rather than leaving it checked only against itself.

Units: all times in seconds, all positions in metres.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import C_LIGHT, as_vector
from .kepler import LinearTrajectory, Trajectory

DIRECTIONS = ("transmit", "receive")


@dataclass(frozen=True)
class LightTimeSolution:
    """Outcome of one light-time solve.

    Attributes
    ----------
    tau_s
        Converged (or truncated) light time in seconds.
    direction
        ``"transmit"`` or ``"receive"``.
    iterates
        Every ``tau`` visited, starting with ``range / c``. ``len(iterates) - 1``
        is the number of iterations performed.
    residual_s
        ``|r_remote(t +/- tau) - r_local| / c - tau`` at the returned ``tau``.
        This is the quantity the iteration drives to zero and it is reported,
        not assumed.
    converged
        Whether ``|residual_s|`` fell below ``tol_s`` within ``max_iter``.
    remote_position_m
        Remote position at the retarded or advanced epoch, metres.
    remote_velocity_mps
        Remote velocity at the same epoch, m/s.
    """

    tau_s: float
    direction: str
    iterates: tuple[float, ...]
    residual_s: float
    converged: bool
    remote_position_m: np.ndarray
    remote_velocity_mps: np.ndarray

    @property
    def iterations(self) -> int:
        """Number of fixed-point updates applied."""
        return len(self.iterates) - 1

    @property
    def range_m(self) -> float:
        """Light-time range ``c * tau``, metres."""
        return C_LIGHT * self.tau_s


def _sign(direction: str) -> float:
    if direction == "transmit":
        return 1.0
    if direction == "receive":
        return -1.0
    raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")


def solve_light_time(
    remote: Trajectory,
    local_position_m: object,
    epoch_s: float,
    direction: str = "transmit",
    tol_s: float = 1.0e-15,
    max_iter: int = 30,
) -> LightTimeSolution:
    """Solve the light-time equation by fixed-point iteration.

    Parameters
    ----------
    remote
        Trajectory of the far terminal, queried at ``epoch_s +/- tau``.
    local_position_m
        Position of the local terminal at ``epoch_s``, metres. The local
        terminal does not move during the solve: both events are at the same
        place for the local terminal by construction, because the point-ahead
        angle is defined at one event.
    epoch_s
        Absolute epoch of the local event, seconds.
    direction
        ``"transmit"`` solves for the advanced epoch, ``"receive"`` for the
        retarded one.
    tol_s
        Convergence tolerance on the light-time residual, seconds. The default
        of 1e-15 s is below the float64 resolution of a 1e-2 s light time, so
        the iteration normally stops on ``max_iter`` only if it is diverging;
        in practice it stops because consecutive iterates become equal.
    max_iter
        Iteration cap. Passing a small value deliberately -- 0 or 1 -- produces
        the truncated solution used for the light-time iteration budget line.

    Notes
    -----
    ``max_iter=0`` returns the zeroth iterate ``range / c`` with its residual
    measured, which is the honest representation of "no light-time iteration at
    all".
    """
    sign = _sign(direction)
    r_local = as_vector(local_position_m, "local_position_m")
    epoch = float(epoch_s)
    if not np.isfinite(epoch):
        raise ValueError(f"epoch_s must be finite, got {epoch_s!r}")
    cap = int(max_iter)
    if cap < 0:
        raise ValueError(f"max_iter must be non-negative, got {max_iter!r}")
    if not (np.isfinite(float(tol_s)) and float(tol_s) >= 0.0):
        raise ValueError(f"tol_s must be non-negative and finite, got {tol_s!r}")

    def evaluate(tau: float) -> tuple[float, np.ndarray, np.ndarray]:
        t = epoch + sign * tau
        position = remote.position(t)
        velocity = remote.velocity(t)
        return float(np.linalg.norm(position - r_local)) / C_LIGHT, position, velocity

    tau = float(np.linalg.norm(remote.position(epoch) - r_local)) / C_LIGHT
    iterates = [tau]
    converged = False
    mapped, position, velocity = evaluate(tau)
    for _ in range(cap):
        mapped, position, velocity = evaluate(tau)
        if abs(mapped - tau) <= float(tol_s):
            converged = True
            tau = mapped
            iterates.append(tau)
            mapped, position, velocity = evaluate(tau)
            break
        tau = mapped
        iterates.append(tau)
    else:
        mapped, position, velocity = evaluate(tau)
        converged = abs(mapped - tau) <= float(tol_s)

    return LightTimeSolution(
        tau_s=tau,
        direction=direction,
        iterates=tuple(iterates),
        residual_s=mapped - tau,
        converged=converged,
        remote_position_m=position,
        remote_velocity_mps=velocity,
    )


def linear_light_time_closed_form(
    trajectory: LinearTrajectory,
    local_position_m: object,
    epoch_s: float,
    direction: str = "transmit",
) -> float:
    """Exact light time for a constant-velocity remote trajectory, seconds.

    With ``d = r_remote(t) - r_local`` and ``v`` the constant remote velocity,
    the transmit equation ``|d + v tau| = c tau`` is the quadratic

        (v.v - c^2) tau^2 + 2 (d.v) tau + d.d = 0,

    whose single positive root is written here as

        tau = C / (sqrt(B^2 + (c^2 - v^2) C) - B),      B = d.v,  C = d.d,

    rather than as ``(B + sqrt(D)) / (c^2 - v^2)``. The two forms are
    algebraically identical; the second subtracts two nearly equal numbers when
    ``B`` is negative and large, which is the receding-geometry case, and loses
    significant digits exactly where a crosslink spends half its pass. The
    receive equation is the same with the sign of ``B`` flipped.

    Raises
    ------
    TypeError
        If ``trajectory`` is not a :class:`~pointahead.kepler.LinearTrajectory`;
        there is no closed form for a Keplerian remote trajectory and silently
        linearising one would be a wrong answer dressed as an exact one.
    """
    if not isinstance(trajectory, LinearTrajectory):
        raise TypeError(
            "linear_light_time_closed_form requires a LinearTrajectory; "
            f"got {type(trajectory).__name__}. A Keplerian trajectory has no closed-form "
            "light time and must go through solve_light_time."
        )
    sign = _sign(direction)
    r_local = as_vector(local_position_m, "local_position_m")
    d = trajectory.position(float(epoch_s)) - r_local
    v = trajectory.velocity(float(epoch_s))
    b = sign * float(np.dot(d, v))
    c_term = float(np.dot(d, d))
    speed_squared = float(np.dot(v, v))
    discriminant = b * b + (C_LIGHT * C_LIGHT - speed_squared) * c_term
    if discriminant < 0.0:  # pragma: no cover - needs |v| > c, refused upstream
        raise ValueError("no real light-time root; the remote trajectory is superluminal")
    return c_term / (float(np.sqrt(discriminant)) - b)


def light_time_residual(
    remote: Trajectory,
    local_position_m: object,
    epoch_s: float,
    tau_s: float,
    direction: str = "transmit",
) -> float:
    """Residual of the light-time equation at a given ``tau``, in seconds.

    Positive means the true light time is longer than ``tau``. This is the
    function the fixed-point iteration zeroes, exposed so a validation script
    can show the residual shrinking per iteration rather than asserting it.
    """
    sign = _sign(direction)
    r_local = as_vector(local_position_m, "local_position_m")
    t = float(epoch_s) + sign * float(tau_s)
    distance = float(np.linalg.norm(remote.position(t) - r_local))
    return distance / C_LIGHT - float(tau_s)
