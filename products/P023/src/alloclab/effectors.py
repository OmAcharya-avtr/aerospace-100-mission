"""Effector configuration models and the general control-effectiveness matrix.

The control allocation problem solved throughout this package is

    tau = B u,    u_min <= u <= u_max                                    (1)

where ``tau`` is the desired body torque [N*m], ``u`` is the vector of
individual effector commands, and ``B`` (3 x m) is the control-effectiveness
matrix whose column ``i`` is the body torque produced per unit of command
``u_i``.  This is the standard control-allocation formulation of
Durham (1993) "Constrained Control Allocation", J. Guidance, Control, and
Dynamics 16(4), 717-725, doi:10.2514/3.21072, and of Bodson (2002)
"Evaluation of Optimization Methods for Control Allocation", J. Guidance,
Control, and Dynamics 25(4), 703-711, doi:10.2514/2.4937.

Assumptions common to every model in this module:

* Rigid body, body-fixed effectors, effectiveness independent of the body
  state (no aerodynamic or flexible-mode dependence).
* Linear, instantaneous effectors: no actuator dynamics, no rate limits, no
  minimum impulse bit, no thruster rise/fall transients.
* Torque only. Net force produced by a thruster cluster is computed but is not
  a controlled variable here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "EffectorSet",
    "general_effector_set",
    "thruster_cluster",
    "reaction_wheel_array",
    "pyramid_reaction_wheels",
    "orthogonal_effectors",
]

_TOL = 1e-9


def _as_2d(matrix: ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got shape {arr.shape}")
    if arr.shape[0] != 3:
        raise ValueError(f"{name} must have 3 rows (body torque axes), got {arr.shape[0]}")
    if arr.shape[1] < 1:
        raise ValueError(f"{name} must have at least one column (effector)")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite entries")
    return arr


@dataclass(eq=False)
class EffectorSet:
    """A set of bounded effectors described by a control-effectiveness matrix.

    Parameters
    ----------
    matrix
        ``(3, m)`` control-effectiveness matrix ``B`` [N*m per command unit].
        Column ``i`` is the body torque produced by one unit of ``u_i``.
    lower, upper
        ``(m,)`` command bounds, in the same command units as ``B``'s columns.
        ``lower[i] == upper[i]`` is allowed and denotes a fixed (failed-off or
        stuck) effector.
    names
        Optional per-effector labels.
    units
        Free-text command unit label, for reporting only (e.g. ``"N"`` for
        thrust, ``"N*m"`` for wheel motor torque).
    """

    matrix: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    names: tuple[str, ...] = field(default=())
    units: str = "command"

    def __post_init__(self) -> None:
        self.matrix = _as_2d(self.matrix, "matrix")
        m = self.matrix.shape[1]
        self.lower = np.asarray(self.lower, dtype=float).reshape(-1)
        self.upper = np.asarray(self.upper, dtype=float).reshape(-1)
        if self.lower.shape != (m,):
            raise ValueError(f"lower must have shape ({m},), got {self.lower.shape}")
        if self.upper.shape != (m,):
            raise ValueError(f"upper must have shape ({m},), got {self.upper.shape}")
        if not np.all(np.isfinite(self.lower)) or not np.all(np.isfinite(self.upper)):
            raise ValueError("command bounds must be finite")
        if np.any(self.upper < self.lower):
            bad = int(np.argmax(self.upper < self.lower))
            raise ValueError(
                f"upper bound must be >= lower bound; effector {bad} has "
                f"lower={self.lower[bad]!r} upper={self.upper[bad]!r}"
            )
        if not self.names:
            self.names = tuple(f"e{i}" for i in range(m))
        else:
            self.names = tuple(str(n) for n in self.names)
            if len(self.names) != m:
                raise ValueError(f"names must have {m} entries, got {len(self.names)}")

    # -- basic properties ------------------------------------------------

    @property
    def n_effectors(self) -> int:
        """Number of effectors ``m``."""
        return self.matrix.shape[1]

    @property
    def span(self) -> np.ndarray:
        """``upper - lower`` per effector [command units]."""
        return self.upper - self.lower

    @property
    def rank(self) -> int:
        """Rank of ``B`` restricted to effectors that can still move.

        Fixed effectors (``lower == upper``) contribute no controllable
        authority and are excluded, so ``rank < 3`` means at least one body
        axis is uncontrollable.
        """
        free = self.free_mask()
        if not np.any(free):
            return 0
        return int(np.linalg.matrix_rank(self.matrix[:, free]))

    def free_mask(self, tol: float = _TOL) -> np.ndarray:
        """Boolean ``(m,)`` mask of effectors whose command can still vary."""
        return self.span > tol

    def torque(self, commands: ArrayLike) -> np.ndarray:
        """Body torque ``B u`` [N*m] produced by ``commands``."""
        u = np.asarray(commands, dtype=float)
        if u.shape[-1] != self.n_effectors:
            raise ValueError(
                f"commands must have last dimension {self.n_effectors}, got {u.shape[-1]}"
            )
        return u @ self.matrix.T

    def clip(self, commands: ArrayLike) -> np.ndarray:
        """Clip ``commands`` into the box ``[lower, upper]``."""
        u = np.asarray(commands, dtype=float)
        return np.clip(u, self.lower, self.upper)

    def bound_violation(self, commands: ArrayLike) -> np.ndarray:
        """Per-command maximum bound violation [command units], 0 if inside.

        Returns a scalar for a 1-D ``commands`` input and a ``(n,)`` array for
        a 2-D ``(n, m)`` batch.
        """
        u = np.asarray(commands, dtype=float)
        if u.shape[-1] != self.n_effectors:
            raise ValueError(
                f"commands must have last dimension {self.n_effectors}, got {u.shape[-1]}"
            )
        viol = np.maximum(np.maximum(self.lower - u, u - self.upper), 0.0)
        return viol.max(axis=-1)

    def within_bounds(self, commands: ArrayLike, tol: float = 1e-9) -> np.ndarray:
        """True where every command lies inside ``[lower, upper]`` to ``tol``."""
        return self.bound_violation(commands) <= tol

    # -- failure modelling -----------------------------------------------

    def with_failures(
        self,
        failed: ArrayLike,
        stuck_at: ArrayLike | None = None,
    ) -> EffectorSet:
        """Return a copy with the listed effectors failed.

        Parameters
        ----------
        failed
            Indices of failed effectors.
        stuck_at
            Command value each failed effector is stuck at. Scalar or
            per-failure array. ``None`` (default) means failed-off, i.e. stuck
            at 0 for two-sided effectors and at ``lower`` for one-sided ones
            whose range does not contain 0.

        The failed effectors keep their columns in ``B`` but get
        ``lower == upper``, so their contribution becomes a fixed bias that
        every allocator must work around rather than a free variable.
        """
        idx = np.atleast_1d(np.asarray(failed, dtype=int))
        if idx.size and (idx.min() < 0 or idx.max() >= self.n_effectors):
            raise ValueError(
                f"failed indices must be in [0, {self.n_effectors - 1}], got {idx.tolist()}"
            )
        lower = self.lower.copy()
        upper = self.upper.copy()
        if stuck_at is None:
            values = np.clip(0.0, self.lower[idx], self.upper[idx])
        else:
            values = np.broadcast_to(np.asarray(stuck_at, dtype=float), idx.shape).copy()
            outside = (values < self.lower[idx] - _TOL) | (values > self.upper[idx] + _TOL)
            if np.any(outside):
                raise ValueError("stuck_at values must lie inside the nominal command bounds")
        lower[idx] = values
        upper[idx] = values
        return EffectorSet(self.matrix.copy(), lower, upper, self.names, self.units)

    def health(self, tol: float = _TOL) -> np.ndarray:
        """``(m,)`` float mask, 1.0 for effectors that can still move, else 0.0."""
        return self.free_mask(tol).astype(float)

    def summary(self) -> str:
        """One-line-per-effector text summary, for the CLI."""
        lines = [
            f"EffectorSet: {self.n_effectors} effectors, rank {self.rank}, units [{self.units}]"
        ]
        for i, name in enumerate(self.names):
            col = self.matrix[:, i]
            lines.append(
                f"  {name:>8s}  b=[{col[0]:+.4f} {col[1]:+.4f} {col[2]:+.4f}]  "
                f"u in [{self.lower[i]:+.4f}, {self.upper[i]:+.4f}]"
            )
        return "\n".join(lines)


def general_effector_set(
    matrix: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
    names: tuple[str, ...] = (),
    units: str = "command",
) -> EffectorSet:
    """Build an :class:`EffectorSet` directly from an effectiveness matrix.

    Use this when the effector hardware is not a thruster cluster or a wheel
    array -- control surfaces, magnetorquers, gimballed engines -- and the
    ``(3, m)`` torque-per-command matrix is known from elsewhere.
    """
    return EffectorSet(np.asarray(matrix, dtype=float), lower, upper, names, units)


def thruster_cluster(
    positions: ArrayLike,
    directions: ArrayLike,
    max_thrust: ArrayLike,
    min_thrust: ArrayLike = 0.0,
    names: tuple[str, ...] = (),
) -> EffectorSet:
    """Effector set for a cluster of on-off-throttleable thrusters.

    The torque produced by thruster ``i`` firing with thrust magnitude
    ``u_i`` [N] is the moment of the thrust force about the body origin,

        L_i = r_i x (F_hat_i * u_i)                                       (2)

    so ``B[:, i] = r_i x F_hat_i`` [m], giving torque in N*m for thrust in N.
    Equation (2) is the elementary rigid-body moment definition; see e.g.
    Markley & Crassidis (2014), "Fundamentals of Spacecraft Attitude
    Determination and Control", Springer, Chapter 7 (actuator models).

    Parameters
    ----------
    positions
        ``(m, 3)`` thruster nozzle positions relative to the body origin
        (normally the centre of mass) [m].
    directions
        ``(m, 3)`` direction of the thrust force **applied to the body**
        (opposite the exhaust plume). Normalised internally.
    max_thrust
        Scalar or ``(m,)`` maximum thrust [N].
    min_thrust
        Scalar or ``(m,)`` minimum thrust [N], default 0. Thrusters are
        one-sided: the command box is ``[min_thrust, max_thrust]`` and cannot
        include negative thrust. A non-zero ``min_thrust`` models a thruster
        that cannot be commanded below a floor once open; it does **not**
        model minimum-impulse-bit or pulse-width behaviour.

    Notes
    -----
    Validity: thrust magnitudes are treated as continuous within the bounds.
    Pulse-width modulation, minimum impulse bit, plume impingement, and
    centre-of-mass migration are all outside this model.
    """
    pos = np.asarray(positions, dtype=float)
    dirs = np.asarray(directions, dtype=float)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"positions must be (m, 3), got {pos.shape}")
    if dirs.shape != pos.shape:
        raise ValueError(f"directions must match positions shape {pos.shape}, got {dirs.shape}")
    norms = np.linalg.norm(dirs, axis=1)
    if np.any(norms < _TOL):
        raise ValueError("every thrust direction must be a non-zero vector")
    unit = dirs / norms[:, None]
    matrix = np.cross(pos, unit).T  # (3, m)

    m = pos.shape[0]
    hi = np.broadcast_to(np.asarray(max_thrust, dtype=float), (m,)).astype(float)
    lo = np.broadcast_to(np.asarray(min_thrust, dtype=float), (m,)).astype(float)
    if np.any(lo < 0.0):
        raise ValueError("min_thrust must be >= 0; thrusters cannot pull")
    if np.any(hi <= 0.0):
        raise ValueError("max_thrust must be > 0")
    return EffectorSet(matrix, lo, hi, names, units="N")


def reaction_wheel_array(
    spin_axes: ArrayLike,
    max_torque: ArrayLike,
    names: tuple[str, ...] = (),
) -> EffectorSet:
    """Effector set for an array of reaction wheels.

    ``u_i`` is the motor torque applied **to wheel i** about its spin axis
    ``a_hat_i`` [N*m]. By conservation of angular momentum the reaction on the
    spacecraft body is equal and opposite,

        L_body = -sum_i a_hat_i * u_i                                     (3)

    so ``B[:, i] = -a_hat_i`` (dimensionless), giving body torque in N*m for
    wheel motor torque in N*m. See Markley & Crassidis (2014), Chapter 7, or
    Wie (2008), "Space Vehicle Dynamics and Control", 2nd ed., AIAA, Chapter 7.

    Bounds are symmetric, ``[-max_torque, +max_torque]``, because a wheel can
    be accelerated or decelerated.

    Notes
    -----
    Validity: torque-limited operation only. Wheel momentum saturation, the
    gyroscopic ``omega x h`` cross-coupling, motor speed-torque roll-off,
    static friction and zero-crossing behaviour are not modelled. A momentum
    envelope check must be done separately.
    """
    axes = np.asarray(spin_axes, dtype=float)
    if axes.ndim != 2 or axes.shape[1] != 3:
        raise ValueError(f"spin_axes must be (m, 3), got {axes.shape}")
    norms = np.linalg.norm(axes, axis=1)
    if np.any(norms < _TOL):
        raise ValueError("every wheel spin axis must be a non-zero vector")
    unit = axes / norms[:, None]
    matrix = -unit.T
    m = axes.shape[0]
    hi = np.broadcast_to(np.asarray(max_torque, dtype=float), (m,)).astype(float)
    if np.any(hi <= 0.0):
        raise ValueError("max_torque must be > 0")
    return EffectorSet(matrix, -hi, hi, names, units="N*m")


def pyramid_reaction_wheels(
    max_torque: float = 0.1,
    half_angle_deg: float = 54.7356103172,
    n_wheels: int = 4,
) -> EffectorSet:
    """Standard ``n``-wheel pyramid array about the body ``+z`` axis.

    Spin axis ``i`` is at ``half_angle_deg`` from ``+z``, at azimuth
    ``2*pi*i/n``. The default half angle is ``arctan(sqrt(2))`` = 54.7356 deg,
    at which the four-wheel array is isotropic: with axes at azimuths 0, 90,
    180 and 270 deg,

        sum_i a_i a_i^T = 2 sin^2(beta) (xx^T + yy^T) + 4 cos^2(beta) zz^T

    which equals ``(4/3) I`` exactly when ``tan^2(beta) = 2``, so the array has
    the same least-squares authority about every body axis. This is the
    standard skewed four-wheel pyramid described in Markley & Crassidis (2014),
    "Fundamentals of Spacecraft Attitude Determination and Control", Springer,
    Chapter 7, and Wie (2008), "Space Vehicle Dynamics and Control", 2nd ed.,
    AIAA, Chapter 7.

    Parameters
    ----------
    max_torque
        Per-wheel motor torque limit [N*m], symmetric.
    half_angle_deg
        Cone half angle from ``+z`` [deg], in ``(0, 90]``.
    n_wheels
        Number of wheels, at least 3 (3 wheels on a cone still span R^3).
    """
    if n_wheels < 3:
        raise ValueError(f"n_wheels must be >= 3 to span three axes, got {n_wheels}")
    if not 0.0 < half_angle_deg <= 90.0:
        raise ValueError(f"half_angle_deg must be in (0, 90], got {half_angle_deg}")
    beta = np.deg2rad(half_angle_deg)
    az = 2.0 * np.pi * np.arange(n_wheels) / n_wheels
    axes = np.stack(
        [np.sin(beta) * np.cos(az), np.sin(beta) * np.sin(az), np.full(n_wheels, np.cos(beta))],
        axis=1,
    )
    names = tuple(f"rw{i + 1}" for i in range(n_wheels))
    return reaction_wheel_array(axes, max_torque, names)


def orthogonal_effectors(max_torque: ArrayLike = 1.0) -> EffectorSet:
    """Three orthogonal, symmetric, unit-effectiveness effectors.

    ``B = I_3`` with bounds ``[-max_torque, +max_torque]``. The attainable
    moment set is then exactly the box ``[-max_torque, max_torque]^3``, which
    is the closed-form reference case used in ``validation/validate_ams.py``.
    """
    hi = np.broadcast_to(np.asarray(max_torque, dtype=float), (3,)).astype(float)
    if np.any(hi <= 0.0):
        raise ValueError("max_torque must be > 0")
    return EffectorSet(np.eye(3), -hi, hi, ("x", "y", "z"), units="N*m")
