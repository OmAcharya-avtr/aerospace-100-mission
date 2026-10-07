"""The plant, the declared disturbance bound and the declared constraint sets.

Model
-----
Discrete-time linear time-invariant plant with a bounded additive disturbance,

    x_{k+1} = A x_k + B u_k + w_k,    w_k in W,                            (1)

with a declared state-constraint polytope ``X`` and a declared input-constraint
polytope ``U``. The word *declared* is load-bearing throughout this package:
every safety statement is conditional on ``w_k in W`` for all ``k``, and
:mod:`simplexguard.boundviolation` exists to measure what happens when that
assumption is false.

Assumptions, which bound what any result in this package means
-------------------------------------------------------------
1. ``A`` and ``B`` are exactly known. No parametric uncertainty is modelled; a
   model error is not representable as an additive ``w`` of a declared bound
   unless the state stays in a region where the two are equivalent, and this
   package does not check that.
2. ``w_k`` is an arbitrary sequence in ``W``, not a stochastic process. The
   guarantee is worst-case over sequences, so no distribution on ``W`` enters
   the switching condition. A distribution enters only in
   :mod:`simplexguard.simulate`, which has to draw something.
3. The full state ``x_k`` is measured exactly. There is no observer and no
   measurement noise. An output-feedback runtime guard needs a robust set in
   the estimation error as well, and that is not implemented here.
4. Control is applied with no delay: ``u_k`` computed from ``x_k`` affects
   ``x_{k+1}``. One step of computation delay changes the switching condition
   and would need the one-step-ahead set, not the current one.

The shipped reference plant
---------------------------
:func:`reference_plant` is a single-axis attitude regulation loop, discretised
from the rigid-body double integrator

    theta_dot_dot = u,

by exact zero-order-hold sampling at ``dt``, which gives

    A = [[1, dt], [0, 1]],    B = [[dt^2/2], [dt]].                        (2)

Equation (2) is the exact ZOH discretisation of a double integrator and not an
approximation of it (Franklin, Powell and Workman, *Digital Control of Dynamic
Systems*, 3rd ed., Addison-Wesley, 1998, chapter 4). The state is
``x = [theta, theta_dot]`` in ``[rad, rad/s]``, the input is a commanded
angular acceleration in ``rad/s^2``, and the disturbance bound represents an
unmodelled angular acceleration of magnitude ``w_accel`` acting for one sample,
entered as the corresponding state increment ``[w_accel dt^2/2, w_accel dt]``.

**The numbers in :func:`reference_plant` are illustrative.** They are chosen so
that the robust invariant set is non-trivial, the guard fires at a measurable
rate and the whole thing runs inside the compute budget. They are not a
measurement of any spacecraft, and nothing in this package qualifies a plant it
was not given.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .polytope import Box, Polytope, box

__all__ = ["Plant", "reference_plant"]


@dataclass(frozen=True)
class Plant:
    """Plant (1) together with its declared disturbance and constraint sets.

    Parameters
    ----------
    A :
        State matrix, shape ``(n, n)``, dimensionless.
    B :
        Input matrix, shape ``(n, m)``.
    disturbance :
        The **declared** set ``W`` containing every admissible ``w_k``, in the
        units of the state. Must be bounded and contain the origin for the
        invariant-set recursion to mean what it says.
    state_constraints :
        The declared polytope ``X``, in the units of the state.
    input_constraints :
        The declared polytope ``U``, in the units of the input.
    dt :
        Sample interval in seconds. Recorded for reporting; the recursion is
        in discrete time and does not use it.
    state_names, input_names :
        Labels with units, used by the CLI and the figures.
    """

    A: np.ndarray
    B: np.ndarray
    disturbance: Polytope
    state_constraints: Polytope
    input_constraints: Polytope
    dt: float = 1.0
    state_names: tuple[str, ...] = field(default=())
    input_names: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        a = np.atleast_2d(np.asarray(self.A, dtype=float))
        b = np.atleast_2d(np.asarray(self.B, dtype=float))
        if a.ndim != 2 or a.shape[0] != a.shape[1]:
            raise ValueError(f"A must be square, got shape {a.shape}")
        if b.shape[0] != a.shape[0]:
            raise ValueError(f"B has {b.shape[0]} rows, A has {a.shape[0]}")
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            raise ValueError("A and B must be finite")
        n, m = a.shape[0], b.shape[1]
        if self.disturbance.dim != n:
            raise ValueError(f"disturbance set has dim {self.disturbance.dim}, expected {n}")
        if self.state_constraints.dim != n:
            raise ValueError(
                f"state constraints have dim {self.state_constraints.dim}, expected {n}"
            )
        if self.input_constraints.dim != m:
            raise ValueError(
                f"input constraints have dim {self.input_constraints.dim}, expected {m}"
            )
        if float(self.dt) <= 0.0:
            raise ValueError(f"dt must be positive, got {self.dt}")
        if not self.disturbance.contains(np.zeros(n), tol=0.0):
            raise ValueError("the declared disturbance set must contain the origin")
        object.__setattr__(self, "A", a)
        object.__setattr__(self, "B", b)
        names = tuple(self.state_names) or tuple(f"x{i}" for i in range(n))
        unames = tuple(self.input_names) or tuple(f"u{i}" for i in range(m))
        if len(names) != n or len(unames) != m:
            raise ValueError("state_names/input_names length does not match the plant")
        object.__setattr__(self, "state_names", names)
        object.__setattr__(self, "input_names", unames)

    @property
    def n_states(self) -> int:
        """State dimension ``n``."""
        return int(self.A.shape[0])

    @property
    def n_inputs(self) -> int:
        """Input dimension ``m``."""
        return int(self.B.shape[1])

    def step(self, x: np.ndarray, u: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
        """One application of (1). ``w`` defaults to zero and is **not** checked
        against the declared set: :mod:`simplexguard.boundviolation` deliberately
        passes disturbances outside it."""
        xv = np.asarray(x, dtype=float).ravel()
        uv = np.atleast_1d(np.asarray(u, dtype=float)).ravel()
        if xv.shape[0] != self.n_states:
            raise ValueError(f"x has length {xv.shape[0]}, expected {self.n_states}")
        if uv.shape[0] != self.n_inputs:
            raise ValueError(f"u has length {uv.shape[0]}, expected {self.n_inputs}")
        nxt = self.A @ xv + self.B @ uv
        if w is not None:
            wv = np.asarray(w, dtype=float).ravel()
            if wv.shape[0] != self.n_states:
                raise ValueError(f"w has length {wv.shape[0]}, expected {self.n_states}")
            nxt = nxt + wv
        return nxt

    def disturbance_is_admissible(self, w: np.ndarray, tol: float = 1e-12) -> bool:
        """True if ``w`` lies in the declared set ``W``."""
        return self.disturbance.contains(w, tol=tol)

    def with_disturbance(self, disturbance: Polytope) -> Plant:
        """A copy with a different declared disturbance set."""
        return Plant(
            A=self.A,
            B=self.B,
            disturbance=disturbance,
            state_constraints=self.state_constraints,
            input_constraints=self.input_constraints,
            dt=self.dt,
            state_names=self.state_names,
            input_names=self.input_names,
        )

    def describe(self) -> str:
        """Human-readable summary of the declared model, for the CLI."""
        lines = [
            f"states              {self.n_states}  {', '.join(self.state_names)}",
            f"inputs              {self.n_inputs}  {', '.join(self.input_names)}",
            f"sample interval     {self.dt:g} s",
            f"A                   {np.array2string(self.A, precision=6)}",
            f"B                   {np.array2string(self.B, precision=6)}",
            "open-loop |eig(A)|  "
            + np.array2string(np.abs(np.linalg.eigvals(self.A)), precision=6),
        ]
        if isinstance(self.disturbance, Box):
            lines.append(
                "declared W (box)    half-widths "
                f"{np.array2string(self.disturbance.half_widths, precision=8)}"
            )
        else:
            lines.append(f"declared W          {self.disturbance.n_halfspaces} halfspaces")
        lines.append(f"declared X          {self.state_constraints.n_halfspaces} halfspaces")
        lines.append(f"declared U          {self.input_constraints.n_halfspaces} halfspaces")
        return "\n".join(lines)


def reference_plant(
    dt: float = 0.05,
    angle_limit_rad: float = 0.30,
    rate_limit_rad_s: float = 0.50,
    accel_limit_rad_s2: float = 3.0,
    disturbance_accel_rad_s2: float = 0.12,
) -> Plant:
    """The shipped illustrative single-axis attitude plant of equation (2).

    Parameters
    ----------
    dt :
        Sample interval, s. Default 0.05 s (20 Hz).
    angle_limit_rad :
        Declared bound on ``|theta|``, rad. Default 0.30 rad (17.2 deg).
    rate_limit_rad_s :
        Declared bound on ``|theta_dot|``, rad/s. Default 0.50 rad/s.
    accel_limit_rad_s2 :
        Declared bound on ``|u|``, rad/s^2. Default 3.0 rad/s^2.
    disturbance_accel_rad_s2 :
        Declared bound on the unmodelled angular acceleration, rad/s^2, entered
        as the one-sample state increment ``[a dt^2/2, a dt]``. Default
        0.12 rad/s^2, which at ``dt = 0.05 s`` is a state increment of
        ``[1.5e-4 rad, 6.0e-3 rad/s]``.

    All five numbers are illustrative, as the module docstring says.
    """
    for name, value in (
        ("dt", dt),
        ("angle_limit_rad", angle_limit_rad),
        ("rate_limit_rad_s", rate_limit_rad_s),
        ("accel_limit_rad_s2", accel_limit_rad_s2),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive, got {value}")
    if not np.isfinite(disturbance_accel_rad_s2) or disturbance_accel_rad_s2 < 0.0:
        raise ValueError(
            f"disturbance_accel_rad_s2 must be finite and non-negative, "
            f"got {disturbance_accel_rad_s2}"
        )
    a = np.array([[1.0, dt], [0.0, 1.0]])
    b = np.array([[0.5 * dt * dt], [dt]])
    w = box(np.array([0.5 * dt * dt, dt]) * disturbance_accel_rad_s2)
    x_set = box([angle_limit_rad, rate_limit_rad_s])
    u_set = box([accel_limit_rad_s2])
    return Plant(
        A=a,
        B=b,
        disturbance=w,
        state_constraints=x_set,
        input_constraints=u_set,
        dt=dt,
        state_names=("theta [rad]", "theta_dot [rad/s]"),
        input_names=("u [rad/s^2]",),
    )
