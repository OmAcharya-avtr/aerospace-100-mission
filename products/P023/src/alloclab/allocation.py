"""Allocation methods: pseudo-inverse, weighted pseudo-inverse, LP and QP.

All four solve, in increasing order of what they respect,

    find u such that  B u = tau,  with  u_min <= u <= u_max               (1)

for the over-actuated case ``m > 3``. The unconstrained generalized inverses
solve only the equality; the LP and QP formulations carry the box constraint
explicitly. Terminology and formulations follow Bodson (2002),
doi:10.2514/2.4937, and Haerkegaard (2002), "Efficient active set algorithms
for solving constrained least squares problems in aircraft control
allocation", Proc. 41st IEEE CDC, 1295-1300, doi:10.1109/cdc.2002.1184694.

Every allocator returns an :class:`AllocationResult` whose ``status`` says what
actually happened. A command that cannot be met is reported as
``status="infeasible"`` with the residual attached; it is never silently
clipped and returned as if it had succeeded.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import linprog, lsq_linear

from .effectors import EffectorSet

__all__ = [
    "AllocationResult",
    "InfeasibleAllocationError",
    "pseudo_inverse_allocate",
    "weighted_pseudo_inverse_allocate",
    "redistributed_pseudo_inverse_allocate",
    "lp_allocate",
    "qp_allocate",
    "allocate",
    "is_attainable",
    "METHODS",
]

METHODS = ("pinv", "wpinv", "rpi", "lp", "qp")

#: Default absolute tolerance on the torque residual, in N*m. Chosen as
#: 1e-8 N*m: far below any physically meaningful torque for the effector
#: scales modelled here, and comfortably above the ~1e-13 N*m round-off of a
#: double-precision solve on a well-conditioned 3 x m system.
DEFAULT_TORQUE_TOL = 1e-8

#: Default absolute tolerance on a command-bound violation, in command units.
DEFAULT_BOUND_TOL = 1e-9


class InfeasibleAllocationError(RuntimeError):
    """Raised when a caller demanded a torque the effector set cannot produce."""


@dataclass
class AllocationResult:
    """Outcome of one allocation.

    Attributes
    ----------
    commands
        ``(m,)`` effector commands [command units of the effector set].
    achieved_torque
        ``B u`` [N*m].
    desired_torque
        The requested torque [N*m].
    residual
        ``tau_desired - B u`` [N*m].
    residual_norm
        2-norm of ``residual`` [N*m].
    bound_violation
        Largest amount by which any command exceeds its bound [command units];
        0.0 when the solution is inside the box.
    feasible
        True when the achieved torque matches the command to ``torque_tol``
        **and** the commands respect their bounds to ``bound_tol``.
    status
        ``"exact"``, ``"saturated"`` or ``"infeasible"``. ``"saturated"`` means
        the solution sits on the boundary of the command box but still meets
        the torque; ``"infeasible"`` means the torque was not met.
    method
        Name of the allocator that produced this result.
    message
        Human-readable note; carries the solver status for LP/QP failures.
    solve_time_s
        Wall-clock time of the solve [s].
    """

    commands: np.ndarray
    achieved_torque: np.ndarray
    desired_torque: np.ndarray
    residual: np.ndarray
    residual_norm: float
    bound_violation: float
    feasible: bool
    status: str
    method: str
    message: str = ""
    solve_time_s: float = 0.0
    extras: dict = field(default_factory=dict)

    def __str__(self) -> str:  # pragma: no cover - formatting only
        return (
            f"{self.method}: status={self.status} residual={self.residual_norm:.3e} N*m "
            f"bound_violation={self.bound_violation:.3e}"
        )


def _check_torque(torque: ArrayLike) -> np.ndarray:
    tau = np.asarray(torque, dtype=float).reshape(-1)
    if tau.shape != (3,):
        raise ValueError(f"torque must have shape (3,), got {np.shape(torque)}")
    if not np.all(np.isfinite(tau)):
        raise ValueError("torque must be finite")
    return tau


def _finish(
    eset: EffectorSet,
    u: np.ndarray,
    tau: np.ndarray,
    method: str,
    torque_tol: float,
    bound_tol: float,
    message: str = "",
    solve_time_s: float = 0.0,
    extras: dict | None = None,
) -> AllocationResult:
    achieved = eset.matrix @ u
    residual = tau - achieved
    rnorm = float(np.linalg.norm(residual))
    viol = float(eset.bound_violation(u))
    meets_torque = rnorm <= torque_tol
    inside = viol <= bound_tol
    if meets_torque and inside:
        on_boundary = bool(
            np.any(
                (np.abs(u - eset.lower) <= bound_tol) | (np.abs(u - eset.upper) <= bound_tol)
            )
        )
        status = "saturated" if on_boundary else "exact"
        feasible = True
    else:
        status = "infeasible"
        feasible = False
        if not message:
            if not inside:
                message = (
                    f"solution violates command bounds by {viol:.3e}; "
                    "this allocator does not enforce them"
                )
            else:
                message = (
                    f"desired torque is outside the attainable moment set; "
                    f"best achievable residual {rnorm:.3e} N*m"
                )
    return AllocationResult(
        commands=u,
        achieved_torque=achieved,
        desired_torque=tau,
        residual=residual,
        residual_norm=rnorm,
        bound_violation=viol,
        feasible=feasible,
        status=status,
        method=method,
        message=message,
        solve_time_s=solve_time_s,
        extras=extras or {},
    )


# ----------------------------------------------------------------------
# 1. Pseudo-inverse
# ----------------------------------------------------------------------


def pseudo_inverse_allocate(
    eset: EffectorSet,
    torque: ArrayLike,
    torque_tol: float = DEFAULT_TORQUE_TOL,
    bound_tol: float = DEFAULT_BOUND_TOL,
) -> AllocationResult:
    """Minimum-2-norm allocation, ignoring command bounds.

    Solves ``min ||u||_2  s.t.  B u = tau`` with the Moore-Penrose inverse,

        u = B^+ tau                                                       (2)

    This is the standard unconstrained generalized-inverse allocation
    (Bodson 2002 sec. II). It has no knowledge of ``u_min``/``u_max``, so its
    result is routinely outside the command box; the returned
    ``bound_violation`` says by how much and the status becomes
    ``"infeasible"`` in that case. It is the right choice only when the
    command is known to be well inside the attainable moment set, or as the
    starting point for :func:`redistributed_pseudo_inverse_allocate`.
    """
    tau = _check_torque(torque)
    t0 = time.perf_counter()
    u = np.linalg.pinv(eset.matrix) @ tau
    dt = time.perf_counter() - t0
    return _finish(eset, u, tau, "pinv", torque_tol, bound_tol, solve_time_s=dt)


# ----------------------------------------------------------------------
# 2. Weighted pseudo-inverse
# ----------------------------------------------------------------------


def weighted_pseudo_inverse_allocate(
    eset: EffectorSet,
    torque: ArrayLike,
    weights: ArrayLike | None = None,
    u_pref: ArrayLike | None = None,
    torque_tol: float = DEFAULT_TORQUE_TOL,
    bound_tol: float = DEFAULT_BOUND_TOL,
) -> AllocationResult:
    """Weighted minimum-norm allocation about a preferred command, no bounds.

    Solves ``min (u - u_p)^T W (u - u_p)  s.t.  B u = tau`` for diagonal
    positive ``W = diag(weights)``, whose solution is

        u = u_p + W^-1 B^T (B W^-1 B^T)^+ (tau - B u_p)                   (3)

    Equation (3) is the weighted generalized inverse of Bodson (2002) sec. II.
    Large ``weights[i]`` discourages use of effector ``i``, which is the usual
    way to express "this thruster is expensive" or "prefer the wheels".

    ``u_pref`` matters for one-sided effectors: with thruster bounds
    ``[0, F_max]`` the unbiased solution is symmetric about zero and therefore
    half negative, whereas ``u_pref = (lower + upper) / 2`` biases it into the
    feasible box. It defaults to the box centre for that reason.

    Bounds are still not enforced. Use :func:`qp_allocate` when they matter.
    """
    tau = _check_torque(torque)
    m = eset.n_effectors
    if weights is None:
        w = np.ones(m)
    else:
        w = np.broadcast_to(np.asarray(weights, dtype=float), (m,)).astype(float)
        if np.any(w <= 0.0):
            raise ValueError("weights must be strictly positive")
        if not np.all(np.isfinite(w)):
            raise ValueError("weights must be finite")
    if u_pref is None:
        up = 0.5 * (eset.lower + eset.upper)
    else:
        up = np.broadcast_to(np.asarray(u_pref, dtype=float), (m,)).astype(float)

    t0 = time.perf_counter()
    winv = np.diag(1.0 / w)
    b = eset.matrix
    core = b @ winv @ b.T
    u = up + winv @ b.T @ (np.linalg.pinv(core) @ (tau - b @ up))
    dt = time.perf_counter() - t0
    return _finish(eset, u, tau, "wpinv", torque_tol, bound_tol, solve_time_s=dt)


# ----------------------------------------------------------------------
# 3. Redistributed pseudo-inverse (a saturation handler, not an optimizer)
# ----------------------------------------------------------------------


def redistributed_pseudo_inverse_allocate(
    eset: EffectorSet,
    torque: ArrayLike,
    weights: ArrayLike | None = None,
    u_pref: ArrayLike | None = None,
    max_iter: int | None = None,
    torque_tol: float = DEFAULT_TORQUE_TOL,
    bound_tol: float = DEFAULT_BOUND_TOL,
) -> AllocationResult:
    """Clip-and-redistribute saturation handling on top of the weighted inverse.

    Iterates: solve the weighted generalized inverse, clamp every command that
    left the box to its violated bound, freeze those effectors, and re-solve
    for the residual torque with the remaining free effectors. This is the
    "redistributed pseudoinverse" of Bodson (2002) sec. V.A, included here
    because it is the method most often found in flight software and because
    it is the natural baseline for the LP/QP allocators.

    It is **not** optimal and it is **not** guaranteed to find a feasible
    command even when one exists (Haerkegaard 2002 sec. 2.2.1 makes the same
    point). When it terminates without meeting the torque, the result carries
    ``status="infeasible"`` and the residual; ``extras["n_iterations"]`` and
    ``extras["n_saturated"]`` report what it did.
    """
    tau = _check_torque(torque)
    m = eset.n_effectors
    limit = m + 1 if max_iter is None else int(max_iter)
    if limit < 1:
        raise ValueError("max_iter must be >= 1")

    if weights is None:
        w = np.ones(m)
    else:
        w = np.broadcast_to(np.asarray(weights, dtype=float), (m,)).astype(float)
        if np.any(w <= 0.0):
            raise ValueError("weights must be strictly positive")
    if u_pref is None:
        up = 0.5 * (eset.lower + eset.upper)
    else:
        up = np.broadcast_to(np.asarray(u_pref, dtype=float), (m,)).astype(float)

    t0 = time.perf_counter()
    u = np.clip(up.copy(), eset.lower, eset.upper)
    frozen = ~eset.free_mask()
    n_iter = 0
    for _ in range(limit):
        n_iter += 1
        free = ~frozen
        if not np.any(free):
            break
        resid = tau - eset.matrix @ u
        if np.linalg.norm(resid) <= torque_tol:
            break
        bf = eset.matrix[:, free]
        winv = np.diag(1.0 / w[free])
        du = winv @ bf.T @ (np.linalg.pinv(bf @ winv @ bf.T) @ resid)
        trial = u.copy()
        trial[free] = u[free] + du
        newly = (trial < eset.lower - bound_tol) | (trial > eset.upper + bound_tol)
        newly &= free
        u = np.clip(trial, eset.lower, eset.upper)
        if not np.any(newly):
            break
        frozen |= newly
    dt = time.perf_counter() - t0
    extras = {"n_iterations": n_iter, "n_saturated": int(np.count_nonzero(frozen))}
    return _finish(
        eset, u, tau, "rpi", torque_tol, bound_tol, solve_time_s=dt, extras=extras
    )


# ----------------------------------------------------------------------
# 4. Linear-programming allocation with actuator bounds
# ----------------------------------------------------------------------


def _lp_min_error_highs(
    eset: EffectorSet, tau: np.ndarray, torque_weights: np.ndarray
) -> tuple[np.ndarray, float, str]:
    """min sum(Wv * s) s.t. -s <= tau - B u <= s, bounds on u, s >= 0."""
    m = eset.n_effectors
    n = m + 3
    c = np.concatenate([np.zeros(m), torque_weights])
    # B u - s <= tau   and   -B u - s <= -tau
    a_ub = np.block(
        [
            [eset.matrix, -np.eye(3)],
            [-eset.matrix, -np.eye(3)],
        ]
    )
    b_ub = np.concatenate([tau, -tau])
    bounds = list(zip(eset.lower, eset.upper, strict=True)) + [(0.0, None)] * 3
    res = linprog(c, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        return np.clip(np.zeros(m), eset.lower, eset.upper), float("inf"), res.message
    x = np.asarray(res.x, dtype=float)
    assert x.size == n
    return x[:m], float(res.fun), "optimal"


def _lp_min_control_highs(
    eset: EffectorSet, tau: np.ndarray, cost: np.ndarray, u_pref: np.ndarray
) -> tuple[np.ndarray, float, str]:
    """min sum(cost * |u - u_pref|) s.t. B u = tau exactly, bounds on u."""
    m = eset.n_effectors
    c = np.concatenate([np.zeros(m), cost])
    # u - t <= u_pref ; -u - t <= -u_pref
    a_ub = np.block([[np.eye(m), -np.eye(m)], [-np.eye(m), -np.eye(m)]])
    b_ub = np.concatenate([u_pref, -u_pref])
    a_eq = np.hstack([eset.matrix, np.zeros((3, m))])
    bounds = list(zip(eset.lower, eset.upper, strict=True)) + [(0.0, None)] * m
    res = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=tau, bounds=bounds, method="highs")
    if not res.success:
        return np.clip(u_pref, eset.lower, eset.upper), float("inf"), res.message
    return np.asarray(res.x, dtype=float)[:m], float(res.fun), "optimal"


def _lp_min_error_pulp(
    eset: EffectorSet, tau: np.ndarray, torque_weights: np.ndarray
) -> tuple[np.ndarray, float, str]:
    """Same LP as :func:`_lp_min_error_highs`, built with PuLP and solved by CBC.

    Present so the HiGHS result can be cross-checked against an independent
    simplex implementation; see ``validation/validate_exact_allocation.py``.
    CBC's default optimality tolerance is looser than HiGHS's, so agreement is
    only expected to about 1e-7 N*m.
    """
    import pulp

    m = eset.n_effectors
    prob = pulp.LpProblem("control_allocation_min_error", pulp.LpMinimize)
    u = [
        prob.add_variable(f"u{i}", float(eset.lower[i]), float(eset.upper[i]))
        for i in range(m)
    ]
    s = [prob.add_variable(f"s{k}", 0.0) for k in range(3)]
    prob += pulp.lpSum(float(torque_weights[k]) * s[k] for k in range(3))
    for k in range(3):
        row = pulp.lpSum(float(eset.matrix[k, i]) * u[i] for i in range(m))
        prob += row - s[k] <= float(tau[k])
        prob += -row - s[k] <= -float(tau[k])
    with warnings.catch_warnings():
        # PuLP 3.3 deprecates PULP_CBC_CMD in favour of COIN_CMD, but COIN_CMD
        # needs a separately installed cbc binary that is not in the target
        # environment. The bundled solver is the only one available here.
        warnings.simplefilter("ignore", DeprecationWarning)
        status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    label = pulp.LpStatus[status]
    if label != "Optimal":
        return np.clip(np.zeros(m), eset.lower, eset.upper), float("inf"), label
    sol = np.array([v.value() for v in u], dtype=float)
    return sol, float(pulp.value(prob.objective)), "optimal"


def lp_allocate(
    eset: EffectorSet,
    torque: ArrayLike,
    objective: str = "min_error",
    torque_weights: ArrayLike | None = None,
    cost: ArrayLike | None = None,
    u_pref: ArrayLike | None = None,
    solver: str = "highs",
    torque_tol: float = DEFAULT_TORQUE_TOL,
    bound_tol: float = DEFAULT_BOUND_TOL,
) -> AllocationResult:
    """Linear-programming allocation honouring the command bounds.

    Two objectives, both linear programs in the sense of Bodson (2002)
    sec. III, which converts control allocation to an LP so that simplex can
    be used:

    ``objective="min_error"``
        ``min sum_k Wv_k |tau_k - (B u)_k|`` subject to the box. Always
        feasible: when ``tau`` is outside the attainable moment set this
        returns the bound-respecting command with the smallest weighted
        1-norm torque error, and the result is reported ``infeasible`` with
        that residual attached. Note that for an attainable command the
        optimum is a whole face of the box, and simplex returns an arbitrary
        vertex of it -- typically with several effectors hard against their
        limits. That is a valid exact allocation but not a low-effort one;
        use ``"min_control"`` or :func:`qp_allocate` when effort matters.

    ``objective="min_control"``
        ``min sum_i c_i |u_i - u_pref_i|`` subject to ``B u = tau`` exactly
        and the box. This is the minimum-control-effort allocation. The
        equality makes the LP itself infeasible when ``tau`` is outside the
        attainable moment set; that is detected and reported as
        ``status="infeasible"`` with ``message`` carrying the solver status.

    Parameters
    ----------
    solver
        ``"highs"`` (default) uses ``scipy.optimize.linprog`` with the HiGHS
        dual simplex. ``"pulp"`` builds the same program with PuLP and solves
        it with CBC; it is slower and exists to cross-check HiGHS.
        ``"pulp"`` supports ``objective="min_error"`` only.

    Notes
    -----
    The returned command is projected into the command box before it is
    reported, because the solvers satisfy variable bounds only to their own
    primal feasibility tolerance; the amount absorbed is in
    ``extras["pre_clip_bound_violation"]``.

    Accuracy floor: HiGHS runs at its default primal feasibility tolerance of
    1e-7, so a returned command can sit that far outside its own box before
    the projection above, which turns into a torque error of up to
    ``1e-7 * ||B||`` N*m. That is the LP's residual floor however small
    ``torque_tol`` is set; CBC's is looser still. For a command whose
    magnitude is itself near that floor the LP will report ``infeasible`` on
    an attainable command purely from solver tolerance.
    :func:`qp_allocate`, solved by a direct bounded least-squares
    factorisation, has no such floor.
    """
    tau = _check_torque(torque)
    m = eset.n_effectors
    wv = (
        np.ones(3)
        if torque_weights is None
        else np.broadcast_to(np.asarray(torque_weights, dtype=float), (3,)).astype(float)
    )
    if np.any(wv <= 0.0):
        raise ValueError("torque_weights must be strictly positive")
    cc = (
        np.ones(m)
        if cost is None
        else np.broadcast_to(np.asarray(cost, dtype=float), (m,)).astype(float)
    )
    if np.any(cc <= 0.0):
        raise ValueError("cost must be strictly positive")
    up = (
        0.5 * (eset.lower + eset.upper)
        if u_pref is None
        else np.broadcast_to(np.asarray(u_pref, dtype=float), (m,)).astype(float)
    )

    t0 = time.perf_counter()
    if objective == "min_error":
        if solver == "highs":
            u, obj, msg = _lp_min_error_highs(eset, tau, wv)
        elif solver == "pulp":
            u, obj, msg = _lp_min_error_pulp(eset, tau, wv)
        else:
            raise ValueError(f"unknown solver {solver!r}; expected 'highs' or 'pulp'")
    elif objective == "min_control":
        if solver != "highs":
            raise ValueError("objective='min_control' is implemented for solver='highs' only")
        u, obj, msg = _lp_min_control_highs(eset, tau, cc, up)
    else:
        raise ValueError(
            f"unknown objective {objective!r}; expected 'min_error' or 'min_control'"
        )
    dt = time.perf_counter() - t0

    # HiGHS and CBC both satisfy variable bounds only to their own primal
    # feasibility tolerance (1e-7 by default), which can leave a command a few
    # 1e-10 outside its own declared box. Project it back and record what was
    # absorbed, so the guarantee "an LP allocation is inside the box" holds
    # exactly and the size of the solver's slop stays visible.
    pre_clip_violation = float(eset.bound_violation(u))
    u = eset.clip(u)

    message = "" if msg == "optimal" else f"LP solver reported: {msg}"
    extras = {
        "objective": objective,
        "lp_objective": obj,
        "solver": solver,
        "pre_clip_bound_violation": pre_clip_violation,
    }
    return _finish(
        eset, u, tau, "lp", torque_tol, bound_tol, message, solve_time_s=dt, extras=extras
    )


# ----------------------------------------------------------------------
# 5. Quadratic-programming allocation with actuator bounds
# ----------------------------------------------------------------------


def _bounded_lsq(
    a: np.ndarray, b: np.ndarray, lower: np.ndarray, upper: np.ndarray, tol: float = 1e-12
) -> np.ndarray:
    """Bounded linear least squares with fixed variables eliminated.

    ``scipy.optimize.lsq_linear`` with ``method="bvls"`` needs a strictly
    positive-width box on every variable, so effectors whose bounds have
    collapsed (failed, stuck) are substituted out and their contribution moved
    to the right-hand side before the solve. BVLS is the bounded-variable
    least-squares active-set algorithm of Stark & Parker (1995), "Bounded
    variable least squares: an algorithm and applications", Computational
    Statistics 10, 129-141; it terminates finitely at the exact optimum, which
    is what makes this the reference solution the learned allocator is
    measured against.
    """
    fixed = (upper - lower) <= tol
    if np.all(fixed):
        return lower.copy()
    x = lower.copy()
    if np.any(fixed):
        b = b - a[:, fixed] @ lower[fixed]
        a_free = a[:, ~fixed]
    else:
        a_free = a
    res = lsq_linear(a_free, b, bounds=(lower[~fixed], upper[~fixed]), method="bvls")
    x[~fixed] = res.x
    return x


def qp_allocate(
    eset: EffectorSet,
    torque: ArrayLike,
    control_weights: ArrayLike | None = None,
    torque_weights: ArrayLike | None = None,
    u_pref: ArrayLike | None = None,
    gamma: float = 1e12,
    torque_tol: float = DEFAULT_TORQUE_TOL,
    bound_tol: float = DEFAULT_BOUND_TOL,
) -> AllocationResult:
    """Bounded weighted-least-squares (quadratic-programming) allocation.

    Solves the mixed-optimization control allocation problem of
    Haerkegaard (2002) eq. (5),

        min_u  ||Wu (u - u_p)||^2 + gamma * ||Wv (B u - tau)||^2
        s.t.   u_min <= u <= u_max                                        (4)

    with ``Wu = diag(control_weights)`` and ``Wv = diag(torque_weights)``.
    Large ``gamma`` prioritises meeting the torque over minimising control
    effort. The penalty is not a hard constraint, so for an attainable command
    the achieved torque residual falls as ``1/gamma``: measured on the
    four-wheel pyramid over 200 random commands, the worst residual is
    8.853e-06 N*m at ``gamma=1e4``, 7.810e-08 at 1e6, 6.999e-12 at 1e10 and
    7.299e-14 at 1e12, with no conditioning breakdown up to 1e16
    (``validation/validate_exact_allocation.py``). The default 1e12 therefore
    sits about four orders of magnitude inside ``DEFAULT_TORQUE_TOL`` while
    leaving the control-effort term enough weight to pick among the exact
    solutions. Lower it only if effort minimisation matters more than
    hitting the torque.

    Problem (4) is a bounded linear least-squares problem in the stacked
    matrix ``[sqrt(gamma) Wv B ; Wu]``, so it is solved exactly by the BVLS
    active-set algorithm rather than by an interior-point approximation --
    the same active-set family Haerkegaard (2002) recommends.

    The penalty is soft, and that has a consequence worth stating plainly: for
    an attainable command the achieved torque is exact only to the extent that
    ``gamma`` outweighs the effort term. If the active set at the optimum
    leaves open a direction ``d`` inside the box whose torque effectiveness
    ``||Wv B d||`` is tiny, the solver will move along it to reduce effort and
    accept a residual of order ``|Wu^2 (u_p - u*) . d| / (gamma ||Wv B d||)``.
    Measured example, pinned in
    ``tests/test_allocation.py::test_weak_effector_degrades_the_qp_residual``:
    an effector direction of effectiveness 1e-4 with the default
    ``u_pref`` = box centre gives a 1.5e-8 N*m residual at ``gamma = 1e12``,
    which is above ``DEFAULT_TORQUE_TOL`` and so is honestly reported as
    ``infeasible``. Passing ``u_pref`` equal to the intended command, or
    raising ``gamma``, removes it.

    When ``tau`` is outside the attainable moment set the QP is still solvable
    (the objective is a penalty, not a constraint) and returns the
    bound-respecting command with the smallest weighted 2-norm torque error;
    the result then carries ``status="infeasible"`` and the residual. The
    command is always inside the box.

    Raises
    ------
    ValueError
        On non-positive weights, non-finite ``gamma``, or a wrongly shaped
        torque.
    """
    tau = _check_torque(torque)
    m = eset.n_effectors
    if not np.isfinite(gamma) or gamma <= 0.0:
        raise ValueError(f"gamma must be finite and > 0, got {gamma!r}")
    wu = (
        np.ones(m)
        if control_weights is None
        else np.broadcast_to(np.asarray(control_weights, dtype=float), (m,)).astype(float)
    )
    if np.any(wu <= 0.0):
        raise ValueError("control_weights must be strictly positive")
    wv = (
        np.ones(3)
        if torque_weights is None
        else np.broadcast_to(np.asarray(torque_weights, dtype=float), (3,)).astype(float)
    )
    if np.any(wv <= 0.0):
        raise ValueError("torque_weights must be strictly positive")
    up = (
        0.5 * (eset.lower + eset.upper)
        if u_pref is None
        else np.broadcast_to(np.asarray(u_pref, dtype=float), (m,)).astype(float)
    )

    t0 = time.perf_counter()
    root = np.sqrt(gamma)
    a = np.vstack([root * (wv[:, None] * eset.matrix), np.diag(wu)])
    b = np.concatenate([root * wv * tau, wu * up])
    u = _bounded_lsq(a, b, eset.lower, eset.upper)
    dt = time.perf_counter() - t0
    extras = {"gamma": float(gamma)}
    return _finish(
        eset, u, tau, "qp", torque_tol, bound_tol, solve_time_s=dt, extras=extras
    )


# ----------------------------------------------------------------------
# Dispatch and feasibility
# ----------------------------------------------------------------------

_DISPATCH = {
    "pinv": pseudo_inverse_allocate,
    "wpinv": weighted_pseudo_inverse_allocate,
    "rpi": redistributed_pseudo_inverse_allocate,
    "lp": lp_allocate,
    "qp": qp_allocate,
}


def allocate(
    eset: EffectorSet, torque: ArrayLike, method: str = "qp", **kwargs
) -> AllocationResult:
    """Dispatch to one of :data:`METHODS` by name.

    ``method`` is one of ``"pinv"``, ``"wpinv"``, ``"rpi"``, ``"lp"``, ``"qp"``.
    Remaining keyword arguments are forwarded to the chosen allocator.
    """
    if method not in _DISPATCH:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    return _DISPATCH[method](eset, torque, **kwargs)


def is_attainable(
    eset: EffectorSet, torque: ArrayLike, tol: float = DEFAULT_TORQUE_TOL
) -> bool:
    """Exact test of whether ``torque`` lies in the attainable moment set.

    Solves the ``min_error`` LP and checks that the 2-norm of the residual at
    the LP optimum is below ``tol``. The LP minimises the weighted 1-norm
    error, and the 1-norm and 2-norm of a vector vanish together, so a zero
    LP optimum is exactly the feasibility certificate; the *value* reported
    for an infeasible command is a 1-norm-optimal residual and is not the
    smallest achievable 2-norm error (use :func:`qp_allocate` for that).

    Exact up to LP solver tolerance. Unlike a convex-hull membership test it
    needs no vertex enumeration, so it stays usable for large effector counts.
    """
    res = lp_allocate(eset, torque, objective="min_error", torque_tol=tol)
    return bool(res.residual_norm <= tol)
