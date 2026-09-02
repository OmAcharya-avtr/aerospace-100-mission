"""Constrained rest-to-rest slew planning.

The planner answers one question: **is there a rest-to-rest attitude path from
``q_start`` to ``q_goal`` that keeps every instrument boresight out of every
keep-out cone, stays inside the wheel array's torque and momentum envelope, and
finishes inside the time budget** -- and if not, which of those it was.

Path model
----------
A path is a sequence of waypoint attitudes joined by eigenaxis rest-to-rest
segments::

    q_start -> q_via_1 -> ... -> q_via_k -> q_goal

The spacecraft comes to rest at every waypoint. That is a modelling decision,
not an approximation: it makes the time and momentum of each segment exactly
the closed forms in :mod:`slewforge.profiles`, and it is what an operational
"slew, settle, slew" sequence actually does. A path that keeps the rate
continuous through the via point would be faster and is out of scope; see
README limitations.

Via parameterisation
--------------------
With ``k`` via points the free parameters are ``k`` rotation vectors
``p_1..p_k`` [rad], each applied as an extra inertial rotation about the
nominal interpolated attitude::

    q_via_i = dq(C p_i) ⊗ slerp(q_start, q_goal, i / (k + 1))

``C`` is the problem's canonical frame (see :func:`canonical_frame`): its first
column is the eigenaxis of the direct slew, its second the component of the
primary boresight perpendicular to that axis, its third their cross product.
Working in ``C`` makes the parameterisation invariant to a rigid rotation of
the whole problem, which is what lets the learned warm-start in
:mod:`slewforge.ml` generalise across orientations instead of memorising them.

Optimisation
------------
For fixed ``k`` the planner minimises total manoeuvre time (plus an optional
momentum term) over ``p`` subject to the keep-out constraint, using SLSQP with
numerical gradients. The keep-out constraint is aggregated with a soft
minimum ``-(1/beta) log sum exp(-beta m_i)``, which is a *lower* bound on
``min_i m_i``, so a path that satisfies the aggregated constraint satisfies
every individual one. The margins ``m_i`` themselves come from the closed-form
arc test in :mod:`slewforge.keepout`, so no violation can hide between samples.

Cold start runs the optimiser from a fixed deterministic set of starting
points and keeps the best result. Warm start runs it once from a supplied
guess and falls back to the cold sweep if that fails. Nothing about the cold
sweep is random, so `validation/` results are reproducible without a seed.

Units: angles rad, time s, torque N*m, momentum N*m*s, inertia kg m^2.

References
----------
B. Wie, *Space Vehicle Dynamics and Control*, 2nd ed., AIAA (2008), Sec. 5.3,
    7.4 -- eigenaxis slews and wheel sizing.
H. C. Frakes, J. D. Turner et al. and the wider constrained-attitude-guidance
    literature; the specific formulation used here follows the quadratic
    keep-out constraint of Y. Kim and M. Mesbahi, "Quadratically constrained
    attitude control via semidefinite programming", *IEEE Trans. Automatic
    Control* **49**(5), 731-735 (2004), in which a keep-out cone is the
    condition ``n(q)·c <= cos gamma`` -- exactly the inequality solved in
    closed form here.
D. Kraft, "A software package for sequential quadratic programming", DFVLR-FB
    88-28 (1988) -- the SLSQP algorithm used through ``scipy.optimize``.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize

from .attitude import (
    axis_angle_from_quat,
    cross3,
    quat_from_rotvec,
    quat_multiply,
    quat_normalize,
    quat_relative,
    quat_rotate,
    quat_slerp,
    unit_vector,
)
from .dynamics import RigidBody, eigenaxis_torque
from .keepout import ArcViolation, KeepOutSet, min_margin_on_arc_raw
from .profiles import PROFILE_NAMES, SlewProfile, make_profile

__all__ = [
    "ActuatorCheck",
    "INFEASIBILITY_REASONS",
    "Instrument",
    "PlanResult",
    "SlewPath",
    "SlewProblem",
    "SlewSegment",
    "canonical_frame",
    "canonical_rotvec",
    "cold_start_points",
    "direct_violations",
    "path_margins",
    "path_min_margin",
    "path_violations",
    "plan",
    "verify_actuators",
]

INFEASIBILITY_REASONS = {
    "start_attitude_violates_keepout": (
        "the boresight already lies inside a keep-out cone at q_start; no slew can fix that"
    ),
    "goal_attitude_violates_keepout": (
        "the requested goal attitude puts a boresight inside a keep-out cone"
    ),
    "keepout_covers_all_directions": "a single cone excludes every direction on the sky",
    "wheel_torque_unavailable": (
        "the wheel array cannot produce torque along the direction the slew needs"
    ),
    "wheel_torque_limit_exceeded": (
        "the exact eigenaxis torque, including the gyroscopic term the scalar sizing "
        "model drops, leaves the wheel torque box part-way through a segment"
    ),
    "wheel_momentum_limit_exceeded": (
        "the momentum the slew must store exceeds the wheel array's momentum envelope"
    ),
    "time_limit_exceeded": "every keep-out-feasible path found is slower than max_time",
    "no_feasible_path_found": (
        "the optimiser found no via point clearing every cone; the geometry may be "
        "blocked or may need more via points than max_via allows"
    ),
}
"""Every value :attr:`PlanResult.reason` can take, with what it means.

The set is closed on purpose: a planner that returns "infeasible" without
saying which constraint failed is not usable in a design loop.
"""


@dataclass(frozen=True)
class Instrument:
    """A boresight fixed in the body frame.

    Parameters
    ----------
    name : str
        Label used in violation reports.
    boresight_body : array_like
        Unit vector in body coordinates, shape ``(3,)``, normalised on
        construction. Dimensionless.
    """

    name: str
    boresight_body: NDArray[np.float64]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string")
        object.__setattr__(self, "boresight_body", unit_vector(self.boresight_body).reshape(3))

    def direction(self, quat: ArrayLike) -> NDArray[np.float64]:
        """Inertial boresight direction at attitude ``quat``, shape ``(3,)``."""
        return quat_rotate(quat, self.boresight_body)


@dataclass(frozen=True)
class SlewProblem:
    """A constrained rest-to-rest slew.

    Parameters
    ----------
    quat_start, quat_goal : array_like
        Attitudes ``(4,)``, scalar-first, body to inertial.
    body : RigidBody
        Must carry a wheel array; the planner sizes every segment against it.
    keepout : KeepOutSet
        Inertial exclusion cones, static for the duration of the slew.
    instruments : sequence of Instrument
        Boresights that must clear every cone. An empty sequence makes the
        problem unconstrained, which is legal and gives the direct eigenaxis
        slew.
    profile : str
        One of :data:`slewforge.profiles.PROFILE_NAMES`.
    rate_limit : float or None
        Body-rate cap [rad/s] on top of the wheel momentum envelope.
    required_margin : float
        Clearance [rad] every boresight must keep from every cone. Zero means
        "touching the boundary is allowed"; a real mission uses a positive
        value to cover attitude-knowledge error.
    max_time : float or None
        Time budget [s]. A path slower than this is reported infeasible with
        reason ``time_limit_exceeded``.
    size_against_envelope : bool
        ``True`` (default) sizes every segment so the **exact** eigenaxis
        torque, gyroscopic term included, stays inside the wheel torque box
        and the rate stays inside the momentum envelope. ``False`` sizes with
        the scalar rule of thumb ``alpha = tau_cap / |J e|`` and the user rate
        limit alone -- the spreadsheet answer. Setting it ``False`` is how the
        failure-mode tests reproduce a mid-slew saturation that the rule of
        thumb does not predict: the plan comes back infeasible with reason
        ``wheel_torque_limit_exceeded`` and the utilisation that caused it.
    momentum_weight : float
        Weight [s per N*m*s] on momentum throughput in the objective
        ``T + w * integral|tau|dt``. Default 0: minimise time alone, and
        report momentum separately.
    """

    quat_start: NDArray[np.float64]
    quat_goal: NDArray[np.float64]
    body: RigidBody
    keepout: KeepOutSet = field(default_factory=KeepOutSet)
    instruments: tuple[Instrument, ...] = ()
    profile: str = "bang_bang"
    rate_limit: float | None = None
    required_margin: float = 0.0
    max_time: float | None = None
    momentum_weight: float = 0.0
    size_against_envelope: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "quat_start", quat_normalize(self.quat_start).reshape(4))
        object.__setattr__(self, "quat_goal", quat_normalize(self.quat_goal).reshape(4))
        if not isinstance(self.body, RigidBody):
            raise TypeError(f"body must be a RigidBody, got {type(self.body).__name__}")
        if self.body.wheels is None:
            raise ValueError("the planner needs a body with a wheel array")
        if not isinstance(self.keepout, KeepOutSet):
            raise TypeError(f"keepout must be a KeepOutSet, got {type(self.keepout).__name__}")
        inst = tuple(self.instruments)
        for i in inst:
            if not isinstance(i, Instrument):
                raise TypeError(f"instruments must be Instrument, got {type(i).__name__}")
        object.__setattr__(self, "instruments", inst)
        if self.profile not in PROFILE_NAMES:
            raise ValueError(f"profile must be one of {PROFILE_NAMES}, got {self.profile!r}")
        if self.rate_limit is not None and (
            not math.isfinite(self.rate_limit) or self.rate_limit <= 0.0
        ):
            raise ValueError(f"rate_limit must be finite and > 0 rad/s, got {self.rate_limit}")
        if not math.isfinite(self.required_margin) or self.required_margin < 0.0:
            raise ValueError(
                f"required_margin must be finite and >= 0 rad, got {self.required_margin}"
            )
        if self.max_time is not None and (not math.isfinite(self.max_time) or self.max_time <= 0.0):
            raise ValueError(f"max_time must be finite and > 0 s, got {self.max_time}")
        if not math.isfinite(self.momentum_weight) or self.momentum_weight < 0.0:
            raise ValueError(f"momentum_weight must be finite and >= 0, got {self.momentum_weight}")
        if not isinstance(self.size_against_envelope, bool):
            raise TypeError("size_against_envelope must be a bool")

    @property
    def slew_angle(self) -> float:
        """Eigenaxis angle of the direct slew [rad], in ``[0, pi]``."""
        return axis_angle_from_quat(quat_relative(self.quat_start, self.quat_goal))[1]

    @property
    def eigenaxis(self) -> NDArray[np.float64]:
        """Direct-slew eigenaxis in inertial coordinates, shape ``(3,)``.

        For a slew angle below 1e-12 rad the axis is undefined; ``[1, 0, 0]``
        is returned and the planner short-circuits to a zero-time path.
        """
        return axis_angle_from_quat(quat_relative(self.quat_start, self.quat_goal))[0]

    def boresights(self, quat: ArrayLike) -> dict[str, NDArray[np.float64]]:
        """Inertial boresight directions at ``quat``, keyed by instrument name."""
        return {i.name: i.direction(quat) for i in self.instruments}

    def attitude_margin(self, quat: ArrayLike) -> float:
        """Worst clearance over all instruments and cones at one attitude [rad].

        ``+inf`` when there are no cones or no instruments.
        """
        if not self.instruments or not self.keepout:
            return float("inf")
        return min(float(self.keepout.margin(i.direction(quat))) for i in self.instruments)

    def attitude_violations(self, quat: ArrayLike) -> tuple[tuple[str, str], ...]:
        """``(instrument, cone)`` pairs violated at one attitude."""
        out: list[tuple[str, str]] = []
        for i in self.instruments:
            for c in self.keepout.violations(i.direction(quat)):
                out.append((i.name, c))
        return tuple(out)


@dataclass(frozen=True)
class SlewSegment:
    """One eigenaxis rest-to-rest leg.

    Attributes
    ----------
    quat_start, quat_end : ndarray
        ``(4,)`` endpoint attitudes.
    axis : ndarray
        ``(3,)`` eigenaxis in inertial coordinates.
    angle : float
        Swept angle [rad].
    profile : SlewProfile
        Timing law, already sized against the wheel envelope.
    peak_accel_available : float
        The acceleration limit used for the sizing [rad/s^2].
    rate_cap : float
        The rate limit used, the smaller of the user cap and the wheel
        momentum envelope [rad/s].
    """

    quat_start: NDArray[np.float64]
    quat_end: NDArray[np.float64]
    axis: NDArray[np.float64]
    angle: float
    profile: SlewProfile
    peak_accel_available: float
    rate_cap: float

    @property
    def duration(self) -> float:
        """Segment time [s]."""
        return self.profile.duration

    def attitude_at(self, t: ArrayLike) -> NDArray[np.float64]:
        """Attitude at time ``t`` [s] into the segment, shape ``(4,)`` or ``(n, 4)``."""
        psi = np.atleast_1d(np.asarray(self.profile.angle_at(t), dtype=float))
        out = np.stack(
            [
                quat_multiply(quat_from_rotvec(self.axis * p), self.quat_start)
                for p in psi
            ]
        )
        return out[0] if np.ndim(t) == 0 else out


@dataclass(frozen=True)
class SlewPath:
    """A sequence of eigenaxis segments with its cost accounting.

    Attributes
    ----------
    segments : tuple of SlewSegment
    """

    segments: tuple[SlewSegment, ...]

    @property
    def waypoints(self) -> NDArray[np.float64]:
        """Attitudes at every rest point, shape ``(k + 2, 4)``."""
        if not self.segments:
            return np.zeros((0, 4))
        return np.stack(
            [self.segments[0].quat_start] + [s.quat_end for s in self.segments]
        )

    @property
    def total_time(self) -> float:
        """Sum of the segment durations [s]. Settling time is not modelled."""
        return float(sum(s.duration for s in self.segments))

    @property
    def total_angle(self) -> float:
        """Sum of the segment angles [rad]; the path length on SO(3)."""
        return float(sum(s.angle for s in self.segments))

    @property
    def peak_momentum(self) -> float:
        """Largest stored momentum over the whole path [N*m*s]."""
        return float(max((s.profile.peak_momentum for s in self.segments), default=0.0))

    @property
    def momentum_throughput(self) -> float:
        """``sum integral |tau| dt`` over the segments [N*m*s].

        The quantity that matters for wheel wear and for how much desaturation
        the manoeuvre implies; it grows with every extra via point even when
        the total angle barely does.
        """
        return float(sum(s.profile.momentum_throughput for s in self.segments))

    def objective(self, momentum_weight: float = 0.0) -> float:
        """``total_time + momentum_weight * momentum_throughput`` [s]."""
        return self.total_time + float(momentum_weight) * self.momentum_throughput


@dataclass(frozen=True)
class ActuatorCheck:
    """Result of verifying a path against the exact wheel limits.

    Attributes
    ----------
    feasible : bool
    max_torque_demand : float
        Largest per-wheel torque the minimum-norm allocation asks for [N*m].
    max_momentum_demand : float
        Largest per-wheel momentum [N*m*s].
    torque_utilisation, momentum_utilisation : float
        Utilisation under the **minimum-norm** allocation, which is what the
        planner sizes against and what a controller using the plain
        pseudo-inverse delivers. Above 1 is saturation and makes the plan
        infeasible.
    exact_torque_utilisation, exact_momentum_utilisation : float
        The same demand tested against the *exact* wheel envelope, i.e.
        allowing any allocation inside the per-wheel box. Never larger than
        the figures above; the gap is the headroom a smarter allocator (see
        AllocLab, P023, implemented independently) would recover.
    worst_segment : int
        Index of the segment where the worst utilisation occurred.
    gyroscopic_fraction : float
        Largest ratio of the gyroscopic torque term to the total torque
        magnitude along the path, dimensionless. This is the term the scalar
        sizing model drops.
    """

    feasible: bool
    max_torque_demand: float
    max_momentum_demand: float
    torque_utilisation: float
    momentum_utilisation: float
    exact_torque_utilisation: float
    exact_momentum_utilisation: float
    worst_segment: int
    gyroscopic_fraction: float


@dataclass
class PlanResult:
    """What the planner decided and how it got there.

    Attributes
    ----------
    feasible : bool
    reason : str or None
        ``None`` when feasible; otherwise a key of
        :data:`INFEASIBILITY_REASONS`.
    detail : str
        A sentence naming the offending cone, instrument or number.
    path : SlewPath or None
    objective, total_time, peak_momentum, momentum_throughput : float
        ``inf`` / ``nan`` when no path was found.
    min_margin : float
        Worst clearance along the returned path [rad]; ``-inf`` if none.
    via_params : ndarray or None
        The optimised parameter vector, shape ``(3 k,)``, in canonical-frame
        coordinates.
    n_via : int
    direct_feasible : bool
        Whether the plain eigenaxis slew already cleared every cone.
    direct_min_margin : float
        Clearance of the direct slew [rad] -- negative is the interesting case
        and is what a spreadsheet never reports.
    solve_time_s : float
        Wall-clock time inside :func:`plan` [s].
    n_objective_evals : int
        Number of path evaluations, the solver-independent cost measure.
    n_starts : int
        Optimiser starting points actually used.
    warm_started : bool
        Whether a warm start was supplied.
    warm_start_accepted : bool
        Whether the warm start produced the returned path without falling
        back to the cold sweep.
    warm_start_confidence : float or None
    actuators : ActuatorCheck or None
    violations : tuple of ArcViolation
        Violating stretches of the *direct* eigenaxis slew, deepest first.
        Non-empty even for a feasible plan whenever the direct slew would have
        violated, because that is the finding worth reporting.
    """

    feasible: bool
    reason: str | None
    detail: str
    path: SlewPath | None
    objective: float
    total_time: float
    peak_momentum: float
    momentum_throughput: float
    min_margin: float
    via_params: NDArray[np.float64] | None
    n_via: int
    direct_feasible: bool
    direct_min_margin: float
    solve_time_s: float
    n_objective_evals: int
    n_starts: int
    warm_started: bool
    warm_start_accepted: bool
    warm_start_confidence: float | None
    actuators: ActuatorCheck | None
    violations: tuple[ArcViolation, ...]

    def summary(self) -> str:
        """One-line human-readable verdict."""
        if self.feasible:
            return (
                f"feasible: {self.n_via} via, T = {self.total_time:.3f} s, "
                f"h_peak = {self.peak_momentum:.4f} N*m*s, "
                f"margin = {math.degrees(self.min_margin):.3f} deg"
            )
        return f"infeasible ({self.reason}): {self.detail}"


def canonical_frame(problem: SlewProblem) -> NDArray[np.float64]:
    """Orthonormal ``(3, 3)`` frame attached to the problem geometry.

    Columns: the direct-slew eigenaxis; the component of the primary
    instrument boresight (at the start attitude) perpendicular to it; their
    cross product. Right-handed and orthonormal by construction.

    Fallbacks, both of which occur in real problems and neither of which may
    return a non-orthogonal frame: a zero slew angle uses the world x axis as
    the first column, and a boresight parallel to the eigenaxis picks the
    world axis least aligned with it for the second.
    """
    e = problem.eigenaxis
    if problem.slew_angle < 1e-12:
        e = np.array([1.0, 0.0, 0.0])
    if problem.instruments:
        b = problem.instruments[0].direction(problem.quat_start)
    else:
        b = np.array([0.0, 0.0, 1.0])
    perp = b - float(np.dot(b, e)) * e
    if float(np.linalg.norm(perp)) < 1e-9:
        alt = np.eye(3)[int(np.argmin(np.abs(e)))]
        perp = alt - float(np.dot(alt, e)) * e
    e2 = perp / float(np.linalg.norm(perp))
    e3 = np.cross(e, e2)
    return np.stack([e, e2, e3], axis=1)


def cold_start_points(n_via: int, magnitudes: Sequence[float] = (0.45,)) -> NDArray[np.float64]:
    """Deterministic starting points for the cold multi-start sweep.

    Returns ``(n_starts, 3 n_via)``: the zero vector, then for every magnitude
    the six signed canonical axis directions, applied to **every** via point at
    once. Deviating all via points the same way pushes the whole middle of the
    path off to one side, which is the shape of a detour around an obstacle;
    perturbing only the first via point leaves a two-via path starting from a
    kink instead. Deterministic, so cold-start results reproduce without a
    seed.
    """
    if n_via < 1:
        raise ValueError(f"n_via must be >= 1, got {n_via}")
    k = int(n_via)
    pts = [np.zeros(3 * k)]
    for mag in magnitudes:
        if mag <= 0.0:
            raise ValueError(f"start magnitudes must be > 0 rad, got {mag}")
        for axis in range(3):
            for sign in (1.0, -1.0):
                one = np.zeros(3)
                one[axis] = sign * float(mag)
                pts.append(np.tile(one, k))
    return np.asarray(pts)


# -- unchecked quaternion kernels ---------------------------------------
#
# The public functions in :mod:`slewforge.attitude` validate shapes and
# finiteness on every call, which is right for an API and wrong for an inner
# loop the optimiser runs tens of thousands of times. These four kernels do the
# same arithmetic on already-validated unit quaternions.
# ``tests/test_planner.py::TestFastKernels`` asserts each is bit-identical to
# its public counterpart over random inputs, so the optimisation cannot drift
# away from the documented behaviour.


def _qmul(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """``a ⊗ b`` for two ``(4,)`` quaternions."""
    return np.array(
        [
            a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
            a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
            a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
            a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
        ]
    )


def _qrot(q: NDArray[np.float64], v: NDArray[np.float64]) -> NDArray[np.float64]:
    """Rotate ``(3,)`` ``v`` by unit ``(4,)`` ``q``, active convention."""
    u = q[1:]
    t = cross3(u, v)
    return v + 2.0 * (q[0] * t + cross3(u, t))


def _qrotvec(p: NDArray[np.float64]) -> NDArray[np.float64]:
    """Unit quaternion from a ``(3,)`` rotation vector [rad]."""
    theta = math.sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2])
    s = 0.5 - theta * theta / 48.0 if theta < 1e-6 else math.sin(0.5 * theta) / theta
    return np.array([math.cos(0.5 * theta), s * p[0], s * p[1], s * p[2]])


def _axis_angle_between(
    qa: NDArray[np.float64], qb: NDArray[np.float64]
) -> tuple[NDArray[np.float64], float]:
    """Eigenaxis ``(3,)`` [inertial] and angle [rad] taking ``qa`` to ``qb``."""
    r = _qmul(qb, np.array([qa[0], -qa[1], -qa[2], -qa[3]]))
    vn = math.sqrt(r[1] * r[1] + r[2] * r[2] + r[3] * r[3])
    angle = 2.0 * math.atan2(vn, abs(r[0]))
    if vn < 1e-12:
        return np.array([1.0, 0.0, 0.0]), 0.0
    sign = 1.0 if r[0] >= 0.0 else -1.0
    return np.array([sign * r[1] / vn, sign * r[2] / vn, sign * r[3] / vn]), angle


def canonical_rotvec(p: ArrayLike) -> NDArray[np.float64]:
    """Rewrite a rotation vector as the equivalent one with angle ``<= pi``.

    A rotation vector and the same vector with ``2 pi k`` added to its length
    describe the same rotation, so the via parameterisation is many-to-one and
    an unconstrained optimiser will happily wander to ``|p| = 10^5`` rad. The
    path is identical, but the parameter is then useless as a learning target
    and useless as a warm start. Every parameter vector the planner returns is
    passed through here first.

    Parameters
    ----------
    p : array_like
        Rotation vector [rad], shape ``(3 k,)`` for ``k`` via points; each
        3-vector block is canonicalised independently.

    Returns
    -------
    ndarray
        Same shape, every block of angle at most ``pi``.
    """
    a = np.asarray(p, dtype=float).reshape(-1)
    if a.size % 3 != 0:
        raise ValueError(f"rotation vector length must be a multiple of 3, got {a.size}")
    out = a.copy()
    for i in range(0, a.size, 3):
        block = a[i : i + 3]
        theta = float(np.linalg.norm(block))
        if theta < 1e-12:
            continue
        wrapped = math.fmod(theta, 2.0 * math.pi)
        if wrapped > math.pi:
            wrapped -= 2.0 * math.pi
        elif wrapped < -math.pi:
            wrapped += 2.0 * math.pi
        out[i : i + 3] = block * (wrapped / theta)
    return out


def _torque_corners(
    profile: SlewProfile, je: NDArray[np.float64], gyro: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Body torques where a profile's per-wheel demand can be largest.

    ``tau = psi_ddot J e + psi_dot^2 (e x J e)``. For bang-bang and
    trapezoidal profiles ``psi_ddot`` is piecewise constant and ``psi_dot^2``
    is monotone on each piece, so the torque traces straight segments in the
    plane spanned by ``J e`` and ``e x J e``; a convex norm on a segment peaks
    at an endpoint, so the five corner states below bound the whole profile
    exactly. The smoothed profile is not piecewise linear and is sampled at 33
    points instead, which is enough for a sinusoid.
    """
    if profile.duration <= 0.0:
        return np.zeros((0, 3))
    if profile.kind == "smoothed":
        t = np.linspace(0.0, profile.duration, 33)
        rate = np.asarray(profile.rate_at(t), dtype=float)
        acc = np.asarray(profile.accel_at(t), dtype=float)
        return acc[:, None] * je[None, :] + (rate**2)[:, None] * gyro[None, :]
    a = profile.peak_accel
    w2 = profile.peak_rate**2
    return np.stack(
        [
            a * je,
            a * je + w2 * gyro,
            w2 * gyro,
            -a * je + w2 * gyro,
            -a * je,
        ]
    )


class _PathBuilder:
    """Turns a via parameter vector into a :class:`SlewPath`, fast.

    The wheel envelope, canonical frame and instrument boresights are computed
    once; each evaluation then costs a handful of quaternion products and the
    closed-form arc margins, with no linear programme and no path sampling.
    """

    def __init__(self, problem: SlewProblem, n_via: int) -> None:
        self.problem = problem
        self.n_via = int(n_via)
        self.frame = canonical_frame(problem)
        wheels = problem.body.wheels
        assert wheels is not None  # guaranteed by SlewProblem validation
        self.wheels = wheels
        self.normals, self.h_unit = wheels.envelope()
        self.pinv = np.linalg.pinv(wheels.distribution)
        self.inertia = problem.body.inertia
        self.evals = 0
        self._cache: dict[bytes, SlewPath] = {}
        self._fracs = [
            (i + 1) / (self.n_via + 1) for i in range(self.n_via)
        ]
        self._nominal = [quat_slerp(problem.quat_start, problem.quat_goal, f) for f in self._fracs]
        self._instr = [i.boresight_body for i in problem.instruments]
        self._cones = [(c.axis, c.half_angle) for c in problem.keepout]

    # -- wheel envelope --------------------------------------------------

    def _cap(self, u: NDArray[np.float64], bound: float) -> float:
        """Capability along unit ``u`` under the **minimum-norm** allocation.

        ``bound / max_i |(A^+ u)_i|``. The planner sizes against this rather
        than against the exact envelope because it is what a controller using
        the plain pseudo-inverse -- the common case -- can actually deliver.
        The exact envelope is larger; :func:`verify_actuators` reports both, so
        the headroom a smarter allocator would recover is visible rather than
        assumed.
        """
        c = self.pinv @ u
        peak = float(np.max(np.abs(c)))
        if peak < 1e-15:
            return 0.0
        return bound / peak

    def _exact_cap(self, u: NDArray[np.float64], bound: float) -> float:
        """Capability along unit ``u`` over *all* allocations (exact zonotope)."""
        if self.normals.shape[0] == 0:
            return self.wheels._directional_capability_lp(u, bound)
        denom = np.abs(self.normals @ u)
        active = denom > 1e-12
        if not np.any(active):
            return self.wheels._directional_capability_lp(u, bound)
        return float(np.min(bound * self.h_unit[active] / denom[active]))

    # -- geometry --------------------------------------------------------

    def waypoints(self, params: ArrayLike) -> list[NDArray[np.float64]]:
        """Attitudes ``[q_start, via..., q_goal]`` for a parameter vector."""
        p = np.asarray(params, dtype=float).reshape(self.n_via, 3)
        out = [self.problem.quat_start]
        for i in range(self.n_via):
            rot = self.frame @ p[i]
            out.append(_qmul(_qrotvec(rot), self._nominal[i]))
        out.append(self.problem.quat_goal)
        return out

    def _wheel_utilisation(self, torques: NDArray[np.float64]) -> float:
        """Torque-box utilisation under the minimum-norm allocation.

        ``max_i |(A^+ tau)_i| / tau_max``. Above 1 means the pseudo-inverse
        allocation saturates a wheel; a different allocation might still
        deliver the torque, which is why :func:`verify_actuators` reports the
        exact envelope figure alongside.
        """
        if torques.size == 0:
            return 0.0
        return float(np.max(np.abs(torques @ self.pinv.T))) / self.wheels.max_torque

    def segment(self, q_a: NDArray[np.float64], q_b: NDArray[np.float64]) -> SlewSegment:
        """Size one eigenaxis leg against the wheel envelope.

        The scalar starting point is ``alpha = tau_cap / |J e|``. When
        ``size_against_envelope`` is set the acceleration and rate limits are
        then reduced until the exact torque
        ``psi_ddot J e + psi_dot^2 (e x J e)`` fits in the per-wheel box at
        every instant. Because the torque is bilinear in ``(psi_ddot,
        psi_dot^2)`` and the per-wheel infinity norm is convex, its maximum
        over a bang-bang or trapezoidal profile is attained at one of at most
        five corner states, which is why this costs a handful of matrix-vector
        products rather than a sampled search.
        """
        axis, angle = _axis_angle_between(q_a, q_b)
        # eigenaxis in body coordinates at the start of the leg
        axis_body = _qrot(np.array([q_a[0], -q_a[1], -q_a[2], -q_a[3]]), axis)
        je = self.inertia @ axis_body
        je_norm = float(np.linalg.norm(je))
        u = je / je_norm
        gyro = cross3(axis_body, je)
        tau_cap = self._cap(u, self.wheels.max_torque)
        h_cap = self._cap(u, self.wheels.max_momentum)
        alpha = tau_cap / je_norm
        rate_cap = self.problem.rate_limit
        if self.problem.size_against_envelope:
            omega_cap = h_cap / je_norm
            rate_cap = omega_cap if rate_cap is None else min(rate_cap, omega_cap)
            # Reserve at most half the torque box for the coast-phase
            # gyroscopic term, so the acceleration loop below always converges.
            u_gyro = self._wheel_utilisation(np.atleast_2d(rate_cap**2 * gyro))
            if u_gyro > 0.5:
                rate_cap = rate_cap * math.sqrt(0.5 / u_gyro)
        if alpha <= 0.0 or (rate_cap is not None and rate_cap <= 0.0):
            raise _NoCapability(axis)
        prof = make_profile(self.problem.profile, angle, alpha, je_norm, rate_cap)
        if self.problem.size_against_envelope and angle > 0.0:
            for _ in range(6):
                util = self._wheel_utilisation(_torque_corners(prof, je, gyro))
                if util <= 1.0 + 1e-9 or util <= 0.0:
                    break
                alpha = alpha / util
                prof = make_profile(self.problem.profile, angle, alpha, je_norm, rate_cap)
        return SlewSegment(q_a, q_b, axis, angle, prof, alpha, float(rate_cap or math.inf))

    def build(self, params: ArrayLike) -> SlewPath:
        """Full path for a parameter vector, memoised on the exact bytes.

        SLSQP evaluates the objective and the constraint at the same point,
        and again at each finite-difference offset, so a one-entry-per-point
        cache halves the work. The cache is cleared once it reaches 4096
        entries so a long sweep cannot grow without bound.
        """
        arr = np.ascontiguousarray(np.asarray(params, dtype=float).reshape(-1))
        key = arr.tobytes()
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        wp = self.waypoints(arr)
        path = SlewPath(tuple(self.segment(wp[i], wp[i + 1]) for i in range(len(wp) - 1)))
        if len(self._cache) >= 4096:
            self._cache.clear()
        self._cache[key] = path
        return path

    def direct_path(self) -> SlewPath:
        """The plain eigenaxis slew, no via points."""
        return SlewPath((self.segment(self.problem.quat_start, self.problem.quat_goal),))

    # -- constraint ------------------------------------------------------

    def margins(self, path: SlewPath) -> list[float]:
        """Closed-form clearance [rad] per (segment, instrument, cone).

        Same values as the public :func:`path_margins`, computed through the
        unchecked kernels.
        """
        if not self._instr or not self._cones:
            return [float("inf")]
        out: list[float] = []
        for seg in path.segments:
            for b_body in self._instr:
                n0 = _qrot(seg.quat_start, b_body)
                for axis, gamma in self._cones:
                    out.append(min_margin_on_arc_raw(n0, seg.axis, axis, gamma, seg.angle))
        return out

    def min_margin(self, path: SlewPath) -> float:
        """Worst clearance along the whole path [rad]."""
        return min(self.margins(path))

    def soft_min_margin(self, path: SlewPath, beta: float = 60.0) -> float:
        """Smooth lower bound on :meth:`min_margin`, for the SQP constraint.

        ``-(1/beta) log sum_i exp(-beta m_i) <= min_i m_i`` for every positive
        ``beta``, so satisfying this satisfies the true constraint. The bound
        is loose by at most ``log(n)/beta``; at ``beta = 60`` and 12 terms that
        is 0.041 rad, which the planner removes by a final exact check.
        """
        m = np.asarray(self.margins(path), dtype=float)
        m = m[np.isfinite(m)]
        if m.size == 0:
            return float("inf")
        lo = float(np.min(m))
        return lo - float(np.log(np.sum(np.exp(-beta * (m - lo)))) / beta)

    def objective(self, path: SlewPath) -> float:
        """Objective value [s] for a path."""
        return path.objective(self.problem.momentum_weight)


class _NoCapability(Exception):
    """Raised when the wheel array cannot torque about a required axis."""

    def __init__(self, axis: NDArray[np.float64]) -> None:
        super().__init__(f"no wheel torque available about axis {np.round(axis, 6).tolist()}")
        self.axis = axis


def verify_actuators(
    problem: SlewProblem, path: SlewPath, n_samples: int = 121
) -> ActuatorCheck:
    """Check a path against the exact wheel torque and momentum limits.

    The path was *sized* with the scalar model, which drops the gyroscopic
    term ``psi_dot^2 (e x J e)``. This function evaluates the exact torque
    ``psi_ddot J e + psi_dot^2 (e x J e)`` at ``n_samples`` points per segment,
    and asks two questions of it: whether the *minimum-norm* allocation fits
    inside the per-wheel box (which decides feasibility, and is what the
    planner sized against) and whether *any* allocation would (the exact
    zonotope test, reported for the headroom it shows).
    For a principal-axis eigenaxis the exact torque equals the scalar model's;
    otherwise it is larger and can leave the box mid-slew.

    Parameters
    ----------
    problem : SlewProblem
    path : SlewPath
    n_samples : int
        Samples per segment, ``>= 2``.

    Returns
    -------
    ActuatorCheck
    """
    if n_samples < 2:
        raise ValueError(f"n_samples must be >= 2, got {n_samples}")
    wheels = problem.body.wheels
    assert wheels is not None
    j = problem.body.inertia
    normals, h_unit = wheels.envelope()
    pinv = np.linalg.pinv(wheels.distribution)
    worst_t = worst_h = worst_et = worst_eh = worst_gyro = 0.0
    max_tau = max_h = 0.0
    worst_seg = 0
    worst_util = -1.0

    def exact_util(vectors: NDArray[np.float64], bound: float) -> float:
        if vectors.size == 0:
            return 0.0
        if normals.shape[0] == 0:
            return float(np.max(np.linalg.norm(vectors, axis=1))) / bound
        return float(np.max(np.abs(vectors @ normals.T) / (bound * h_unit)))

    for k, seg in enumerate(path.segments):
        if seg.duration <= 0.0:
            continue
        t = np.linspace(0.0, seg.duration, n_samples)
        rate = np.asarray(seg.profile.rate_at(t), dtype=float)
        accel = np.asarray(seg.profile.accel_at(t), dtype=float)
        axis_body = quat_rotate(
            np.concatenate([[seg.quat_start[0]], -seg.quat_start[1:]]), seg.axis
        )
        tau = eigenaxis_torque(j, axis_body, rate, accel)
        gyro = np.cross(axis_body, j @ axis_body)[None, :] * (rate**2)[:, None]
        mag = np.linalg.norm(tau, axis=1)
        frac = np.where(mag > 1e-15, np.linalg.norm(gyro, axis=1) / np.maximum(mag, 1e-300), 0.0)
        h_body = (j @ axis_body)[None, :] * rate[:, None]
        c_tau = float(np.max(np.abs(tau @ pinv.T)))
        c_h = float(np.max(np.abs(h_body @ pinv.T)))
        seg_t = c_tau / wheels.max_torque
        seg_h = c_h / wheels.max_momentum
        if max(seg_t, seg_h) > worst_util:
            worst_util = max(seg_t, seg_h)
            worst_seg = k
        worst_t = max(worst_t, seg_t)
        worst_h = max(worst_h, seg_h)
        worst_et = max(worst_et, exact_util(tau, wheels.max_torque))
        worst_eh = max(worst_eh, exact_util(h_body, wheels.max_momentum))
        worst_gyro = max(worst_gyro, float(np.max(frac)) if frac.size else 0.0)
        max_tau = max(max_tau, c_tau)
        max_h = max(max_h, c_h)
    return ActuatorCheck(
        feasible=bool(worst_t <= 1.0 + 1e-9 and worst_h <= 1.0 + 1e-9),
        max_torque_demand=max_tau,
        max_momentum_demand=max_h,
        torque_utilisation=worst_t,
        momentum_utilisation=worst_h,
        exact_torque_utilisation=worst_et,
        exact_momentum_utilisation=worst_eh,
        worst_segment=worst_seg,
        gyroscopic_fraction=worst_gyro,
    )


def path_margins(problem: SlewProblem, path: SlewPath) -> list[float]:
    """Closed-form clearance [rad] for every (segment, instrument, cone) triple.

    ``[inf]`` when the problem has no cones or no instruments.
    """
    if not problem.instruments or not problem.keepout:
        return [float("inf")]
    out: list[float] = []
    for seg in path.segments:
        for inst in problem.instruments:
            n0 = inst.direction(seg.quat_start)
            for cone in problem.keepout:
                out.append(cone.min_margin_on_arc(n0, seg.axis, seg.angle))
    return out


def path_min_margin(problem: SlewProblem, path: SlewPath) -> float:
    """Worst clearance along a whole path [rad]. Negative means a violation."""
    return min(path_margins(problem, path))


def path_violations(problem: SlewProblem, path: SlewPath) -> tuple[ArcViolation, ...]:
    """Violating stretches of every segment of a path, deepest first.

    ``psi`` in each :class:`~slewforge.keepout.ArcViolation` is measured from
    the start of its own segment, not from the start of the path.
    """
    out: list[ArcViolation] = []
    for seg in path.segments:
        for inst in problem.instruments:
            n0 = inst.direction(seg.quat_start)
            out.extend(problem.keepout.arc_violations(n0, seg.axis, seg.angle, inst.name))
    out.sort(key=lambda v: -v.depth)
    return tuple(out)


def direct_violations(problem: SlewProblem) -> tuple[ArcViolation, ...]:
    """Violating stretches of the plain eigenaxis slew, deepest first.

    This is the diagnostic the rule-of-thumb approach never produces: it names
    the cone, the instrument, where along the slew the violation starts and
    ends, and how deep it goes.
    """
    if problem.slew_angle < 1e-12 or not problem.instruments or not problem.keepout:
        return ()
    e = problem.eigenaxis
    out: list[ArcViolation] = []
    for inst in problem.instruments:
        n0 = inst.direction(problem.quat_start)
        out.extend(problem.keepout.arc_violations(n0, e, problem.slew_angle, inst.name))
    out.sort(key=lambda v: -v.depth)
    return tuple(out)


def _empty_result(
    reason: str,
    detail: str,
    direct_margin: float,
    elapsed: float,
    violations,
    n_starts: int = 0,
    n_evals: int = 0,
) -> PlanResult:
    return PlanResult(
        feasible=False,
        reason=reason,
        detail=detail,
        path=None,
        objective=float("inf"),
        total_time=float("inf"),
        peak_momentum=float("nan"),
        momentum_throughput=float("nan"),
        min_margin=float("-inf"),
        via_params=None,
        n_via=0,
        direct_feasible=False,
        direct_min_margin=direct_margin,
        solve_time_s=elapsed,
        n_objective_evals=n_evals,
        n_starts=n_starts,
        warm_started=False,
        warm_start_accepted=False,
        warm_start_confidence=None,
        actuators=None,
        violations=violations,
    )


def _optimise(
    builder: _PathBuilder,
    p0: NDArray[np.float64],
    required: float,
    maxiter: int,
) -> tuple[NDArray[np.float64] | None, SlewPath | None, float]:
    """One SLSQP run from ``p0``. Returns ``(params, path, objective)``."""

    def obj(p):
        builder.evals += 1
        try:
            return builder.objective(builder.build(p))
        except _NoCapability:
            return 1e9

    def con(p):
        try:
            return builder.soft_min_margin(builder.build(p)) - required
        except _NoCapability:
            return -1e3

    p0 = canonical_rotvec(p0)
    # Box bounds keep the search inside one covering of SO(3). Without them
    # SLSQP drifts to |p| of order 1e5 rad -- the same path, an unusable
    # parameter. pi per component still double-covers, which canonical_rotvec
    # removes from the returned answer.
    bounds = [(-math.pi, math.pi)] * p0.size
    res = minimize(
        obj,
        p0,
        method="SLSQP",
        bounds=bounds,
        constraints=[{"type": "ineq", "fun": con}],
        options={"maxiter": maxiter, "ftol": 1e-6},
    )
    for cand in (canonical_rotvec(res.x), p0):
        try:
            path = builder.build(cand)
        except _NoCapability:
            continue
        if builder.min_margin(path) >= required:
            return np.asarray(cand, dtype=float), path, builder.objective(path)
    return None, None, float("inf")


def plan(
    problem: SlewProblem,
    warm_start: ArrayLike | None = None,
    warm_start_confidence: float | None = None,
    max_via: int = 2,
    maxiter: int = 25,
    start_magnitudes: Sequence[float] = (0.45,),
    cold_fallback: bool = True,
) -> PlanResult:
    """Plan a constrained rest-to-rest slew.

    Parameters
    ----------
    problem : SlewProblem
    warm_start : array_like or None
        Initial via parameter vector, shape ``(3,)`` for one via point. When
        given the optimiser runs once from it; if that fails and
        ``cold_fallback`` is ``True`` the deterministic cold sweep follows.
    warm_start_confidence : float or None
        Recorded on the result; the planner does not act on it, so a
        badly-calibrated confidence cannot change what is returned.
    max_via : int
        Largest number of via points to try, ``>= 1``. The planner escalates
        from 1 only when 1 fails.
    maxiter : int
        SLSQP iteration cap per start. The default 25 was chosen by measurement,
        not by habit: over 14 seeded problems, 40 iterations cost 1.283 s per
        plan and 25 cost 0.701 s for a mean objective 1.10 % worse
        (`validation/validate_planner_settings.py`).
    start_magnitudes : sequence of float
        Deviation magnitudes [rad] for the cold sweep; see
        :func:`cold_start_points`.
    cold_fallback : bool
        Whether a failed warm start falls back to the cold sweep. Setting it
        ``False`` measures the warm start on its own, which is what
        `validation/validate_warm_start.py` reports separately.

    Returns
    -------
    PlanResult
        Never raises for an infeasible problem: infeasibility is data.
    """
    t0 = time.perf_counter()
    if max_via < 1:
        raise ValueError(f"max_via must be >= 1, got {max_via}")
    required = problem.required_margin
    viol = direct_violations(problem)

    if problem.keepout.covers_sphere():
        return _empty_result(
            "keepout_covers_all_directions",
            "a cone with half-angle >= pi excludes the whole sky",
            float("-inf"),
            time.perf_counter() - t0,
            viol,
        )
    m_start = problem.attitude_margin(problem.quat_start)
    if m_start < required:
        bad = problem.attitude_violations(problem.quat_start)
        return _empty_result(
            "start_attitude_violates_keepout",
            f"clearance at q_start is {math.degrees(m_start):.4f} deg, "
            f"required {math.degrees(required):.4f} deg; violated: {bad}",
            float("-inf"),
            time.perf_counter() - t0,
            viol,
        )
    m_goal = problem.attitude_margin(problem.quat_goal)
    if m_goal < required:
        bad = problem.attitude_violations(problem.quat_goal)
        return _empty_result(
            "goal_attitude_violates_keepout",
            f"clearance at q_goal is {math.degrees(m_goal):.4f} deg, "
            f"required {math.degrees(required):.4f} deg; violated: {bad}",
            float("-inf"),
            time.perf_counter() - t0,
            viol,
        )

    builder1 = _PathBuilder(problem, 1)
    try:
        direct = builder1.direct_path()
    except _NoCapability as exc:
        return _empty_result(
            "wheel_torque_unavailable", str(exc), float("-inf"), time.perf_counter() - t0, viol
        )
    direct_margin = builder1.min_margin(direct)

    def finish(
        path: SlewPath,
        params: NDArray[np.float64] | None,
        n_via: int,
        n_starts: int,
        warm_ok: bool,
        evals: int,
    ) -> PlanResult:
        act = verify_actuators(problem, path)
        margin = path_min_margin(problem, path)
        wheels = problem.body.wheels
        assert wheels is not None
        feasible, reason, detail = True, None, ""
        if not act.feasible:
            feasible = False
            if act.torque_utilisation > 1.0 + 1e-9:
                reason = "wheel_torque_limit_exceeded"
                detail = (
                    f"segment {act.worst_segment} demands {act.max_torque_demand:.6g} N*m "
                    f"per wheel against a {wheels.max_torque:.6g} N*m limit "
                    f"({act.torque_utilisation:.3f} of the box); the gyroscopic term is "
                    f"{act.gyroscopic_fraction * 100:.2f} % of the total torque"
                )
            else:
                reason = "wheel_momentum_limit_exceeded"
                detail = (
                    f"segment {act.worst_segment} demands {act.max_momentum_demand:.6g} "
                    f"N*m*s per wheel against a "
                    f"{wheels.max_momentum:.6g} N*m*s limit"
                )
        elif problem.max_time is not None and path.total_time > problem.max_time:
            feasible = False
            reason = "time_limit_exceeded"
            detail = (
                f"best keep-out-feasible path takes {path.total_time:.4f} s against a "
                f"{problem.max_time:.4f} s budget"
            )
        return PlanResult(
            feasible=feasible,
            reason=reason,
            detail=detail,
            path=path,
            objective=path.objective(problem.momentum_weight),
            total_time=path.total_time,
            peak_momentum=path.peak_momentum,
            momentum_throughput=path.momentum_throughput,
            min_margin=margin,
            via_params=params,
            n_via=n_via,
            direct_feasible=bool(direct_margin >= required),
            direct_min_margin=direct_margin,
            solve_time_s=time.perf_counter() - t0,
            n_objective_evals=evals,
            n_starts=n_starts,
            warm_started=warm_start is not None,
            warm_start_accepted=warm_ok,
            warm_start_confidence=(
                None if warm_start_confidence is None else float(warm_start_confidence)
            ),
            actuators=act,
            violations=viol,
        )

    if direct_margin >= required:
        return finish(direct, None, 0, 0, False, 0)

    best: tuple[float, NDArray[np.float64], SlewPath, int] | None = None
    starts_used = 0
    warm_ok = False
    builders = [builder1]

    for n_via in range(1, max_via + 1):
        builder = builder1
        if n_via != 1:
            builder = _PathBuilder(problem, n_via)
            builders.append(builder)
        candidates: list[NDArray[np.float64]] = []
        if warm_start is not None and n_via == 1:
            w = np.asarray(warm_start, dtype=float).reshape(-1)
            if w.size != 3:
                raise ValueError(f"warm_start must have 3 elements, got {w.size}")
            candidates.append(w)
        cold = cold_start_points(n_via, start_magnitudes)
        if warm_start is not None and n_via == 1:
            p, path, val = _optimise(builder, candidates[0], required, maxiter)
            starts_used += 1
            if path is not None:
                warm_ok = True
                best = (val, p, path, n_via)
            if best is not None or not cold_fallback:
                break
        for p0 in cold:
            p, path, val = _optimise(builder, p0, required, maxiter)
            starts_used += 1
            if path is not None and (best is None or val < best[0]):
                best = (val, p, path, n_via)
        if best is not None:
            break

    evals = sum(b.evals for b in builders)
    if best is None:
        return _empty_result(
            "no_feasible_path_found",
            f"{starts_used} optimiser starts up to {max_via} via point(s) found no path "
            f"clearing every cone; the direct slew clears by "
            f"{math.degrees(direct_margin):.4f} deg (negative means it violates)",
            direct_margin,
            time.perf_counter() - t0,
            viol,
            n_starts=starts_used,
            n_evals=evals,
        )
    val, p, path, n_via = best
    return finish(path, p, n_via, starts_used, warm_ok, evals)
