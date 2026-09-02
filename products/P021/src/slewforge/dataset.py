"""Deterministic generation of constrained-slew planning problems and labels.

Every problem is synthetic. There is no flight data anywhere in this package,
and no claim is made that the distribution below resembles any real mission's
slew catalogue; see ``DATASET_CARD.md``.

Sampling
--------
Start and goal attitudes are drawn from a Haar-uniform distribution on SO(3)
(Shoemake 1992, via normalised Gaussian quaternions). One to three keep-out
cones are placed on Haar-uniform axes with half-angles drawn uniformly from a
configurable range. A problem is **kept only if it is interesting**: both
endpoints must clear every cone, and the direct eigenaxis slew must violate at
least one. Problems whose direct slew is already clear need no via point and
would train the model to predict zero.

Labels come from the cold multi-start planner in :mod:`slewforge.planner`, so
generating a dataset is expensive (about 0.7 s per accepted problem on the
2-core build machine) and is parallelised over processes.

Features
--------
:func:`problem_features` returns 28 numbers in the problem's own canonical
frame (:func:`slewforge.planner.canonical_frame`), so that rotating an entire
problem leaves every feature unchanged. That invariance is the reason a model
trained on a few hundred problems has any chance of generalising: without it
the model would have to learn SO(3) equivariance from data.

References
----------
K. Shoemake, "Uniform Random Rotations", in *Graphics Gems III*, Academic
    Press (1992) -- Haar-uniform sampling of SO(3).
"""

from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .dynamics import RigidBody
from .keepout import KeepOutCone, KeepOutSet
from .planner import Instrument, SlewProblem, canonical_frame, plan
from .wheels import pyramid_wheels

__all__ = [
    "FEATURE_NAMES",
    "MAX_FEATURE_CONES",
    "PlanningDataset",
    "generate_dataset",
    "generate_problems",
    "problem_features",
    "reference_spacecraft",
]

MAX_FEATURE_CONES = 3
"""Cones represented in the feature vector. A problem with more cones has only
its three worst (by direct-path margin) described; the planner still respects
all of them, so the model is merely under-informed, never wrong."""


def _feature_names() -> tuple[str, ...]:
    names = ["slew_angle", "boresight_cone_angle", "direct_min_margin", "n_cones"]
    for k in range(MAX_FEATURE_CONES):
        names += [
            f"cone{k}_axis_x",
            f"cone{k}_axis_y",
            f"cone{k}_axis_z",
            f"cone{k}_half_angle",
            f"cone{k}_min_margin",
            f"cone{k}_psi_frac",
            f"cone{k}_width_frac",
            f"cone{k}_present",
        ]
    return tuple(names)


FEATURE_NAMES = _feature_names()
"""The 28 feature names, in order. Canonical-frame coordinates throughout."""


def reference_spacecraft(
    inertia: tuple[float, float, float] = (120.0, 100.0, 80.0),
    max_torque: float = 0.15,
    max_momentum: float = 12.0,
) -> RigidBody:
    """The spacecraft used for every problem in the reference dataset.

    A 120/100/80 kg m^2 diagonal inertia with a standard four-wheel pyramid at
    0.15 N*m and 12 N*m*s per wheel. Representative of a several-hundred-kilo
    imaging satellite; the exact values are a modelling choice, not a
    measurement of any real vehicle.
    """
    return RigidBody(np.diag(inertia), pyramid_wheels(max_torque, max_momentum))


def _haar_quaternion(rng: np.random.Generator) -> NDArray[np.float64]:
    q = rng.normal(size=4)
    return q / float(np.linalg.norm(q))


def _haar_direction(rng: np.random.Generator) -> NDArray[np.float64]:
    v = rng.normal(size=3)
    return v / float(np.linalg.norm(v))


def generate_problems(
    n_problems: int,
    seed: int,
    body: RigidBody | None = None,
    boresight_body: tuple[float, float, float] = (1.0, 0.0, 0.0),
    half_angle_range: tuple[float, float] = (math.radians(15.0), math.radians(50.0)),
    n_cone_choices: tuple[int, ...] = (1, 2, 3),
    min_slew_angle: float = math.radians(20.0),
    max_attempts_per_problem: int = 400,
) -> list[SlewProblem]:
    """Sample ``n_problems`` accepted planning problems, deterministically.

    A problem is accepted when both endpoint attitudes clear every cone and the
    direct eigenaxis slew violates at least one -- roughly one draw in ten with
    the default ranges.

    Parameters
    ----------
    n_problems : int
        Number to return, ``>= 1``.
    seed : int
        Seeds ``numpy.random.default_rng``; the same seed always gives the same
        problems.
    body : RigidBody or None
        Defaults to :func:`reference_spacecraft`.
    boresight_body : tuple of float
        The single instrument boresight in body coordinates.
    half_angle_range : tuple of float
        Uniform range for cone half-angles [rad].
    n_cone_choices : tuple of int
        Cone counts drawn uniformly.
    min_slew_angle : float
        Reject slews shorter than this [rad]; a 2 deg slew has no interesting
        geometry.
    max_attempts_per_problem : int
        Raises ``RuntimeError`` if the acceptance rate is so low that this
        budget is exhausted, rather than looping forever.

    Returns
    -------
    list of SlewProblem
    """
    if n_problems < 1:
        raise ValueError(f"n_problems must be >= 1, got {n_problems}")
    lo, hi = half_angle_range
    if not 0.0 < lo <= hi < math.pi:
        raise ValueError(f"half_angle_range must satisfy 0 < lo <= hi < pi, got {half_angle_range}")
    rng = np.random.default_rng(seed)
    spacecraft = body if body is not None else reference_spacecraft()
    inst = Instrument("telescope", np.asarray(boresight_body, dtype=float))
    out: list[SlewProblem] = []
    attempts = 0
    budget = max_attempts_per_problem * n_problems
    while len(out) < n_problems:
        attempts += 1
        if attempts > budget:
            raise RuntimeError(
                f"only {len(out)} of {n_problems} problems accepted in {budget} draws; "
                "widen half_angle_range or lower min_slew_angle"
            )
        q0 = _haar_quaternion(rng)
        q1 = _haar_quaternion(rng)
        cones = tuple(
            KeepOutCone(_haar_direction(rng), float(rng.uniform(lo, hi)), f"cone{k}")
            for k in range(int(rng.choice(n_cone_choices)))
        )
        problem = SlewProblem(q0, q1, spacecraft, KeepOutSet(cones), (inst,))
        if problem.slew_angle < min_slew_angle:
            continue
        if problem.attitude_margin(q0) <= 0.0 or problem.attitude_margin(q1) <= 0.0:
            continue
        n0 = inst.direction(q0)
        if problem.keepout.min_margin_on_arc(n0, problem.eigenaxis, problem.slew_angle) >= 0.0:
            continue
        out.append(problem)
    return out


def problem_features(problem: SlewProblem) -> NDArray[np.float64]:
    """Canonical-frame feature vector, shape ``(28,)``.

    Order is :data:`FEATURE_NAMES`. Units: angles in radians, margins in
    radians, fractions dimensionless. Cones are sorted worst-margin first and
    padded with zeros (and a zero present flag) beyond
    :data:`MAX_FEATURE_CONES`.

    Invariance: every entry is built from inner products taken in the
    canonical frame, so applying the same rotation to ``q_start``, ``q_goal``
    and every cone axis leaves the vector unchanged to floating-point
    precision. ``tests/test_dataset.py::test_features_are_rotation_invariant``
    checks that over random rotations.
    """
    frame = canonical_frame(problem)
    delta = problem.slew_angle
    e = problem.eigenaxis
    inst = problem.instruments[0]
    n0 = inst.direction(problem.quat_start)
    beta = math.acos(min(1.0, max(-1.0, float(np.dot(n0, e)))))

    rows: list[tuple[float, NDArray[np.float64]]] = []
    for cone in problem.keepout:
        margin = cone.min_margin_on_arc(n0, e, delta)
        a, b, _c = cone.arc_coefficients(n0, e)
        psi_star = math.atan2(b, a)
        if psi_star < 0.0:
            psi_star += 2.0 * math.pi
        psi_frac = psi_star / delta if delta > 0.0 else 0.0
        intervals = cone.violation_intervals(n0, e, delta)
        width = sum(hi - lo for lo, hi in intervals)
        width_frac = width / delta if delta > 0.0 else 0.0
        axis_c = frame.T @ cone.axis
        rows.append(
            (
                margin,
                np.array(
                    [
                        axis_c[0],
                        axis_c[1],
                        axis_c[2],
                        cone.half_angle,
                        margin,
                        min(psi_frac, 2.0),
                        width_frac,
                        1.0,
                    ]
                ),
            )
        )
    rows.sort(key=lambda r: r[0])
    direct_margin = rows[0][0] if rows else float(np.pi)
    feats = [delta, beta, direct_margin, float(len(problem.keepout))]
    for k in range(MAX_FEATURE_CONES):
        feats.extend(rows[k][1] if k < len(rows) else np.zeros(8))
    return np.asarray(feats, dtype=float)


@dataclass(frozen=True)
class PlanningDataset:
    """A labelled split.

    Attributes
    ----------
    features : ndarray
        ``(n, 28)``.
    targets : ndarray
        ``(n, 3)`` cold-optimised via parameters in canonical coordinates
        [rad].
    solve_time : ndarray
        ``(n,)`` cold solve wall-clock time [s].
    objective : ndarray
        ``(n,)`` cold objective [s].
    evals : ndarray
        ``(n,)`` cold path evaluations.
    problems : tuple of SlewProblem
        The problems themselves, in the same order, so a benchmark can replan
        them.
    n_attempted : int
        Problems planned, including those the cold planner failed on and which
        are therefore absent from the arrays.
    """

    features: NDArray[np.float64]
    targets: NDArray[np.float64]
    solve_time: NDArray[np.float64]
    objective: NDArray[np.float64]
    evals: NDArray[np.float64]
    problems: tuple[SlewProblem, ...]
    n_attempted: int

    def __len__(self) -> int:
        return int(self.features.shape[0])

    @property
    def label_rate(self) -> float:
        """Fraction of attempted problems the cold planner solved with one via."""
        return len(self) / self.n_attempted if self.n_attempted else 0.0


def _label_one(problem: SlewProblem):
    """Plan one problem with the cold multi-start planner (worker function)."""
    result = plan(problem, max_via=1)
    if not result.feasible or result.via_params is None or result.n_via != 1:
        return None
    return (
        problem_features(problem),
        np.asarray(result.via_params, dtype=float),
        result.solve_time_s,
        result.objective,
        float(result.n_objective_evals),
    )


def generate_dataset(
    n_problems: int,
    seed: int,
    n_jobs: int = 2,
    **problem_kwargs,
) -> PlanningDataset:
    """Generate problems and label them with the cold planner.

    Parameters
    ----------
    n_problems : int
        Problems to attempt.
    seed : int
        Passed to :func:`generate_problems`.
    n_jobs : int
        Worker processes. ``1`` runs in-process, which is what the tests use;
        ``2`` matches the build machine. Labels are identical either way --
        the cold planner is deterministic -- so parallelism changes only the
        wall-clock time, never the dataset.
    **problem_kwargs
        Forwarded to :func:`generate_problems`.

    Returns
    -------
    PlanningDataset
        Problems the cold planner could not solve with a single via point are
        dropped; :attr:`PlanningDataset.label_rate` reports how many.
    """
    if n_jobs < 1:
        raise ValueError(f"n_jobs must be >= 1, got {n_jobs}")
    problems = generate_problems(n_problems, seed, **problem_kwargs)
    if n_jobs == 1:
        labelled = [_label_one(p) for p in problems]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            labelled = list(pool.map(_label_one, problems, chunksize=4))
    keep = [(p, r) for p, r in zip(problems, labelled, strict=True) if r is not None]
    if not keep:
        raise RuntimeError("the cold planner solved none of the generated problems")
    feats = np.stack([r[0] for _, r in keep])
    targ = np.stack([r[1] for _, r in keep])
    return PlanningDataset(
        features=feats,
        targets=targ,
        solve_time=np.array([r[2] for _, r in keep]),
        objective=np.array([r[3] for _, r in keep]),
        evals=np.array([r[4] for _, r in keep]),
        problems=tuple(p for p, _ in keep),
        n_attempted=len(problems),
    )
