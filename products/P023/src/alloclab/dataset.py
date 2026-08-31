"""Deterministic synthetic dataset generation for the learned allocator.

Everything here is synthetic. There is no flight data, no test-stand data and
no hardware in this package; the labels are the exact QP allocations produced
by :func:`alloclab.allocation.qp_allocate`, so the learned allocator is being
trained to imitate a solver, not to model a physical system. See
``DATASET_CARD.md``.

Regeneration is bit-for-bit reproducible from the integer ``seed`` through
``numpy.random.default_rng``.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .allocation import qp_allocate
from .ams import attainable_moment_set
from .effectors import EffectorSet, thruster_cluster

__all__ = [
    "AllocationDataset",
    "reference_thruster_cluster",
    "generate_dataset",
    "torque_scale",
]


def reference_thruster_cluster(
    max_thrust: float = 1.0, arm: float = 0.5
) -> EffectorSet:
    """The eight-thruster reference cluster used throughout the AI benchmark.

    Six thrusters form three antiparallel couples giving +/- torque about each
    body axis; a seventh and eighth sit on a corner and fire tangentially,
    adding a skew torque direction with all three components non-zero. The set
    is genuinely over-actuated (m = 8 for three controlled axes) and every
    effector is one-sided, ``u in [0, max_thrust]``.

    ==== ===================== ================== ==========================
    id   position r [m]        thrust dir (body)  torque column [m]
    ==== ===================== ================== ==========================
    t1   (0, arm, 0)           (0, 0, +1)         (+arm, 0, 0)
    t2   (0, arm, 0)           (0, 0, -1)         (-arm, 0, 0)
    t3   (arm, 0, 0)           (0, 0, -1)         (0, +arm, 0)
    t4   (arm, 0, 0)           (0, 0, +1)         (0, -arm, 0)
    t5   (arm, 0, 0)           (0, +1, 0)         (0, 0, +arm)
    t6   (arm, 0, 0)           (0, -1, 0)         (0, 0, -arm)
    t7   (arm, arm, arm)       (+1, -1, 0)/sqrt2  +arm (1, 1, -2)/sqrt2
    t8   (arm, arm, arm)       (-1, +1, 0)/sqrt2  -arm (1, 1, -2)/sqrt2
    ==== ===================== ================== ==========================

    Columns follow eq. (2) of :mod:`alloclab.effectors`, ``B[:, i] = r_i x
    F_hat_i``. Torque is in N*m for thrust in N and arm in m.

    The four distinct generator directions are in general position -- no two
    parallel, no three coplanar -- so the attainable moment set has exactly
    ``4 * 3 + 2 = 14`` vertices, which is the closed-form check in
    ``validation/validate_ams.py``.

    This is a plausible arrangement, not a flight configuration from any real
    vehicle; plume impingement, minimum impulse bit and centre-of-mass offset
    are all ignored.
    """
    if arm <= 0.0:
        raise ValueError(f"arm must be > 0, got {arm}")
    s = 1.0 / np.sqrt(2.0)
    positions = np.array(
        [
            [0.0, arm, 0.0],
            [0.0, arm, 0.0],
            [arm, 0.0, 0.0],
            [arm, 0.0, 0.0],
            [arm, 0.0, 0.0],
            [arm, 0.0, 0.0],
            [arm, arm, arm],
            [arm, arm, arm],
        ]
    )
    directions = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [s, -s, 0.0],
            [-s, s, 0.0],
        ]
    )
    names = tuple(f"t{i + 1}" for i in range(8))
    return thruster_cluster(positions, directions, max_thrust, 0.0, names)


def torque_scale(eset: EffectorSet) -> float:
    """Largest torque magnitude on the AMS boundary [N*m].

    Used to normalise the learned allocator's inputs. Computed from the AMS
    vertices, so it is the exact circumscribed radius of the attainable set.
    """
    ams = attainable_moment_set(eset)
    return float(np.max(np.linalg.norm(ams.vertices, axis=1)))


@dataclass
class AllocationDataset:
    """One generated dataset.

    Attributes
    ----------
    torques
        ``(n, 3)`` commanded body torques [N*m].
    health
        ``(n, m)`` float mask, 1.0 where the effector can still move.
    commands
        ``(n, m)`` QP reference allocations [command units].
    residual_norm
        ``(n,)`` torque residual of the QP label [N*m]; non-zero exactly on
        the samples whose command is outside the degraded AMS.
    attainable
        ``(n,)`` bool, True when the QP label meets the command.
    seed
        The seed used, for regeneration.
    """

    torques: np.ndarray
    health: np.ndarray
    commands: np.ndarray
    residual_norm: np.ndarray
    attainable: np.ndarray
    seed: int

    def __len__(self) -> int:
        return int(self.torques.shape[0])

    @property
    def attainable_fraction(self) -> float:
        """Fraction of samples whose command the degraded set could meet."""
        return float(np.mean(self.attainable))


def _failure_combinations(m: int, max_failures: int) -> list[tuple[int, ...]]:
    combos: list[tuple[int, ...]] = [()]
    for k in range(1, max_failures + 1):
        combos.extend(combinations(range(m), k))
    return combos


def generate_dataset(
    eset: EffectorSet,
    n_samples: int,
    seed: int,
    max_failures: int = 2,
    failure_prob: float = 0.5,
    magnitude_range: tuple[float, float] = (0.0, 1.05),
    gamma: float = 1e12,
) -> AllocationDataset:
    """Generate ``n_samples`` (torque, health) -> QP command examples.

    Sampling, per example:

    1. Draw the number of failed effectors: 0 with probability
       ``1 - failure_prob``, otherwise uniform on ``1..max_failures``; then
       draw which ones, uniformly without replacement. Failed effectors are
       failed-off (pinned to 0 for the one-sided thruster set).
    2. Draw a direction uniformly on the unit sphere (normalised Gaussian).
    3. Draw a magnitude fraction ``rho`` uniformly on ``magnitude_range`` and
       set ``tau = rho * boundary_scale(direction) * direction``, where the
       boundary scale is that of the **degraded** attainable moment set. So
       ``rho <= 1`` is attainable and ``rho > 1`` is not; the default upper
       limit 1.05 puts about 5% of the samples just outside the set, which is
       where saturation behaviour has to be learned.
    4. Label with :func:`alloclab.allocation.qp_allocate` at ``u_pref =
       lower`` -- the minimum-effort preference, which for a one-sided
       thruster set means minimum total thrust.

    The AMS of each distinct failure combination is computed once and cached,
    so the cost is ``C`` convex hulls plus ``n_samples`` bounded least-squares
    solves.

    Raises
    ------
    ValueError
        On non-positive ``n_samples``, out-of-range probabilities, a
        ``magnitude_range`` that is not increasing and non-negative, or
        ``max_failures`` outside ``0..m``.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be > 0, got {n_samples}")
    m = eset.n_effectors
    if not 0 <= max_failures <= m:
        raise ValueError(f"max_failures must be in [0, {m}], got {max_failures}")
    if not 0.0 <= failure_prob <= 1.0:
        raise ValueError(f"failure_prob must be in [0, 1], got {failure_prob}")
    lo_r, hi_r = magnitude_range
    if not 0.0 <= lo_r < hi_r:
        raise ValueError(f"magnitude_range must satisfy 0 <= lo < hi, got {magnitude_range}")

    rng = np.random.default_rng(int(seed))
    combos = _failure_combinations(m, max_failures)
    cache: dict[tuple[int, ...], tuple[EffectorSet, object]] = {}
    for combo in combos:
        degraded = eset.with_failures(list(combo)) if combo else eset
        cache[combo] = (degraded, attainable_moment_set(degraded))

    torques = np.zeros((n_samples, 3))
    health = np.zeros((n_samples, m))
    commands = np.zeros((n_samples, m))
    residual = np.zeros(n_samples)
    attainable = np.zeros(n_samples, dtype=bool)

    single_and_up = [c for c in combos if c]
    for i in range(n_samples):
        if max_failures == 0 or rng.random() >= failure_prob or not single_and_up:
            combo: tuple[int, ...] = ()
        else:
            k = int(rng.integers(1, max_failures + 1))
            combo = tuple(sorted(rng.choice(m, size=k, replace=False).tolist()))
        degraded, ams = cache[combo]

        d = rng.normal(size=3)
        n = np.linalg.norm(d)
        while n < 1e-12:  # pragma: no cover - probability ~0
            d = rng.normal(size=3)
            n = np.linalg.norm(d)
        d = d / n
        if ams.degenerate:
            scale = 0.0
        else:
            scale = ams.boundary_scale(d)
            if not np.isfinite(scale) or scale < 0.0:
                scale = 0.0
        rho = rng.uniform(lo_r, hi_r)
        tau = rho * scale * d

        res = qp_allocate(degraded, tau, u_pref=degraded.lower, gamma=gamma)
        torques[i] = tau
        health[i] = degraded.health()
        commands[i] = res.commands
        residual[i] = res.residual_norm
        attainable[i] = res.feasible

    return AllocationDataset(
        torques=torques,
        health=health,
        commands=commands,
        residual_norm=residual,
        attainable=attainable,
        seed=int(seed),
    )
