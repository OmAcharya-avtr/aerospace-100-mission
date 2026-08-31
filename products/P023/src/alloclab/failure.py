"""Effector-failure handling and explicit infeasibility reporting.

The point of this module is the distinction that saturation logic usually
loses: a torque command that the *degraded* effector set can still produce
must be met exactly, and one that it cannot must be reported as unmet with the
size of the shortfall, not clipped and returned as though it had succeeded.

The reallocation itself is nothing more than re-running an allocator against
the degraded effectiveness matrix; that a failed effector is handled by
recomputing the allocation rather than by patching the nominal command is the
argument of Haerkegaard (2003), "Resolving actuator redundancy -- control
allocation vs. linear quadratic control", Proc. European Control Conference,
1826-1831, doi:10.23919/ecc.2003.7085231, sec. "Actuator failure".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .allocation import (
    DEFAULT_TORQUE_TOL,
    AllocationResult,
    InfeasibleAllocationError,
    allocate,
    lp_allocate,
)
from .ams import attainable_moment_set
from .effectors import EffectorSet

__all__ = ["FailureReport", "reallocate_after_failure", "failure_margin"]


@dataclass
class FailureReport:
    """Nominal-versus-degraded comparison for a set of failed effectors.

    Attributes
    ----------
    failed
        Indices of the failed effectors.
    nominal
        Allocation on the healthy set.
    degraded
        Allocation on the degraded set. ``degraded.status == "infeasible"``
        when the remaining effectors cannot meet the command.
    attainable
        Whether the command lies in the degraded attainable moment set, as
        decided by the exact LP feasibility test.
    residual_norm
        2-norm of the unmet torque after reallocation [N*m]; 0 to solver
        tolerance when ``attainable``.
    remaining_rank
        Rank of the degraded effectiveness matrix over effectors that can
        still move. Below 3, at least one body axis has no authority left.
    volume_ratio
        Degraded AMS volume divided by the nominal AMS volume, dimensionless.
        ``None`` when either set is degenerate.
    """

    failed: tuple[int, ...]
    nominal: AllocationResult
    degraded: AllocationResult
    attainable: bool
    residual_norm: float
    remaining_rank: int
    volume_ratio: float | None

    def __str__(self) -> str:  # pragma: no cover - formatting only
        vr = "n/a" if self.volume_ratio is None else f"{self.volume_ratio:.4f}"
        return (
            f"failed={list(self.failed)} attainable={self.attainable} "
            f"residual={self.residual_norm:.3e} N*m rank={self.remaining_rank} "
            f"AMS volume ratio={vr}"
        )


def reallocate_after_failure(
    eset: EffectorSet,
    torque: ArrayLike,
    failed: ArrayLike,
    method: str = "qp",
    stuck_at: ArrayLike | None = None,
    require_feasible: bool = False,
    torque_tol: float = DEFAULT_TORQUE_TOL,
    compute_volume: bool = True,
    **kwargs,
) -> FailureReport:
    """Reallocate ``torque`` after the listed effectors fail.

    The degraded set is built with :meth:`EffectorSet.with_failures`, which
    pins the failed commands (to 0 by default, or to ``stuck_at``) rather than
    deleting their columns, so a thruster stuck open still contributes its
    torque bias and the remaining effectors have to cancel it.

    Feasibility is decided **before** looking at the allocator's output, by the
    exact LP test :func:`alloclab.allocation.is_attainable`, so the report
    distinguishes "the allocator failed" from "no command could have worked".

    Parameters
    ----------
    method
        Any name in :data:`alloclab.allocation.METHODS`. Bounds are only
        enforced by ``"lp"``, ``"qp"`` and (heuristically) ``"rpi"``.
    require_feasible
        When True, raise :class:`InfeasibleAllocationError` instead of
        returning a report whose ``degraded`` result is infeasible.
    compute_volume
        Compute the nominal and degraded AMS volumes for ``volume_ratio``.
        Costs two convex hulls; set False in a tight loop.

    Raises
    ------
    InfeasibleAllocationError
        If ``require_feasible`` and the command cannot be met.
    """
    tau = np.asarray(torque, dtype=float).reshape(-1)
    if tau.shape != (3,):
        raise ValueError(f"torque must have shape (3,), got {np.shape(torque)}")
    idx = tuple(int(i) for i in np.atleast_1d(np.asarray(failed, dtype=int)))
    degraded_set = eset.with_failures(idx, stuck_at=stuck_at)

    nominal = allocate(eset, tau, method=method, **kwargs)
    degraded = allocate(degraded_set, tau, method=method, **kwargs)

    lp_probe = lp_allocate(degraded_set, tau, objective="min_error", torque_tol=torque_tol)
    attainable = bool(lp_probe.residual_norm <= torque_tol)

    if attainable and not degraded.feasible:
        degraded.message = (
            f"{degraded.message + '; ' if degraded.message else ''}"
            f"command IS attainable by the degraded set (LP residual "
            f"{lp_probe.residual_norm:.3e} N*m) but method '{method}' did not find it"
        ).strip()
    elif not attainable:
        degraded.message = (
            f"command is NOT attainable by the degraded set (exact LP feasibility test: "
            f"optimal weighted 1-norm torque error "
            f"{lp_probe.extras['lp_objective']:.6e} N*m, which is zero only for an "
            f"attainable command); best 2-norm residual from method '{method}' is "
            f"{degraded.residual_norm:.6e} N*m"
        )
        degraded.status = "infeasible"
        degraded.feasible = False

    volume_ratio = None
    if compute_volume:
        nom_ams = attainable_moment_set(eset)
        deg_ams = attainable_moment_set(degraded_set)
        if not nom_ams.degenerate and nom_ams.volume > 0.0:
            volume_ratio = float(deg_ams.volume / nom_ams.volume)

    report = FailureReport(
        failed=idx,
        nominal=nominal,
        degraded=degraded,
        attainable=attainable,
        residual_norm=float(degraded.residual_norm),
        remaining_rank=degraded_set.rank,
        volume_ratio=volume_ratio,
    )
    if require_feasible and not degraded.feasible:
        raise InfeasibleAllocationError(
            f"torque {tau.tolist()} N*m cannot be allocated after failure of effectors "
            f"{list(idx)}: {degraded.message} "
            f"(bound violation {degraded.bound_violation:.3e}, "
            f"remaining rank {degraded_set.rank})"
        )
    return report


def failure_margin(
    eset: EffectorSet, torque: ArrayLike, failed: ArrayLike, stuck_at: ArrayLike | None = None
) -> float:
    """How much the command could grow before the degraded set cannot meet it.

    Returns ``rho / ||tau||`` where ``rho`` is the distance from the origin to
    the boundary of the degraded attainable moment set along ``tau``
    (Durham 1993's direct-allocation scale). A value below 1 means the command
    is already outside the degraded set; ``inf`` is returned for a zero
    command.

    Raises
    ------
    RuntimeError
        If the degraded AMS is degenerate (rank below 3), where the direction
        scale is not defined.
    """
    tau = np.asarray(torque, dtype=float).reshape(-1)
    if tau.shape != (3,):
        raise ValueError(f"torque must have shape (3,), got {np.shape(torque)}")
    norm = float(np.linalg.norm(tau))
    if norm == 0.0:
        return float("inf")
    degraded_set = eset.with_failures(failed, stuck_at=stuck_at)
    ams = attainable_moment_set(degraded_set)
    return ams.boundary_scale(tau) / norm
