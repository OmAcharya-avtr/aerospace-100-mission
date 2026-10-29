"""Text rendering. No formatting logic anywhere else, and no printing in the library.

Every function here returns a list of lines. The CLI of :mod:`aligncal.__main__`
is the only place in this package that calls ``print``.
"""

from __future__ import annotations

import numpy as np

from .batch import BatchSolution
from .observability import ObservabilityReport
from .recursive import RecursiveSolution

__all__ = ["ARCSEC_PER_RAD", "compare_lines", "observability_lines", "solution_lines"]

#: Arcseconds per radian, ``180 * 3600 / pi``. Defined once.
ARCSEC_PER_RAD: float = 180.0 * 3600.0 / np.pi


def observability_lines(report: ObservabilityReport) -> list[str]:
    """The observability verdict, as text. Unobservable degrees of freedom first."""
    return list(report.summary())


def solution_lines(solution: BatchSolution | RecursiveSolution) -> list[str]:
    """One estimate with the verdict that gated it, as text."""
    return list(solution.summary())


def compare_lines(
    batch_solution: BatchSolution,
    recursive_solution: RecursiveSolution,
    *,
    truth_rad: np.ndarray | None = None,
) -> list[str]:
    """Batch against recursive on the observable subspace, with truth if available.

    The comparison is made per degree of freedom, because a comparison of
    vector magnitudes hides exactly the direction the record cannot support.
    """
    sensor = batch_solution.observability
    lines = [
        f"BATCH AGAINST RECURSIVE for {batch_solution.name}",
        f"{'dof':>4s} {'verdict':>13s} {'batch arcsec':>14s} {'mekf arcsec':>14s} "
        f"{'difference':>14s} {'CRB arcsec':>13s} {'mekf sigma':>13s}",
    ]
    for dof in sensor.degrees_of_freedom:
        batch_component = float(np.dot(batch_solution.theta_rad, dof.direction))
        mekf_component = float(np.dot(recursive_solution.theta_rad, dof.direction))
        crb = batch_solution.sigma_rad_in_direction(dof.direction)
        mekf_sigma = recursive_solution.sigma_rad_in_direction(dof.direction)
        lines.append(
            f"{dof.index:>4d} {dof.verdict:>13s} "
            f"{batch_component * ARCSEC_PER_RAD:>14.4f} "
            f"{mekf_component * ARCSEC_PER_RAD:>14.4f} "
            f"{(mekf_component - batch_component) * ARCSEC_PER_RAD:>14.6f} "
            f"{'inf' if not np.isfinite(crb) else f'{crb * ARCSEC_PER_RAD:.4f}':>13s} "
            f"{mekf_sigma * ARCSEC_PER_RAD:>13.4f}"
        )
    if truth_rad is not None:
        truth = np.asarray(truth_rad, dtype=float).reshape(3)
        lines.append("  against the injected truth, per degree of freedom:")
        for dof in sensor.degrees_of_freedom:
            true_component = float(np.dot(truth, dof.direction))
            batch_error = float(np.dot(batch_solution.theta_rad, dof.direction)) - true_component
            mekf_error = (
                float(np.dot(recursive_solution.theta_rad, dof.direction)) - true_component
            )
            note = (
                " [withheld by the batch solution; its error here is the whole truth]"
                if dof.index in batch_solution.withheld
                else ""
            )
            lines.append(
                f"    dof {dof.index}: truth {true_component * ARCSEC_PER_RAD:+10.4f}, "
                f"batch error {batch_error * ARCSEC_PER_RAD:+10.4f}, "
                f"mekf error {mekf_error * ARCSEC_PER_RAD:+10.4f} arcsec{note}"
            )
    return lines
