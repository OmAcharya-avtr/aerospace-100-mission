"""Numerical diagnostics: facet growth, iteration stalling, tolerance sensitivity.

These are not conveniences.  The three failure modes of the maximal-robust-
invariant-set recursion in floating point are

1. **facet growth** -- every iteration appends `m` rows and only redundancy
   removal keeps the representation finite;
2. **stalling** -- the shrink margin of equation (9) decays geometrically with
   a ratio set by the slowest closed-loop pole, so a pole near the unit circle
   needs an iteration count no cap can be chosen for in advance;
3. **tolerance sensitivity** -- the redundancy test of equation (2) is an
   inequality with a slack, and a slack large enough to drop a facet that
   genuinely bounds the set changes the computed answer.

This module measures all three on a given system and returns plain records, so
that the README and VALIDATION tables are generated from measurements rather
than written by hand.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .invariant import (
    CONVERGED,
    InvariantSetResult,
    maximal_robust_invariant_set,
    verify_robust_invariance,
)
from .polytope import Polytope

__all__ = ["GrowthRow", "ToleranceRow", "growth_table", "tolerance_sweep"]


@dataclass(frozen=True)
class GrowthRow:
    """One row of the per-iteration growth table.

    Attributes
    ----------
    k : int
        Iteration index.
    halfspaces_raw : int
        Rows before redundancy removal: `m_{k-1} + m_{k-1}` by construction,
        so this doubles every iteration if nothing is removed.
    halfspaces : int
        Rows after removal.
    removed : int
        `halfspaces_raw - halfspaces`.
    vertices : int or None
    volume : float or None
        In (state unit)^n.
    volume_ratio : float or None
        `volume_k / volume_{k-1}`, dimensionless.
    shrink_margin : float
        Equation (9) margin, in `b` units.
    margin_ratio : float or None
        `shrink_margin_k / shrink_margin_{k-1}`, dimensionless.  This is the
        stalling diagnostic: a ratio close to 1 means the recursion needs many
        more iterations.
    """

    k: int
    halfspaces_raw: int
    halfspaces: int
    removed: int
    vertices: int | None
    volume: float | None
    volume_ratio: float | None
    shrink_margin: float
    margin_ratio: float | None


def growth_table(result: InvariantSetResult) -> list[GrowthRow]:
    """Turn a result's history into the measured growth table."""
    rows: list[GrowthRow] = []
    prev_vol: float | None = None
    prev_margin: float | None = None
    for rec in result.history:
        vol_ratio = None
        if rec.volume is not None and prev_vol is not None and prev_vol > 0.0:
            vol_ratio = rec.volume / prev_vol
        margin_ratio = None
        if (
            prev_margin is not None
            and np.isfinite(prev_margin)
            and abs(prev_margin) > 0.0
            and np.isfinite(rec.shrink_margin)
        ):
            margin_ratio = rec.shrink_margin / prev_margin
        rows.append(
            GrowthRow(
                k=rec.k,
                halfspaces_raw=rec.halfspaces_raw,
                halfspaces=rec.halfspaces,
                removed=rec.halfspaces_raw - rec.halfspaces,
                vertices=rec.vertices,
                volume=rec.volume,
                volume_ratio=vol_ratio,
                shrink_margin=rec.shrink_margin,
                margin_ratio=margin_ratio,
            )
        )
        if rec.volume is not None:
            prev_vol = rec.volume
        prev_margin = rec.shrink_margin
    return rows


@dataclass(frozen=True)
class ToleranceRow:
    """One row of the redundancy-tolerance sweep.

    Attributes
    ----------
    redundancy_tol : float
        The `tol` of equation (2), in `b` units.
    termination : str
    iterations : int
    facets : int or None
    volume : float or None
        In (state unit)^n.
    invariance_margin : float or None
        `verify_robust_invariance` worst facet margin on the returned set, in
        `b` units, computed independently of the recursion.  **Positive means
        the returned set is not actually robustly invariant**, which is the
        failure a too-large redundancy tolerance produces.
    volume_vs_reference : float or None
        `volume / volume(reference row)`, dimensionless, where the reference
        row is the smallest tolerance in the sweep.  A value above 1 means
        this tolerance produced a strictly larger set, i.e. a different
        answer.
    """

    redundancy_tol: float
    termination: str
    iterations: int
    facets: int | None
    volume: float | None
    invariance_margin: float | None
    volume_vs_reference: float | None


def tolerance_sweep(
    A: np.ndarray,
    X: Polytope,
    W: Polytope,
    tolerances: list[float],
    *,
    max_iter: int = 60,
    convergence_tol: float = 1e-9,
) -> list[ToleranceRow]:
    """Run the recursion once per redundancy tolerance and compare answers.

    Parameters
    ----------
    A, X, W
        As in :func:`invariantset.maximal_robust_invariant_set`.
    tolerances : list of float
        Redundancy tolerances to try, in `b` units.  The first element after
        sorting ascending is used as the reference answer.
    max_iter, convergence_tol
        Held fixed across the sweep so that only the redundancy tolerance
        varies.

    Returns
    -------
    list of ToleranceRow
        Sorted by ascending tolerance.
    """
    if not tolerances:
        raise ValueError("tolerances must be a non-empty list")
    ordered = sorted(float(t) for t in tolerances)
    rows: list[ToleranceRow] = []
    reference_volume: float | None = None
    for tol in ordered:
        res = maximal_robust_invariant_set(
            A,
            X,
            W,
            max_iter=max_iter,
            convergence_tol=convergence_tol,
            redundancy_tol=tol,
            track_geometry=False,
        )
        facets = None if res.polytope is None else res.polytope.n_halfspaces
        volume = None
        margin = None
        if res.polytope is not None and res.termination == CONVERGED:
            volume = float(res.polytope.volume())
            margin = float(verify_robust_invariance(A, res.polytope, W)[1])
        if reference_volume is None and volume is not None:
            reference_volume = volume
        ratio = None
        if volume is not None and reference_volume is not None and reference_volume > 0:
            ratio = volume / reference_volume
        rows.append(
            ToleranceRow(
                redundancy_tol=tol,
                termination=res.termination,
                iterations=res.iterations,
                facets=facets,
                volume=volume,
                invariance_margin=margin,
                volume_vs_reference=ratio,
            )
        )
    return rows
