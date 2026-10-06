"""How far the fade statistics can move before the answer changes.

A rate optimiser that returns an optimum and stops has told the user almost
nothing, because the input it was given -- a scintillation index, a link
margin -- is itself an estimate with an error bar. The question that matters
is whether a 10 % error in that estimate changes the MODCOD, and this module
answers it by resolving the boundary rather than by sampling near it.

Method
------
For a scalar parameter ``theta`` and a rebuild function ``theta -> RateProblem``:

1. Solve at ``theta_0`` and record the canonical support (the *signature*).
2. Walk a geometric grid outward in each direction until the signature changes
   or the scan bound is reached.
3. Where it changed, bisect between the last-unchanged and first-changed grid
   point until the bracket is narrower than ``rel_tol`` of ``theta_0``.

The reported interval is therefore accurate to ``rel_tol``, and the boundary
is a solved root, not a grid point. Two honest caveats, both reported in the
returned record rather than left for the user to discover:

* **A change narrower than one grid step can be missed.** If the signature
  changes and changes back between two grid points, the scan does not see it.
  ``grid_points`` is reported so the step size is known;
  ``refine_bracket=True`` on a suspicious result tightens it.
* **An interval that hits the scan bound is a lower bound, not a width.**
  ``bounded_below_by_scan`` and ``bounded_above_by_scan`` say so, and
  ``is_flat`` is only ever True on the strength of the scanned range.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .problem import InfeasibleProblem, RateProblem
from .select import select_rate

__all__ = [
    "SensitivityReport",
    "margin_sensitivity",
    "scintillation_sensitivity",
    "stability_interval",
    "target_sensitivity",
]

Signature = tuple[int, ...] | None


@dataclass(frozen=True)
class SensitivityReport:
    """The stability interval of a decision in one parameter.

    Attributes
    ----------
    parameter
        Name of the swept parameter.
    nominal
        ``theta_0``, in the parameter's own units.
    lower, upper
        Interval over which the canonical support is unchanged, same units.
        ``lower`` may be the scan floor and ``upper`` the scan ceiling; see
        the two ``bounded_*_by_scan`` flags.
    signature
        The canonical support at ``theta_0``, as MODCOD indices.
    relative_width
        ``(upper - lower) / nominal``, dimensionless. Undefined (NaN) when
        ``nominal`` is zero.
    downward_headroom, upward_headroom
        ``(nominal - lower) / nominal`` and ``(upper - nominal) / nominal``,
        the fractional error in ``theta`` that can be tolerated in each
        direction.
    knife_edge_at
        The relative perturbation used for the knife-edge test, e.g. 0.10.
    is_knife_edge
        True when a perturbation of ``knife_edge_at`` in **either** direction
        changes the support. This is the headline flag.
    is_flat
        True when neither direction changes the support anywhere in the
        scanned range. Conditional on the scan range, which is reported.
    bounded_below_by_scan, bounded_above_by_scan
        True when the corresponding endpoint is the scan bound rather than a
        resolved boundary.
    boundary_signature_below, boundary_signature_above
        The support just outside each resolved boundary, or None when that
        side was not resolved. ``None`` as a *signature* means the problem
        became infeasible there, which is reported as a change.
    grid_points
        Number of grid points per direction in the coarse scan.
    """

    parameter: str
    nominal: float
    lower: float
    upper: float
    signature: Signature
    relative_width: float
    downward_headroom: float
    upward_headroom: float
    knife_edge_at: float
    is_knife_edge: bool
    is_flat: bool
    bounded_below_by_scan: bool
    bounded_above_by_scan: bool
    boundary_signature_below: Signature
    boundary_signature_above: Signature
    grid_points: int


def _signature(problem: RateProblem) -> Signature:
    """Canonical support, or None when the instance is infeasible."""
    try:
        return select_rate(problem).support
    except InfeasibleProblem:
        return None


def stability_interval(
    rebuild: Callable[[float], RateProblem],
    nominal: float,
    *,
    parameter: str,
    lower_bound: float,
    upper_bound: float,
    grid_points: int = 48,
    rel_tol: float = 1e-4,
    knife_edge_at: float = 0.10,
) -> SensitivityReport:
    """Resolve the interval in ``theta`` over which the decision is unchanged.

    Parameters
    ----------
    rebuild
        ``theta -> RateProblem``. Must be pure.
    nominal
        ``theta_0``.
    parameter
        Label for the report.
    lower_bound, upper_bound
        Scan range. Must bracket ``nominal`` strictly.
    grid_points
        Coarse-scan points per direction, >= 2.
    rel_tol
        Bisection stops when the bracket is narrower than
        ``rel_tol * max(abs(nominal), 1)``.
    knife_edge_at
        Relative perturbation for the knife-edge test, in (0, 1).
    """
    if not lower_bound < nominal < upper_bound:
        raise ValueError(
            f"nominal {nominal!r} must lie strictly inside "
            f"({lower_bound!r}, {upper_bound!r})"
        )
    if grid_points < 2:
        raise ValueError(f"grid_points must be >= 2, got {grid_points}")
    if not 0.0 < knife_edge_at < 1.0:
        raise ValueError(f"knife_edge_at must be in (0, 1), got {knife_edge_at!r}")

    base = _signature(rebuild(nominal))
    scale = max(abs(nominal), 1.0)
    tol = rel_tol * scale

    def walk(end: float) -> tuple[float, bool, Signature]:
        grid = np.linspace(nominal, end, grid_points + 1)[1:]
        last_same = nominal
        for value in grid:
            if _signature(rebuild(float(value))) != base:
                lo, hi = last_same, float(value)
                while hi - lo > tol if hi > lo else lo - hi > tol:
                    mid = 0.5 * (lo + hi)
                    if _signature(rebuild(mid)) == base:
                        lo = mid
                    else:
                        hi = mid
                return lo, False, _signature(rebuild(hi))
            last_same = float(value)
        return float(end), True, None

    upper, up_bounded, sig_above = walk(upper_bound)
    lower, lo_bounded, sig_below = walk(lower_bound)

    rel_width = (upper - lower) / nominal if nominal != 0.0 else float("nan")
    down = (nominal - lower) / nominal if nominal != 0.0 else float("nan")
    up = (upper - nominal) / nominal if nominal != 0.0 else float("nan")

    knife = False
    for factor in (1.0 - knife_edge_at, 1.0 + knife_edge_at):
        probe = nominal * factor
        if lower_bound < probe < upper_bound and _signature(rebuild(probe)) != base:
            knife = True

    return SensitivityReport(
        parameter=parameter,
        nominal=float(nominal),
        lower=float(lower),
        upper=float(upper),
        signature=base,
        relative_width=float(rel_width),
        downward_headroom=float(down),
        upward_headroom=float(up),
        knife_edge_at=float(knife_edge_at),
        is_knife_edge=bool(knife),
        is_flat=bool(up_bounded and lo_bounded),
        bounded_below_by_scan=bool(lo_bounded),
        bounded_above_by_scan=bool(up_bounded),
        boundary_signature_below=sig_below,
        boundary_signature_above=sig_above,
        grid_points=int(grid_points),
    )


def scintillation_sensitivity(
    problem: RateProblem,
    *,
    span: float = 4.0,
    **kwargs: object,
) -> SensitivityReport:
    """Stability of the decision in the scintillation index.

    Requires ``problem.fade`` to expose ``scintillation_index`` and to be
    reconstructible from it -- true for
    :class:`~coderateopt.fade.LognormalFade` and
    :class:`~coderateopt.fade.GammaGammaFade`. ``span`` sets the scan range to
    ``[s0 / span, s0 * span]``.
    """
    fade = problem.fade
    s0 = getattr(fade, "scintillation_index", None)
    if s0 is None:
        raise TypeError(
            f"{type(fade).__name__} has no scintillation_index; sweep the margin instead"
        )
    kind = type(fade)
    if hasattr(kind, "from_scintillation"):
        ratio = fade.beta / fade.alpha
        rebuild_fade = lambda s: kind.from_scintillation(s, ratio=ratio)  # noqa: E731
    else:
        median = getattr(fade, "median_preserving", False)
        rebuild_fade = lambda s: kind(s, median_preserving=median)  # noqa: E731
    return stability_interval(
        lambda s: problem.with_fade(rebuild_fade(s)),
        float(s0),
        parameter="scintillation_index",
        lower_bound=float(s0) / span,
        upper_bound=float(s0) * span,
        **kwargs,  # type: ignore[arg-type]
    )


def margin_sensitivity(
    problem: RateProblem, *, span_db: float = 12.0, **kwargs: object
) -> SensitivityReport:
    """Stability of the decision in the nominal link margin, dB."""
    m0 = problem.margin_db
    return stability_interval(
        problem.with_margin,
        float(m0),
        parameter="margin_db",
        lower_bound=float(m0) - span_db,
        upper_bound=float(m0) + span_db,
        **kwargs,  # type: ignore[arg-type]
    )


def target_sensitivity(problem: RateProblem, **kwargs: object) -> SensitivityReport:
    """Stability of the decision in the availability target."""
    t0 = problem.availability_target
    return stability_interval(
        problem.with_target,
        float(t0),
        parameter="availability_target",
        lower_bound=1e-6,
        upper_bound=1.0 - 1e-9,
        **kwargs,  # type: ignore[arg-type]
    )
