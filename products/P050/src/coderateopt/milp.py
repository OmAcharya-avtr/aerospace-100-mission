"""Mixed-integer encoding of the rate-selection problem, solved with HiGHS.

``scipy.optimize.milp`` is a branch-and-bound mixed-integer linear programme
solved by HiGHS (Huangfu and Hall, "Parallelizing the dual revised simplex
method", *Mathematical Programming Computation* 10(1), 2018). It returns a
global optimum for a feasible, bounded MILP, subject to its own integrality
and feasibility tolerances; it is not a heuristic. Because every variable here
is bounded in [0, 1] and the feasible set is a bounded polytope intersected
with a cardinality constraint, the instance is always bounded, and the only
failure mode that matters is infeasibility, which is detected and reported
rather than papered over.

Variable vector, length ``2M``: ``[x_0 .. x_{M-1}, y_0 .. y_{M-1}]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from .lp import restricted_lp
from .problem import InfeasibleProblem, RateProblem, Solution

__all__ = ["MILP_OPTIONS", "MilpEncoding", "assert_feasible", "build_milp", "solve_milp"]

#: Options passed to every ``scipy.optimize.milp`` call in this package.
#:
#: ``mip_rel_gap=0.0`` is not decoration. ``scipy.optimize.milp`` leaves the
#: relative MIP gap at the HiGHS default of 1e-4 when the option is not set,
#: which means it stops as soon as it has a solution within 0.01 % of the
#: bound. On this problem that is enough to return the wrong *decision*: at
#: scintillation index 0.5281008782676444, margin 12 dB, target 0.99, K = 2,
#: long-run mode, the illustrative table, the default returns the single entry
#: ``ook-r1/2`` at goodput 0.495001474400 bits/symbol, while the true optimum
#: is the mix ``(ook-r1/2, ook-r2/3)`` at 0.495018082635 -- a relative
#: difference of 3.355e-05, comfortably inside the default gap. The goodput
#: error is negligible; the support change is not, because every sensitivity
#: boundary in :mod:`coderateopt.sensitivity` is found by watching the support
#: change. With the default gap those boundaries are noise. This was found by
#: the enumeration cross-check, which is the entire argument for having one.
#: Measured in ``validation/validate_mip_gap.py``.
MILP_OPTIONS: dict[str, float] = {"mip_rel_gap": 0.0}

#: Time fractions below this are treated as zero when reading a solver result.
SUPPORT_FLOOR = 1e-9


@dataclass(frozen=True)
class MilpEncoding:
    """The exact arrays handed to ``scipy.optimize.milp``, exposed for tests.

    Attributes
    ----------
    c
        Objective coefficients for minimisation, length ``2M``. The first
        ``M`` are ``-R_m A_m``; the ``y`` block is zero.
    constraint_matrix
        Dense constraint matrix, shape ``(n_rows, 2M)``.
    lower, upper
        Row bounds matching ``constraint_matrix``.
    integrality
        0 for ``x``, 1 for ``y``.
    variable_lower, variable_upper
        Variable bounds, length ``2M``.
    row_labels
        One label per row, so a failing constraint can be named.
    """

    c: np.ndarray
    constraint_matrix: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    integrality: np.ndarray
    variable_lower: np.ndarray
    variable_upper: np.ndarray
    row_labels: tuple[str, ...]


def assert_feasible(problem: RateProblem) -> None:
    """Raise :class:`InfeasibleProblem` when no assignment meets the target.

    The best long-run availability any assignment can reach is ``max_m A_m``,
    because ``sum_m A_m x_m`` over the simplex is maximised at a vertex and a
    vertex is a single MODCOD, which ``K >= 1`` always permits. The same bound
    applies in ``"per_interval"`` mode, where the constraint is at least as
    tight. So this one comparison decides feasibility exactly, in both modes,
    with no solver call.
    """
    avail = problem.availabilities()
    best = int(np.argmax(avail))
    if avail[best] + 1e-12 < problem.availability_target:
        shortfall = problem.availability_target - float(avail[best])
        raise InfeasibleProblem(
            f"availability target {problem.availability_target:.6f} is unreachable: the best "
            f"MODCOD in the table is {problem.modcods.names[best]!r} at availability "
            f"{float(avail[best]):.6f} (threshold {problem.modcods.thresholds_db[best]:.2f} dB), "
            f"short by {shortfall:.6f}. Raise the link margin, lower the target, or add a "
            f"lower-threshold MODCOD; the margin needed by "
            f"{problem.modcods.names[best]!r} at this target is "
            f"{_required_margin(problem, best):.2f} dB "
            f"({_required_margin(problem, best) - problem.margin_db:+.2f} dB from the current "
            f"{problem.margin_db:.2f} dB).",
            best_availability=float(avail[best]),
            best_modcod=problem.modcods.names[best],
            target=problem.availability_target,
            mode=problem.mode,
        )


def _required_margin(problem: RateProblem, index: int) -> float:
    """Margin in dB at which MODCOD ``index`` would just meet the target."""
    quantile = problem.fade.quantile_db(problem.availability_target)
    return float(problem.modcods.thresholds_db[index] - quantile)


def build_milp(problem: RateProblem) -> MilpEncoding:
    """Assemble the MILP for ``problem``.

    Rows, in order:

    * ``sum_m x_m = 1``
    * ``x_m - y_m <= 0`` for each ``m``
    * ``sum_m y_m <= K``
    * ``x_m - d y_m >= 0`` for each ``m`` (minimum dwell, only when ``d > 0``)
    * ``sum_m A_m x_m >= A_min`` (``"long_run"`` mode only)

    In ``"per_interval"`` mode the availability constraint is enforced by
    fixing ``x_m = y_m = 0`` for every ``m`` with ``A_m < A_min`` through the
    variable bounds, which is exactly the indicator implication
    ``x_m > 0 => A_m >= A_min`` with no big-M needed.
    """
    m = problem.n_modcods
    avail = problem.availabilities()
    rates = problem.modcods.rates
    allowed = problem.allowed_mask()

    c = np.concatenate([-rates * avail, np.zeros(m)])

    rows: list[np.ndarray] = []
    lower: list[float] = []
    upper: list[float] = []
    labels: list[str] = []

    row = np.zeros(2 * m)
    row[:m] = 1.0
    rows.append(row)
    lower.append(1.0)
    upper.append(1.0)
    labels.append("time_fractions_sum_to_one")

    for i in range(m):
        row = np.zeros(2 * m)
        row[i] = 1.0
        row[m + i] = -1.0
        rows.append(row)
        lower.append(-np.inf)
        upper.append(0.0)
        labels.append(f"link_x_le_y[{problem.modcods.names[i]}]")

    row = np.zeros(2 * m)
    row[m:] = 1.0
    rows.append(row)
    lower.append(-np.inf)
    upper.append(float(problem.effective_k))
    labels.append("table_cardinality")

    dwell = problem.min_dwell_fraction
    if dwell > 0.0:
        for i in range(m):
            row = np.zeros(2 * m)
            row[i] = 1.0
            row[m + i] = -dwell
            rows.append(row)
            lower.append(0.0)
            upper.append(np.inf)
            labels.append(f"min_dwell[{problem.modcods.names[i]}]")

    if problem.mode == "long_run":
        row = np.zeros(2 * m)
        row[:m] = avail
        rows.append(row)
        lower.append(float(problem.availability_target))
        upper.append(np.inf)
        labels.append("long_run_availability")

    var_lower = np.zeros(2 * m)
    var_upper = np.ones(2 * m)
    var_upper[:m] = allowed.astype(float)
    var_upper[m:] = allowed.astype(float)

    return MilpEncoding(
        c=c,
        constraint_matrix=np.vstack(rows),
        lower=np.asarray(lower, dtype=float),
        upper=np.asarray(upper, dtype=float),
        integrality=np.concatenate([np.zeros(m), np.ones(m)]),
        variable_lower=var_lower,
        variable_upper=var_upper,
        row_labels=tuple(labels),
    )


def solve_milp(problem: RateProblem, *, tolerance: float = 1e-9) -> Solution:
    """Solve ``problem`` with ``scipy.optimize.milp``.

    Returns a :class:`~coderateopt.problem.Solution` with ``canonical=False``:
    among tied optima HiGHS returns one, and which one is not part of its
    contract. Use :func:`coderateopt.select.select_rate` when the choice among
    ties must be reproducible.

    Uses :data:`MILP_OPTIONS`, which sets ``mip_rel_gap=0.0``. Read that
    constant's docstring before calling ``scipy.optimize.milp`` yourself: the
    default gap returns the wrong decision on instances of this problem.

    **The continuous part of the answer is recomputed, not taken from HiGHS.**
    Branch and bound decides the support, which is what the integer variables
    are for; the time fractions on that support are then re-solved exactly by
    :func:`coderateopt.lp.restricted_lp`. The reason is measured: HiGHS
    accepts a solution that violates the availability row within its primal
    feasibility tolerance, and on an instance where that row is tight the
    violation buys goodput that is not actually available. At margin 15.50 dB,
    scintillation index 0.05, target 0.990000, K = 2, long-run mode, the
    illustrative table, the raw HiGHS allocation achieved availability
    0.989999983417085 -- short of the target by 1.66e-08 -- and reported
    goodput 1.43572616018188 against the true optimum 1.435725437609653, a
    relative excess of 5.0e-07. The polish step removes it; the residual
    disagreement with enumeration after polishing is at the 1e-15 level
    (``validation/validate_milp_vs_exhaustive.py``).

    Raises
    ------
    InfeasibleProblem
        When the availability target cannot be met. Never returns a
        least-bad assignment: an infeasible instance has no answer, and
        returning the closest thing to one is how a tool gets quoted as
        saying a link closes when it does not.
    RuntimeError
        When the solver reports neither success nor infeasibility.
    """
    assert_feasible(problem)
    enc = build_milp(problem)
    result = milp(
        c=enc.c,
        constraints=LinearConstraint(enc.constraint_matrix, enc.lower, enc.upper),
        integrality=enc.integrality,
        bounds=Bounds(enc.variable_lower, enc.variable_upper),
        options=dict(MILP_OPTIONS),
    )
    if result.status == 2:  # pragma: no cover - guarded by assert_feasible
        raise InfeasibleProblem(
            f"HiGHS reported the MILP infeasible: {result.message}",
            best_availability=float(np.max(problem.availabilities())),
            best_modcod=problem.modcods.names[int(np.argmax(problem.availabilities()))],
            target=problem.availability_target,
            mode=problem.mode,
        )
    if result.status != 0 or result.x is None:
        raise RuntimeError(
            f"scipy.optimize.milp returned status {result.status}: {result.message}"
        )
    x = np.asarray(result.x[: problem.n_modcods], dtype=float)
    support = tuple(int(i) for i in np.flatnonzero(x > SUPPORT_FLOOR))
    polished = restricted_lp(problem, support)
    if polished is not None:
        x = polished[1]
    return _solution_from_x(problem, x, method="milp", canonical=False, tolerance=tolerance)


def _solution_from_x(
    problem: RateProblem,
    x: np.ndarray,
    *,
    method: str,
    canonical: bool,
    tolerance: float,
    tied_supports: tuple[tuple[int, ...], ...] = (),
) -> Solution:
    """Clean a raw time-fraction vector and wrap it in a :class:`Solution`."""
    x = np.where(np.abs(x) < 1e-10, 0.0, x)
    total = float(x.sum())
    if total <= 0.0:  # pragma: no cover - cannot happen for a feasible instance
        raise RuntimeError("solver returned an all-zero time-fraction vector")
    x = x / total
    avail = problem.availabilities()
    support = tuple(int(i) for i in np.flatnonzero(x > 0.0))
    return Solution(
        time_fractions=x,
        support=support,
        expected_goodput=float(np.dot(problem.modcods.rates * avail, x)),
        achieved_availability=float(np.dot(avail, x)),
        worst_interval_availability=float(np.min(avail[list(support)])),
        method=method,
        canonical=canonical,
        tied_supports=tied_supports,
        tolerance=tolerance,
    )
