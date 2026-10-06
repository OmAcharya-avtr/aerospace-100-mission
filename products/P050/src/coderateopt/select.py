"""The one entry point most callers want.

:func:`select_rate` solves the instance with the MILP and then, when the
instance is small enough for enumeration, canonicalises the choice among tied
optima so that repeated runs and different scipy versions give the same
answer. Sensitivity analysis depends on that: a boundary found by watching the
support change is meaningless if the support can change for free.
"""

from __future__ import annotations

from typing import Literal

from .exhaustive import count_subsets, solve_closed_form_k2, solve_exhaustive
from .milp import solve_milp
from .problem import RateProblem, Solution

__all__ = ["CANONICAL_SUBSET_BUDGET", "select_rate"]

#: Enumeration is used for canonicalisation only below this many subsets.
CANONICAL_SUBSET_BUDGET = 20_000

Method = Literal["milp", "exhaustive", "closed_form", "auto"]


def select_rate(
    problem: RateProblem,
    *,
    method: Method = "auto",
    canonicalise: bool = True,
    tolerance: float = 1e-9,
) -> Solution:
    """Solve ``problem`` and return the optimal MODCOD mix.

    Parameters
    ----------
    problem
        The instance.
    method
        ``"milp"`` uses ``scipy.optimize.milp`` only. ``"exhaustive"`` uses
        subset enumeration only. ``"closed_form"`` uses the solver-free vertex
        enumeration, which requires an effective cardinality limit of at most
        2. ``"auto"`` (default) solves with the MILP and canonicalises by
        enumeration when that is cheap.
    canonicalise
        With ``method="auto"``, whether to canonicalise ties. Turning it off
        saves the enumeration pass and returns ``canonical=False``.
    tolerance
        Absolute goodput tolerance for optimality and tie comparisons.

    Raises
    ------
    InfeasibleProblem
        When the availability target cannot be met by any assignment.
    """
    if method == "milp":
        return solve_milp(problem, tolerance=tolerance)
    if method == "exhaustive":
        return solve_exhaustive(problem, tolerance=tolerance)
    if method == "closed_form":
        return solve_closed_form_k2(problem, tolerance=tolerance)
    if method != "auto":
        raise ValueError(
            f"method must be 'milp', 'exhaustive', 'closed_form' or 'auto', got {method!r}"
        )

    milp_solution = solve_milp(problem, tolerance=tolerance)
    if not canonicalise:
        return milp_solution
    if count_subsets(problem.n_modcods, problem.effective_k) > CANONICAL_SUBSET_BUDGET:
        return milp_solution
    canonical = solve_exhaustive(problem, tolerance=tolerance)
    if abs(canonical.expected_goodput - milp_solution.expected_goodput) > max(
        tolerance, 1e-7 * abs(milp_solution.expected_goodput)
    ):
        raise RuntimeError(
            "MILP and enumeration disagree on the optimum: "
            f"milp={milp_solution.expected_goodput!r} "
            f"enumeration={canonical.expected_goodput!r}. This is a defect in the MILP "
            "encoding or in the enumeration, not a tolerance to be widened; please report it "
            "with the instance that produced it."
        )
    return canonical
