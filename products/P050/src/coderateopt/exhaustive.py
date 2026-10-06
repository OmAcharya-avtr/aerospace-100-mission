"""Exhaustive enumeration: the independent check on the MILP encoding.

Three things live here, deliberately sharing as little machinery with
:mod:`coderateopt.milp` as possible:

``solve_exhaustive``
    Enumerate every MODCOD subset of size at most ``K``, solve the restricted
    linear programme on each, and keep the best. No integer variables, no
    linking constraints, no cardinality row -- the combinatorial part is done
    by ``itertools.combinations``. If this and the MILP disagree, the MILP
    encoding is wrong, and that is the only cheap way to find out.

``solve_closed_form_k2``
    For ``K <= 2`` the feasible set is a line segment and its vertices can be
    written down. This path calls **no solver at all**: it evaluates the
    objective at the enumerated vertices in NumPy. It is the third opinion,
    and the one that does not share HiGHS with the other two.

``canonical_solution``
    Among optimal subsets, pick the one whose sorted index tuple is
    lexicographically smallest, and report all the others as ties. Canonical
    MODCOD order (ascending threshold, then descending rate, then name) makes
    that choice reproducible, which the MILP alone does not promise.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from .lp import restricted_lp
from .milp import assert_feasible
from .problem import RateProblem, Solution

__all__ = [
    "canonical_solution",
    "count_subsets",
    "enumerate_subsets",
    "restricted_lp",
    "solve_closed_form_k2",
    "solve_exhaustive",
]


def count_subsets(n_modcods: int, max_entries: int) -> int:
    """Number of subsets enumerated: ``sum_{k=1..K} C(M, k)``."""
    from math import comb

    k = min(max_entries, n_modcods)
    return sum(comb(n_modcods, j) for j in range(1, k + 1))


def enumerate_subsets(problem: RateProblem) -> list[tuple[int, ...]]:
    """All allowed subsets of size 1..K, in lexicographic order."""
    allowed = np.flatnonzero(problem.allowed_mask())
    out: list[tuple[int, ...]] = []
    for size in range(1, problem.effective_k + 1):
        out.extend(combinations((int(i) for i in allowed), size))
    return out


def solve_exhaustive(problem: RateProblem, *, tolerance: float = 1e-9) -> Solution:
    """Enumerate subsets and return the canonical optimum.

    Cost is ``sum_{k=1..K} C(M, k)`` linear programmes. At ``M = 9, K = 2``
    that is 45; at ``M = 30, K = 3`` it is 4525; at ``M = 100, K = 5`` it is
    about 7.9 x 10**7, which is why the MILP path exists. The crossover
    measured in this repository is in ``validation/validate_milp_vs_exhaustive.py``.
    """
    assert_feasible(problem)
    results: list[tuple[tuple[int, ...], float, np.ndarray]] = []
    for subset in enumerate_subsets(problem):
        got = restricted_lp(problem, subset)
        if got is None:
            continue
        value, x = got
        results.append((subset, value, x))
    if not results:  # pragma: no cover - guarded by assert_feasible
        raise RuntimeError("enumeration found no feasible subset for a feasible problem")
    return canonical_solution(problem, results, method="exhaustive", tolerance=tolerance)


def canonical_solution(
    problem: RateProblem,
    results: list[tuple[tuple[int, ...], float, np.ndarray]],
    *,
    method: str,
    tolerance: float,
) -> Solution:
    """Pick the canonical optimum from enumerated ``(subset, value, x)`` triples.

    The support actually reported is the *realised* support ``{m : x_m > 0}``,
    not the enumerated subset, because an LP optimum on a size-2 subset may
    put zero weight on one of them. Ties are compared on the realised support,
    so a size-2 subset that collapses onto one entry is not reported as a
    distinct tied alternative to that entry.
    """
    from .milp import _solution_from_x

    best = max(v for _, v, _ in results)
    optimal: dict[tuple[int, ...], np.ndarray] = {}
    for _, value, x in results:
        if value + tolerance >= best:
            x = np.where(np.abs(x) < 1e-10, 0.0, x)
            support = tuple(int(i) for i in np.flatnonzero(x > 0.0))
            optimal.setdefault(support, x)
    chosen = min(optimal)
    ties = tuple(sorted(s for s in optimal if s != chosen))
    return _solution_from_x(
        problem,
        optimal[chosen],
        method=method,
        canonical=True,
        tolerance=tolerance,
        tied_supports=ties,
    )


def solve_closed_form_k2(problem: RateProblem, *, tolerance: float = 1e-9) -> Solution:
    """Solver-free optimum for ``K <= 2``, by enumerating the vertices.

    With ``sum x = 1`` the feasible set of a two-entry mix ``(i, j)`` is the
    segment ``x_i in [d, 1 - d]``, ``x_j = 1 - x_i``, cut by the long-run
    availability constraint ``A_i x_i + A_j (1 - x_i) >= A_min``, which is a
    half-line in ``x_i`` with its boundary at
    ``x_i = (A_min - A_j) / (A_i - A_j)``. A linear objective on an interval is
    optimised at one of its two endpoints, so the two endpoints of the
    intersected interval (plus the ``M`` single-entry points) are the complete
    candidate list. They are evaluated directly in NumPy with no call into any
    optimiser, which is why this function is an independent check rather than
    a reimplementation.

    This is a *complete* solver, not an approximation, whenever
    ``min_dwell_fraction == 0``: the module docstring's second structural
    result says the long-run optimum never needs more than two entries.

    Raises
    ------
    ValueError
        If ``problem.max_entries > 2``; this closed form does not generalise.
    """
    if problem.effective_k > 2:
        raise ValueError(
            f"solve_closed_form_k2 handles max_entries <= 2, got {problem.max_entries}"
        )
    assert_feasible(problem)
    avail = problem.availabilities()
    goodput = problem.modcods.rates * avail
    target = problem.availability_target
    long_run = problem.mode == "long_run"
    allowed = np.flatnonzero(problem.allowed_mask())
    n = problem.n_modcods

    results: list[tuple[tuple[int, ...], float, np.ndarray]] = []

    def record(weights: dict[int, float]) -> None:
        x = np.zeros(n)
        for i, w in weights.items():
            x[i] = w
        if np.any(x < -1e-12):
            return
        if long_run and float(np.dot(avail, x)) + 1e-12 < target:
            return
        results.append((tuple(sorted(weights)), float(np.dot(goodput, x)), x))

    for i in allowed:
        record({int(i): 1.0})
    dwell = problem.min_dwell_fraction
    if problem.effective_k >= 2 and 2.0 * dwell <= 1.0:
        for i, j in combinations((int(k) for k in allowed), 2):
            lo, hi = dwell, 1.0 - dwell
            if long_run and abs(avail[i] - avail[j]) > 1e-15:
                tight = (target - avail[j]) / (avail[i] - avail[j])
                if avail[i] > avail[j]:
                    lo = max(lo, tight)
                else:
                    hi = min(hi, tight)
            if lo > hi + 1e-12:
                continue
            for frac in (max(lo, 0.0), min(hi, 1.0)):
                record({i: float(frac), j: float(1.0 - frac)})
    if not results:  # pragma: no cover - guarded by assert_feasible
        raise RuntimeError("vertex enumeration found no feasible point")
    return canonical_solution(problem, results, method="closed_form_k2", tolerance=tolerance)
