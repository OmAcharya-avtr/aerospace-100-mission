"""The restricted linear programme, shared by the MILP and the enumeration.

Once the *set* of MODCODs is fixed, choosing the time fractions is a linear
programme in at most ``K`` variables with two rows. Both solver paths need it:
the enumeration solves one per subset, and the MILP solves one at the end to
place its chosen support exactly on the constraint. It lives in its own module
so neither imports the other.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from .problem import RateProblem

__all__ = ["restricted_lp"]


def restricted_lp(
    problem: RateProblem, subset: tuple[int, ...]
) -> tuple[float, np.ndarray] | None:
    """Best goodput using only ``subset``, or None when that subset is infeasible.

    The restricted problem is a plain LP over the simplex on ``subset``:
    maximise ``sum_{m in S} R_m A_m x_m`` subject to ``sum x = 1``,
    ``x_m >= d`` for ``m in S``, and, in ``"long_run"`` mode,
    ``sum_{m in S} A_m x_m >= A_min``. Fixing the subset is what turns the
    indicator implication ``x_m > 0 => x_m >= d`` into a plain lower bound,
    which is the whole reason enumeration needs no integer variables.

    Size-1 subsets are evaluated directly; larger ones go to
    ``scipy.optimize.linprog`` (HiGHS), with no integer variables anywhere.

    Returns ``(goodput, x)`` with ``x`` a full-length time-fraction vector.
    """
    avail = problem.availabilities()
    goodput = problem.modcods.rates * avail
    dwell = problem.min_dwell_fraction
    x = np.zeros(problem.n_modcods)
    if len(subset) * dwell > 1.0 + 1e-12:
        return None

    if len(subset) == 1:
        i = subset[0]
        if problem.mode == "long_run" and avail[i] + 1e-12 < problem.availability_target:
            return None
        x[i] = 1.0
        return float(goodput[i]), x

    idx = np.asarray(subset, dtype=int)
    c = -goodput[idx]
    a_eq = np.ones((1, idx.size))
    b_eq = np.array([1.0])
    if problem.mode == "long_run":
        a_ub = -avail[idx].reshape(1, -1)
        b_ub = np.array([-problem.availability_target])
    else:
        a_ub, b_ub = None, None
    res = linprog(
        c=c,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=[(dwell, 1.0)] * idx.size,
        method="highs",
    )
    if not res.success:
        return None
    x[idx] = np.asarray(res.x, dtype=float)
    return float(-res.fun), x
