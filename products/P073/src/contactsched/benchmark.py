"""Comparison harnesses: method against method, predictor against baseline.

Nothing here decides anything. These functions run the methods that exist, score
them with the same code, and return rows. Keeping the scoring in one place is the
point: a greedy assignment and an exact assignment measured by two different
pieces of code would not be comparable, and a learned predictor measured by a
different function from its baseline would not be either.

Timings returned by :func:`compare_methods` are wall clock on a shared
two-core machine and are unstable at the tens-of-per-cent level. They belong in
a table marked unstable and never in a prose conclusion.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .assignment import Assignment, objective_value
from .exact import FairnessPolicy, solve_exact
from .fairness import FairnessReport, analyse
from .greedy import schedule_greedy
from .problem import SchedulingProblem, check_feasible


@dataclass(frozen=True)
class MethodResult:
    """One method's output on one problem, with its score and its cost.

    Attributes
    ----------
    assignment : Assignment
        What the method produced.
    report : FairnessReport
        Fairness accounting for that assignment.
    violations : tuple of str
        Feasibility violations, empty for a correct method. Measured rather than
        assumed: a method that returns an infeasible schedule should be caught
        by the harness that scores it, not trusted.
    elapsed_s : float
        Wall clock, unstable, table-only.
    """

    assignment: Assignment
    report: FairnessReport
    violations: tuple[str, ...]
    elapsed_s: float


def run_method(
    problem: SchedulingProblem,
    method: str,
    objective_mode: str = "weighted_contacts",
    policy: FairnessPolicy | None = None,
    time_limit_s: float = 60.0,
) -> MethodResult:
    """Run one named method and score it.

    Parameters
    ----------
    problem : SchedulingProblem
        The instance.
    method : str
        ``"greedy"``, ``"greedy-demand"``, ``"exact"`` or ``"exact-fair"``. The
        last enforces the declared minimum contact counts.
    objective_mode : str
        Scoring mode, as in :func:`contactsched.assignment.objective_value`.
    policy : FairnessPolicy or None
        Overrides the policy implied by ``method``.
    time_limit_s : float
        Solver limit, seconds.

    Raises
    ------
    ValueError
        On an unknown method name.
    """
    start = time.perf_counter()
    if method == "greedy":
        assignment = schedule_greedy(problem, objective_mode, "priority")
    elif method == "greedy-demand":
        assignment = schedule_greedy(problem, objective_mode, "demand")
    elif method == "exact":
        assignment = solve_exact(
            problem, objective_mode, policy or FairnessPolicy(), "highs", time_limit_s
        )
    elif method == "exact-fair":
        assignment = solve_exact(
            problem,
            objective_mode,
            policy or FairnessPolicy(enforce_min_contacts=True),
            "highs",
            time_limit_s,
        )
    else:
        raise ValueError(
            f"unknown method {method!r}; expected 'greedy', 'greedy-demand', "
            "'exact' or 'exact-fair'"
        )
    elapsed = time.perf_counter() - start
    return MethodResult(
        assignment=assignment,
        report=analyse(problem, assignment),
        violations=tuple(check_feasible(problem, assignment.selected)),
        elapsed_s=float(elapsed),
    )


def compare_methods(
    problem: SchedulingProblem,
    methods: tuple[str, ...] = ("greedy", "greedy-demand", "exact", "exact-fair"),
    objective_mode: str = "weighted_contacts",
    time_limit_s: float = 60.0,
) -> dict[str, MethodResult]:
    """Run several methods on one problem and return them keyed by name."""
    return {
        method: run_method(problem, method, objective_mode, None, time_limit_s)
        for method in methods
    }


def optimality_gap(greedy: Assignment, exact: Assignment) -> float:
    """Relative shortfall of ``greedy`` against ``exact``, dimensionless.

    ``(exact - greedy) / exact``, and 0.0 when the exact objective is zero. A
    negative value would mean the heuristic beat the proven optimum, which is
    impossible over the same candidate grid and is therefore a defect signal;
    callers assert on it rather than clipping it.
    """
    if exact.objective == 0.0:
        return 0.0
    return float((exact.objective - greedy.objective) / exact.objective)


def gap_distribution(
    problems: list[SchedulingProblem],
    objective_mode: str = "weighted_contacts",
    time_limit_s: float = 60.0,
) -> dict[str, float | list[float]]:
    """Distribution of the greedy optimality gap over an instance family.

    Returns
    -------
    dict
        ``gaps`` is the full list, in instance order, because a mean alone hides
        the shape and the specification for this batch asks for the
        distribution. The summary keys are ``n``, ``mean``, ``median``, ``p90``,
        ``max``, ``wins`` (instances where greedy matched the optimum) and
        ``win_fraction``.
    """
    gaps: list[float] = []
    for problem in problems:
        greedy = schedule_greedy(problem, objective_mode, "priority")
        exact = solve_exact(problem, objective_mode, FairnessPolicy(), "highs", time_limit_s)
        gaps.append(optimality_gap(greedy, exact))
    array = np.asarray(gaps, dtype=float)
    wins = int(np.sum(array <= 1e-12))
    return {
        "gaps": gaps,
        "n": float(len(gaps)),
        "mean": float(array.mean()) if len(array) else float("nan"),
        "median": float(np.median(array)) if len(array) else float("nan"),
        "p90": float(np.quantile(array, 0.9)) if len(array) else float("nan"),
        "max": float(array.max()) if len(array) else float("nan"),
        "wins": float(wins),
        "win_fraction": float(wins / len(array)) if len(array) else float("nan"),
    }


def fairness_cost(
    unconstrained: MethodResult, constrained: MethodResult, objective_mode: str
) -> dict[str, float]:
    """What the declared fairness constraints cost, in objective and in equality.

    This is the number the product exists to put on the table. A fairness
    constraint is not free: forcing a minimum contact count for a starved
    request takes antenna time from somewhere, and the objective falls. Reporting
    the new schedule without the price is the thing this function refuses to do.

    Returns
    -------
    dict
        ``objective_unconstrained``, ``objective_constrained``,
        ``objective_loss``, ``objective_loss_fraction``, ``jain_gain``,
        ``max_min_gain``, ``starved_before``, ``starved_after``,
        ``seconds_short_before``, ``seconds_short_after``. ``objective_loss`` is
        ``nan`` when the constrained solve was infeasible, because there is no
        schedule to price.
    """
    del objective_mode
    if constrained.assignment.status == "infeasible":
        return {
            "objective_unconstrained": float(unconstrained.assignment.objective),
            "objective_constrained": float("nan"),
            "objective_loss": float("nan"),
            "objective_loss_fraction": float("nan"),
            "jain_gain": float("nan"),
            "max_min_gain": float("nan"),
            "starved_before": float(len(unconstrained.report.starved)),
            "starved_after": float("nan"),
            "seconds_short_before": float(unconstrained.report.total_seconds_short),
            "seconds_short_after": float("nan"),
        }
    loss = unconstrained.assignment.objective - constrained.assignment.objective
    return {
        "objective_unconstrained": float(unconstrained.assignment.objective),
        "objective_constrained": float(constrained.assignment.objective),
        "objective_loss": float(loss),
        "objective_loss_fraction": float(
            loss / unconstrained.assignment.objective
            if unconstrained.assignment.objective
            else 0.0
        ),
        "jain_gain": float(constrained.report.jain - unconstrained.report.jain),
        "max_min_gain": float(constrained.report.max_min - unconstrained.report.max_min),
        "starved_before": float(len(unconstrained.report.starved)),
        "starved_after": float(len(constrained.report.starved)),
        "seconds_short_before": float(unconstrained.report.total_seconds_short),
        "seconds_short_after": float(constrained.report.total_seconds_short),
    }


def score_predictor(
    name: str,
    y_true: np.ndarray,
    scores: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    calibrated: bool,
    bins: int = 10,
) -> dict[str, float | str | bool]:
    """One row of the predictor comparison table.

    ``scores`` drives the rank statistics and ``probabilities`` the calibration
    statistics; for most predictors they are the same array, and for
    :class:`contactsched.predictors.RatioRankingBaseline` they are not, which is
    why they are separate arguments.
    """
    from .predictors import (
        average_precision,
        brier_score,
        expected_calibration_error,
        roc_auc,
        threshold_metrics,
    )

    row: dict[str, float | str | bool] = {
        "predictor": name,
        "calibrated_claim": bool(calibrated),
        "roc_auc": roc_auc(y_true, scores),
        "average_precision": average_precision(y_true, scores),
        "brier": brier_score(y_true, probabilities),
        "ece": expected_calibration_error(y_true, probabilities, bins),
        "ece_bins": float(bins),
    }
    row.update(threshold_metrics(y_true, probabilities, threshold))
    return row


def objective_of(problem: SchedulingProblem, assignment: Assignment) -> float:
    """Re-score an assignment with the harness's own code, for cross-checking.

    A method reports its own objective. This recomputes it from the problem and
    the selected indices, so that a method that mis-reports its score is caught.
    ``tests/test_benchmark.py`` asserts the two agree for every method.
    """
    return objective_value(problem, assignment.selected, assignment.objective_mode)
