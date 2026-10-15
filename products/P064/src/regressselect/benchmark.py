"""Benchmark the three strategies on the same held-out revisions.

Three strategies, scored on the same evaluation revisions, with the same test
universe:

``dependency``
    Baseline 1. Selects every test that reaches a changed module in the import
    graph; the remaining tests are appended in node-id order so the strategy
    also yields a total order and can be given an APFD. Its *selection* is
    what matters, and its cost is reported with it.

``history``
    Baseline 2. The smoothed co-failure prioritiser.

``learned``
    The gradient-boosted selector.

Two numbers decide the question the product asks, and both are reported for
all three strategies at every budget:

``apfd``
    Average Percentage of Faults Detected (Rothermel, Untch, Chu and Harrold,
    IEEE TSE 27(10), 2001), averaged over the evaluation revisions that have
    at least one failing test. The detection matrix for a revision maps one
    fault -- "the regression introduced at this revision" -- to the set of
    tests that failed at it, so APFD here is the ``m = 1`` case and reduces to
    ``1 - TF/n + 1/(2n)`` for ``TF`` the rank of the first test that would have
    caught the regression. That is stated rather than left implicit because
    ``m = 1`` APFD and multi-fault APFD are not the same quantity.

``safety_loss``
    Fraction of the tests that actually failed which a budget-truncated
    selection would not have run, averaged over the same revisions.

Averaging is over regression revisions only. A revision with no failure has an
undefined safety loss and including it as a zero is how a selector comes to
look safe: at this history's 36.3 per cent regression rate, counting green
revisions as zero loss would divide every reported loss by roughly 2.8.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .history import History, Revision
from .importgraph import ImportGraph
from .inventory import module_of_node
from .metrics import apfd, apfd_single_fault, safety_loss, selection_cost
from .mutation import FaultMatrix
from .prioritise import CoFailurePrioritiser
from .selector import LearnedSelector

__all__ = [
    "StrategyScore",
    "BudgetPoint",
    "MutationPoint",
    "dependency_order",
    "evaluate_strategies",
    "evaluate_on_fault_matrix",
    "fault_matrix_apfd",
]


@dataclass(frozen=True)
class BudgetPoint:
    """One strategy scored at one budget.

    ``budget`` is the fraction of the suite the strategy is allowed to run
    (dimensionless, in (0, 1]). ``safety_loss`` and ``regressions_caught`` are
    dimensionless fractions; ``tests_run_fraction`` is the realised cost,
    which for the dependency baseline is set by the import graph and not by
    the budget. APFD is a property of the whole order and is therefore
    reported once per strategy, on :class:`StrategyScore`, not per budget.
    """

    strategy: str
    budget: float
    tests_run_fraction: float
    safety_loss: float
    regressions_caught: float
    revisions_scored: int


@dataclass(frozen=True)
class StrategyScore:
    """A strategy's full-order APFD and its per-budget points."""

    strategy: str
    apfd_full_order: float
    revisions_scored: int
    points: tuple[BudgetPoint, ...]


def dependency_order(
    graph: ImportGraph, revision: Revision, nodes: Sequence[str]
) -> tuple[list[str], set[str]]:
    """Baseline 1 order and selection for one revision.

    Returns ``(order, selected)``. ``selected`` is every node whose test module
    reaches a changed module; ``order`` places those first, in node-id order,
    then the rest. The within-group order is lexical because the dependency
    baseline carries no information with which to rank inside the selection,
    and inventing one would attribute the resulting APFD to the wrong thing.
    """
    changed = [m for m in revision.changed_modules if m in graph.edges]
    selected_modules = graph.select_tests(changed) if changed else set()
    selected = {n for n in nodes if module_of_node(n) in selected_modules}
    rest = [n for n in nodes if n not in selected]
    return sorted(selected) + sorted(rest), selected


def _truncate(order: Sequence[str], budget: float) -> list[str]:
    k = max(1, int(math.ceil(budget * len(order))))
    return list(order[:k])


def evaluate_strategies(
    train: History,
    test: History,
    graph: ImportGraph,
    budgets: Sequence[float] = (0.10, 0.25, 0.50, 0.75, 1.00),
    selector: LearnedSelector | None = None,
) -> dict[str, StrategyScore]:
    """Score all three strategies on ``test`` after fitting on ``train``.

    ``budgets`` are fractions of the suite. The dependency baseline ignores the
    budget by construction -- its selection is whatever the import graph says --
    so its ``tests_run_fraction`` is the measured cost at every budget and its
    reported safety loss at every budget is the same number. That is the point
    the README makes about it, and it is reported rather than hidden by leaving
    the baseline out of the budget sweep.
    """
    for b in budgets:
        if not 0.0 < b <= 1.0:
            raise ValueError(f"budget {b} must lie in (0, 1]")
    nodes = list(test.test_ids)
    prioritiser = CoFailurePrioritiser().fit(train)
    learned = selector if selector is not None else LearnedSelector().fit(train, graph)

    regression_revs = [r for r in test.revisions if r.is_regression]
    if not regression_revs:
        raise ValueError("the evaluation history contains no revision with a failing test")

    orders: dict[str, dict[str, list[str]]] = {"dependency": {}, "history": {}, "learned": {}}
    dep_selected: dict[str, set[str]] = {}
    for rev in regression_revs:
        order, selected = dependency_order(graph, rev, nodes)
        orders["dependency"][rev.revision] = order
        dep_selected[rev.revision] = selected
        orders["history"][rev.revision] = [
            rt.node_id for rt in prioritiser.rank(rev.changed_modules, nodes)
        ]
        orders["learned"][rev.revision] = [rt.node_id for rt in learned.rank(rev, nodes)]

    results: dict[str, StrategyScore] = {}
    for name in ("dependency", "history", "learned"):
        full_apfd = [
            apfd_single_fault(orders[name][r.revision], r.failed_tests) for r in regression_revs
        ]
        points: list[BudgetPoint] = []
        for budget in budgets:
            losses: list[float] = []
            costs: list[float] = []
            caught: list[float] = []
            for rev in regression_revs:
                order = orders[name][rev.revision]
                if name == "dependency":
                    chosen = sorted(dep_selected[rev.revision])
                else:
                    chosen = _truncate(order, budget)
                losses.append(safety_loss(chosen, rev.failed_tests))
                costs.append(selection_cost(chosen, nodes))
                caught.append(1.0 if set(chosen) & set(rev.failed_tests) else 0.0)
            points.append(
                BudgetPoint(
                    strategy=name,
                    budget=budget,
                    tests_run_fraction=sum(costs) / len(costs),
                    safety_loss=sum(losses) / len(losses),
                    regressions_caught=sum(caught) / len(caught),
                    revisions_scored=len(regression_revs),
                )
            )
        results[name] = StrategyScore(
            strategy=name,
            apfd_full_order=sum(full_apfd) / len(full_apfd),
            revisions_scored=len(regression_revs),
            points=tuple(points),
        )
    return results


@dataclass(frozen=True)
class MutationPoint:
    """One strategy scored at one budget against a measured fault matrix.

    ``faults_detected_all`` is the fraction of *every* seeded fault that the
    selection would catch; its ceiling is the full-suite detection rate, which
    is below 1 whenever the suite misses a seeded fault. ``faults_detected_detectable``
    divides instead by the faults the full suite does catch, so it reaches 1
    when the strategy loses nothing relative to running everything. Both are
    reported because quoting only the second is how a selection strategy comes
    to look perfect on a suite with holes in it.
    """

    strategy: str
    budget: float
    tests_run_fraction: float
    faults_detected_all: float
    faults_detected_detectable: float
    full_suite_upper_bound: float


def evaluate_on_fault_matrix(
    matrix: FaultMatrix,
    graph: ImportGraph,
    prioritiser: CoFailurePrioritiser,
    selector: LearnedSelector,
    budgets: Sequence[float] = (0.10, 0.25, 0.50, 0.75, 1.00),
    churn_lines: int = 1,
) -> dict[str, tuple[MutationPoint, ...]]:
    """Score the three strategies against a measured seeded-fault matrix.

    ``churn_lines`` is the churn, in lines, attributed to each seeded fault.
    Every fault in ``fixtures/faults/faults.json`` is a single-line textual
    substitution, so 1 is the truthful value; a larger value is a sensitivity
    check on the churn features and must be reported as such, because the
    learned selector leans on churn and a one-line change is the case where it
    has least to go on.
    """
    nodes = list(matrix.node_ids)
    upper = matrix.full_suite_detection()
    out: dict[str, tuple[MutationPoint, ...]] = {}
    detectable = set(matrix.detected_fault_ids)
    for name in ("dependency", "history", "learned"):
        points: list[MutationPoint] = []
        for budget in budgets:
            costs: list[float] = []
            hits_all = 0
            hits_detectable = 0
            for fault_id in matrix.fault_ids:
                changed = tuple(matrix.changed_modules[fault_id])
                rev = Revision(
                    revision=fault_id,
                    changed_modules=changed,
                    churn=dict.fromkeys(changed, churn_lines),
                )
                if name == "dependency":
                    _, selected = dependency_order(graph, rev, nodes)
                    chosen = sorted(selected)
                elif name == "history":
                    order = [rt.node_id for rt in prioritiser.rank(changed, nodes)]
                    chosen = _truncate(order, budget)
                else:
                    order = [rt.node_id for rt in selector.rank(rev, nodes)]
                    chosen = _truncate(order, budget)
                costs.append(selection_cost(chosen, nodes))
                found = bool(set(chosen) & set(matrix.detecting.get(fault_id, ())))
                hits_all += int(found)
                if fault_id in detectable:
                    hits_detectable += int(found)
            points.append(
                MutationPoint(
                    strategy=name,
                    budget=budget,
                    tests_run_fraction=sum(costs) / len(costs),
                    faults_detected_all=hits_all / len(matrix.fault_ids),
                    faults_detected_detectable=(
                        hits_detectable / len(detectable) if detectable else 0.0
                    ),
                    full_suite_upper_bound=upper,
                )
            )
        out[name] = tuple(points)
    return out


def fault_matrix_apfd(
    matrix: FaultMatrix,
    graph: ImportGraph,
    prioritiser: CoFailurePrioritiser,
    selector: LearnedSelector,
    churn_lines: int = 1,
) -> dict[str, float]:
    """Multi-fault APFD over the measured fault matrix, one order per strategy.

    The order is produced for a single synthetic change that touches every
    module carrying a seeded fault, so one order is scored against all the
    faults the suite can detect. Undetectable faults are excluded, because
    APFD is undefined for a fault with no detecting test; their count is in
    ``MutationPoint.full_suite_upper_bound``.
    """
    nodes = list(matrix.node_ids)
    detectable = {f: matrix.detecting[f] for f in matrix.detected_fault_ids}
    if not detectable:
        raise ValueError("no seeded fault is detected by the suite; APFD is undefined")
    changed = tuple(sorted({m for ms in matrix.changed_modules.values() for m in ms}))
    rev = Revision(
        revision="union", changed_modules=changed, churn=dict.fromkeys(changed, churn_lines)
    )
    dep_order, _ = dependency_order(graph, rev, nodes)
    orders = {
        "dependency": dep_order,
        "history": [rt.node_id for rt in prioritiser.rank(changed, nodes)],
        "learned": [rt.node_id for rt in selector.rank(rev, nodes)],
    }
    return {name: apfd(order, detectable) for name, order in orders.items()}
