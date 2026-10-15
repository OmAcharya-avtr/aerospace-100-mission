"""Integration and regression tests for the three-strategy benchmark.

The numbers pinned here are the ones reported in README.md,
validation/VALIDATION.md and MODEL_CARD.md. They are pinned so that a change
which moves a published number fails the suite instead of silently
republishing a different one. Tolerances: 1e-6 for the two deterministic
baselines, 5e-3 for the learned selector, because the learned number depends
on the gradient-boosting implementation and scikit-learn is bounded to one
minor version for exactly that reason.
"""

from __future__ import annotations

import pytest

from regressselect.benchmark import (
    dependency_order,
    evaluate_on_fault_matrix,
    evaluate_strategies,
    fault_matrix_apfd,
)
from regressselect.mutation import FaultMatrix

MEASURED = {
    "dependency": {"apfd": 0.8361486, "cost": 0.4076577, "loss": 0.1720721},
    "history": {"apfd": 0.7956081},
    "learned": {"apfd": 0.8147523},
}


@pytest.fixture(scope="module")
def scores(split, graph, selector):
    train, test = split
    return evaluate_strategies(
        train, test, graph, budgets=(0.10, 0.25, 0.50, 0.75, 1.00), selector=selector
    )


def test_thirty_seven_evaluation_revisions_are_scored(scores):
    for score in scores.values():
        assert score.revisions_scored == 37


def test_dependency_baseline_apfd_and_cost_are_the_published_numbers(scores):
    dep = scores["dependency"]
    assert dep.apfd_full_order == pytest.approx(MEASURED["dependency"]["apfd"], abs=1e-6)
    point = dep.points[0]
    assert point.tests_run_fraction == pytest.approx(MEASURED["dependency"]["cost"], abs=1e-6)
    assert point.safety_loss == pytest.approx(MEASURED["dependency"]["loss"], abs=1e-6)


def test_history_baseline_apfd_is_the_published_number(scores):
    assert scores["history"].apfd_full_order == pytest.approx(
        MEASURED["history"]["apfd"], abs=1e-6
    )


def test_learned_selector_apfd_is_the_published_number(scores):
    assert scores["learned"].apfd_full_order == pytest.approx(
        MEASURED["learned"]["apfd"], abs=5e-3
    )


def test_the_dependency_baseline_wins_on_apfd(scores):
    # This is the headline research result and it is asserted so that a future
    # change which reverses it has to change this test deliberately.
    assert scores["dependency"].apfd_full_order > scores["learned"].apfd_full_order
    assert scores["learned"].apfd_full_order > scores["history"].apfd_full_order


def test_the_dependency_baseline_has_the_lowest_safety_loss_at_matched_cost(split, graph, selector):
    train, test = split
    matched = evaluate_strategies(train, test, graph, budgets=(0.4167,), selector=selector)
    dep = matched["dependency"].points[0]
    hist = matched["history"].points[0]
    learned = matched["learned"].points[0]
    assert dep.tests_run_fraction < hist.tests_run_fraction
    assert dep.safety_loss < learned.safety_loss
    assert learned.safety_loss < hist.safety_loss


def test_safety_loss_falls_to_zero_at_full_budget(scores):
    for name in ("history", "learned"):
        last = scores[name].points[-1]
        assert last.budget == pytest.approx(1.0)
        assert last.safety_loss == pytest.approx(0.0, abs=1e-12)
        assert last.regressions_caught == pytest.approx(1.0, abs=1e-12)


def test_dependency_safety_loss_is_not_zero_and_that_is_the_point(scores):
    # Baseline 1 is safe with respect to the import graph only. The fixture
    # history declares an out-of-closure coupling rate of 0.010 and a flake
    # rate of 0.003 per test per revision, so its measured loss is positive.
    assert scores["dependency"].points[0].safety_loss > 0.0


def test_safety_loss_is_non_increasing_in_the_budget(scores):
    for name in ("history", "learned"):
        losses = [p.safety_loss for p in scores[name].points]
        assert all(b <= a + 1e-12 for a, b in zip(losses, losses[1:], strict=False)), name


def test_budget_validation(split, graph, selector):
    train, test = split
    for bad in (0.0, -0.1, 1.2):
        with pytest.raises(ValueError, match="must lie in"):
            evaluate_strategies(train, test, graph, budgets=(bad,), selector=selector)


def test_dependency_order_places_the_selection_first(graph, history):
    nodes = list(history.test_ids)
    revision = next(r for r in history.revisions if r.changed_modules == ("sampleproj.orbit",))
    order, selected = dependency_order(graph, revision, nodes)
    assert sorted(order) == sorted(nodes)
    assert set(order[: len(selected)]) == selected


def test_fault_matrix_evaluation_reproduces_the_published_upper_bound(
    graph, prioritiser, selector, product_root
):
    import json

    raw = json.loads((product_root / "fixtures/faults/fault_matrix.json").read_text())
    matrix = FaultMatrix(
        fault_ids=tuple(raw["fault_ids"]),
        node_ids=tuple(raw["node_ids"]),
        detecting={k: tuple(v) for k, v in raw["detecting"].items()},
        changed_modules={k: tuple(v) for k, v in raw["changed_modules"].items()},
    )
    assert len(matrix.fault_ids) == 14
    assert len(matrix.detected_fault_ids) == 10
    assert matrix.full_suite_detection() == pytest.approx(10.0 / 14.0, abs=1e-12)
    points = evaluate_on_fault_matrix(matrix, graph, prioritiser, selector, churn_lines=1)
    dep = points["dependency"][0]
    # 87 selected nodes summed over 14 faults, out of 24 nodes each
    assert dep.tests_run_fraction == pytest.approx(87.0 / (24.0 * 14.0), abs=1e-6)
    assert dep.faults_detected_detectable == pytest.approx(1.0, abs=1e-12)
    assert dep.faults_detected_all == pytest.approx(10.0 / 14.0, abs=1e-12)
    apfds = fault_matrix_apfd(matrix, graph, prioritiser, selector, churn_lines=1)
    assert apfds["dependency"] > apfds["learned"] > apfds["history"]
