"""Edge cases and input validation across the public surface."""

from __future__ import annotations

from pathlib import Path

import pytest

import regressselect
from regressselect.benchmark import (
    evaluate_on_fault_matrix,
    evaluate_strategies,
    fault_matrix_apfd,
)
from regressselect.history import History, Revision
from regressselect.importgraph import build_import_graph
from regressselect.mutation import FaultMatrix
from regressselect.prioritise import CoFailurePrioritiser
from regressselect.selector import LearnedSelector


def test_version_is_declared():
    assert regressselect.__version__ == "0.1.0"


def test_public_names_are_importable():
    for name in regressselect.__all__:
        assert hasattr(regressselect, name), name


def test_benchmark_rejects_an_evaluation_history_with_no_failure(split, graph, selector):
    train, _ = split
    green = History(
        test_ids=train.test_ids,
        revisions=tuple(
            Revision(f"g{i}", ("sampleproj.orbit",), {"sampleproj.orbit": 2}) for i in range(4)
        ),
    )
    with pytest.raises(ValueError, match="no revision with a failing test"):
        evaluate_strategies(train, green, graph, budgets=(1.0,), selector=selector)


def test_a_revision_touching_a_module_outside_the_graph_selects_nothing(graph, history):
    from regressselect.benchmark import dependency_order

    nodes = list(history.test_ids)
    revision = Revision("x", ("not_a_module",), {"not_a_module": 5})
    order, selected = dependency_order(graph, revision, nodes)
    assert selected == set()
    assert sorted(order) == sorted(nodes)


def test_fault_matrix_with_no_detectable_fault_rejects_apfd(graph, prioritiser, selector):
    matrix = FaultMatrix(
        fault_ids=("FA",),
        node_ids=("tests/test_orbit.py::test_period_matches_vis_viva",),
        detecting={"FA": ()},
        changed_modules={"FA": ("sampleproj.orbit",)},
    )
    assert matrix.full_suite_detection() == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ValueError, match="no seeded fault is detected"):
        fault_matrix_apfd(matrix, graph, prioritiser, selector)
    points = evaluate_on_fault_matrix(matrix, graph, prioritiser, selector, budgets=(1.0,))
    assert points["learned"][0].faults_detected_all == pytest.approx(0.0, abs=1e-12)
    assert points["learned"][0].faults_detected_detectable == pytest.approx(0.0, abs=1e-12)


def test_empty_fault_matrix_is_rejected():
    empty = FaultMatrix((), (), {}, {})
    with pytest.raises(ValueError, match="fault matrix is empty"):
        empty.full_suite_detection()


def test_single_module_project_graph(tmp_path: Path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "pkg" / "only.py").write_text("V = 1\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_only.py").write_text(
        "from pkg.only import V\n\n\ndef test_v():\n    assert V == 1\n", encoding="utf-8"
    )
    graph = build_import_graph(tmp_path)
    assert graph.select_tests(["pkg.only"]) == {"tests.test_only"}


def test_prioritiser_handles_a_test_absent_from_the_fitted_inventory(split):
    train, _ = split
    pri = CoFailurePrioritiser().fit(train)
    # A node the fit never saw still gets a finite base rate rather than a KeyError.
    rate = pri.score("tests/test_new.py::test_fresh", ["sampleproj.quat"])
    assert 0.0 < rate < 1.0


def test_selector_rank_accepts_an_explicit_node_subset(split, graph, selector):
    _, test = split
    subset = list(selector.node_ids)[:5]
    ranked = selector.rank(test.revisions[0], subset)
    assert sorted(rt.node_id for rt in ranked) == sorted(subset)


def test_selector_with_one_estimator_still_ranks(split, graph):
    from regressselect.selector import SelectorConfig

    train, test = split
    sel = LearnedSelector(config=SelectorConfig(n_estimators=1, max_depth=1)).fit(train, graph)
    ranked = sel.rank(test.revisions[0])
    assert len(ranked) == 24
