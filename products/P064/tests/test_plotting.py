"""Plotting tests. Agg backend; no figure is displayed."""

from __future__ import annotations

from pathlib import Path

import pytest

matplotlib = pytest.importorskip("matplotlib")

from regressselect.benchmark import (  # noqa: E402
    evaluate_on_fault_matrix,
    evaluate_strategies,
)
from regressselect.mutation import FaultMatrix  # noqa: E402
from regressselect.plotting import (  # noqa: E402
    plot_fault_detection,
    plot_feature_importances,
    plot_safety_loss_vs_cost,
)


def test_backend_is_agg():
    assert matplotlib.get_backend().lower() == "agg"


def test_safety_loss_figure_is_written(tmp_path: Path, split, graph, selector):
    train, test = split
    scores = evaluate_strategies(train, test, graph, budgets=(0.25, 1.0), selector=selector)
    out = plot_safety_loss_vs_cost(scores, tmp_path / "a.png", title="t")
    assert out.is_file()
    assert out.stat().st_size > 5000


def test_fault_detection_figure_is_written(
    tmp_path: Path, graph, prioritiser, selector, product_root: Path
):
    import json

    raw = json.loads((product_root / "fixtures/faults/fault_matrix.json").read_text())
    matrix = FaultMatrix(
        fault_ids=tuple(raw["fault_ids"]),
        node_ids=tuple(raw["node_ids"]),
        detecting={k: tuple(v) for k, v in raw["detecting"].items()},
        changed_modules={k: tuple(v) for k, v in raw["changed_modules"].items()},
    )
    points = evaluate_on_fault_matrix(
        matrix, graph, prioritiser, selector, budgets=(0.25, 1.0), churn_lines=1
    )
    out = plot_fault_detection(points, tmp_path / "b.png")
    assert out.is_file()
    assert out.stat().st_size > 5000


def test_feature_importance_figure_is_written(tmp_path: Path, selector):
    out = plot_feature_importances(selector.feature_importances(), tmp_path / "c.png")
    assert out.is_file()
    assert out.stat().st_size > 5000


def test_nested_output_directory_is_created(tmp_path: Path, selector):
    out = plot_feature_importances(
        selector.feature_importances(), tmp_path / "deep" / "deeper" / "d.png"
    )
    assert out.is_file()
