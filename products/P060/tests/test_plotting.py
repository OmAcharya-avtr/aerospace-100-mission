"""Plotting smoke tests: the figures are produced and are non-empty files."""

from __future__ import annotations

import pytest

matplotlib = pytest.importorskip("matplotlib", reason="matplotlib is an optional extra")
matplotlib.use("Agg")

from traceaudit import audit, parse_junit_xml, parse_requirements_file  # noqa: E402
from traceaudit.plotting import (  # noqa: E402
    plot_confusion,
    plot_coverage_figures,
    plot_finding_breakdown,
    plot_trace_matrix,
)


@pytest.fixture
def result(sample_project):
    doc = parse_requirements_file(sample_project / "docs" / "REQUIREMENTS.md",
                                  relative_to=sample_project)
    report = parse_junit_xml(sample_project / "junit.xml", relative_to=sample_project)
    return audit(doc, report, test_root=sample_project / "tests")


@pytest.mark.verifies("REQ-010")
def test_trace_matrix_figure_is_written(result, tmp_path):
    out = plot_trace_matrix(result, tmp_path / "matrix.png")
    assert out.is_file()
    assert out.stat().st_size > 5000


@pytest.mark.verifies("REQ-011")
def test_coverage_figure_is_written(result, tmp_path):
    out = plot_coverage_figures({"sample": result}, tmp_path / "coverage.png")
    assert out.is_file()
    assert out.stat().st_size > 5000


@pytest.mark.verifies("REQ-013")
def test_finding_breakdown_figure_is_written(result, tmp_path):
    out = plot_finding_breakdown({"sample": result}, tmp_path / "findings.png")
    assert out.is_file()
    assert out.stat().st_size > 5000


@pytest.mark.verifies("REQ-018")
def test_confusion_figure_is_written(tmp_path):
    out = plot_confusion({"tp": 5, "fp": 3, "fn": 3, "tn": 12},
                         tmp_path / "confusion.png", subtitle="n = 23")
    assert out.is_file()
    assert out.stat().st_size > 5000


@pytest.mark.verifies("REQ-010")
def test_trace_matrix_handles_an_empty_result(tmp_path):
    from traceaudit import TestReport, TraceConfig, parse_requirements_text

    doc = parse_requirements_text("", source="s.md")
    report = TestReport(cases=(), source="j.xml", kind="junit-xml")
    empty = audit(doc, report, config=TraceConfig(heuristic_assertions=False))
    out = plot_trace_matrix(empty, tmp_path / "empty.png")
    assert out.is_file()
