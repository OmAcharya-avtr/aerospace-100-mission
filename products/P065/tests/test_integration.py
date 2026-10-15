"""End-to-end: load, check, cross-link, render, in one pass."""

from __future__ import annotations

import json
import time

import matplotlib

matplotlib.use("Agg")

from hazardtrace import (  # noqa: E402
    format_json,
    format_markdown,
    format_report,
    load_analysis,
    load_junit,
    load_requirements,
    render_control_structure_png,
    render_mermaid,
    run_checks,
)


def test_worked_example_end_to_end(tmp_path, cases_dir, fixtures_dir):
    analysis = load_analysis(cases_dir / "p024_bdot_detumble.yaml")
    requirements = load_requirements(
        fixtures_dir / "p024_bdot" / "docs" / "REQUIREMENTS.md"
    )
    tests = load_junit(fixtures_dir / "p024_bdot" / "junit.xml")
    report = run_checks(analysis, requirements=requirements, tests=tests)

    assert len(report.findings) == 11
    assert report.exit_code == 1

    text = format_report(report)
    payload = json.loads(format_json(report))
    markdown = format_markdown(report)
    mermaid = render_mermaid(analysis, report=report)
    png = render_control_structure_png(report, tmp_path / "structure.png")

    assert len(payload["findings"]) == len(report.findings)
    for rendering in (text, markdown):
        assert "INCOMPLETE" in rendering
    assert "CA-DIPOLE" in mermaid
    assert png.exists()


def test_the_whole_pipeline_runs_well_inside_the_compute_budget(cases_dir, fixtures_dir):
    # The declared budget for this portfolio is 3 minutes per run on 2 cores.
    # The pipeline here is counting over tens of records plus one 305-case junit
    # parse, so the margin is several orders of magnitude; this test exists to
    # catch an accidental quadratic, not to measure performance.
    start = time.perf_counter()
    for _ in range(20):
        report = run_checks(
            load_analysis(cases_dir / "p024_bdot_detumble.yaml"),
            requirements=load_requirements(
                fixtures_dir / "p024_bdot" / "docs" / "REQUIREMENTS.md"
            ),
            tests=load_junit(fixtures_dir / "p024_bdot" / "junit.xml"),
        )
        assert report.exit_code == 1
    elapsed = time.perf_counter() - start
    assert elapsed < 30.0


def test_all_shipped_cases_render_without_error(tmp_path, cases_dir):
    for path in sorted(cases_dir.glob("*.yaml")):
        analysis = load_analysis(path)
        report = run_checks(analysis)
        assert format_report(report)
        assert json.loads(format_json(report))
        assert format_markdown(report)
        assert render_mermaid(analysis, report=report).startswith("flowchart TD")
        out = render_control_structure_png(report, tmp_path / f"{path.stem}.png")
        assert out.exists()


def test_cross_links_only_tighten_the_verdict(cases_dir, fixtures_dir):
    # Supplying a requirements document and a test report can only add findings,
    # never remove them: the two cross-link checks are the only ones they enable.
    analysis = load_analysis(cases_dir / "p024_bdot_detumble.yaml")
    without = run_checks(analysis)
    with_links = run_checks(
        analysis,
        requirements=load_requirements(
            fixtures_dir / "p024_bdot" / "docs" / "REQUIREMENTS.md"
        ),
        tests=load_junit(fixtures_dir / "p024_bdot" / "junit.xml"),
    )
    assert set(without.findings).issubset(set(with_links.findings))
    added = {f.code for f in with_links.findings} - {f.code for f in without.findings}
    assert added == {"HT012", "HT013"}
