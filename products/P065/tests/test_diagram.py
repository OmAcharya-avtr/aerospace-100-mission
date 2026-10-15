"""Diagram renderers: Mermaid text and the matplotlib PNG."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from hazardtrace import (  # noqa: E402
    load_analysis,
    parse_analysis,
    render_control_structure_png,
    render_mermaid,
    run_checks,
)


def test_mermaid_declares_every_component_and_path(cases_dir):
    analysis = load_analysis(cases_dir / "p024_bdot_detumble.yaml")
    text = render_mermaid(analysis)
    assert text.startswith("flowchart TD\n")
    for component in analysis.structure.components:
        assert component.id in text
    # Hand count: 6 components, 2 control actions (2 edges each) and 1 feedback
    # path (2 edges) gives 6 + 4 + 2 = 12 declaration lines plus the header.
    assert len(analysis.structure.components) == 6
    assert len(text.splitlines()) == 13


def test_mermaid_is_byte_identical_between_runs(cases_dir):
    analysis = load_analysis(cases_dir / "p024_bdot_detumble.yaml")
    assert render_mermaid(analysis) == render_mermaid(analysis)


def test_mermaid_marks_gap_actions_when_given_a_report(cases_dir):
    analysis = load_analysis(cases_dir / "p024_bdot_detumble.yaml")
    report = run_checks(analysis)
    text = render_mermaid(analysis, report=report)
    assert text.count("%% enumeration gap") == 2
    assert "%% enumeration gap" not in render_mermaid(analysis)


def test_mermaid_escapes_quotes_and_trims_long_labels():
    text = parse_analysis(
        "meta:\n  title: t\ncontrol_structure:\n  controllers:\n"
        '    - id: C\n      name: \'a "quoted" name that is very much longer than '
        "the label budget allows'\n"
    )
    rendered = render_mermaid(text, max_label_chars=20)
    assert '"' not in rendered.split("C[")[1].split("]")[0].replace('"', "", 2)
    assert "…" in rendered


def test_mermaid_on_an_empty_structure():
    rendered = render_mermaid(parse_analysis("meta:\n  title: t\n"))
    assert rendered == "flowchart TD\n"


def test_png_is_written(tmp_path, cases_dir):
    report = run_checks(load_analysis(cases_dir / "p024_bdot_detumble.yaml"))
    out = render_control_structure_png(report, tmp_path / "nested" / "structure.png")
    assert out.exists()
    assert out.stat().st_size > 10_000
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_png_handles_a_structure_with_no_feedback(tmp_path):
    report = run_checks(
        parse_analysis(
            "meta:\n  title: t\ncontrol_structure:\n  controllers:\n    - id: C\n"
            "      name: c\n  actuators:\n    - id: A\n      name: a\n"
            "  controlled_processes:\n    - id: P\n      name: p\n  control_actions:\n"
            "    - id: CA\n      name: ca\n      controller: C\n      actuator: A\n"
            "      process: P\n"
        )
    )
    out = render_control_structure_png(report, tmp_path / "open_loop.png")
    assert out.exists()


def test_png_handles_an_empty_structure(tmp_path):
    report = run_checks(parse_analysis("meta:\n  title: t\n"))
    out = render_control_structure_png(report, tmp_path / "empty.png")
    assert out.exists()
