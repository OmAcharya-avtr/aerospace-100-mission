"""Mermaid and PNG rendering."""

from __future__ import annotations

import pytest

from watchdogspec import render_escalation_png, render_mermaid, run_checks


def test_mermaid_names_every_task_and_ladder(worked_spec):
    text = render_mermaid(worked_spec)
    assert text.startswith("flowchart LR")
    for task in worked_spec.tasks:
        assert task.id in text
    for ladder in worked_spec.ladders:
        assert ladder.id in text


def test_mermaid_marks_tasks_with_an_error(worked_spec, worked_report):
    text = render_mermaid(worked_spec, worked_report)
    assert "TASK-FDIR (error)" in text
    assert "TASK-GNC (error)" not in text


def test_mermaid_is_deterministic(worked_spec):
    assert render_mermaid(worked_spec) == render_mermaid(worked_spec)


def test_mermaid_emits_one_escalate_edge_per_distinct_ladder(worked_spec):
    text = render_mermaid(worked_spec)
    assert text.count("-->|escalate|") == 2


def test_png_is_written_and_is_a_png(tmp_path, worked_spec, worked_report):
    target = tmp_path / "nested" / "escalation.png"
    written = render_escalation_png(worked_spec, worked_report, target, dpi=70)
    assert written == target
    assert target.exists()
    assert target.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_png_rejects_a_report_with_no_timed_task(minimal_spec):
    from watchdogspec import Spec

    broken = Spec(
        system=minimal_spec.system,
        supervisor=minimal_spec.supervisor,
        ladders=minimal_spec.ladders,
        tasks=tuple(
            type(task)(
                id=task.id,
                heartbeat_period_ms=task.heartbeat_period_ms,
                missed_beats_tolerated=task.missed_beats_tolerated,
                liveness_deadline_ms=task.liveness_deadline_ms,
                escalation="absent",
            )
            for task in minimal_spec.tasks
        ),
    )
    with pytest.raises(ValueError, match="no timed task"):
        render_escalation_png(broken, run_checks(broken), "unused.png")
