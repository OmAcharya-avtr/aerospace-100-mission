"""End to end: YAML on disk, through every stage, to a rendered report.

Also the cross-check the product exists for: the closed-form worst case
against the seeded discrete-event simulation, run over the shipped worked
example rather than over a synthetic task.
"""

from __future__ import annotations

import json

import pytest

from conftest import WORKED
from watchdogspec import (
    format_json,
    format_markdown,
    format_report,
    load_spec,
    render_mermaid,
    run_checks,
    simulate_detection,
    simulate_false_trips,
    spec_false_trip_rates,
    spec_timing,
    wilson_interval,
)
from watchdogspec.falsetrip import false_trip_probability


def test_full_pipeline_from_file_to_every_renderer():
    spec = load_spec(WORKED)
    report = run_checks(spec)
    text = format_report(report)
    payload = json.loads(format_json(report))
    markdown = format_markdown(report)
    mermaid = render_mermaid(spec, report)

    assert report.exit_code == 1
    assert len(report.timings) == 4
    for task_id in report.timings:
        assert task_id in text
        assert task_id in markdown
        assert task_id in mermaid
        assert task_id in payload["timings"]


def test_timing_from_spec_matches_timing_from_the_report():
    spec = load_spec(WORKED)
    report = run_checks(spec)
    standalone = spec_timing(spec)
    assert set(standalone) == set(report.timings)
    for task_id, timing in standalone.items():
        other = report.timings[task_id]
        assert timing.detection_latency_ms == pytest.approx(
            other.detection_latency_ms, rel=1e-15
        )
        assert timing.full_ladder_latency_ms == pytest.approx(
            other.full_ladder_latency_ms, rel=1e-15
        )


@pytest.mark.parametrize("task_id", ["TASK-GNC", "TASK-IMU", "TASK-TLM", "TASK-FDIR"])
def test_closed_form_bounds_the_simulation_on_every_worked_example_task(task_id):
    spec = load_spec(WORKED)
    report = run_checks(spec)
    task = spec.task(task_id)
    assert task is not None
    closed = report.timings[task_id].detection_latency_ms

    adversarial = simulate_detection(task, spec.supervisor, seed=0, trials=1, adversarial=True)
    assert adversarial.max_latency_from_receipt_ms == pytest.approx(closed, abs=1e-6)

    for seed in range(4):
        result = simulate_detection(task, spec.supervisor, seed=seed, trials=250)
        assert result.violations == 0
        assert result.max_latency_from_receipt_ms <= closed + 1e-9


def test_declared_false_trip_rate_matches_the_simulation_for_the_telemetry_task():
    spec = load_spec(WORKED)
    task = spec.task("TASK-TLM")
    assert task is not None
    closed = spec_false_trip_rates(spec)["TASK-TLM"].per_gap
    assert closed == pytest.approx(
        false_trip_probability(1000.0, 250.0, 90.0, 0), rel=1e-15
    )
    result = simulate_false_trips(task, seed=20261016, gaps=100_000)
    low, high = wilson_interval(result.false_trips, result.gaps)
    assert low <= closed <= high, f"{closed} outside {low}..{high}"


def test_the_two_shipped_findings_survive_the_whole_pipeline():
    report = run_checks(load_spec(WORKED))
    text = format_report(report)
    assert "WS003" in text
    assert "WS005" in text
    assert "TASK-FDIR" in text
    assert "TASK-TLM" in text
