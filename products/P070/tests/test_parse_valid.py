"""Parsing a well-formed specification."""

from __future__ import annotations

import pytest

from conftest import MINIMAL, PLANTED, WORKED
from watchdogspec import SPEC_VERSION, Action, load_spec, parse_spec

GOOD = """
spec_version: 1
system: "s"
supervisor:
  poll_period_ms: 10.0
ladders:
  - id: L
    rungs:
      - action: reset
        latency_ms: 1.0
tasks:
  - id: T
    heartbeat_period_ms: 20.0
    missed_beats_tolerated: 1
    liveness_deadline_ms: 100.0
    escalation: L
"""


def test_minimum_document_parses_with_documented_defaults():
    spec = parse_spec(GOOD)
    assert spec.spec_version == SPEC_VERSION
    assert spec.supervisor.poll_jitter_ms == 0.0
    assert spec.supervisor.declaration_overhead_ms == 0.0
    assert spec.supervisor.name == "supervisor"
    assert spec.tasks[0].heartbeat_jitter_ms == 0.0
    assert spec.tasks[0].jitter_sigma_ms == 0.0
    assert spec.tasks[0].false_trip_budget_per_hour is None
    assert spec.ladders[0].rungs[0].recovery_cost == 0.0
    assert spec.source == "<string>"


def test_source_is_recorded():
    assert parse_spec(GOOD, source="x.yaml").source == "x.yaml"


def test_integers_are_accepted_where_a_number_is_expected():
    document = GOOD.replace("poll_period_ms: 10.0", "poll_period_ms: 10")
    assert parse_spec(document).supervisor.poll_period_ms == pytest.approx(10.0, abs=1e-12)


@pytest.mark.parametrize("path", [MINIMAL, PLANTED, WORKED])
def test_every_shipped_specification_parses(path):
    spec = load_spec(path)
    assert spec.tasks
    assert spec.ladders
    assert spec.source == str(path)


def test_worked_example_contents(worked_spec):
    assert worked_spec.system == "small-satellite flight computer supervision"
    assert [task.id for task in worked_spec.tasks] == [
        "TASK-GNC",
        "TASK-IMU",
        "TASK-TLM",
        "TASK-FDIR",
    ]
    assert worked_spec.supervisor.poll_period_ms == pytest.approx(5.0, abs=1e-12)
    full = worked_spec.ladder("LADDER-FULL")
    assert full is not None
    assert [rung.action for rung in full.rungs] == [
        Action.LOG,
        Action.RESTART_TASK,
        Action.SAFE_MODE,
        Action.RESET,
    ]
    # 2 + 60 + 150 + 400 = 612 ms.
    assert full.total_latency_ms == pytest.approx(612.0, abs=1e-12)
    # 0.1 + 5 + 40 + 250 = 295.1 declared cost units.
    assert full.total_cost == pytest.approx(295.1, abs=1e-9)


def test_planted_fixture_has_twelve_tasks_and_four_ladders(planted_spec):
    assert len(planted_spec.tasks) == 13
    assert len(planted_spec.ladders) == 4


def test_every_action_keyword_parses():
    for action in Action:
        document = GOOD.replace("action: reset", f"action: {action.value}")
        assert parse_spec(document).ladders[0].rungs[0].action is action


def test_notes_and_descriptions_may_be_empty():
    document = GOOD + "notes: ''\n"
    assert parse_spec(document).notes == ""
