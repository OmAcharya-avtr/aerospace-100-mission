"""Shared fixtures. Every spec fixture is one of the shipped example files."""

from __future__ import annotations

from pathlib import Path

import pytest

from watchdogspec import Action, Ladder, Rung, Spec, Supervisor, Task, load_spec, run_checks

ROOT = Path(__file__).resolve().parent.parent
SPECS = ROOT / "examples" / "specs"

MINIMAL = SPECS / "minimal_consistent.yaml"
PLANTED = SPECS / "planted_defects.yaml"
WORKED = SPECS / "smallsat_fc.yaml"


@pytest.fixture
def minimal_spec() -> Spec:
    """The shipped internally consistent example."""
    return load_spec(MINIMAL)


@pytest.fixture
def planted_spec() -> Spec:
    """The shipped fixture carrying exactly one instance of each check."""
    return load_spec(PLANTED)


@pytest.fixture
def worked_spec() -> Spec:
    """The shipped worked example."""
    return load_spec(WORKED)


@pytest.fixture
def planted_report():
    """Checks run over the planted fixture."""
    return run_checks(load_spec(PLANTED))


@pytest.fixture
def worked_report():
    """Checks run over the worked example."""
    return run_checks(load_spec(WORKED))


def make_ladder(*latencies: float, ladder_id: str = "L") -> Ladder:
    """A ladder of the given rung latencies in ms, severities ascending."""
    actions = [Action.LOG, Action.RESTART_TASK, Action.SAFE_MODE, Action.RESET]
    rungs = tuple(
        Rung(action=actions[min(index, len(actions) - 1)], latency_ms=value, recovery_cost=index)
        for index, value in enumerate(latencies)
    )
    return Ladder(id=ladder_id, rungs=rungs)


def make_task(
    period: float = 20.0,
    jitter: float = 2.0,
    k: int = 1,
    deadline: float = 1000.0,
    sigma: float = 0.0,
    task_id: str = "T",
    escalation: str = "L",
    budget: float | None = None,
) -> Task:
    """A task with the given declared numbers, all in ms."""
    return Task(
        id=task_id,
        heartbeat_period_ms=period,
        missed_beats_tolerated=k,
        liveness_deadline_ms=deadline,
        escalation=escalation,
        heartbeat_jitter_ms=jitter,
        jitter_sigma_ms=sigma,
        false_trip_budget_per_hour=budget,
    )


def make_supervisor(poll: float = 10.0, jitter: float = 2.0, overhead: float = 1.0) -> Supervisor:
    """A supervisor with observation delay ``poll + jitter + overhead`` ms."""
    return Supervisor(poll_period_ms=poll, poll_jitter_ms=jitter, declaration_overhead_ms=overhead)
