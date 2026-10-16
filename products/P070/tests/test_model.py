"""The declared data model: derived quantities and the action ordering."""

from __future__ import annotations

import pytest

from conftest import make_ladder, make_supervisor, make_task
from watchdogspec import Action, Ladder, Rung, Spec


def test_action_severity_order_is_strictly_increasing():
    order = [Action.LOG, Action.NOTIFY, Action.RESTART_TASK, Action.SAFE_MODE, Action.RESET]
    severities = [action.severity for action in order]
    assert severities == sorted(severities)
    assert len(set(severities)) == len(severities)


@pytest.mark.parametrize(
    ("action", "recovers"),
    [
        (Action.LOG, False),
        (Action.NOTIFY, False),
        (Action.RESTART_TASK, True),
        (Action.SAFE_MODE, True),
        (Action.RESET, True),
    ],
)
def test_only_the_three_state_changing_actions_recover(action, recovers):
    assert action.recovers is recovers


def test_beat_window_is_period_plus_jitter():
    # Hand calculation: P = 20, J = 2 ms, so W = 22 ms.
    task = make_task(period=20.0, jitter=2.0)
    assert task.beat_window_ms == pytest.approx(22.0, abs=1e-12)


def test_tolerated_silence_is_k_times_the_window():
    # Hand calculation: k = 3, W = 22 ms, so k*W = 66 ms.
    task = make_task(period=20.0, jitter=2.0, k=3)
    assert task.tolerated_silence_ms == pytest.approx(66.0, abs=1e-12)


def test_tolerated_silence_is_zero_when_no_beat_is_tolerated():
    assert make_task(k=0).tolerated_silence_ms == 0.0


def test_ladder_totals_and_cumulative_latency():
    # Hand calculation: rungs 2, 20, 50, 100 ms sum to 172 ms; the cumulative
    # series is 2, 22, 72, 172 ms. Costs 0, 1, 2, 3 sum to 6.
    ladder = make_ladder(2.0, 20.0, 50.0, 100.0)
    assert ladder.total_latency_ms == pytest.approx(172.0, abs=1e-12)
    assert ladder.total_cost == pytest.approx(6.0, abs=1e-12)
    cumulative = ladder.cumulative_latency_ms()
    assert len(cumulative) == 4
    for value, expected in zip(cumulative, (2.0, 22.0, 72.0, 172.0), strict=True):
        assert value == pytest.approx(expected, abs=1e-12)


def test_supervisor_observation_delay_is_the_sum_of_its_three_terms():
    # Hand calculation: Q + Jq + O = 10 + 2 + 1 = 13 ms.
    assert make_supervisor().observation_delay_ms == pytest.approx(13.0, abs=1e-12)


def test_spec_lookup_by_identifier():
    spec = Spec(
        system="s",
        supervisor=make_supervisor(),
        ladders=(make_ladder(1.0),),
        tasks=(make_task(),),
    )
    assert spec.ladder("L") is not None
    assert spec.ladder("absent") is None
    assert spec.task("T") is not None
    assert spec.task("absent") is None


def test_model_objects_are_frozen():
    rung = Rung(action=Action.LOG, latency_ms=1.0)
    with pytest.raises(AttributeError):
        rung.latency_ms = 2.0  # type: ignore[misc]


def test_ladder_with_one_rung_is_representable():
    ladder = Ladder(id="single", rungs=(Rung(action=Action.RESET, latency_ms=5.0),))
    assert ladder.total_latency_ms == pytest.approx(5.0, abs=1e-12)
    assert ladder.cumulative_latency_ms() == (5.0,)
