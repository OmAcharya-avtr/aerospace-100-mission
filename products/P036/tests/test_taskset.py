"""Task-model and period/deadline arithmetic tests."""

from __future__ import annotations

import pytest

from rtclock.taskset import PeriodicTask, TaskSet, hyperperiod_s


def test_implicit_deadline_defaults_to_period():
    t = PeriodicTask("a", 0.01, 0.002)
    assert t.deadline_s == 0.01
    assert t.has_implicit_deadline is True
    assert t.utilization == pytest.approx(0.2, abs=1e-15)
    assert t.density == pytest.approx(0.2, abs=1e-15)


# Hand computation: C = 2 ms, T = 10 ms, D = 5 ms.
#   U = 2/10 = 0.2 ; density = 2/5 = 0.4
def test_constrained_deadline_density_hand_value():
    t = PeriodicTask("a", 0.01, 0.002, deadline_s=0.005)
    assert t.has_implicit_deadline is False
    assert t.utilization == pytest.approx(0.2, abs=1e-15)
    assert t.density == pytest.approx(0.4, abs=1e-15)


# Hand computation: ceil(window / T) releases in [0, window).
#   T = 7: window 0 -> 0; 1 -> 1; 7 -> 1; 7.1 -> 2; 14 -> 2; 20 -> 3
@pytest.mark.parametrize(
    ("window", "expected"), [(0.0, 0), (1.0, 1), (7.0, 1), (7.1, 2), (14.0, 2), (20.0, 3)]
)
def test_releases_in_hand_values(window, expected):
    assert PeriodicTask("a", 7.0, 1.0).releases_in(window) == expected


def test_releases_in_rejects_negative_window():
    with pytest.raises(ValueError, match="window_s must be >= 0"):
        PeriodicTask("a", 7.0, 1.0).releases_in(-1.0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"name": "", "period_s": 1.0, "wcet_s": 0.1}, "non-empty string"),
        ({"name": "a", "period_s": 0.0, "wcet_s": 0.1}, "period_s must be > 0"),
        ({"name": "a", "period_s": -1.0, "wcet_s": 0.1}, "period_s must be > 0"),
        ({"name": "a", "period_s": 1.0, "wcet_s": 0.0}, "wcet_s must be > 0"),
        ({"name": "a", "period_s": float("inf"), "wcet_s": 0.1}, "period_s must be finite"),
        ({"name": "a", "period_s": 1.0, "wcet_s": 0.1, "deadline_s": 2.0}, "arbitrary-deadline"),
        ({"name": "a", "period_s": 1.0, "wcet_s": 0.5, "deadline_s": 0.2}, "can never complete"),
        ({"name": "a", "period_s": 1.0, "wcet_s": 0.1, "deadline_s": 0.0}, "deadline_s must be"),
    ],
)
def test_periodic_task_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        PeriodicTask(**kwargs)


def test_periodic_task_rejects_non_numeric():
    with pytest.raises(TypeError, match="real number"):
        PeriodicTask("a", "1.0", 0.1)  # type: ignore[arg-type]


def test_task_set_rejects_duplicate_names_and_wrong_types():
    with pytest.raises(ValueError, match="duplicated"):
        TaskSet([PeriodicTask("a", 1.0, 0.1), PeriodicTask("a", 2.0, 0.1)])
    with pytest.raises(TypeError, match="must be PeriodicTask"):
        TaskSet(["a"])  # type: ignore[list-item]


def test_rate_monotonic_assigns_shortest_period_highest_priority():
    ts = TaskSet(
        [
            PeriodicTask("slow", 20.0, 1.0),
            PeriodicTask("fast", 5.0, 1.0),
            PeriodicTask("mid", 10.0, 1.0),
        ]
    ).rate_monotonic()
    names = [t.name for t in ts.by_priority()]
    assert names == ["fast", "mid", "slow"]
    assert [t.priority for t in ts.by_priority()] == [2, 1, 0]


def test_deadline_monotonic_differs_from_rate_monotonic_when_d_is_not_t():
    ts = TaskSet(
        [
            PeriodicTask("long_period_tight_deadline", 20.0, 1.0, deadline_s=2.0),
            PeriodicTask("short_period_loose_deadline", 5.0, 1.0, deadline_s=5.0),
        ]
    )
    assert [t.name for t in ts.rate_monotonic().by_priority()][0] == "short_period_loose_deadline"
    assert (
        [t.name for t in ts.deadline_monotonic().by_priority()][0]
        == "long_period_tight_deadline"
    )


def test_rate_monotonic_ties_broken_by_original_order():
    ts = TaskSet([PeriodicTask("first", 10.0, 1.0), PeriodicTask("second", 10.0, 1.0)])
    assert [t.name for t in ts.rate_monotonic().by_priority()] == ["first", "second"]


def test_priority_queries():
    ts = TaskSet(
        [PeriodicTask("a", 1.0, 0.1), PeriodicTask("b", 2.0, 0.1), PeriodicTask("c", 3.0, 0.1)]
    ).rate_monotonic()
    b = next(t for t in ts if t.name == "b")
    assert [t.name for t in ts.higher_priority_than(b)] == ["a"]
    assert [t.name for t in ts.lower_priority_than(b)] == ["c"]


def test_priority_queries_require_an_assignment():
    ts = TaskSet([PeriodicTask("a", 1.0, 0.1)])
    with pytest.raises(ValueError, match="have no priority"):
        ts.by_priority()
    with pytest.raises(ValueError, match="no priority assigned"):
        ts.higher_priority_than(ts[0])
    with pytest.raises(ValueError, match="no priority assigned"):
        ts.lower_priority_than(ts[0])


def test_task_set_sequence_protocol():
    ts = TaskSet([PeriodicTask("a", 1.0, 0.1), PeriodicTask("b", 2.0, 0.2)])
    assert len(ts) == 2
    assert ts[0].name == "a"
    assert [t.name for t in ts] == ["a", "b"]
    # Hand computation: 0.1/1 + 0.2/2 = 0.1 + 0.1 = 0.2
    assert ts.total_utilization == pytest.approx(0.2, abs=1e-15)


# Hand computation: lcm(7, 12, 20) on a 1 ns grid.
#   7 = 7, 12 = 2^2*3, 20 = 2^2*5 -> lcm = 2^2*3*5*7 = 420
def test_hyperperiod_hand_value():
    assert hyperperiod_s([7.0, 12.0, 20.0]) == pytest.approx(420.0, rel=1e-12)
    # Harmonic set: lcm(4, 8, 16) = 16
    assert hyperperiod_s([4.0, 8.0, 16.0]) == pytest.approx(16.0, rel=1e-12)
    # Milliseconds on a 1 ns grid: lcm(10 ms, 25 ms) = 50 ms
    assert hyperperiod_s([0.010, 0.025]) == pytest.approx(0.050, rel=1e-12)


def test_hyperperiod_validation():
    with pytest.raises(ValueError, match="must be non-empty"):
        hyperperiod_s([])
    with pytest.raises(ValueError, match="quantum_s must be > 0"):
        hyperperiod_s([1.0], quantum_s=0.0)
    with pytest.raises(ValueError, match="every period must be > 0"):
        hyperperiod_s([1.0, -1.0])
    with pytest.raises(ValueError, match="rounds to zero"):
        hyperperiod_s([1e-12], quantum_s=1e-9)
