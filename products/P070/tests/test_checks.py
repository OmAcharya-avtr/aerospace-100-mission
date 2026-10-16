"""Behaviour of the twelve checks, including the documented precedence."""

from __future__ import annotations

import pytest

from conftest import make_ladder, make_supervisor, make_task
from watchdogspec import (
    CHECKS,
    Action,
    Ladder,
    Rung,
    Severity,
    Spec,
    check_spec,
    run_checks,
)


def build(tasks, ladders=None, supervisor=None) -> Spec:
    """A one-off spec from the given tasks and ladders."""
    return Spec(
        system="unit",
        supervisor=supervisor or make_supervisor(),
        ladders=tuple(ladders or (make_ladder(2.0, 20.0, 50.0, 100.0),)),
        tasks=tuple(tasks),
    )


def codes(report) -> list[str]:
    return [finding.code for finding in report.findings]


def test_catalogue_has_twelve_checks_with_unique_codes():
    assert len(CHECKS) == 12
    assert sorted(CHECKS) == [f"WS{index:03d}" for index in range(1, 13)]


def test_check_spec_returns_the_catalogue_entry():
    assert check_spec("WS002").severity is Severity.ERROR
    assert "tolerated silence" in check_spec("WS002").summary


def test_check_spec_raises_on_an_unknown_code():
    with pytest.raises(KeyError):
        check_spec("WS999")


def test_clean_task_raises_nothing():
    report = run_checks(build([make_task(deadline=1000.0)]))
    assert codes(report) == []
    assert report.is_consistent is True
    assert report.exit_code == 0


def test_ws001_fires_when_detection_alone_misses():
    # W = 110, WCDL = 2 * 110 + 13 = 233 against D = 200.
    report = run_checks(build([make_task(period=100.0, jitter=10.0, k=1, deadline=200.0)]))
    assert codes(report) == ["WS001"]


def test_ws002_fires_and_suppresses_ws001():
    # W = 55, k*W = 220 >= D = 200. Detection also misses, but the specific
    # diagnosis is the tolerance and only that is reported.
    report = run_checks(build([make_task(period=50.0, jitter=5.0, k=4, deadline=200.0)]))
    assert codes(report) == ["WS002"]


def test_ws002_does_not_fire_when_the_tolerance_is_zero():
    # k = 0 tolerates no silence at all, so it cannot be what breaks a
    # deadline, however small the deadline is.
    report = run_checks(build([make_task(period=500.0, jitter=50.0, k=0, deadline=400.0)]))
    assert "WS002" not in codes(report)


def test_ws003_fires_when_detection_fits_but_the_ladder_does_not():
    report = run_checks(build([make_task(period=20.0, jitter=2.0, k=1, deadline=100.0)]))
    assert codes(report) == ["WS003"]


def test_ws003_is_suppressed_when_detection_already_missed():
    report = run_checks(build([make_task(period=100.0, jitter=10.0, k=1, deadline=200.0)]))
    assert "WS003" not in codes(report)


def test_ws004_fires_exactly_at_the_deadline():
    # WCDL = 57 and the ladder totals 172, so the completion is at 229 ms.
    report = run_checks(build([make_task(period=20.0, jitter=2.0, k=1, deadline=229.0)]))
    assert codes(report) == ["WS004"]
    assert report.is_consistent is True


def test_ws004_widens_with_the_tolerance():
    # A slack of 0.5 ms is met at the default tolerance and marginal at 1 ms.
    task = make_task(period=20.0, jitter=2.0, k=1, deadline=229.5)
    assert codes(run_checks(build([task]))) == []
    assert codes(run_checks(build([task]), tol_ms=1.0)) == ["WS004"]


def test_ws005_fires_only_when_a_budget_is_declared():
    noisy = {"period": 1000.0, "jitter": 200.0, "k": 0, "sigma": 80.0, "deadline": 5000.0}
    assert codes(run_checks(build([make_task(**noisy)]))) == []
    assert codes(run_checks(build([make_task(**noisy, budget=0.5)]))) == ["WS005"]


def test_ws005_does_not_fire_inside_the_budget():
    report = run_checks(
        build([make_task(period=1000.0, jitter=200.0, k=0, sigma=80.0,
                         deadline=5000.0, budget=1000.0)])
    )
    assert codes(report) == []


def test_ws006_fires_and_suppresses_ws001():
    report = run_checks(build([make_task(period=500.0, jitter=50.0, k=0, deadline=400.0)]))
    assert codes(report) == ["WS006"]


def test_ws007_fires_when_the_poll_is_coarser_than_the_window():
    report = run_checks(build([make_task(period=4.0, jitter=1.0, k=1, deadline=500.0)]))
    assert codes(report) == ["WS007"]


def test_ws008_fires_when_jitter_reaches_the_period():
    report = run_checks(build([make_task(period=10.0, jitter=10.0, k=0, deadline=1000.0)]))
    assert codes(report) == ["WS008"]


def test_ws009_fires_on_a_ladder_that_never_recovers():
    ladder = Ladder(
        id="L",
        rungs=(
            Rung(action=Action.LOG, latency_ms=1.0),
            Rung(action=Action.NOTIFY, latency_ms=2.0),
        ),
    )
    report = run_checks(build([make_task(deadline=1000.0)], ladders=[ladder]))
    assert codes(report) == ["WS009"]


def test_ws010_fires_once_per_ladder_on_the_first_inversion():
    ladder = Ladder(
        id="L",
        rungs=(
            Rung(action=Action.SAFE_MODE, latency_ms=1.0),
            Rung(action=Action.LOG, latency_ms=1.0),
            Rung(action=Action.RESET, latency_ms=1.0),
        ),
    )
    report = run_checks(build([make_task(deadline=1000.0)], ladders=[ladder]))
    assert codes(report).count("WS010") == 1


def test_ws011_fires_on_a_cost_inversion():
    ladder = Ladder(
        id="L",
        rungs=(
            Rung(action=Action.LOG, latency_ms=1.0, recovery_cost=9.0),
            Rung(action=Action.RESET, latency_ms=1.0, recovery_cost=1.0),
        ),
    )
    report = run_checks(build([make_task(deadline=1000.0)], ladders=[ladder]))
    assert codes(report) == ["WS011"]


def test_ws012_fires_and_the_task_is_not_timed():
    report = run_checks(build([make_task(escalation="absent")]))
    assert codes(report) == ["WS012"]
    assert report.timings == {}
    assert report.false_trips == {}


def test_exit_code_is_one_only_for_errors():
    warning_only = run_checks(build([make_task(period=4.0, jitter=1.0, k=1, deadline=500.0)]))
    assert warning_only.count_by_severity() == {"warning": 1}
    assert warning_only.exit_code == 0
    error = run_checks(build([make_task(period=100.0, jitter=10.0, k=1, deadline=200.0)]))
    assert error.exit_code == 1


def test_findings_are_sorted_by_subject_then_code(planted_report):
    subjects = [finding.subject for finding in planted_report.findings]
    assert subjects == sorted(subjects)


def test_report_counts_agree_with_the_finding_list(planted_report):
    assert sum(planted_report.count_by_code().values()) == len(planted_report.findings)
    assert sum(planted_report.count_by_severity().values()) == len(planted_report.findings)
