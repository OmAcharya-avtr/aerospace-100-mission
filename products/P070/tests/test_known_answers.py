"""Known answers over the two shipped fixtures.

The planted fixture is built to carry exactly one instance of every check, so
the expected mapping of code to subject is a literal written out below and not
derived from the code under test. Every timing number in this file is
hand-calculated and the calculation appears in the comment above it; the
supervisor observation delay is Q + Jq + O = 10 + 2 + 1 = 13 ms throughout the
planted fixture and 5 + 1 + 0.5 = 6.5 ms throughout the worked example.
"""

from __future__ import annotations

import pytest

from watchdogspec import Margin, Severity

#: code -> subject, written by hand from examples/specs/planted_defects.yaml.
EXPECTED_PLANTED: dict[str, str] = {
    "WS001": "T-WS001",
    "WS002": "T-WS002",
    "WS003": "T-WS003",
    "WS004": "T-WS004",
    "WS005": "T-WS005",
    "WS006": "T-WS006",
    "WS007": "T-WS007",
    "WS008": "T-WS008",
    "WS009": "L-NORECOVER",
    "WS010": "L-DISORDER",
    "WS011": "L-COSTDROP",
    "WS012": "T-WS012",
}


def test_planted_fixture_raises_exactly_one_finding_per_check(planted_report):
    assert len(planted_report.findings) == 12
    assert planted_report.count_by_code() == dict.fromkeys(EXPECTED_PLANTED, 1)


def test_planted_fixture_names_the_expected_subject_for_each_code(planted_report):
    actual = {finding.code: finding.subject for finding in planted_report.findings}
    assert actual == EXPECTED_PLANTED


def test_planted_fixture_severity_split(planted_report):
    # Errors: WS001, WS002, WS003, WS006, WS008, WS009, WS012 -> 7.
    # Warnings: WS004, WS005, WS007, WS010, WS011 -> 5.
    assert planted_report.count_by_severity() == {"error": 7, "warning": 5}
    assert planted_report.exit_code == 1
    assert planted_report.is_consistent is False


def test_planted_clean_task_has_no_finding(planted_report):
    assert all(finding.subject != "T-CLEAN" for finding in planted_report.findings)


def test_planted_clean_task_timing_known_answer(planted_report):
    # T-CLEAN: W = 22, WCDL = 3 * 22 + 13 = 79, ladder 79 + 172 = 251 ms,
    # D = 1000 ms, slack 749 ms, all four rungs reachable.
    timing = planted_report.timings["T-CLEAN"]
    assert timing.beat_window_ms == pytest.approx(22.0, abs=1e-9)
    assert timing.detection_latency_ms == pytest.approx(79.0, abs=1e-9)
    assert timing.full_ladder_latency_ms == pytest.approx(251.0, abs=1e-9)
    assert timing.ladder_slack_ms == pytest.approx(749.0, abs=1e-9)
    assert timing.deepest_rung_reachable == 4


def test_planted_boundary_task_is_marginal_not_met_or_missed(planted_report):
    # T-WS004: D is exactly the worst-case completion time, 229 ms. The slack
    # is 0.0 and the verdict is deliberately not a boolean.
    timing = planted_report.timings["T-WS004"]
    assert timing.full_ladder_latency_ms == pytest.approx(229.0, abs=1e-9)
    assert abs(timing.ladder_slack_ms) <= planted_report.tol_ms
    assert timing.ladder_margin is Margin.MARGINAL
    assert timing.detection_margin is Margin.MET


def test_planted_tolerance_exceeds_deadline_is_detected(planted_report):
    # The defect the specification asks for by name: T-WS002 tolerates four
    # missed beats of a 55 ms window, a silence of 220 ms, against a deadline
    # of 200 ms that the tolerance is declared to protect.
    finding = next(f for f in planted_report.findings if f.code == "WS002")
    assert finding.subject == "T-WS002"
    assert finding.severity is Severity.ERROR
    assert "tolerance exceeds the deadline it is declared to protect" in finding.message
    timing = planted_report.timings["T-WS002"]
    assert timing.tolerated_silence_ms == pytest.approx(220.0, abs=1e-9)
    assert timing.deadline_ms == pytest.approx(200.0, abs=1e-9)
    assert timing.tolerance_slack_ms == pytest.approx(-20.0, abs=1e-9)
    assert timing.tolerance_margin is Margin.MISSED


def test_worked_example_findings(worked_report):
    # Published with its two findings rather than tuned until silent.
    assert {(f.code, f.subject) for f in worked_report.findings} == {
        ("WS003", "TASK-FDIR"),
        ("WS005", "TASK-TLM"),
    }
    assert worked_report.count_by_severity() == {"error": 1, "warning": 1}
    assert worked_report.exit_code == 1


def test_worked_example_gnc_timing_known_answer(worked_report):
    # TASK-GNC: W = 23, WCDL = 3 * 23 + 6.5 = 75.5 ms; LADDER-FULL totals
    # 2 + 60 + 150 + 400 = 612 ms, so completion is at 687.5 ms against
    # D = 800 ms, a slack of 112.5 ms, and all four rungs are reachable.
    timing = worked_report.timings["TASK-GNC"]
    assert timing.beat_window_ms == pytest.approx(23.0, abs=1e-9)
    assert timing.detection_latency_ms == pytest.approx(75.5, abs=1e-9)
    assert timing.full_ladder_latency_ms == pytest.approx(687.5, abs=1e-9)
    assert timing.ladder_slack_ms == pytest.approx(112.5, abs=1e-9)
    assert timing.deepest_rung_reachable == 4
    assert timing.total_cost == pytest.approx(295.1, abs=1e-9)


def test_worked_example_fdir_reaches_only_its_first_rung(worked_report):
    # TASK-FDIR: W = 115, WCDL = 4 * 115 + 6.5 = 466.5 ms against D = 500 ms,
    # so detection fits with 33.5 ms to spare. Rung 1 finishes at 468.5 ms and
    # fits; rung 2 finishes at 528.5 ms and does not. Deepest reachable is 1.
    timing = worked_report.timings["TASK-FDIR"]
    assert timing.detection_latency_ms == pytest.approx(466.5, abs=1e-9)
    assert timing.detection_slack_ms == pytest.approx(33.5, abs=1e-9)
    assert timing.rungs[0].finish_ms == pytest.approx(468.5, abs=1e-9)
    assert timing.rungs[1].finish_ms == pytest.approx(528.5, abs=1e-9)
    assert timing.deepest_rung_reachable == 1
    assert timing.full_ladder_latency_ms == pytest.approx(1078.5, abs=1e-9)


def test_worked_example_imu_timing_known_answer(worked_report):
    # TASK-IMU: W = 6, WCDL = 2 * 6 + 6.5 = 18.5 ms; LADDER-FAST totals
    # 1 + 25 + 90 = 116 ms, so completion is at 134.5 ms against D = 300 ms.
    timing = worked_report.timings["TASK-IMU"]
    assert timing.detection_latency_ms == pytest.approx(18.5, abs=1e-9)
    assert timing.full_ladder_latency_ms == pytest.approx(134.5, abs=1e-9)
    assert timing.deepest_rung_reachable == 3


def test_minimal_example_is_consistent(minimal_spec):
    from watchdogspec import run_checks

    report = run_checks(minimal_spec)
    assert report.findings == ()
    assert report.exit_code == 0
    assert report.is_consistent is True
    # TASK-CTRL: W = 23, WCDL = 3 * 23 + 6.5 = 75.5 ms, ladder 462 ms,
    # completion 537.5 ms against D = 800 ms.
    timing = report.timings["TASK-CTRL"]
    assert timing.detection_latency_ms == pytest.approx(75.5, abs=1e-9)
    assert timing.full_ladder_latency_ms == pytest.approx(537.5, abs=1e-9)
