"""The recorded history: loading, validation, windows and summaries.

The hand-checkable fixture is the ``tiny`` history from ``conftest.py``: two
tests over two sessions, in which test 0 fails once for 2000 ms and then passes
for 1000 ms in session 0 and passes for 1000 ms in session 1, and test 1 passes
for 500 ms in both. Every number below is checked against that by hand.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from flakerank.history import (
    SCHEMA,
    RunHistory,
    history_from_mapping,
    history_sha256,
    load_history,
)

from _support import HISTORY_PATH, tiny_history

# ---------------------------------------------------------------------------
# Known answers on the hand-checkable fixture
# ---------------------------------------------------------------------------


def test_tiny_shape(tiny: RunHistory) -> None:
    """Two tests, two sessions, five attempts."""
    assert (tiny.n_tests, tiny.n_sessions, tiny.n_attempts) == (2, 2, 5)
    assert tiny.retry_budget == 3


def test_tiny_summary_of_the_failing_test(tiny: RunHistory) -> None:
    """Test 0: 3 attempts, 1 failed, p = 1/3, wasted 2.0 s, total 4.0 s.

    Hand arithmetic: attempts 2000 + 1000 + 1000 = 4000 ms = 4.0 s; failed
    attempts 2000 ms = 2.0 s; p = 1 failure / 3 attempts = 0.3333333333333333;
    mean attempt = 4000/3 ms = 1.3333333333333333 s; mean failing = 2.0 s;
    sessions = 2; sessions with a failure = 1.
    """
    first = tiny.summaries()[0]
    assert first.name == "suite/test_0"
    assert (first.attempts, first.failed_attempts, first.sessions) == (3, 1, 2)
    assert first.failure_probability == pytest.approx(1.0 / 3.0, rel=1e-15, abs=0.0)
    assert first.wasted_seconds == pytest.approx(2.0, rel=0.0, abs=1e-12)
    assert first.total_seconds == pytest.approx(4.0, rel=0.0, abs=1e-12)
    assert first.mean_attempt_seconds == pytest.approx(4.0 / 3.0, rel=1e-15, abs=0.0)
    assert first.mean_failing_seconds == pytest.approx(2.0, rel=0.0, abs=1e-12)
    assert first.sessions_with_failure == 1
    assert first.max_attempt_seconds == pytest.approx(2.0, rel=0.0, abs=1e-12)
    assert first.min_attempt_seconds == pytest.approx(1.0, rel=0.0, abs=1e-12)


def test_tiny_summary_of_the_clean_test(tiny: RunHistory) -> None:
    """Test 1: 2 attempts, 0 failures, p = 0, wasted 0.0 s, 0.5 s each."""
    second = tiny.summaries()[1]
    assert (second.attempts, second.failed_attempts) == (2, 0)
    assert second.failure_probability == 0.0
    assert second.wasted_seconds == 0.0
    assert np.isnan(second.mean_failing_seconds)
    # With no observed failure there is no measured failing runtime, so the
    # mean over all attempts is substituted; see TestSummary.cost_runtime_seconds.
    assert second.cost_runtime_seconds == pytest.approx(0.5, rel=0.0, abs=1e-12)


def test_tiny_cost_runtime_uses_the_failing_mean_when_there_is_one(tiny: RunHistory) -> None:
    """For a test that did fail, the cost runtime is the FAILING-attempt mean."""
    first = tiny.summaries()[0]
    assert first.cost_runtime_seconds == pytest.approx(first.mean_failing_seconds,
                                                       rel=0.0, abs=0.0)
    assert first.cost_runtime_seconds != pytest.approx(first.mean_attempt_seconds,
                                                       rel=1e-6, abs=0.0)


def test_tiny_session_wasted_seconds(tiny: RunHistory) -> None:
    """Session 0 wasted 2.0 s and session 1 wasted nothing."""
    per_session = tiny.session_wasted_seconds()
    assert per_session.shape == (2,)
    assert per_session[0] == pytest.approx(2.0, rel=0.0, abs=1e-12)
    assert per_session[1] == 0.0


def test_tiny_session_test_table(tiny: RunHistory) -> None:
    """The (session, test) table puts the 2.0 s in exactly one cell."""
    table = tiny.session_test_wasted_seconds()
    assert table.shape == (2, 2)
    assert table[0, 0] == pytest.approx(2.0, rel=0.0, abs=1e-12)
    assert table.sum() == pytest.approx(2.0, rel=0.0, abs=1e-12)


def test_tiny_window(tiny: RunHistory) -> None:
    """Session window [1, 2) holds the two session-1 attempts only."""
    window = tiny.session_window(1, 2)
    assert window.n_attempts == 2
    assert window.summaries()[0].failed_attempts == 0


def test_restrict_tests_keeps_only_the_named_test(tiny: RunHistory) -> None:
    """Restricting to test 1 drops test 0's three attempts."""
    restricted = tiny.restrict_tests([1])
    assert restricted.n_attempts == 2
    assert [summary.name for summary in restricted.summaries()] == ["suite/test_1"]


def test_minimum_attempts_drops_thin_tests(tiny: RunHistory) -> None:
    """A threshold above a test's attempt count removes it from the summaries."""
    assert len(tiny.summaries(minimum_attempts=3)) == 1
    assert len(tiny.summaries(minimum_attempts=4)) == 0


# ---------------------------------------------------------------------------
# Input validation -- one test per rejection in the module docstring
# ---------------------------------------------------------------------------


def test_unknown_schema_rejected() -> None:
    """A loader that accepts an unknown schema will one day misread a file."""
    with pytest.raises(ValueError, match="unsupported history schema"):
        history_from_mapping(tiny_history(schema="flakerank.history/999"))


def test_missing_schema_rejected() -> None:
    """A missing schema is as bad as a wrong one."""
    payload = tiny_history()
    del payload["schema"]
    with pytest.raises(ValueError, match="unsupported history schema"):
        history_from_mapping(payload)


def test_wrong_duration_unit_rejected() -> None:
    """Durations are exact integer milliseconds; any other unit is rejected."""
    with pytest.raises(ValueError, match="duration_unit must be 'millisecond'"):
        history_from_mapping(tiny_history(duration_unit="second"))


@pytest.mark.parametrize("budget", [0, -1, 1.5, "3"])
def test_bad_retry_budget_rejected(budget: object) -> None:
    """The retry budget is an integer >= 1."""
    with pytest.raises(ValueError, match="retry_budget must be an integer >= 1"):
        history_from_mapping(tiny_history(retry_budget=budget))  # type: ignore[arg-type]


def test_unequal_attempt_array_lengths_rejected() -> None:
    """Five parallel arrays must be parallel."""
    payload = tiny_history()
    payload["attempts"]["duration_ms"] = payload["attempts"]["duration_ms"][:-1]
    with pytest.raises(ValueError, match="unequal lengths"):
        history_from_mapping(payload)


def test_empty_attempt_array_rejected() -> None:
    """A history with no attempts has nothing to rank."""
    payload = tiny_history()
    for name in ("session", "test", "attempt", "failed", "duration_ms"):
        payload["attempts"][name] = []
    with pytest.raises(ValueError, match="is empty"):
        history_from_mapping(payload)


def test_missing_attempt_column_rejected() -> None:
    """Every one of the five columns is required by name."""
    payload = tiny_history()
    del payload["attempts"]["failed"]
    with pytest.raises(ValueError, match="attempts.failed is missing"):
        history_from_mapping(payload)


def test_session_index_out_of_range_rejected() -> None:
    """An attempt cannot belong to a session the history does not have."""
    payload = tiny_history()
    payload["attempts"]["session"][0] = 7
    with pytest.raises(ValueError, match="attempts.session must index"):
        history_from_mapping(payload)


def test_test_index_out_of_range_rejected() -> None:
    """An attempt cannot belong to a test the history does not have."""
    payload = tiny_history()
    payload["attempts"]["test"][0] = 9
    with pytest.raises(ValueError, match="attempts.test must index"):
        history_from_mapping(payload)


def test_non_binary_failed_flag_rejected() -> None:
    """An outcome is pass or fail, so the flag is 0 or 1."""
    payload = tiny_history()
    payload["attempts"]["failed"][0] = 2
    with pytest.raises(ValueError, match="must contain only 0"):
        history_from_mapping(payload)


def test_negative_duration_rejected() -> None:
    """A duration is a non-negative number of milliseconds."""
    payload = tiny_history()
    payload["attempts"]["duration_ms"][0] = -5
    with pytest.raises(ValueError, match="duration_ms must be >= 0"):
        history_from_mapping(payload)


def test_fractional_duration_rejected() -> None:
    """A fractional millisecond would reintroduce representation error."""
    payload = tiny_history()
    payload["attempts"]["duration_ms"][0] = 1000.5
    with pytest.raises(ValueError, match="must contain integers only"):
        history_from_mapping(payload)


def test_attempt_number_not_one_based_rejected() -> None:
    """Attempt numbers are 1-based and consecutive within a group."""
    payload = tiny_history()
    payload["attempts"]["attempt"][0] = 0
    with pytest.raises(ValueError, match="must be 1-based"):
        history_from_mapping(payload)


def test_non_consecutive_attempt_number_rejected() -> None:
    """A gap in the attempt numbering means a lost record."""
    payload = tiny_history()
    payload["attempts"]["attempt"][1] = 3
    with pytest.raises(ValueError, match="where 2 was expected"):
        history_from_mapping(payload)


def test_attempt_after_a_pass_rejected() -> None:
    """Stop-at-first-pass forbids a further attempt after a passing one."""
    payload = tiny_history(
        attempts=[
            (0, 0, 1, 0, 1000),
            (0, 0, 2, 1, 2000),
        ]
    )
    with pytest.raises(ValueError, match="after a PASSING"):
        history_from_mapping(payload)


def test_group_longer_than_the_retry_budget_rejected() -> None:
    """A group of four attempts contradicts a budget of three."""
    payload = tiny_history(
        retry_budget=3,
        attempts=[
            (0, 0, 1, 1, 100),
            (0, 0, 2, 1, 100),
            (0, 0, 3, 1, 100),
            (0, 0, 4, 0, 100),
        ],
    )
    with pytest.raises(ValueError, match="above the declared retry_budget"):
        history_from_mapping(payload)


def test_duplicate_test_names_rejected() -> None:
    """Names identify tests, so they must be unique."""
    payload = tiny_history()
    payload["tests"][1]["name"] = payload["tests"][0]["name"]
    with pytest.raises(ValueError, match="test names must be unique"):
        history_from_mapping(payload)


def test_empty_tests_list_rejected() -> None:
    """A history with no tests is not a history."""
    payload = tiny_history()
    payload["tests"] = []
    with pytest.raises(ValueError, match="non-empty list of test objects"):
        history_from_mapping(payload)


def test_missing_history_file_message_names_the_generator() -> None:
    """The error tells the reader how to recreate the file."""
    with pytest.raises(FileNotFoundError, match="data/generate_history.py"):
        load_history(HISTORY_PATH.parent / "no_such_history.json")


def test_window_bounds_validated(tiny: RunHistory) -> None:
    """A window must be a non-empty range inside the history."""
    with pytest.raises(ValueError, match="session window must satisfy"):
        tiny.session_window(1, 1)
    with pytest.raises(ValueError, match="session window must satisfy"):
        tiny.session_window(0, 99)


def test_restrict_tests_validated(tiny: RunHistory) -> None:
    """An empty or out-of-range selection is rejected."""
    with pytest.raises(ValueError, match="at least one test"):
        tiny.restrict_tests([])
    with pytest.raises(ValueError, match="test index out of range"):
        tiny.restrict_tests([5])


def test_minimum_attempts_validated(tiny: RunHistory) -> None:
    """A threshold below 1 is meaningless."""
    with pytest.raises(ValueError, match="minimum_attempts must be >= 1"):
        tiny.summaries(minimum_attempts=0)


# ---------------------------------------------------------------------------
# The committed history itself
# ---------------------------------------------------------------------------


def test_committed_history_loads_and_validates(history: RunHistory) -> None:
    """The committed file passes every check the loader makes."""
    assert history.n_tests == 72
    assert history.n_sessions == 330
    assert history.retry_budget == 3
    assert history.n_attempts == 23836


def test_committed_history_schema_literal(history: RunHistory) -> None:
    """The schema string is the one the loader declares."""
    payload = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    assert payload["schema"] == SCHEMA


def test_committed_history_is_integer_milliseconds(history: RunHistory) -> None:
    """Durations are exact integers, so a sum carries no input error.

    The largest total is 2953413293 ms, far below 2**53, so every partial sum
    of these integers is exactly representable in double precision.
    """
    assert np.issubdtype(history.attempt_duration_ms.dtype, np.integer)
    assert int(history.attempt_duration_ms.sum()) == 2953413293
    assert int(history.attempt_duration_ms.sum()) < 2**53


def test_committed_history_failure_count(history: RunHistory) -> None:
    """1417 of 23836 recorded attempts failed: a pooled rate of 0.0594478939...

    Hand arithmetic: 1417 / 23836 = 0.059447893941936565.
    """
    failures = int(np.count_nonzero(history.attempt_failed))
    assert failures == 1417
    assert failures / history.n_attempts == pytest.approx(
        0.059447893941936565, rel=1e-15, abs=0.0
    )


def test_committed_history_pinned_case_present(history: RunHistory) -> None:
    """The 1-in-44 case the product was motivated by is in the history."""
    index = history.test_names.index("soak/test_validation_probe_1_in_44")
    assert history.true_failure_probability[index] == pytest.approx(
        1.0 / 44.0, rel=1e-15, abs=0.0
    )


def test_committed_history_digest_is_stable() -> None:
    """The digest of the committed file, pinned so a silent edit is caught.

    ``validation/validate_history.py`` pins the same value and is the
    authoritative check; this test makes a change to the data file fail the
    suite as well, because a change to the data changes every number in the
    documentation.
    """
    assert history_sha256(HISTORY_PATH) == (
        "aa12f8cc861ba66008a0d3762cd68fa24a24b013f897da67c8151996c531b18a"
    )


def test_committed_history_wasted_total(history: RunHistory) -> None:
    """Wasted time over the whole history: 253493.035 s from 1417 failures.

    Hand arithmetic: the generator transcript records 253493035 ms of
    failed-attempt time, which is 253493.035 s.
    """
    failed = history.attempt_failed.astype(bool)
    wasted_ms = int(history.attempt_duration_ms[failed].sum())
    assert wasted_ms == 253493035
    assert history.duration_seconds[failed].sum() == pytest.approx(
        253493.035, rel=0.0, abs=1e-6
    )
