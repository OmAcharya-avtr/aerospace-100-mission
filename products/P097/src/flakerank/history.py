"""The recorded run history: schema, loader, validation and per-test summaries.

Everything this product asserts a number about comes from a *recorded* history
on disk, generated once by ``data/generate_history.py`` and committed. Nothing
here times anything live.

That is a design constraint rather than a convenience. This product is about
intermittency, and a product about intermittency whose own checks are
intermittent has refuted itself. The mission this product was built in lost a
whole product to checks that compared two wall-clock readings on a shared
two-core runner, and five intermittent failures plus a sixth caught in flight
all traced to that one class. So:

* no function in this package calls :func:`time.perf_counter` or any other
  clock;
* every duration is read from the committed history as an **integer number of
  milliseconds**, which is exactly representable in double precision up to
  2**53 ms, so a sum of durations carries no representation error from the
  input and only the rounding of the accumulation itself;
* every test and every validation script asserts on those recorded integers.

THE SCHEMA
----------

A history is a JSON object with a columnar attempt table, because a row-per-
attempt layout for 24000 attempts does not fit the mission's 1 MB limit on
committed data.

``schema``
    The literal ``"flakerank.history/1"``. A loader that accepts an unknown
    schema string silently is a loader that will one day read a different file
    as though it were this one.
``retry_budget``
    Maximum attempts per invocation, ``m >= 1``. The recorded history obeys it
    and :func:`load_history` checks that it does.
``duration_unit``
    Documentation only; the loader requires it to be ``"millisecond"``.
``tests``
    One object per test: ``name``, ``suite``, and the generator's ground-truth
    parameters ``true_failure_probability`` and ``true_mean_ms``. The
    ground-truth fields are the *generator's* parameters, not measurements, and
    exist so that negative controls can be anchored on a true value rather than
    on an output. Nothing in :mod:`flakerank.baselines` or
    :mod:`flakerank.predictor` reads them.
``sessions``
    One object per CI session: ``index`` and ``day``. Sessions are ordered in
    time by ``index``, which is the unit the held-out TIME split cuts on.
``attempts``
    Five equal-length integer arrays: ``session``, ``test``, ``attempt``
    (1-based within a session for that test), ``failed`` (0 or 1) and
    ``duration_ms``.

WHAT THE LOADER REJECTS
-----------------------

Each of these is an input-validation test in ``tests/test_history.py``:

* an unknown or missing ``schema``;
* a ``duration_unit`` other than ``"millisecond"``;
* attempt arrays of unequal length, or empty;
* a session or test index outside its table;
* a ``failed`` value other than 0 or 1;
* a negative duration;
* an attempt number that is not 1-based consecutive within its
  ``(session, test)`` group;
* an attempt recorded *after* a passing attempt in the same group, which the
  stop-at-first-pass policy forbids;
* a group longer than ``retry_budget``;
* a ``retry_budget`` below 1.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "SCHEMA",
    "RunHistory",
    "TestSummary",
    "history_sha256",
    "load_history",
]

#: The only schema string this loader accepts.
SCHEMA = "flakerank.history/1"

#: Milliseconds per second. Durations are stored as exact integer milliseconds.
_MS_PER_S = 1000.0


@dataclass(frozen=True)
class TestSummary:
    """Per-test measurements over a history or a window of one.

    All durations are seconds. Counts are dimensionless.

    Attributes:
        name: the test's identifier.
        suite: the suite it belongs to.
        sessions: number of distinct sessions in which the test was invoked.
        attempts: total attempts recorded, including retries.
        failed_attempts: attempts recorded as failures.
        failure_probability: ``failed_attempts / attempts``, the per-attempt
            flake rate. This is the quantity baseline 1 ranks by.
        wasted_seconds: summed duration of the FAILED attempts. This is the
            ground truth baseline 3 accumulates.
        total_seconds: summed duration of all attempts.
        mean_attempt_seconds: mean duration over all attempts.
        mean_failing_seconds: mean duration over failing attempts, or ``nan``
            when the test never failed in this window.
        sd_attempt_seconds: sample standard deviation of attempt durations
            (``ddof=1``), or ``0.0`` for a single attempt.
        max_attempt_seconds: longest recorded attempt.
        min_attempt_seconds: shortest recorded attempt.
        first_session: lowest session index in which the test was invoked.
        last_session: highest session index in which the test was invoked.
        sessions_with_failure: number of distinct sessions containing at least
            one failing attempt.
    """

    name: str
    suite: str
    sessions: int
    attempts: int
    failed_attempts: int
    failure_probability: float
    wasted_seconds: float
    total_seconds: float
    mean_attempt_seconds: float
    mean_failing_seconds: float
    sd_attempt_seconds: float
    max_attempt_seconds: float
    min_attempt_seconds: float
    first_session: int
    last_session: int
    sessions_with_failure: int

    @property
    def cost_runtime_seconds(self) -> float:
        """The runtime equation (3) of :mod:`flakerank.geometric` needs.

        The mean FAILING-attempt duration when the window contains at least one
        failure, and the mean over all attempts when it does not. The fallback
        is the honest one available: with no observed failure there is no
        measured failing runtime, and the alternative -- reporting ``nan`` --
        would propagate into a ranking. The substitution is recorded in
        ``README.md`` under Limitations because it biases the analytic cost of a
        never-observed-failing test by whatever factor failing attempts differ
        from passing ones, and for such a test the analytic cost is zero anyway
        since its measured ``p`` is zero.
        """
        if self.failed_attempts > 0:
            return self.mean_failing_seconds
        return self.mean_attempt_seconds


@dataclass(frozen=True)
class RunHistory:
    """A recorded CI run history as five parallel integer attempt arrays.

    Attributes:
        retry_budget: maximum attempts per invocation recorded in the history.
        test_names: test identifier per test index.
        test_suites: suite name per test index.
        true_failure_probability: the GENERATOR's per-attempt failure
            probability per test index. Ground truth for negative controls
            only; no baseline and no model reads it.
        true_mean_ms: the generator's mean attempt duration in milliseconds per
            test index. Ground truth for controls only.
        session_days: calendar day per session index.
        attempt_session: session index of each attempt.
        attempt_test: test index of each attempt.
        attempt_number: 1-based attempt number within its ``(session, test)``
            group.
        attempt_failed: 1 for a failing attempt, 0 for a passing one.
        attempt_duration_ms: attempt duration, exact integer milliseconds.
        source: the file the history was read from, relative where possible, or
            ``"<in-memory>"``.
    """

    retry_budget: int
    test_names: tuple[str, ...]
    test_suites: tuple[str, ...]
    true_failure_probability: np.ndarray
    true_mean_ms: np.ndarray
    session_days: np.ndarray
    attempt_session: np.ndarray
    attempt_test: np.ndarray
    attempt_number: np.ndarray
    attempt_failed: np.ndarray
    attempt_duration_ms: np.ndarray
    source: str = "<in-memory>"

    @property
    def n_tests(self) -> int:
        """Number of tests in the history."""
        return len(self.test_names)

    @property
    def n_sessions(self) -> int:
        """Number of sessions in the history."""
        return int(self.session_days.size)

    @property
    def n_attempts(self) -> int:
        """Number of recorded attempts."""
        return int(self.attempt_session.size)

    @property
    def duration_seconds(self) -> np.ndarray:
        """Attempt durations in seconds, as ``float64``."""
        return self.attempt_duration_ms.astype(np.float64) / _MS_PER_S

    def session_window(self, start: int, stop: int) -> RunHistory:
        """The sub-history whose session indices lie in ``[start, stop)``.

        The test and session tables are kept whole so that indices remain
        comparable across windows; only the attempt arrays are filtered. A test
        with no attempt in the window summarises as a zero-attempt test and is
        dropped by :meth:`summaries`.

        Args:
            start: first session index included.
            stop: first session index excluded.

        Returns:
            A new :class:`RunHistory`.

        Raises:
            ValueError: if ``stop <= start`` or either bound is outside the
                history.
        """
        if not isinstance(start, (int, np.integer)) or not isinstance(stop, (int, np.integer)):
            raise TypeError("start and stop must be integer session indices")
        start, stop = int(start), int(stop)
        if not 0 <= start < stop <= self.n_sessions:
            raise ValueError(
                f"session window must satisfy 0 <= start < stop <= {self.n_sessions}, "
                f"got start={start}, stop={stop}"
            )
        keep = (self.attempt_session >= start) & (self.attempt_session < stop)
        return RunHistory(
            retry_budget=self.retry_budget,
            test_names=self.test_names,
            test_suites=self.test_suites,
            true_failure_probability=self.true_failure_probability,
            true_mean_ms=self.true_mean_ms,
            session_days=self.session_days,
            attempt_session=self.attempt_session[keep],
            attempt_test=self.attempt_test[keep],
            attempt_number=self.attempt_number[keep],
            attempt_failed=self.attempt_failed[keep],
            attempt_duration_ms=self.attempt_duration_ms[keep],
            source=self.source,
        )

    def restrict_tests(self, test_indices: Sequence[int]) -> RunHistory:
        """The sub-history containing only the given test indices.

        Args:
            test_indices: test indices to keep. Order is irrelevant; duplicates
                are ignored.

        Returns:
            A new :class:`RunHistory` with the same tables and a filtered
            attempt set.

        Raises:
            ValueError: for an empty selection or an index outside the table.
        """
        wanted = sorted({int(index) for index in test_indices})
        if not wanted:
            raise ValueError("test_indices must select at least one test")
        if wanted[0] < 0 or wanted[-1] >= self.n_tests:
            raise ValueError(
                f"test index out of range: the history has {self.n_tests} tests, got "
                f"indices spanning [{wanted[0]}, {wanted[-1]}]"
            )
        keep = np.isin(self.attempt_test, np.asarray(wanted, dtype=np.int64))
        return RunHistory(
            retry_budget=self.retry_budget,
            test_names=self.test_names,
            test_suites=self.test_suites,
            true_failure_probability=self.true_failure_probability,
            true_mean_ms=self.true_mean_ms,
            session_days=self.session_days,
            attempt_session=self.attempt_session[keep],
            attempt_test=self.attempt_test[keep],
            attempt_number=self.attempt_number[keep],
            attempt_failed=self.attempt_failed[keep],
            attempt_duration_ms=self.attempt_duration_ms[keep],
            source=self.source,
        )

    def summaries(self, *, minimum_attempts: int = 1) -> tuple[TestSummary, ...]:
        """Per-test measurements, in test-index order, for tests with attempts.

        Args:
            minimum_attempts: tests with fewer recorded attempts than this are
                omitted. The default of 1 omits only tests absent from the
                window.

        Returns:
            A tuple of :class:`TestSummary`, ordered by test index.

        Raises:
            ValueError: for ``minimum_attempts < 1``.
        """
        if minimum_attempts < 1:
            raise ValueError(f"minimum_attempts must be >= 1, got {minimum_attempts}")
        seconds = self.duration_seconds
        failed = self.attempt_failed.astype(bool)
        out: list[TestSummary] = []
        for index in range(self.n_tests):
            rows = self.attempt_test == index
            attempts = int(np.count_nonzero(rows))
            if attempts < minimum_attempts:
                continue
            durations = seconds[rows]
            row_failed = failed[rows]
            row_sessions = self.attempt_session[rows]
            failures = int(np.count_nonzero(row_failed))
            failing_durations = durations[row_failed]
            out.append(
                TestSummary(
                    name=self.test_names[index],
                    suite=self.test_suites[index],
                    sessions=int(np.unique(row_sessions).size),
                    attempts=attempts,
                    failed_attempts=failures,
                    failure_probability=failures / attempts,
                    wasted_seconds=float(failing_durations.sum()),
                    total_seconds=float(durations.sum()),
                    mean_attempt_seconds=float(durations.mean()),
                    mean_failing_seconds=(
                        float(failing_durations.mean()) if failures else float("nan")
                    ),
                    sd_attempt_seconds=(
                        float(durations.std(ddof=1)) if attempts > 1 else 0.0
                    ),
                    max_attempt_seconds=float(durations.max()),
                    min_attempt_seconds=float(durations.min()),
                    first_session=int(row_sessions.min()),
                    last_session=int(row_sessions.max()),
                    sessions_with_failure=int(np.unique(row_sessions[row_failed]).size),
                )
            )
        return tuple(out)

    def session_wasted_seconds(self) -> np.ndarray:
        """Wasted seconds per session, shape ``(n_sessions,)``.

        The resampling unit of the ground-truth accumulation is the session, so
        this is the vector the bootstrap in :mod:`flakerank.resample` resamples.
        """
        out = np.zeros(self.n_sessions, dtype=np.float64)
        failed = self.attempt_failed.astype(bool)
        np.add.at(out, self.attempt_session[failed], self.duration_seconds[failed])
        return out

    def session_test_wasted_seconds(self) -> np.ndarray:
        """Wasted seconds per ``(session, test)``, shape ``(n_sessions, n_tests)``.

        Used by the per-test bootstrap, which resamples whole sessions so that
        every test's interval comes from the same resample of the same unit.
        """
        out = np.zeros((self.n_sessions, self.n_tests), dtype=np.float64)
        failed = self.attempt_failed.astype(bool)
        np.add.at(
            out,
            (self.attempt_session[failed], self.attempt_test[failed]),
            self.duration_seconds[failed],
        )
        return out


def _as_int_array(values: Any, name: str) -> np.ndarray:
    """Coerce a JSON array to ``int64``, rejecting anything non-integral."""
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(
            f"attempts.{name} must be a one-dimensional array, got shape {array.shape}"
        )
    if array.size == 0:
        raise ValueError(
            f"attempts.{name} is empty; a history with no attempts has nothing to rank"
        )
    if not np.issubdtype(array.dtype, np.integer):
        as_float = array.astype(np.float64)
        if not np.all(np.isfinite(as_float)) or not np.all(as_float == np.floor(as_float)):
            raise ValueError(
                f"attempts.{name} must contain integers only; durations are stored as "
                "exact integer milliseconds so that a sum of them carries no "
                "representation error from the input"
            )
        array = as_float.astype(np.int64)
    return array.astype(np.int64)


def _validate_attempt_groups(history: RunHistory) -> None:
    """Check the stop-at-first-pass policy and the attempt numbering."""
    order = np.lexsort(
        (history.attempt_number, history.attempt_test, history.attempt_session)
    )
    sessions = history.attempt_session[order]
    tests = history.attempt_test[order]
    numbers = history.attempt_number[order]
    failed = history.attempt_failed[order]
    expected = 1
    previous: tuple[int, int] | None = None
    for position in range(sessions.size):
        key = (int(sessions[position]), int(tests[position]))
        if key != previous:
            expected = 1
            previous = key
        elif failed[position - 1] == 0:
            raise ValueError(
                f"session {key[0]}, test {key[1]} records an attempt after a PASSING "
                "attempt. The recorded retry policy stops at the first pass, so this "
                "history is not consistent with its own retry_budget."
            )
        if int(numbers[position]) != expected:
            raise ValueError(
                f"session {key[0]}, test {key[1]} has attempt number "
                f"{int(numbers[position])} where {expected} was expected; attempt "
                "numbers must be 1-based and consecutive within a (session, test) group"
            )
        if expected > history.retry_budget:
            raise ValueError(
                f"session {key[0]}, test {key[1]} records {expected} attempts, above "
                f"the declared retry_budget of {history.retry_budget}"
            )
        expected += 1


def load_history(path: str | Path) -> RunHistory:
    """Read, validate and return a recorded run history.

    Args:
        path: path to the JSON history produced by ``data/generate_history.py``.

    Returns:
        A validated :class:`RunHistory`.

    Raises:
        FileNotFoundError: if the path does not exist.
        ValueError: for any violation listed in the module docstring. The
            message names the field and what was expected.
    """
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(
            f"no run history at {target.name!r}. It is generated once by "
            "data/generate_history.py and committed; run that script from the "
            "product root to recreate it."
        )
    payload = json.loads(target.read_text(encoding="utf-8"))
    return history_from_mapping(payload, source=target.name)


def history_from_mapping(payload: dict[str, Any], *, source: str = "<in-memory>") -> RunHistory:
    """Validate an already-parsed history mapping. See :func:`load_history`."""
    schema = payload.get("schema")
    if schema != SCHEMA:
        raise ValueError(
            f"unsupported history schema {schema!r}; this loader reads {SCHEMA!r} only. "
            "A loader that accepts an unknown schema silently will one day read a "
            "different file as though it were this one."
        )
    unit = payload.get("duration_unit")
    if unit != "millisecond":
        raise ValueError(
            f"duration_unit must be 'millisecond', got {unit!r}. Durations are stored "
            "as exact integer milliseconds so that a sum of them carries no "
            "representation error from the input."
        )
    retry_budget = payload.get("retry_budget")
    if not isinstance(retry_budget, int) or isinstance(retry_budget, bool) or retry_budget < 1:
        raise ValueError(
            f"retry_budget must be an integer >= 1, got {retry_budget!r}; it is the "
            "maximum number of ATTEMPTS per invocation"
        )
    tests = payload.get("tests")
    sessions = payload.get("sessions")
    attempts = payload.get("attempts")
    if not isinstance(tests, list) or not tests:
        raise ValueError("tests must be a non-empty list of test objects")
    if not isinstance(sessions, list) or not sessions:
        raise ValueError("sessions must be a non-empty list of session objects")
    if not isinstance(attempts, dict):
        raise ValueError("attempts must be an object of five equal-length integer arrays")

    columns = {}
    for name in ("session", "test", "attempt", "failed", "duration_ms"):
        if name not in attempts:
            raise ValueError(f"attempts.{name} is missing")
        columns[name] = _as_int_array(attempts[name], name)
    lengths = {name: int(column.size) for name, column in columns.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"attempt arrays have unequal lengths: {lengths}")

    history = RunHistory(
        retry_budget=retry_budget,
        test_names=tuple(str(item["name"]) for item in tests),
        test_suites=tuple(str(item.get("suite", "")) for item in tests),
        true_failure_probability=np.asarray(
            [float(item.get("true_failure_probability", float("nan"))) for item in tests],
            dtype=np.float64,
        ),
        true_mean_ms=np.asarray(
            [float(item.get("true_mean_ms", float("nan"))) for item in tests], dtype=np.float64
        ),
        session_days=np.asarray([int(item["day"]) for item in sessions], dtype=np.int64),
        attempt_session=columns["session"],
        attempt_test=columns["test"],
        attempt_number=columns["attempt"],
        attempt_failed=columns["failed"],
        attempt_duration_ms=columns["duration_ms"],
        source=source,
    )

    if len(set(history.test_names)) != history.n_tests:
        raise ValueError("test names must be unique")
    if history.attempt_session.min() < 0 or history.attempt_session.max() >= history.n_sessions:
        raise ValueError(
            f"attempts.session must index the {history.n_sessions} sessions, got range "
            f"[{int(history.attempt_session.min())}, {int(history.attempt_session.max())}]"
        )
    if history.attempt_test.min() < 0 or history.attempt_test.max() >= history.n_tests:
        raise ValueError(
            f"attempts.test must index the {history.n_tests} tests, got range "
            f"[{int(history.attempt_test.min())}, {int(history.attempt_test.max())}]"
        )
    if not np.all(np.isin(history.attempt_failed, (0, 1))):
        raise ValueError("attempts.failed must contain only 0 (passed) and 1 (failed)")
    if history.attempt_duration_ms.min() < 0:
        raise ValueError(
            f"attempts.duration_ms must be >= 0, got a minimum of "
            f"{int(history.attempt_duration_ms.min())}; a duration is a non-negative "
            "number of milliseconds"
        )
    if history.attempt_number.min() < 1:
        raise ValueError("attempts.attempt must be 1-based, so >= 1")
    _validate_attempt_groups(history)
    return history


def history_sha256(path: str | Path) -> str:
    """The SHA-256 hex digest of a history file, for pinning in evidence.

    Args:
        path: file to hash.

    Returns:
        64 lower-case hex characters.
    """
    target = Path(path)
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 16), b""):
            digest.update(block)
    return digest.hexdigest()
