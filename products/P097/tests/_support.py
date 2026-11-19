"""Hand-checkable fixtures and constants shared by the test modules.

Kept out of ``conftest.py`` because ``conftest`` is imported by pytest as a
plugin and is not importable as a module from a sibling test file without a
package ``__init__.py``; adding one would change how pytest resolves the test
modules. This file is an ordinary module on the test directory's own
``sys.path`` entry.
"""

from __future__ import annotations

from pathlib import Path

#: The product root, anchored on this file and never on the caller's cwd.
ROOT = Path(__file__).resolve().parent.parent

#: The committed history, relative to the product root.
HISTORY_PATH = ROOT / "data" / "run_history.json"

#: The declared seed, matching ``data/generate_history.py``.
SEED = 97044

#: SHA-256 of the committed history, pinned so a silent edit is caught.
HISTORY_SHA256 = "aa12f8cc861ba66008a0d3762cd68fa24a24b013f897da67c8151996c531b18a"


def tiny_history(
    *,
    retry_budget: int = 3,
    attempts: list[tuple[int, int, int, int, int]] | None = None,
    n_tests: int = 2,
    n_sessions: int = 2,
    schema: str = "flakerank.history/1",
    duration_unit: str = "millisecond",
) -> dict:
    """Build a minimal history mapping for the input-validation tests.

    ``attempts`` rows are ``(session, test, attempt, failed, duration_ms)``.
    The default is a two-session, two-test history in which test 0 fails once
    for 2000 ms and then passes for 1000 ms in session 0 and passes for 1000 ms
    in session 1, and test 1 passes for 500 ms in both. Hand-checkable: test 0
    wastes exactly 2.0 s and has a measured p of 1/3.
    """
    if attempts is None:
        attempts = [
            (0, 0, 1, 1, 2000),
            (0, 0, 2, 0, 1000),
            (0, 1, 1, 0, 500),
            (1, 0, 1, 0, 1000),
            (1, 1, 1, 0, 500),
        ]
    return {
        "schema": schema,
        "retry_budget": retry_budget,
        "duration_unit": duration_unit,
        "tests": [
            {
                "name": f"suite/test_{index}",
                "suite": "suite",
                "true_failure_probability": 0.5 if index == 0 else 0.0,
                "true_mean_ms": 1000.0,
            }
            for index in range(n_tests)
        ],
        "sessions": [{"index": index, "day": index} for index in range(n_sessions)],
        "attempts": {
            "session": [row[0] for row in attempts],
            "test": [row[1] for row in attempts],
            "attempt": [row[2] for row in attempts],
            "failed": [row[3] for row in attempts],
            "duration_ms": [row[4] for row in attempts],
        },
    }
