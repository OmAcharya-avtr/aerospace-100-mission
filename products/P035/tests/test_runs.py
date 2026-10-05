"""Tests for the vectorised persistence-run detection."""

from __future__ import annotations

import numpy as np
import pytest

from telemetryool.runs import any_run, first_run_index, run_counter_trace


def test_first_run_index_hand_values() -> None:
    """Rows, with persistence = 3:

    row 0: T T T F F        -> run completes at index 2
    row 1: F T T T F        -> run completes at index 3
    row 2: T T F T T        -> no run of 3 -> -1
    row 3: F F F F F        -> -1
    row 4: T T T T T        -> first completion at index 2, not 4
    """
    breach = np.array(
        [
            [1, 1, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [1, 1, 0, 1, 1],
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
        ],
        dtype=bool,
    )
    assert first_run_index(breach, 3).tolist() == [2, 3, -1, -1, 2]


def test_persistence_one_is_the_first_true() -> None:
    breach = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 0]], dtype=bool)
    assert first_run_index(breach, 1).tolist() == [2, 0, -1]


def test_any_run_matches_first_run_index() -> None:
    rng = np.random.default_rng(0)
    breach = rng.random((200, 40)) < 0.3
    for persistence in (1, 2, 3, 5):
        assert np.array_equal(
            any_run(breach, persistence), first_run_index(breach, persistence) >= 0
        )


def test_run_longer_than_window_never_fires() -> None:
    breach = np.ones((3, 4), dtype=bool)
    assert first_run_index(breach, 5).tolist() == [-1, -1, -1]


def test_run_counter_trace_hand_values() -> None:
    """breach = T T T F T T, persistence = 2 -> counter 1 2 2 0 1 2 (saturating at 2)."""
    breach = np.array([1, 1, 1, 0, 1, 1], dtype=bool)
    assert run_counter_trace(breach, 2).tolist() == [1, 2, 2, 0, 1, 2]


def test_run_counter_trace_no_saturation_below_persistence() -> None:
    breach = np.ones(5, dtype=bool)
    assert run_counter_trace(breach, 10).tolist() == [1, 2, 3, 4, 5]


def test_first_run_index_against_reference_implementation() -> None:
    """Cross-check the vectorised path against a plain Python loop."""
    rng = np.random.default_rng(7)
    breach = rng.random((120, 25)) < 0.4

    def reference(row: np.ndarray, persistence: int) -> int:
        count = 0
        for i, bit in enumerate(row):
            count = count + 1 if bit else 0
            if count >= persistence:
                return i
        return -1

    for persistence in (1, 2, 4, 7):
        expected = [reference(row, persistence) for row in breach]
        assert first_run_index(breach, persistence).tolist() == expected


def test_input_validation() -> None:
    with pytest.raises(ValueError, match="must be 2-D"):
        first_run_index(np.ones(5, dtype=bool), 2)
    with pytest.raises(ValueError, match="persistence must be an integer >= 1"):
        first_run_index(np.ones((2, 5), dtype=bool), 0)
    with pytest.raises(ValueError, match="persistence must be an integer >= 1"):
        first_run_index(np.ones((2, 5), dtype=bool), 2.5)
    with pytest.raises(ValueError, match="must be 1-D"):
        run_counter_trace(np.ones((2, 5), dtype=bool), 2)
    with pytest.raises(ValueError, match="persistence must be >= 1"):
        run_counter_trace(np.ones(5, dtype=bool), 0)
