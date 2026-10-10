"""Fixture corpus, part 1: test functions that verify nothing.

AST-PARSED ONLY. Nothing in fixtures/heuristic/ is ever executed; these files
are input to the assertion heuristic and to its hand-labelled error-rate
measurement. The hand labels are in labels.json and were written by reading
these functions before the heuristic was run against them.
"""

import pytest


def test_empty_body():
    pass


def test_only_calls_code():
    monitor = {"momentum": 0.0}
    monitor["momentum"] = 1.5


def test_builds_a_value_and_drops_it():
    samples = [1.0, 2.0, 3.0]
    total = sum(samples)
    formatted = f"{total:.3f}"
    del formatted


def test_assert_true_constant():
    assert True


def test_assert_one_constant():
    assert 1


def test_vacuous_self_comparison():
    value = 3.0
    assert value == value


def test_always_true_predicate():
    result = [1, 2, 3]
    assert len(result) >= 0


def test_assert_short_circuited_by_true():
    value = -1.0
    assert value > 0 or True


_ = pytest
