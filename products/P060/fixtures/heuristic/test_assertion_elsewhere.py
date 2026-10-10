"""Fixture corpus, part 3: tests whose verification is not in the function body.

These are the heuristic's false positives by construction. Every function here
does verify something, but the assertion lives in a module-level helper, in a
fixture, or inside a verifier object's method, and the heuristic does not
follow calls.

AST-PARSED ONLY; see the docstring of test_truly_empty.py.
"""

import pytest


def _check_nonneg(value):
    """Module-level assertion helper."""
    assert value >= 0


class Verifier:
    def must_equal(self, got, expected):
        if got != expected:
            raise AssertionError(f"{got} != {expected}")


@pytest.fixture
def checker():
    def _check(value):
        assert value is not None

    return _check


def test_delegates_to_module_helper():
    _check_nonneg(3.0)


def test_delegates_to_fixture(checker):
    checker(3.0)


def test_delegates_to_verifier_object():
    Verifier().must_equal(2 + 2, 4)
