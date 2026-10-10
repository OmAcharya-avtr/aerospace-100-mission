"""Fixture corpus, part 2: test functions that do verify something.

AST-PARSED ONLY; see the docstring of test_truly_empty.py.
"""

import unittest

import numpy as np
import pytest
from hamcrest import assert_that, equal_to


def test_plain_assert():
    assert 2 + 2 == 4


def test_pytest_raises():
    with pytest.raises(ValueError):
        raise ValueError("out of declared range")


def test_pytest_warns():
    with pytest.warns(UserWarning):
        pass


def test_numpy_testing():
    np.testing.assert_allclose([1.0, 2.0], [1.0, 2.0], rtol=1e-12)


def test_assert_inside_loop():
    for value in (1.0, 2.0, 3.0):
        assert value > 0.0


def test_assert_inside_with():
    with open(__file__, encoding="utf-8") as handle:
        assert handle.readable()


def test_assert_inside_try():
    try:
        value = 1 / 1
    except ZeroDivisionError:
        value = 0.0
    assert value == 1.0


def test_hamcrest_style():
    assert_that(1 + 1, equal_to(2))


def test_nested_function_asserts():
    def check(value):
        assert value > 0

    check(3.0)


def test_pytest_fail_branch():
    value = 1
    if value != 1:
        pytest.fail(f"value was {value}")


class TestUnittestStyle(unittest.TestCase):
    def test_unittest_assert_equal(self):
        self.assertEqual(2 + 2, 4)

    def test_unittest_assert_raises(self):
        with self.assertRaises(ValueError):
            raise ValueError("bad")
