"""Run the doctests embedded in the public API docstrings."""

import doctest

import pytest

from interleavekit import block, convolutional, helical, metrics, srandom

MODULES = [block, convolutional, helical, metrics, srandom]


@pytest.mark.parametrize("module", MODULES, ids=lambda m: m.__name__)
def test_doctests(module):
    results = doctest.testmod(module, verbose=False, report=False)
    assert results.failed == 0, f"{results.failed} doctest failures in {module.__name__}"
