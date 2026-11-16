"""A pytest plugin that makes a bare ``rel=`` call to ``pytest.approx`` raise.

``pytest.approx(expected, rel=r)`` passes if EITHER the relative or the absolute
bound is met, and the absolute default is ``1e-12``. So for any quantity of
order 1e-5, every relative tolerance tighter than 2e-8 is silently replaced by
the default and the assertion no longer tests what it says it tests. An earlier
product in this mission found exactly 3 of 271 assertions depending on that
default, one of which was hiding a real 2.086218e-10 relative discrepancy.

This plugin is loaded by ``validate_approx_audit.py`` with ``-p``. It replaces
``pytest.approx`` with a wrapper that raises when ``rel`` is given and ``abs``
is not, so the audit is a measurement of the whole suite rather than a grep.
"""

from __future__ import annotations

import pytest

_ORIGINAL = pytest.approx


class BareRelativeToleranceError(AssertionError):
    """Raised when ``pytest.approx`` is called with ``rel=`` and no ``abs=``."""


def _audited(expected, rel=None, abs=None, nan_ok=False):  # noqa: A002
    if rel is not None and abs is None:
        raise BareRelativeToleranceError(
            "pytest.approx was called with rel= and no abs=. The absolute default is "
            "1e-12 and passes on its own, so the relative bound is not what is being "
            "tested. Pass an explicit abs=, and abs=0.0 when the relative bound is the "
            "one you mean."
        )
    return _ORIGINAL(expected, rel=rel, abs=abs, nan_ok=nan_ok)


def pytest_configure(config):
    """Install the audited wrapper before collection."""
    pytest.approx = _audited


def pytest_unconfigure(config):
    """Restore the original."""
    pytest.approx = _ORIGINAL
