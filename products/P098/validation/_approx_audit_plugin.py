"""A pytest plugin that makes a bare ``rel=`` in ``pytest.approx`` an error.

``pytest.approx(expected, rel=r)`` passes if EITHER the relative or the
absolute bound is met, and the absolute default is ``1e-12``. For a quantity of
order 1e-5 every relative tolerance tighter than 2e-8 is therefore silently
replaced by the default and the assertion no longer tests what it says it
tests. An earlier product of this mission found exactly 3 of 271 assertions
depending on the default, one of which was hiding a real 2.086e-10 relative
discrepancy.

Loaded with ``-p _approx_audit_plugin``, this plugin replaces
:func:`pytest.approx` with a wrapper that raises when ``rel`` is given and
``abs`` is not. A suite that passes under it has no assertion resting on the
hidden default, and that is a measurement rather than a reading of the source.
"""

from __future__ import annotations

import pytest

_ORIGINAL = pytest.approx


def _strict_approx(expected: object, rel: object = None, abs: object = None, nan_ok: bool = False):  # noqa: A002
    """``pytest.approx`` with the hidden absolute default forbidden."""
    if rel is not None and abs is None:
        raise AssertionError(
            "pytest.approx was called with rel= and no abs=. The absolute "
            "default is 1e-12 and would silently replace the relative bound "
            "for any quantity of order 1e-5 or smaller. Pass abs=0.0 when the "
            "relative bound is the one you mean."
        )
    return _ORIGINAL(expected, rel=rel, abs=abs, nan_ok=nan_ok)


def pytest_configure(config: pytest.Config) -> None:
    """Install the strict wrapper for the whole session."""
    pytest.approx = _strict_approx  # type: ignore[assignment]
