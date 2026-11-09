"""A pytest plugin that audits every ``pytest.approx`` call in the suite.

``pytest.approx(expected, rel=r)`` passes if EITHER the relative or the
absolute bound is met, and the absolute default is ``1e-12``. So for any
quantity of order 1e-5 every relative tolerance tighter than 2e-8 is silently
replaced by the default and the assertion no longer tests what it says it
tests. One product of this mission found 3 of 271 assertions depending on that
default, and one of the three was hiding a real 2.1e-10 relative discrepancy.

This matters more here than in most products, because identifiability work is
full of quantities that are small by construction: singular values at the
1e-16 level, sums of sensitivities that vanish identically, profile curvatures
of order 1e-10, and principal angles of order 1e-6 degrees.

Two modes, selected by ``TWINDIVERGE_APPROX_AUDIT``:

``strict``
    Any call that passes ``rel=`` without an explicit ``abs=`` raises
    immediately, naming the file and line. This is the static question: is
    there any call in the suite that relies on the default at all?

``zero``
    Every call's ``abs`` is forced to ``0.0``, whatever it was given. The suite
    is then re-run and the failures are counted. This is the stronger question:
    for how many assertions is the ABSOLUTE bound the one doing the work? A
    failure here is not necessarily a defect -- a comparison against an exact
    zero has no relative bound and must use an absolute one -- but every such
    assertion has to be accounted for by name.

The plugin is loaded with ``-p`` and is never imported by the package or by the
test suite itself.
"""

from __future__ import annotations

import os

import pytest

MODE = os.environ.get("TWINDIVERGE_APPROX_AUDIT", "")
_original = pytest.approx


class BareRelativeTolerance(AssertionError):
    """Raised in strict mode when ``rel=`` is passed without ``abs=``."""


def _strict(expected, rel=None, abs=None, nan_ok=False):  # noqa: A002 - pytest's name
    if rel is not None and abs is None:
        import traceback

        frame = traceback.extract_stack()[-2]
        raise BareRelativeTolerance(
            f"pytest.approx(rel={rel!r}) with no explicit abs= at "
            f"{os.path.basename(frame.filename)}:{frame.lineno}; the hidden "
            "absolute default of 1e-12 would apply"
        )
    return _original(expected, rel=rel, abs=abs, nan_ok=nan_ok)


def _zero(expected, rel=None, abs=None, nan_ok=False):  # noqa: A002 - pytest's name
    return _original(expected, rel=rel, abs=0.0, nan_ok=nan_ok)


def pytest_configure(config) -> None:
    """Install the chosen wrapper before collection."""
    _ = config
    if MODE == "strict":
        pytest.approx = _strict
    elif MODE == "zero":
        pytest.approx = _zero
