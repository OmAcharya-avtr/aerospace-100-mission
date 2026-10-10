"""Recording requirement claims into junit XML from a pytest run.

The convention this package reads is the one `pytest-requirements` 0.3.1 uses:
a junit ``<property name="requirement_id" value="REQ-003"/>`` on the test case.
pytest writes such a property for every ``record_property`` call and for every
entry a plugin appends to ``item.user_properties``.

To emit them without installing a plugin, put this in your ``conftest.py``::

    import pytest
    from traceaudit.markers import record_claims

    def pytest_configure(config):
        config.addinivalue_line(
            "markers", "verifies(req_id): requirement this test verifies"
        )

    @pytest.hookimpl(tryfirst=True)
    def pytest_runtest_setup(item):
        record_claims(item)

and mark tests with ``@pytest.mark.verifies("REQ-003")``.  One marker may name
several ids, and a test may carry several markers.

This module imports nothing from pytest; it only touches the ``iter_markers``
and ``user_properties`` attributes that a pytest ``Item`` provides, so it is
importable and testable without pytest installed.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

DEFAULT_MARKER = "verifies"
DEFAULT_PROPERTY = "requirement_id"


def claims_from_marker_args(args: Iterable[Any]) -> tuple[str, ...]:
    """Flatten marker arguments into a tuple of requirement id strings.

    ``@pytest.mark.verifies("REQ-001")``, ``verifies("REQ-001", "REQ-002")``
    and ``verifies(["REQ-001", "REQ-002"])`` all give the same result.
    """
    out: list[str] = []
    for arg in args:
        if isinstance(arg, str):
            out.append(arg)
        elif isinstance(arg, Iterable):
            out.extend(str(x) for x in arg)
        else:
            raise TypeError(
                f"marker argument must be a string or an iterable of strings, "
                f"got {type(arg).__name__}"
            )
    seen: dict[str, None] = {}
    for item in out:
        seen.setdefault(item.strip(), None)
    return tuple(k for k in seen if k)


def record_claims(
    item: Any, *, marker_name: str = DEFAULT_MARKER, property_name: str = DEFAULT_PROPERTY
) -> tuple[str, ...]:
    """Copy every ``@pytest.mark.<marker_name>`` id onto the item's properties.

    Parameters
    ----------
    item:
        A pytest ``Item``; anything with ``iter_markers`` and
        ``user_properties`` works, which is what makes this unit-testable.
    marker_name:
        Marker to read.
    property_name:
        junit property name to write.  The default matches what
        :mod:`traceaudit.testreports` reads by default.

    Returns
    -------
    tuple of str
        The ids recorded, in order.
    """
    recorded: list[str] = []
    existing = {
        value for name, value in getattr(item, "user_properties", []) if name == property_name
    }
    for marker in item.iter_markers(name=marker_name):
        for claim in claims_from_marker_args(marker.args):
            if claim in existing or claim in recorded:
                continue
            item.user_properties.append((property_name, claim))
            recorded.append(claim)
    return tuple(recorded)
