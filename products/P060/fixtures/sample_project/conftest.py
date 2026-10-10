"""conftest for the fixture project — the three-line recipe from the README."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from traceaudit.markers import record_claims  # noqa: E402


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "verifies(req_id): requirement this test verifies"
    )


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    record_claims(item)
