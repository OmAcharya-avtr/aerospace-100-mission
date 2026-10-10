"""Register the `verifies` marker and record its ids into junit XML.

This is the three-line recipe documented in `traceaudit.markers` and in
README.md, used on this package's own suite so that `traceaudit` can audit
itself. It is also the reason `python -m pytest tests/ --junitxml=junit.xml`
produces a report with requirement claims in it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from traceaudit.markers import record_claims

PRODUCT_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = PRODUCT_ROOT / "fixtures"
SAMPLE = FIXTURES / "sample_project"
HEURISTIC = FIXTURES / "heuristic"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "verifies(req_id): requirement id this test verifies"
    )


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: pytest.Item) -> None:
    record_claims(item)


@pytest.fixture(scope="session")
def product_root() -> Path:
    """Root of this product directory."""
    return PRODUCT_ROOT


@pytest.fixture(scope="session")
def sample_project() -> Path:
    """The bundled sample project fixture directory."""
    return SAMPLE


@pytest.fixture(scope="session")
def heuristic_corpus() -> Path:
    """The bundled hand-labelled assertion corpus."""
    return HEURISTIC
