"""Shared fixtures. Paths are resolved from this file so the suite runs from
any working directory.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from regressselect.history import load_history
from regressselect.importgraph import build_import_graph
from regressselect.inventory import collect_tests
from regressselect.prioritise import CoFailurePrioritiser
from regressselect.selector import LearnedSelector

PRODUCT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PROJECT = PRODUCT_ROOT / "fixtures" / "sample_project"
HISTORY_JSON = PRODUCT_ROOT / "fixtures" / "history" / "history.json"
FAULTS_JSON = PRODUCT_ROOT / "fixtures" / "faults" / "faults.json"
FAULT_MATRIX_JSON = PRODUCT_ROOT / "fixtures" / "faults" / "fault_matrix.json"
SAMPLE_JUNIT = SAMPLE_PROJECT / "junit.xml"


@pytest.fixture(scope="session")
def product_root() -> Path:
    return PRODUCT_ROOT


@pytest.fixture(scope="session")
def sample_project() -> Path:
    if not SAMPLE_PROJECT.is_dir():
        pytest.fail(f"bundled sample project is missing: {SAMPLE_PROJECT}")
    return SAMPLE_PROJECT


@pytest.fixture(scope="session")
def graph(sample_project: Path):
    return build_import_graph(sample_project)


@pytest.fixture(scope="session")
def inventory(sample_project: Path):
    return collect_tests(sample_project)


@pytest.fixture(scope="session")
def history():
    if not HISTORY_JSON.is_file():
        pytest.fail(
            f"committed history fixture is missing: {HISTORY_JSON}. "
            "Regenerate with python fixtures/regenerate_fixtures.py"
        )
    return load_history(HISTORY_JSON)


@pytest.fixture(scope="session")
def split(history):
    return history.split_chronological(0.70)


@pytest.fixture(scope="session")
def prioritiser(split):
    return CoFailurePrioritiser().fit(split[0])


@pytest.fixture(scope="session")
def selector(split, graph):
    return LearnedSelector().fit(split[0], graph)
