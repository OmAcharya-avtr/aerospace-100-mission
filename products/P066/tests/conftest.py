"""Shared fixtures. ``src`` is put on the path so the suite runs from a cold clone."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
CASES = ROOT / "examples" / "cases"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from budgetroll import Budget, load  # noqa: E402


@pytest.fixture(scope="session")
def cases_dir() -> Path:
    """Directory holding the shipped example declarations."""
    return CASES


@pytest.fixture(scope="session")
def clean_mass() -> Budget:
    """The negative control: a declaration that raises no finding."""
    return load(CASES / "clean_mass.yaml")


@pytest.fixture(scope="session")
def planted() -> Budget:
    """The fixture built to raise every check."""
    return load(CASES / "planted_findings.yaml")


@pytest.fixture(scope="session")
def cubesat_power() -> Budget:
    """The worked CubeSat orbit-average power budget."""
    return load(CASES / "cubesat_power.yaml")


@pytest.fixture(scope="session")
def cubesat_link() -> Budget:
    """The worked CubeSat S-band downlink degradation budget."""
    return load(CASES / "cubesat_link.yaml")
