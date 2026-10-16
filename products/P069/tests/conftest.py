"""Shared fixtures: the committed campaign fixtures and a small hand-checkable space."""

from __future__ import annotations

from pathlib import Path

import pytest

from scenariocov import (
    Parameter,
    ScenarioSpace,
    load_campaign,
    load_class_document,
    load_space,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.fixture(scope="session")
def fixture_dir() -> Path:
    """Path to the committed fixtures directory."""
    return FIXTURES


@pytest.fixture(scope="session")
def rendezvous_space() -> ScenarioSpace:
    """The committed 7-parameter rendezvous scenario space."""
    return load_space(FIXTURES / "space.json")


@pytest.fixture(scope="session")
def rendezvous_campaign() -> tuple[dict[str, object], ...]:
    """The committed 240-scenario campaign with five planted gaps."""
    return load_campaign(FIXTURES / "campaign_executed.json")


@pytest.fixture(scope="session")
def candidate_pool() -> tuple[dict[str, object], ...]:
    """The committed 400-scenario candidate pool."""
    return load_campaign(FIXTURES / "candidate_pool.json")


@pytest.fixture(scope="session")
def declared_classes():
    """The committed equivalence classes and boundaries."""
    return load_class_document(FIXTURES / "classes.json")


@pytest.fixture
def tiny_space() -> ScenarioSpace:
    """A 3-parameter space with 2, 2 and 3 levels; every count here is hand-checkable.

    Level counts (2, 2, 3) give 12 full combinations, |T_1| = 7 and
    |T_2| = 2*2 + 2*3 + 2*3 = 16.
    """
    return ScenarioSpace(
        [
            Parameter.categorical("mode", ["safe", "nominal"]),
            Parameter.continuous("speed", [0.0, 1.0, 2.0], units="m/s"),
            Parameter.categorical("light", ["sun", "dusk", "dark"]),
        ]
    )
