"""Shared fixtures.

The module-scoped fixtures exist because the build container has one CPU core:
rebuilding the Walker ephemeris or the synthetic dataset per test would push
the suite past its time budget.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from constellink.constellation import GroundStation, walker_delta
from constellink.contacts import all_contact_windows
from constellink.graph import ContactGraph
from constellink.synthdata import DatasetConfig, generate_dataset

EPOCH = datetime(2026, 4, 1, tzinfo=UTC)


@pytest.fixture(scope="module")
def epoch() -> datetime:
    """Fixed element epoch used across the suite."""
    return EPOCH


@pytest.fixture(scope="module")
def station() -> GroundStation:
    """A single southern-hemisphere station."""
    return GroundStation("AWARUA", -46.53, 168.38, 0.01, 10.0)


@pytest.fixture(scope="module")
def small_constellation():
    """An 8/2/1 Walker shell at 550 km -- small enough for a fast suite."""
    return walker_delta(8, 2, 1, 53.0, 550.0, EPOCH, max_epoch_age_days=2.0)


@pytest.fixture(scope="module")
def small_graph(small_constellation, station):
    """Contact graph of the small shell over 2 hours, 60 s scan."""
    t1 = EPOCH + timedelta(hours=2)
    eph = small_constellation.ephemeris(EPOCH, t1, 60.0)
    windows = all_contact_windows(eph, small_constellation.satellites, [station])
    return ContactGraph(windows=windows, t0=EPOCH, t1=t1)


@pytest.fixture(scope="module")
def small_dataset():
    """A deliberately small synthetic dataset (6 h horizon) for model tests."""
    return generate_dataset(DatasetConfig(horizon_hours=6.0, seed=7))
