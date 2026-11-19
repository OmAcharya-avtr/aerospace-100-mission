"""Shared fixtures. Session-scoped where the work is propagation or a fit.

The committed record and the SGP4 propagator are built once per session: the
build environment has two cores shared with four other concurrent builds, so
a fixture that re-propagated per test would cost minutes for nothing.
"""

from __future__ import annotations

import pytest

from passyield.dataset import STATIONS, load_passes, station_by_name, window_of
from passyield.decompose import PassEvents
from passyield.geometry import campaign_epoch, find_passes, satellite

#: The one declared seed, the same value as ``validation/_bootstrap.SEED``.
SEED = 99099


@pytest.fixture(scope="session")
def satrec():
    """The propagator for the committed element set."""
    return satellite()


@pytest.fixture(scope="session")
def station():
    """A mid-latitude station from the catalogue."""
    return station_by_name("GS-01")


@pytest.fixture(scope="session")
def windows(satrec, station):
    """Every contact window of ``GS-01`` in two days from the element epoch."""
    epoch_jd, epoch_fr = campaign_epoch(satrec)
    found = find_passes(satrec, station, epoch_jd, epoch_fr, 2.0, coarse_step_s=5.0)
    assert found, "the two-day scan found no window; the fixture is useless"
    return found


@pytest.fixture(scope="session")
def long_window(windows):
    """The longest window of the two-day scan."""
    return max(windows, key=lambda item: item.duration_s)


@pytest.fixture(scope="session")
def events():
    """A declared event timeline with every term non-zero."""
    return PassEvents(
        acquisition_s=40.0,
        resync_intervals_s=((120.0, 18.0), (250.0, 9.0)),
        idle_intervals_s=((160.0, 25.0),),
    )


@pytest.fixture(scope="session")
def record():
    """The committed per-pass record."""
    return load_passes()


@pytest.fixture(scope="session")
def catalogue():
    """The station catalogue."""
    return STATIONS


@pytest.fixture(scope="session")
def record_window(record):
    """A window reconstructed from the committed record."""
    return window_of(record[0]), record[0]
