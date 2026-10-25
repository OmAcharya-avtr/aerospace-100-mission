"""Shared fixtures.

Two kinds of problem appear in this suite. A *hand-built* problem has windows
and requests written out as literals, so that the expected answer can be worked
out on paper and put in the test; those are used wherever an exact value is
asserted. A *generated* problem comes from
:func:`contactsched.dataset.generate_instance` with a fixed seed, is large enough
to exercise the solver, and is used where the assertion is an invariant rather
than a value.
"""

from __future__ import annotations

import pytest

from contactsched.dataset import generate_instance
from contactsched.ephemeris import Scenario, default_fleet
from contactsched.problem import ContactRequest, build_problem
from contactsched.station import GroundStation, HorizonMask, default_network
from contactsched.visibility import ContactWindow, all_windows

SMALL_SEEDS = (200, 202, 208, 211, 213, 216, 223, 226, 230, 235, 237, 240, 244, 245, 248)
"""Seeds whose reduced instances have 8 to 17 candidates; see ``small_instances``."""


@pytest.fixture(scope="session")
def scenario() -> Scenario:
    """A six-hour horizon starting 2026-04-30T00:00:00."""
    return Scenario.from_calendar(2026, 4, 30, 0, 0, 0.0, duration_s=21600.0)


@pytest.fixture(scope="session")
def network() -> tuple[GroundStation, ...]:
    """The declared four-antenna network."""
    return default_network()


@pytest.fixture(scope="session")
def fleet() -> tuple:
    """The declared six-spacecraft fleet."""
    return default_fleet()


@pytest.fixture(scope="session")
def windows(scenario, network, fleet) -> tuple[ContactWindow, ...]:
    """Every window of the declared network and fleet over the six-hour horizon."""
    return tuple(all_windows(scenario, network, fleet, step_s=45.0, tolerance_s=0.5))


@pytest.fixture
def toy_stations() -> tuple[GroundStation, ...]:
    """Two antennas with round turnaround times, for hand-built problems."""
    return (
        GroundStation("A1", 0.0, 0.0, 0.0, HorizonMask(floor_deg=5.0), turnaround_s=100.0),
        GroundStation("A2", 10.0, 20.0, 0.0, HorizonMask(floor_deg=5.0), turnaround_s=0.0),
    )


@pytest.fixture
def toy_windows() -> tuple[ContactWindow, ...]:
    """Four hand-written windows on two antennas.

    ``A1`` carries one long window shared by two spacecraft and one short one;
    ``A2`` carries one window for each spacecraft. Numbers are round so that
    candidate start times are exact in binary and every expected value in the
    tests can be written down.
    """
    return (
        ContactWindow("A1", "SC-1", "M1", 0.0, 1000.0, 45.0, 500.0, 10.0),
        ContactWindow("A1", "SC-2", "M2", 200.0, 1200.0, 60.0, 700.0, 10.0),
        ContactWindow("A1", "SC-1", "M1", 4000.0, 4400.0, 20.0, 4200.0, 10.0),
        ContactWindow("A2", "SC-2", "M2", 100.0, 900.0, 70.0, 500.0, 10.0),
    )


@pytest.fixture
def toy_requests() -> tuple[ContactRequest, ...]:
    """Two requests, one per spacecraft, with different priorities."""
    return (
        ContactRequest("R1", "M1", "SC-1", priority=1, duration_s=200.0,
                       min_contacts=1, max_contacts=2),
        ContactRequest("R2", "M2", "SC-2", priority=2, duration_s=200.0,
                       min_contacts=2, max_contacts=2),
    )


@pytest.fixture
def toy_problem(toy_stations, toy_requests, toy_windows):
    """A hand-built problem with a 400 s candidate grid."""
    return build_problem(toy_stations, toy_requests, toy_windows, slot_step_s=400.0)


@pytest.fixture(scope="session")
def instance_seed7():
    """A generated instance over a twelve-hour horizon, seed 7."""
    return generate_instance(7, horizon_s=43200.0)


@pytest.fixture(scope="session")
def small_instances():
    """Generated instances small enough for brute-force enumeration.

    Sizes are forced down to two antennas, three spacecraft and three requests
    over a six-hour horizon with a 600 s candidate grid. The seeds were chosen,
    by the scan recorded in ``validation/validate_exact_output.txt``, as those
    whose candidate count lands between 8 and 17 -- large enough to have
    structure, small enough that ``2 ** n`` enumeration finishes. They are
    listed rather than ranged so a failure names the seed.
    """
    out = []
    for seed in SMALL_SEEDS:
        out.append(
            generate_instance(
                seed,
                n_stations=2,
                n_spacecraft=3,
                n_requests=3,
                horizon_s=21600.0,
                slot_step_s=600.0,
            )
        )
    return out
