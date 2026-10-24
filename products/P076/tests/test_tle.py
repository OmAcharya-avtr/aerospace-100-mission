"""Terminal states from two-line element sets, through ``sgp4``.

The element sets below are the ones distributed in the ``sgp4`` package's own
documentation and test material for ISS (ZARYA), reproduced here so the test is
self-contained and needs no network. They are used to exercise the conversion
from SGP4 output to this package's state objects; no numerical claim in this
repository depends on them being current, and they are not current.
"""

from __future__ import annotations

import numpy as np
import pytest

from pointahead.angle import closed_form_point_ahead
from pointahead.geometry import norm
from pointahead.tle import TEME_FRAME, TleTerminal, crosslink_from_tles

LINE1 = "1 25544U 98067A   19343.69339541  .00001764  00000-0  38792-4 0  9991"
LINE2 = "2 25544  51.6439 211.2001 0007417  17.6667  85.6398 15.50103472202482"

# The same object with its node shifted, used only to make a second terminal at
# a different place in the same orbit. It is not a real catalogue entry.
LINE2_SHIFTED = "2 25544  51.6439 211.2001 0007417  17.6667 125.6398 15.50103472202482"

JD = 2458826.0
FR = 0.19339541


def test_tle_terminal_builds_from_two_lines() -> None:
    terminal = TleTerminal.from_lines("iss", LINE1, LINE2)
    assert terminal.name == "iss"
    assert terminal.satellite.satnum == 25544


def test_a_malformed_element_set_is_refused() -> None:
    # sgp4 2.27 sets Satrec.error rather than raising for a line pair it cannot
    # make sense of, so a wrapper that only caught exceptions would hand back a
    # satellite object that silently produces nothing.
    with pytest.raises(ValueError, match="sgp4 (rejected|could not parse)"):
        TleTerminal.from_lines("broken", "not a tle", "not a tle either")


def test_an_out_of_range_propagation_epoch_is_refused() -> None:
    terminal = TleTerminal.from_lines("iss", LINE1, LINE2)
    # Three centuries past the element-set epoch drives the SGP4 propagator past
    # its own validity and it reports an error code rather than a silly state.
    with pytest.raises(ValueError, match="sgp4 propagation"):
        terminal.state(JD + 110000.0, 0.0)


def test_the_state_is_in_metres_and_metres_per_second() -> None:
    terminal = TleTerminal.from_lines("iss", LINE1, LINE2)
    state = terminal.state(JD, FR)
    radius = norm(state.position_m)
    speed = state.speed_mps
    # A low Earth orbit: radius between 6.5e6 and 7.2e6 m, speed near 7.7e3 m/s.
    # sgp4 reports kilometres, so a missing conversion would put these at 6.8e3
    # and 7.7, which is what this test is for.
    assert 6.5e6 < radius < 7.2e6
    assert 7.0e3 < speed < 8.0e3


def test_a_crosslink_between_two_tle_terminals_carries_the_teme_label() -> None:
    local = TleTerminal.from_lines("a", LINE1, LINE2)
    remote = TleTerminal.from_lines("b", LINE1, LINE2_SHIFTED)
    link = crosslink_from_tles(local, remote, JD, FR, label="tle pair")
    assert link.frame == TEME_FRAME
    assert link.label == "tle pair"
    assert link.range_m > 1.0e5
    angle = closed_form_point_ahead(link).total_rad
    # Two terminals in the same plane and at nearly the same radius have a small
    # transverse relative velocity, so the angle is of order a few microradians.
    assert 1.0e-7 < angle < 1.0e-4


def test_the_two_trajectory_models_are_both_accepted() -> None:
    local = TleTerminal.from_lines("a", LINE1, LINE2)
    remote = TleTerminal.from_lines("b", LINE1, LINE2_SHIFTED)
    for model in ("kepler", "linear"):
        link = crosslink_from_tles(local, remote, JD, FR, trajectory_model=model)
        assert link.trajectory_model == model


def test_propagation_moves_the_terminal() -> None:
    terminal = TleTerminal.from_lines("iss", LINE1, LINE2)
    first = terminal.state(JD, FR)
    second = terminal.state(JD, FR + 60.0 / 86400.0)
    separation = float(np.linalg.norm(second.position_m - first.position_m))
    # One minute at 7.7 km/s is of order 450 km of arc.
    assert 3.0e5 < separation < 5.0e5
