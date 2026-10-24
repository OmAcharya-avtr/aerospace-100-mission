"""Shared fixtures. Every path is anchored on ``__file__``, never on the caller's cwd."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pointahead.budget import BudgetInputs
from pointahead.kepler import MU_EARTH, LinearTrajectory
from pointahead.scenario import (
    DEFAULT_BEAM,
    CircularOrbit,
    geo_to_leo_pair,
    leo_crossing_pair,
    leo_trailing_pair,
)
from pointahead.terminal import BeamContext, Crosslink, TerminalState

ROOT = Path(__file__).resolve().parent.parent
CASE_DIR = ROOT / "examples" / "cases"

C = 299792458.0


@pytest.fixture
def hand_link() -> Crosslink:
    """A hand-calculable crosslink with a straight-line remote trajectory.

    Local terminal at ``[1e7, 0, 0]`` m with zero velocity, so no aberration.
    Remote terminal at ``[1e7, 2e6, 0]`` m, so the separation is ``[0, 2e6, 0]``
    and the line of sight is ``+y``. Remote velocity ``[7500, 0, 0]`` m/s is
    entirely transverse.

    Hand calculation: ``theta = 2 |v_rel_perp| / c = 15000 / 299792458``
    ``= 5.003462...e-05`` rad ``= 50.03462`` urad.
    """
    return Crosslink(
        local=TerminalState([1.0e7, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0, "local"),
        remote=TerminalState([1.0e7, 2.0e6, 0.0], [7500.0, 0.0, 0.0], 0.0, "remote"),
        trajectory_model="linear",
        label="hand",
        frame="hand-built inertial",
    )


@pytest.fixture
def hand_point_ahead_rad() -> float:
    """``2 * 7500 / c``, the hand-calculated answer for ``hand_link``."""
    return 15000.0 / C


@pytest.fixture
def radial_link() -> Crosslink:
    """A crosslink whose relative velocity is purely along the line of sight.

    The point-ahead angle must be exactly zero: the apparent source direction
    and the required transmit direction are both parallel to the separation
    vector, whatever the light times are.
    """
    return Crosslink(
        local=TerminalState([1.0e7, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0, "local"),
        remote=TerminalState([1.0e7, 2.0e6, 0.0], [0.0, 4000.0, 0.0], 0.0, "remote"),
        trajectory_model="linear",
        label="radial",
        frame="hand-built inertial",
    )


@pytest.fixture
def relativistic_light_time() -> tuple[LinearTrajectory, np.ndarray]:
    """A light-time case with an exact rational answer.

    Remote one light-second away along ``+x`` moving at ``0.6 c`` along ``+y``,
    local at the origin. The transmit light time solves
    ``|[c, 0.6 c tau, 0]| = c tau``, so ``tau = 1 / sqrt(1 - 0.36) = 1.25`` s
    exactly.
    """
    trajectory = LinearTrajectory(np.array([C, 0.0, 0.0]), np.array([0.0, 0.6 * C, 0.0]), 0.0)
    return trajectory, np.zeros(3)


@pytest.fixture
def crossing_link() -> Crosslink:
    """The built-in ``leo_crossing`` geometry under the Kepler model."""
    return leo_crossing_pair()


@pytest.fixture
def trailing_link() -> Crosslink:
    """The built-in ``leo_trailing`` geometry under the Kepler model."""
    return leo_trailing_pair()


@pytest.fixture
def geo_link() -> Crosslink:
    """The built-in ``geo_relay_to_leo`` geometry under the Kepler model."""
    return geo_to_leo_pair()


@pytest.fixture
def default_beam() -> BeamContext:
    """1550 nm, 25 mm waist radius."""
    return DEFAULT_BEAM


@pytest.fixture
def default_inputs(default_beam: BeamContext) -> BudgetInputs:
    """A declared configuration: 5 cm/s per axis, 30 s old, one iteration."""
    return BudgetInputs(
        sigma_velocity_per_axis_mps=0.05,
        ephemeris_age_s=30.0,
        light_time_iterations=1,
        beam=default_beam,
        aberration_route="momentum_boost",
        angle_method="closed_form",
        loss_budget_db=1.0,
    )


@pytest.fixture
def circular_orbit() -> CircularOrbit:
    """A 600 km equatorial circular orbit at the ascending node."""
    return CircularOrbit(600.0e3, 0.0, 0.0, 0.0)


@pytest.fixture
def mu() -> float:
    """WGS-84 Earth gravitational parameter, m^3/s^2."""
    return MU_EARTH


@pytest.fixture
def case_dir() -> Path:
    """Directory holding the shipped YAML cases."""
    return CASE_DIR
