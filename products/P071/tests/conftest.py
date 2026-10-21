"""Shared fixtures. Every path is anchored on ``__file__``, never on the caller's cwd."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conjprob.frames import covariance_from_rtn_sigmas
from conjprob.geometry import ObjectState

REPO_ROOT = Path(__file__).resolve().parents[1]
CASES_DIR = REPO_ROOT / "examples" / "cases"


@pytest.fixture
def repo_root() -> Path:
    """Repository root, resolved from this file."""
    return REPO_ROOT


@pytest.fixture
def cases_dir() -> Path:
    """Directory holding the shipped declared conjunction cases."""
    return CASES_DIR


@pytest.fixture
def leo_primary() -> ObjectState:
    """Primary of the shipped leo-crossing case, built in code rather than read."""
    position = np.array([7078137.0, 0.0, 0.0])
    velocity = np.array([0.0, 7504.0, 0.0])
    return ObjectState(
        position_m=position,
        velocity_mps=velocity,
        covariance_m2=covariance_from_rtn_sigmas(position, velocity, [20.0, 100.0, 20.0]),
        name="primary",
    )


@pytest.fixture
def leo_secondary() -> ObjectState:
    """Secondary of the shipped leo-crossing case."""
    position = np.array([7078167.0, 110.0, 40.0])
    velocity = np.array([0.0, 5306.934315511, 5306.934315511])
    return ObjectState(
        position_m=position,
        velocity_mps=velocity,
        covariance_m2=covariance_from_rtn_sigmas(position, velocity, [50.0, 300.0, 50.0]),
        name="secondary",
    )


@pytest.fixture
def leo_radius_m() -> float:
    """Combined hard-body radius of the shipped leo-crossing case, metres."""
    return 12.0
