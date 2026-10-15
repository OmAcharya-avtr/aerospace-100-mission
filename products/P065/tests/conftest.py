"""Shared paths and builders for the test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CASES = ROOT / "examples" / "cases"
FIXTURES = ROOT / "fixtures"

MINIMAL = """
meta:
  title: minimal
losses:
  - id: L-1
    description: a loss
hazards:
  - id: H-1
    description: a hazard
    losses: [L-1]
constraints:
  - id: SC-1
    description: a constraint
    hazards: [H-1]
    requirements: [REQ-001]
control_structure:
  controllers:
    - id: CTRL-1
      name: a controller
  actuators:
    - id: ACT-1
      name: an actuator
  sensors:
    - id: SNS-1
      name: a sensor
  controlled_processes:
    - id: PROC-1
      name: a process
  control_actions:
    - id: CA-1
      name: an action
      controller: CTRL-1
      actuator: ACT-1
      process: PROC-1
  feedback:
    - id: FB-1
      name: a feedback path
      sensor: SNS-1
      controller: CTRL-1
      process: PROC-1
"""


@pytest.fixture
def cases_dir() -> Path:
    """Directory holding the shipped example cases."""
    return CASES


@pytest.fixture
def fixtures_dir() -> Path:
    """Directory holding the shipped cross-link fixtures."""
    return FIXTURES


@pytest.fixture
def minimal_yaml() -> str:
    """A well-formed analysis with one control action and no UCAs."""
    return MINIMAL
