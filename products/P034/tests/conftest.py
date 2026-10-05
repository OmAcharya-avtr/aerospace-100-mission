"""Shared fixtures. The suite never touches the network or the filesystem."""

from __future__ import annotations

import numpy as np
import pytest

from faultinject.faults import Injection
from faultinject.harness import fault_rng, nominal_trace, run_case
from faultinject.taxonomy import FaultKind


@pytest.fixture(scope="session")
def nominal():
    """Fault-free trace at seed 7, 150 steps."""
    return nominal_trace(7, 150)


@pytest.fixture
def rng() -> np.random.Generator:
    return fault_rng(1)


@pytest.fixture
def bias_injection() -> Injection:
    """Constant 0.5 m position bias over steps 40..139."""
    return Injection.create(FaultKind.SENSOR_BIAS, "pos", {"offset": 0.5}, 40, 100)


@pytest.fixture
def bias_trace(bias_injection):
    return run_case([bias_injection], 7, 150)
