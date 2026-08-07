"""Shared fixtures.

The trained model is expensive relative to a unit test, so it is built once per
session with a deliberately small configuration (240 scenarios x 16 altitudes
instead of 700 x 28).  Tests that assert statistical behaviour (interval
coverage) therefore use correspondingly wide tolerances; the production-size
numbers live in ``validation/benchmark_results.md``.
"""

from __future__ import annotations

import numpy as np
import pytest

from cncast.model import CnCastModel, train_default_model

SMALL_N_SCENARIOS = 240
SMALL_N_ALTITUDES = 16


@pytest.fixture(scope="session")
def small_model() -> tuple[CnCastModel, dict]:
    """A small, seeded, conformally calibrated model plus its data splits."""
    return train_default_model(
        n_scenarios=SMALL_N_SCENARIOS,
        n_altitudes=SMALL_N_ALTITUDES,
        random_state=7,
    )


@pytest.fixture(scope="session")
def fine_grid() -> np.ndarray:
    """A 1 m-resolution altitude grid over 0-20 km, m."""
    return np.linspace(0.0, 20_000.0, 20_001)
