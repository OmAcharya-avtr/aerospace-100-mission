"""Shared fixtures. Every sample is seeded, so every test is deterministic."""

from __future__ import annotations

import numpy as np
import pytest

from calibaudit.synthetic import get_spec, sample_forecast

#: Small sample used wherever a test needs data but not statistical power.
SMALL_N = 400
#: Larger sample used where a test asserts a statistical quantity.
LARGE_N = 8000


@pytest.fixture(scope="session")
def calibrated_small():
    return sample_forecast(get_spec("calibrated"), SMALL_N, seed=56001)


@pytest.fixture(scope="session")
def calibrated_large():
    return sample_forecast(get_spec("calibrated"), LARGE_N, seed=56002)


@pytest.fixture(scope="session")
def overconfident_small():
    return sample_forecast(get_spec("overconfident"), SMALL_N, seed=56003)


@pytest.fixture(scope="session")
def overconfident_large():
    return sample_forecast(get_spec("overconfident"), LARGE_N, seed=56004)


@pytest.fixture(scope="session")
def rare_large():
    return sample_forecast(get_spec("calibrated_rare"), LARGE_N, seed=56005)


@pytest.fixture
def rng():
    return np.random.default_rng(56006)
