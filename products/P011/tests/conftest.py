"""Shared fixtures.  Deliberately small grids: the whole suite must stay fast."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from waveforge.loop import AOConfig, AOSystem  # noqa: E402
from waveforge.pupil import PupilGrid  # noqa: E402


@pytest.fixture(scope="session")
def small_pupil() -> PupilGrid:
    """32 x 32 samples across a 0.5 m aperture."""
    return PupilGrid(32, 0.5)


@pytest.fixture(scope="session")
def tiny_config() -> AOConfig:
    """A deliberately small but complete AO configuration."""
    return AOConfig(
        n_pix=32,
        n_sub=4,
        n_act=5,
        screen_pixels=256,
        n_subharmonics=3,
        seed=11,
    )


@pytest.fixture(scope="session")
def tiny_system(tiny_config: AOConfig) -> AOSystem:
    """An assembled small AO system, built once per session."""
    return AOSystem(tiny_config)


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    """A fixed-seed generator for tests that need randomness."""
    return np.random.default_rng(20260829)
