"""Shared fixtures. Every random test is seeded; nothing here is flaky by design."""

from __future__ import annotations

import numpy as np
import pytest

from photoncount.simulate import DetectorSpec


@pytest.fixture
def rng() -> np.random.Generator:
    """A fixed-seed generator. Tests that need independence make their own."""
    return np.random.default_rng(20261006)


@pytest.fixture
def spec_ideal() -> DetectorSpec:
    """An ideal counter: no dead time, no afterpulsing."""
    return DetectorSpec()


@pytest.fixture
def spec_paralyzable() -> DetectorSpec:
    """A paralyzable detector with afterpulsing whose delay exceeds the dead time."""
    return DetectorSpec(
        dead_time_s=1e-7,
        model="paralyzable",
        afterpulse_probability=0.05,
        afterpulse_mean_delay_s=5e-7,
    )
