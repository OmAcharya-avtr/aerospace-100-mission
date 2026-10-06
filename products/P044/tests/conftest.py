"""Shared fixtures and Hypothesis settings for the aperturediv test suite."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, settings

settings.register_profile(
    "aperturediv",
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile("aperturediv")


@pytest.fixture
def rng() -> np.random.Generator:
    """A fixed-seed generator, so every test is reproducible."""
    return np.random.default_rng(44044)


@pytest.fixture
def line_array_correlation() -> np.ndarray:
    """Log-irradiance correlation of four apertures at 0.05 m, rho_c 0.10 m."""
    from aperturediv.correlation import correlation_matrix, equispaced_positions

    return correlation_matrix(equispaced_positions(4, 0.05), 0.10, "gaussian")
