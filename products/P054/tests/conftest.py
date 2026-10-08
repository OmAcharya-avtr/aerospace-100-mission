"""Shared pytest configuration for the rareverify test suite."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from hypothesis import HealthCheck, settings

settings.register_profile(
    "rareverify",
    deadline=None,
    max_examples=40,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile("rareverify")


def pytest_configure(config: pytest.Config) -> None:
    """Silence the scikit-learn kernel-bound warnings the surrogate provokes.

    The Gaussian-process length scale saturating at its upper bound on an
    exactly linear limit state is expected and is documented in
    rareverify.surrogate; it is not a defect to be fixed by widening bounds
    until the warning disappears.
    """
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.*")
    warnings.filterwarnings(
        "ignore", message=".*close to the specified upper bound.*"
    )
    warnings.filterwarnings(
        "ignore", message=".*close to the specified lower bound.*"
    )


@pytest.fixture
def rng() -> np.random.Generator:
    """A fixed-seed generator so every test is reproducible."""
    return np.random.default_rng(20261008)
