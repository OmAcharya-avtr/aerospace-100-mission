"""Shared fixtures.

Samples are session-scoped because nothing in the suite mutates them and
because generating the larger ones is the slowest thing here apart from the
model fits.
"""

from __future__ import annotations

import numpy as np
import pytest

from transfergap import synthetic
from transfergap.discrepancy import GapSample


@pytest.fixture(scope="session")
def small_field() -> GapSample:
    """15 paired measurements from an exchangeable campaign: the hard case."""
    return synthetic.generate(15, "random", seed=101)


@pytest.fixture(scope="session")
def field_sample() -> GapSample:
    """60 paired measurements from an exchangeable campaign."""
    return synthetic.generate(60, "random", seed=202)


@pytest.fixture(scope="session")
def swept_sample() -> GapSample:
    """120 paired measurements flown to a monotone Mach card: NOT exchangeable."""
    return synthetic.generate(120, "swept", seed=303)


@pytest.fixture(scope="session")
def pool() -> GapSample:
    """A 3000-point held-out pool from the same envelope, for coverage."""
    return synthetic.generate(3000, "random", seed=404)


@pytest.fixture(scope="session")
def truth() -> synthetic.PopulationTruth:
    """The population truth at a reduced Monte Carlo size, for speed."""
    return synthetic.population_truth(draws=120_000, seed=505)


@pytest.fixture
def simple_gaps() -> np.ndarray:
    """``[1, 2, 3, 4]``: the sample every hand calculation in this suite uses."""
    return np.array([1.0, 2.0, 3.0, 4.0])
