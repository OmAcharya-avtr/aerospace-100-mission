"""Shared fixtures.

Residual simulation and classifier fitting are the expensive steps, so both are
session-scoped. Nothing mutable is shared: every detector object in this
package is frozen and every statistic function is pure.
"""

from __future__ import annotations

import numpy as np
import pytest

from twininvalidate import (
    AssetChange,
    DriftClassifier,
    StreamSpec,
    calibration_set,
    reference_twin,
    simulate_residuals,
    training_set,
)


@pytest.fixture(scope="session")
def twin():
    """The shipped reference twin."""
    return reference_twin()


@pytest.fixture(scope="session")
def filt(twin):
    """Its fixed-gain steady-state residual generator."""
    return twin.steady_state()


@pytest.fixture(scope="session")
def in_control():
    """In-control residual streams, 120 runs x 2000 samples, seed 53001."""
    return simulate_residuals(
        StreamSpec(change=AssetChange(), n_runs=120, n_samples=2000, seed=53001)
    )


@pytest.fixture(scope="session")
def stepped():
    """Streams with the declared parameter step present from sample 0."""
    return simulate_residuals(
        StreamSpec(
            change=AssetChange("parameter_step", onset=0, magnitude=-0.01),
            n_runs=120,
            n_samples=1500,
            seed=53100,
        )
    )


@pytest.fixture(scope="session")
def classifier():
    """A fitted drift classifier, built from the declared seeded datasets."""
    train = training_set(n_runs_in_control=60, n_runs_per_scenario=20, n_samples=500)
    cal = calibration_set(n_runs_in_control=30, n_runs_per_scenario=10, n_samples=500)
    return DriftClassifier(n_estimators=60, max_depth=8).fit(train.x, train.y, cal.x, cal.y)


@pytest.fixture(scope="session")
def rng():
    """A fixed generator, so every test that samples is reproducible."""
    return np.random.default_rng(53053)
