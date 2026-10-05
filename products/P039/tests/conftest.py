"""Shared fixtures. Every dataset built here is deliberately tiny.

The test suite runs on a single shared CPU core, so no test may build the
population sizes the validation scripts use. The sizes here are the smallest
that still exercise the code path; statistical claims are made in
``validation/``, not here.
"""

from __future__ import annotations

import numpy as np
import pytest

from latencynet.dataset import build_dataset
from latencynet.pipeline import make_lognormal_pipeline


@pytest.fixture(scope="session")
def three_stage_spec():
    """The P033 cross-check pipeline: 3 lognormal stages, independent."""
    return make_lognormal_pipeline(
        (60e-6, 180e-6, 30e-6), (12e-6, 40e-6, 6e-6), latent_rho=0.0
    )


@pytest.fixture(scope="session")
def probe_trace(three_stage_spec):
    """A 256-pass probe trace of the three-stage pipeline."""
    from latencynet.pipeline import sample_stage_latencies

    return sample_stage_latencies(three_stage_spec, 256, 7)


@pytest.fixture(scope="session")
def tiny_dataset():
    """A 60-pipeline correlated population, reference samples kept small."""
    return build_dataset(
        "correlated",
        n_train=32,
        n_calibration=14,
        n_test=14,
        seed=11,
        n_probe=64,
        n_reference=3000,
        n_reference_test=3000,
    )


@pytest.fixture(scope="session")
def rng():
    """A fixed generator for tests that need one."""
    return np.random.default_rng(2026)
