"""Shared fixtures.  Sizes here are chosen to keep every test file inside the
30 s budget stated in the README compute section on a single contended core."""

from __future__ import annotations

import numpy as np
import pytest

from telemetryool.synthetic import NominalModel, equicorrelation


@pytest.fixture
def rng() -> np.random.Generator:
    """A fixed-seed generator, so every test that uses randomness is reproducible."""
    return np.random.default_rng(20261005)


@pytest.fixture
def iid_model() -> NominalModel:
    """Four independent unit-variance channels; the chart design hypothesis."""
    return NominalModel(4)


@pytest.fixture
def correlated_model() -> NominalModel:
    """Four channels with equicorrelation 0.6, no serial correlation."""
    return NominalModel(4, correlation=equicorrelation(4, 0.6))
