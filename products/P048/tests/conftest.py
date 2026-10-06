"""Shared fixtures. Seeds are fixed so every test is deterministic."""

from __future__ import annotations

import numpy as np
import pytest

from softdecode.channel import GammaGammaFading, LognormalFading
from softdecode.csi import MultiplicativeCsiError
from softdecode.detection import DetectionModel
from softdecode.ldpc import make_regular_ldpc


@pytest.fixture(scope="session")
def lognormal() -> LognormalFading:
    return LognormalFading(0.3)


@pytest.fixture(scope="session")
def gammagamma() -> GammaGammaFading:
    return GammaGammaFading(4.0, 2.0)


@pytest.fixture(scope="session")
def detection() -> DetectionModel:
    return DetectionModel(1.0)


@pytest.fixture(scope="session")
def code():
    return make_regular_ldpc()


@pytest.fixture(scope="session")
def csi_error() -> MultiplicativeCsiError:
    return MultiplicativeCsiError(2.0, 1.0)


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(20261006)
