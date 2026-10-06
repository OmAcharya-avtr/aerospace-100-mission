"""Shared fixtures.

The reference channel path is session-scoped: generating it costs about 0.1 s on
the single contended core this suite runs on, and dozens of tests do not need
dozens of regenerations.
"""

from __future__ import annotations

import numpy as np
import pytest

from codedfade.channel import ChannelConfig, generate_amplitude
from codedfade.convolutional import ConvolutionalCode
from codedfade.reedsolomon import ReedSolomon

#: Reference channel: 1 Mbaud, 200-symbol correlation length, moderate turbulence.
REFERENCE_CONFIG = ChannelConfig(
    scintillation_index=0.6,
    correlation_time_s=2.0e-4,
    sample_rate_hz=1.0e6,
    marginal="lognormal",
    kernel="exp",
    seed=0,
)

#: Amplitude threshold used throughout the suite and by the X1 cross-check.
REFERENCE_THRESHOLD = 0.6


@pytest.fixture(scope="session")
def config() -> ChannelConfig:
    return REFERENCE_CONFIG


@pytest.fixture(scope="session")
def amplitude(config: ChannelConfig) -> np.ndarray:
    return generate_amplitude(config, 200_000)


@pytest.fixture(scope="session")
def rs_small() -> ReedSolomon:
    """RS(15,11) over GF(2^4), t = 2: small enough for exhaustive tests."""
    return ReedSolomon(15, 11, 4)


@pytest.fixture(scope="session")
def rs_work() -> ReedSolomon:
    """RS(31,21) over GF(2^5), t = 5: the code the link sweep uses."""
    return ReedSolomon(31, 21, 5)


@pytest.fixture(scope="session")
def conv() -> ConvolutionalCode:
    """The standard rate-1/2, K = 3, (0o7, 0o5) code."""
    return ConvolutionalCode()


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20261006)
