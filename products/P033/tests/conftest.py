"""Shared fixtures. Sizes are chosen so the whole suite stays inside the
repository's compute budget: one CPU core, every test file under a minute."""

from __future__ import annotations

import numpy as np
import pytest

from edgeinfer.budget import Budget
from edgeinfer.dataset import hand_counted_cnn, hand_counted_mlp, random_cnn, random_mlp
from edgeinfer.roofline import DeviceModel


@pytest.fixture
def rng() -> np.random.Generator:
    """A fixed generator, so every test that uses it is reproducible."""
    return np.random.default_rng(20260401)


@pytest.fixture
def device() -> DeviceModel:
    """A declared device with round peaks, so hand arithmetic is easy.

    1 GFLOP/s and 1 GB/s put the ridge point at 1 FLOP/B exactly, so a node's
    bound side can be read off its arithmetic intensity.
    """
    return DeviceModel(
        name="test-device",
        peak_flops=1e9,
        peak_bandwidth_bytes_s=1e9,
        source="declared for tests; not a real device",
    )


@pytest.fixture
def hand_mlp():
    """The fixed MLP whose costs are hand-counted in ``test_ops.py``."""
    return hand_counted_mlp()


@pytest.fixture
def hand_cnn():
    """The fixed CNN whose costs are hand-counted in ``test_ops.py``."""
    return hand_counted_cnn()


@pytest.fixture
def small_mlp(rng: np.random.Generator):
    """A slightly larger MLP for integration paths."""
    return random_mlp(rng, "small_mlp", n_in=16, widths=(24, 12), n_out=4)


@pytest.fixture
def small_cnn(rng: np.random.Generator):
    """A slightly larger CNN for integration paths."""
    return random_cnn(rng, "small_cnn", spatial=12, channels=(4, 8), kernel=3, n_out=4)


@pytest.fixture
def loose_budget() -> Budget:
    """A budget nothing small will exceed."""
    return Budget(
        name="loose",
        latency_s=1.0,
        peak_memory_bytes=64 * 1024 * 1024,
        median_latency_s=0.5,
        power_w=7.0,
    )
