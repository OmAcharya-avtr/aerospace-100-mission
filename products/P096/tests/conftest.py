"""Shared fixtures. The corpus is loaded once per session because it is 750 kB."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session", autouse=True)
def _repository_root() -> Path:
    """Run every test with the working directory at the repository root.

    The corpus loader takes relative paths by design, so that no absolute path
    is ever constructed or printed. Anchoring here on ``__file__`` means the
    suite passes from any directory pytest is invoked from.
    """
    os.chdir(ROOT)
    return ROOT


@pytest.fixture(scope="session")
def corpus():
    """The committed corpus, loaded once."""
    from handoverloss.corpus import load_corpus

    return load_corpus()


@pytest.fixture
def rng() -> np.random.Generator:
    """A fixed-seed generator, so every test that uses randomness is reproducible."""
    return np.random.default_rng(96_096)


@pytest.fixture
def worked_sequence():
    """The hand-calculated handover used by the known-answer tests.

    R_s = 2e8 bit/s, R_d = 1e9 bit/s, B = 3.2e10 bits, q0 = 2.9e10 bits,
    T_acq = 40 s, T_rst = 12 s, T_ramp = 60 s, H = 700 s.
    """
    from handoverloss.sequence import HandoverSequence

    return HandoverSequence(
        source_rate_bps=2.0e8,
        downlink_rate_bps=1.0e9,
        buffer_capacity_bits=3.2e10,
        initial_occupancy_bits=2.9e10,
        acquisition_s=40.0,
        restart_s=12.0,
        ramp_s=60.0,
        horizon_s=700.0,
    )
