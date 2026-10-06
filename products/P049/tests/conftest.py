"""Shared fixtures.

The hand-checked series are tiny and written out literally so that every
known-answer test can be verified by eye from the test source.
"""

from __future__ import annotations

import numpy as np
import pytest

from linkoutage.channel import lognormal_amplitude_series


@pytest.fixture
def hand_series() -> np.ndarray:
    """``[1.0, 0.5, 0.5, 1.0, 1.0, 0.4, 1.0]``, threshold 0.6, fs 1.0 Hz.

    mask      = [F, T, T, F, F, T, F]
    runs      = [1,3) length 2;  [5,6) length 1
    down-crossings at 1 and 5  -> 2
    up-crossings   at 3 and 6  -> 2
    complete fades: both; durations 2.0 s and 1.0 s -> mean 1.5 s
    in-fade samples 3 of 7 -> outage 3/7
    record duration (N-1)/fs = 6.0 s -> LCR = 2/6 Hz
    """
    return np.array([1.0, 0.5, 0.5, 1.0, 1.0, 0.4, 1.0])


@pytest.fixture
def censored_series() -> np.ndarray:
    """``[0.4, 0.4, 1.0, 1.0, 0.3]``, threshold 0.6, fs 2.0 Hz.

    mask = [T, T, F, F, T]
    runs = [0,2) left-censored length 2;  [4,5) right-censored length 1
    down-crossings: only at index 4  -> 1  (index 0 has no predecessor)
    up-crossings  : only at index 2  -> 1
    complete fades: none -> mean fade duration is nan
    record duration = 4/2 = 2.0 s -> LCR = 0.5 Hz
    outage fraction = 3/5
    """
    return np.array([0.4, 0.4, 1.0, 1.0, 0.3])


@pytest.fixture(scope="session")
def short_channel():
    """A 200 000-sample realisation of the documented channel, seed 2049."""
    return lognormal_amplitude_series(
        200_000, fs_hz=1.0e6, tau_s=2.0e-4, si=0.6, seed=2049
    )
