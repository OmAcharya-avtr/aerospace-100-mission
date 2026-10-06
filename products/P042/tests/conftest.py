"""Shared fixtures.

The MODCOD table is session-scoped: measuring it runs a Monte Carlo BER sweep
over four constellations, which costs about five seconds on the single contended
core this suite runs on, and a hundred tests do not need a hundred sweeps. The
fitted predictors are session-scoped for the same reason, and are deliberately
fitted with very few trees -- these tests check interfaces, shapes, invariants
and calibration *direction*, not model quality, which is what
``validation/validate_predictor.py`` is for.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from acmpilot.channel import ChannelConfig, snr_db_path
from acmpilot.modcod import ModcodTable, measure_thresholds
from acmpilot.predictor import GaussMarkovPredictor, QuantilePredictor, make_lag_features

#: The threshold table committed by ``validation/validate_modcod_thresholds.py``.
COMMITTED_TABLE = Path(__file__).resolve().parent.parent / "validation" / "modcod_thresholds.json"


@pytest.fixture(scope="session")
def measured():
    """``(table, curves)`` measured in-process with the shipped default seed."""
    return measure_thresholds(seed=20261006)


@pytest.fixture(scope="session")
def table(measured) -> ModcodTable:
    """The measured MODCOD table."""
    return measured[0]


@pytest.fixture(scope="session")
def curves(measured):
    """Measured uncoded BER curves keyed by modulation name."""
    return measured[1]


@pytest.fixture(scope="session")
def committed_table() -> ModcodTable | None:
    """The committed threshold table, or ``None`` if it is not present."""
    if not COMMITTED_TABLE.exists():
        return None
    return ModcodTable.load_json(COMMITTED_TABLE)


@pytest.fixture(scope="session")
def config() -> ChannelConfig:
    """The reference channel used throughout the tests."""
    return ChannelConfig(slot_s=1e-3, tau_c_s=10e-3, sigma_i2=0.5, mean_snr_db=14.0)


@pytest.fixture(scope="session")
def snr_path(config) -> np.ndarray:
    """A 6000-slot reference SNR path, seed 1."""
    return snr_db_path(config, 6000, 1)


@pytest.fixture(scope="session")
def fitted(config):
    """``(analytic, learned, delay_slots, n_lags)`` fitted cheaply for interface tests."""
    delay_slots = 10
    n_lags = 4
    train = np.concatenate([snr_db_path(config, 4000, s) for s in (101, 102)])
    features, target, _ = make_lag_features(train, delay_slots=delay_slots, n_lags=n_lags)
    analytic = GaussMarkovPredictor(delay_slots=delay_slots).fit(train)
    learned = QuantilePredictor(
        delay_slots=delay_slots, n_estimators=12, max_depth=3
    ).fit(features, target)
    return analytic, learned, delay_slots, n_lags


@pytest.fixture
def rng() -> np.random.Generator:
    """A fresh seeded generator per test."""
    return np.random.default_rng(12345)


@pytest.fixture
def tiny_table() -> ModcodTable:
    """A hand-built three-mode table with round thresholds, for known-answer tests.

    Thresholds 3, 5, 7 dB; spectral efficiencies 1, 2, 3 bit/symbol. Nothing here
    is measured -- the point is that the arithmetic of selection and accounting
    can be checked by hand.
    """
    from acmpilot.coding import ReedSolomonCode
    from acmpilot.modcod import Modcod

    modcods = (
        Modcod(name="A", modulation="BPSK", code=ReedSolomonCode(255, 255 - 2)),
        Modcod(name="B", modulation="QPSK", code=ReedSolomonCode(255, 255 - 2)),
        Modcod(name="C", modulation="8PSK", code=ReedSolomonCode(255, 255 - 2)),
    )
    return ModcodTable(
        modcods=modcods,
        thresholds_db=np.array([3.0, 5.0, 7.0]),
        threshold_sigma_db=np.array([0.0, 0.0, 0.0]),
        target_ber=1e-6,
        provenance="hand-built fixture, not measured",
    )
