"""Shared fixtures. The learned models are session-scoped because fitting costs seconds."""

from __future__ import annotations

import numpy as np
import pytest

from boundcompress import dataset, learned
from boundcompress.quantise import make_spec, quantise

#: Short arrays keep the suite inside the compute budget while still exercising
#: every predictor's warm-up and every escape path.
TEST_SAMPLES = 1024


@pytest.fixture(scope="session")
def signals() -> dict[str, np.ndarray]:
    """Every built-in signal at :data:`TEST_SAMPLES` samples, default seeds."""
    return {
        name: signal.values
        for name, signal in dataset.generate_all(samples=TEST_SAMPLES).items()
    }


@pytest.fixture(scope="session")
def positive_signals(signals: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Only the strictly positive signals, which are the ones ``mode='rel'`` accepts."""
    return {name: signals[name] for name in dataset.POSITIVE_SIGNALS}


@pytest.fixture(scope="session")
def training_symbols() -> list[np.ndarray]:
    """Quantised symbol arrays from the disjoint training realisations."""
    arrays: list[np.ndarray] = []
    for signal in dataset.training_signals(samples=512).values():
        span = float(signal.values.max() - signal.values.min())
        for fraction in (1e-3, 1e-4):
            spec = make_spec(signal.values, "abs", fraction * span)
            symbols, _, _ = quantise(signal.values, spec)
            arrays.append(symbols)
    return arrays


@pytest.fixture(scope="session")
def learned_linear(training_symbols: list[np.ndarray]) -> learned.LearnedPredictor:
    """Fitted ridge ensemble. Session-scoped: fitting is the expensive part."""
    return learned.LearnedPredictor(kind="linear", members=3, max_rows=4000).fit(
        training_symbols
    )


@pytest.fixture(scope="session")
def learned_mlp(training_symbols: list[np.ndarray]) -> learned.LearnedPredictor:
    """Fitted perceptron ensemble, deliberately small so the suite stays fast."""
    return learned.LearnedPredictor(
        kind="mlp", members=2, hidden=8, max_iter=120, max_rows=4000
    ).fit(training_symbols)
