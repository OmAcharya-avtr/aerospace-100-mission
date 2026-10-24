"""Baseline coder tests. Every size is a real compressed length, round-trip verified."""

from __future__ import annotations

import numpy as np
import pytest

from boundcompress import baselines
from boundcompress.container import FIXED_HEADER_BYTES
from boundcompress.predictors import get_predictor
from boundcompress.quantise import make_spec, quantise


@pytest.fixture
def pieces(signals: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = signals["battery_voltage"]
    spec = make_spec(values, "abs", 1e-3)
    symbols, _, _ = quantise(values, spec)
    residuals = get_predictor("linear").residuals(symbols)
    return values, symbols, residuals


def test_numcodecs_and_zfpy_are_not_installed() -> None:
    """Recorded as a test so the deviation cannot be quietly outgrown.

    The specification names ``numcodecs`` and ``zfpy`` as baseline 2. Neither
    is installed in the pinned environment and nothing was installed to get
    them, so no rate-distortion number in this repository comes from either.
    If a future environment does have them, this test fails and the README's
    alternatives table and VALIDATION.md must be revisited rather than left
    claiming an absence that no longer holds.
    """
    import importlib.util

    assert importlib.util.find_spec("numcodecs") is None
    assert importlib.util.find_spec("zfpy") is None


@pytest.mark.parametrize("method", baselines.LOSSY_METHODS + baselines.LOSSLESS_METHODS)
def test_every_baseline_round_trips(
    method: str, pieces: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> None:
    values, symbols, residuals = pieces
    result = baselines.measure(method, values=values, symbols=symbols, residuals=residuals)
    assert result.roundtrip_ok
    assert result.payload_bytes > 0
    assert result.total_bytes == result.payload_bytes + FIXED_HEADER_BYTES
    assert result.bits_per_sample == pytest.approx(
        8.0 * result.total_bytes / values.size, rel=1e-15
    )


def test_varint_stream_round_trips() -> None:
    rng = np.random.default_rng(13)
    residuals = rng.integers(-100_000, 100_000, size=1000)
    blob = baselines.varint_stream(residuals)
    assert np.array_equal(baselines.varint_unstream(blob, residuals.size), residuals)


def test_symbol_stream_is_eight_bytes_per_symbol() -> None:
    symbols = np.arange(10, dtype=np.int64)
    assert len(baselines.symbol_stream(symbols)) == 80


def test_unknown_method_is_refused(
    pieces: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> None:
    values, symbols, residuals = pieces
    with pytest.raises(baselines.BaselineError, match="unknown baseline method"):
        baselines.measure("brotli_residual", values=values, symbols=symbols,
                          residuals=residuals)


def test_measure_all_covers_every_declared_method(
    pieces: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> None:
    values, symbols, residuals = pieces
    results = baselines.measure_all(values=values, symbols=symbols, residuals=residuals)
    assert set(results) == set(baselines.LOSSY_METHODS + baselines.LOSSLESS_METHODS)
    assert all(result.roundtrip_ok for result in results.values())


def test_roundtrip_helper_agrees_with_measure(
    pieces: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> None:
    values, symbols, residuals = pieces
    assert baselines.roundtrip("zlib_residual", values=values, symbols=symbols,
                               residuals=residuals)


def test_residual_coders_beat_the_naive_symbol_coders(
    pieces: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> None:
    """Prediction plus varints beats int64 dumps. Measured, with a wide margin.

    Asserted as an inequality rather than a ratio, because the ratio is a
    measurement that moves with the signal and belongs in a table.
    """
    values, symbols, residuals = pieces
    results = baselines.measure_all(values=values, symbols=symbols, residuals=residuals)
    assert results["zlib_residual"].total_bytes < results["zlib_symbols"].total_bytes
    assert results["lzma_residual"].total_bytes < results["lzma_symbols"].total_bytes


def test_entropy_floor_bytes_known_answer() -> None:
    """4 bits per symbol over 1024 symbols is 512 bytes. Hand-calculated."""
    assert baselines.entropy_floor_bytes(4.0, 1024) == pytest.approx(512.0, rel=1e-15)
