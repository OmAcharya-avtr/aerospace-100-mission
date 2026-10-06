"""Analytic stream sampling, pattern enumeration and the noise scale."""

from __future__ import annotations

import numpy as np
import pytest

from slotsync.pulses import nyquist_raised_cosine, triangular
from slotsync.stream import (
    antipodal_patterns,
    ook_patterns,
    sample_matrix,
    sample_noise_sigma,
    sample_stream,
)


def test_isolated_symbol_reproduces_the_pulse() -> None:
    pulse = triangular(1.0)
    times = np.linspace(-1.5, 1.5, 61)
    values = sample_stream(times, np.array([1.0]), pulse, first_symbol_index=0)
    assert np.allclose(values, pulse.amplitude(times))


def test_superposition_hand_computed() -> None:
    # Triangle of half-width 1. Symbols a[0] = 1, a[1] = -1 at t = 0.5:
    # p(0.5) = 0.5 and p(-0.5) = 0.5, so x(0.5) = 1*0.5 + (-1)*0.5 = 0.
    pulse = triangular(1.0)
    value = sample_stream(np.array([0.5]), np.array([1.0, -1.0]), pulse)[0]
    assert value == pytest.approx(0.0, abs=1e-15)
    # At t = 0.25: p(0.25) = 0.75, p(-0.75) = 0.25 -> 0.75 - 0.25 = 0.5.
    assert sample_stream(np.array([0.25]), np.array([1.0, -1.0]), pulse)[0] == pytest.approx(0.5)


def test_symbol_offset_shifts_the_waveform() -> None:
    pulse = triangular(1.0)
    times = np.linspace(-1.0, 2.0, 41)
    direct = sample_stream(times, np.array([1.0]), pulse, symbol_offset=0.3)
    shifted = sample_stream(times - 0.3, np.array([1.0]), pulse)
    assert np.allclose(direct, shifted)


def test_sample_matrix_times_data_equals_sample_stream() -> None:
    pulse = nyquist_raised_cosine(0.5, 4.0)
    rng = np.random.default_rng(3)
    symbols = rng.choice([-1.0, 1.0], size=11)
    indices = np.arange(11, dtype=float)
    times = np.linspace(0.0, 10.0, 37)
    matrix = sample_matrix(times, indices, pulse)
    assert np.allclose(matrix @ symbols, sample_stream(times, symbols, pulse))


def test_pattern_enumeration_is_complete_and_distinct() -> None:
    for length in (1, 2, 5, 8):
        patterns = antipodal_patterns(length)
        assert patterns.shape == (2**length, length)
        assert len({tuple(row) for row in patterns}) == 2**length
        assert set(np.unique(patterns)) == {-1.0, 1.0}
        unipolar = ook_patterns(length)
        assert unipolar.shape == (2**length, length)
        assert set(np.unique(unipolar)) == {0.0, 1.0}


def test_pattern_enumeration_rejects_an_unreasonable_length() -> None:
    for factory in (antipodal_patterns, ook_patterns):
        with pytest.raises(ValueError, match=r"\[1, 20\]"):
            factory(21)
        with pytest.raises(ValueError, match=r"\[1, 20\]"):
            factory(0)


def test_noise_sigma_hand_computed() -> None:
    # 0 dB -> sigma = 1; 20 dB -> sigma = 0.1; -20 dB -> sigma = 10.
    assert sample_noise_sigma(0.0) == pytest.approx(1.0)
    assert sample_noise_sigma(20.0) == pytest.approx(0.1)
    assert sample_noise_sigma(-20.0) == pytest.approx(10.0)
    with pytest.raises(ValueError, match="finite"):
        sample_noise_sigma(float("inf"))


def test_non_one_dimensional_symbols_are_rejected() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        sample_stream(np.array([0.0]), np.zeros((2, 2)), triangular(1.0))
