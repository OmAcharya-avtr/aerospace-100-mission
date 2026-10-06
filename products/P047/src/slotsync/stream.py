"""Analytic sampling of a pulse-amplitude-modulated stream.

The received waveform is modelled as

    ``x(t) = sum_m a_m * p(t - m - tau)``

with ``t`` and ``tau`` in symbol periods, ``a_m`` the data symbols and ``p`` a
finite-support shape from :mod:`slotsync.pulses`.  Samples are evaluated
**analytically at the requested instants** rather than by interpolating a dense
oversampled waveform.  That choice matters: interpolation would add its own
error and would colour the additive noise, and both would contaminate the
loop-theory comparison that this package exists to make.  The cost is that the
channel is a pure pulse-shaping channel - no filter, no dispersion - which is
stated as a limitation in the README.

Noise model
-----------
Additive white Gaussian noise is added **independently to each evaluated
sample**, with standard deviation ``sigma`` set by
:func:`sample_noise_sigma` from a sample signal-to-noise ratio.  This is the
model in which the classical loop-noise expression is derived: the detector sees
a signal term plus an independent Gaussian term per sample.  It is valid when
the detector's sample spacing is at least the correlation time of the receiver
filter's output noise; for closely spaced early and late gates on a real
matched-filter output the noise samples are correlated and this model
understates the correlation.  That is a documented limitation, not a hidden
assumption.

``sample_snr_db`` is defined as ``10 log10(1 / sigma^2)`` with the pulse
normalised to unit peak, i.e. it is the ratio of the **peak pulse amplitude
squared** to the per-sample noise variance.  It is *not* ``Es/N0`` on an optical
channel: converting to ``Es/N0`` requires the receiver filter and the photon
statistics, neither of which this package models, and no such conversion is
claimed anywhere.
"""

from __future__ import annotations

import numpy as np

from .pulses import PulseShape

__all__ = [
    "antipodal_patterns",
    "ook_patterns",
    "sample_matrix",
    "sample_noise_sigma",
    "sample_stream",
]


def sample_stream(
    times: np.ndarray,
    symbols: np.ndarray,
    pulse: PulseShape,
    *,
    first_symbol_index: int = 0,
    symbol_offset: float = 0.0,
) -> np.ndarray:
    """Evaluate ``x(t) = sum_m a_m p(t - m - symbol_offset)`` at ``times``.

    Parameters
    ----------
    times
        Sample instants in symbol periods.
    symbols
        Data symbols; ``symbols[i]`` is the symbol at index
        ``first_symbol_index + i``.
    pulse
        Pulse shape.
    first_symbol_index
        Index of ``symbols[0]``.
    symbol_offset
        True timing offset ``tau`` of the transmitter clock, symbol periods.

    Returns
    -------
    numpy.ndarray
        Waveform samples, dimensionless, same shape as ``times``.
    """
    t = np.asarray(times, dtype=float)
    a = np.asarray(symbols, dtype=float)
    if a.ndim != 1:
        raise ValueError(f"symbols must be one-dimensional, got shape {a.shape}")
    indices = first_symbol_index + np.arange(a.size)
    delays = t[..., None] - indices - symbol_offset
    return np.tensordot(pulse.amplitude(delays), a, axes=([-1], [0]))


def sample_matrix(
    times: np.ndarray,
    symbol_indices: np.ndarray,
    pulse: PulseShape,
    *,
    symbol_offset: float = 0.0,
) -> np.ndarray:
    """Matrix ``M[..., m] = p(times[...] - symbol_indices[m] - symbol_offset)``.

    Separating the pulse evaluation from the data makes an exact ensemble
    average over an enumerated set of data patterns a single matrix product;
    :func:`slotsync.scurve.scurve` relies on that.
    """
    t = np.asarray(times, dtype=float)
    m = np.asarray(symbol_indices, dtype=float)
    return pulse.amplitude(t[..., None] - m - symbol_offset)


def antipodal_patterns(length: int) -> np.ndarray:
    """All ``2**length`` antipodal (``+-1``) data patterns, shape ``(2**length, length)``.

    Raises
    ------
    ValueError
        If ``length`` exceeds 20, which would need more than a million rows.
    """
    if not 1 <= length <= 20:
        raise ValueError(f"length must lie in [1, 20], got {length}")
    bits = ((np.arange(1 << length)[:, None] >> np.arange(length)) & 1).astype(float)
    return 2.0 * bits - 1.0


def ook_patterns(length: int) -> np.ndarray:
    """All ``2**length`` unipolar on-off (``0``/``1``) patterns."""
    if not 1 <= length <= 20:
        raise ValueError(f"length must lie in [1, 20], got {length}")
    return ((np.arange(1 << length)[:, None] >> np.arange(length)) & 1).astype(float)


def sample_noise_sigma(sample_snr_db: float) -> float:
    """Per-sample noise standard deviation from the sample SNR in dB.

    ``sigma = 10**(-sample_snr_db / 20)`` with the pulse peak normalised to 1.
    See the module docstring for what this SNR is and is not.
    """
    if not np.isfinite(sample_snr_db):
        raise ValueError(f"sample_snr_db must be finite, got {sample_snr_db!r}")
    return float(10.0 ** (-sample_snr_db / 20.0))
