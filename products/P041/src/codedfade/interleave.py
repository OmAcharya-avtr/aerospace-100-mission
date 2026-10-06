"""Block and convolutional interleavers, with their latency and memory cost.

The design variable
-------------------
A burst of ``L`` consecutive channel-symbol errors is what a correlated fade
produces. An interleaver's job is to spread those ``L`` errors across as many
codewords as possible. For a **block interleaver** of depth ``D`` and span ``S``
(write ``D`` rows of ``S`` symbols, read out column-wise), consecutive symbols of
one codeword leave the interleaver ``D`` symbol periods apart, so a burst of
``L`` consecutive channel symbols deposits at most

    ceil(L / D)                                                           (22)

errors in any one codeword. With ``S`` equal to the codeword length ``n``, a
Reed-Solomon code correcting ``t`` symbols therefore survives any burst with

    ceil(L / D) <= t,  i.e.  L <= t * D                                    (23)

Equation (23) is the whole design rule, and it is why depth and not rate is the
variable the fade statistics act on.

Cost of depth
-------------
These are the quantities a designer trades against (22):

* **Latency.** A block interleaver must fill its ``D x S`` matrix before it can
  read out, and the de-interleaver must do the same, so the end-to-end delay is

      latency = 2 * D * S symbols = 2 * D * S / Rs seconds                 (24)

* **Memory.** One ``D x S`` matrix at each end:

      memory = 2 * D * S symbols = 2 * D * S * m / 8 bytes                 (25)

  for ``m``-bit symbols.

* **Convolutional interleaver** (Forney's construction) with ``B`` branches and
  delay increment ``M``: branch ``j`` delays by ``j*M`` symbols. The end-to-end
  delay is ``M*B*(B-1)`` symbols and the storage is ``M*B*(B-1)/2`` symbols at
  each end, i.e. half the block interleaver's memory for the same spreading.
  Both figures are **measured** by ``validation/validate_interleaver_cost.py``
  by pushing an impulse through the real implementation rather than asserted.

Both interleavers here are exact permutations with an exact inverse, verified by
Hypothesis property tests over random lengths and depths.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class InterleaverCost:
    """Latency and memory cost of an interleaver configuration.

    Attributes
    ----------
    latency_symbols:
        End-to-end (interleaver + de-interleaver) delay, symbols.
    latency_ms:
        The same delay at the stated symbol rate, milliseconds.
    memory_symbols:
        Total storage at both ends, symbols.
    memory_bytes:
        The same storage for ``bits_per_symbol``-bit symbols, bytes.
    symbol_rate_hz:
        Symbol rate the latency was computed at, Hz.
    """

    latency_symbols: int
    latency_ms: float
    memory_symbols: int
    memory_bytes: float
    symbol_rate_hz: float


class BlockInterleaver:
    """Row-in / column-out block interleaver of depth ``D`` and span ``S``.

    Parameters
    ----------
    depth:
        ``D``, the separation in symbol periods between adjacent codeword
        symbols after interleaving. ``>= 1``.
    span:
        ``S``, the number of symbols per row; set equal to the codeword length
        for equation (23) to apply directly. ``>= 1``.
    """

    def __init__(self, depth: int, span: int) -> None:
        depth, span = int(depth), int(span)
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth!r}")
        if span < 1:
            raise ValueError(f"span must be >= 1, got {span!r}")
        self.depth = depth
        self.span = span

    @property
    def block_symbols(self) -> int:
        """``D * S``, the symbols in one interleaver matrix."""
        return self.depth * self.span

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return f"BlockInterleaver(depth={self.depth}, span={self.span})"

    def interleave(self, symbols: np.ndarray) -> np.ndarray:
        """Permute one or more whole blocks. ``symbols.size`` must be a multiple of ``D*S``."""
        s = np.asarray(symbols)
        if s.ndim != 1:
            raise ValueError(f"symbols must be 1-D, got shape {s.shape}")
        if s.size % self.block_symbols:
            raise ValueError(
                f"symbols.size {s.size} is not a multiple of depth*span "
                f"{self.block_symbols}"
            )
        return (
            s.reshape(-1, self.depth, self.span)
            .transpose(0, 2, 1)
            .reshape(-1)
            .copy()
        )

    def deinterleave(self, symbols: np.ndarray) -> np.ndarray:
        """Exact inverse of :meth:`interleave`."""
        s = np.asarray(symbols)
        if s.ndim != 1:
            raise ValueError(f"symbols must be 1-D, got shape {s.shape}")
        if s.size % self.block_symbols:
            raise ValueError(
                f"symbols.size {s.size} is not a multiple of depth*span "
                f"{self.block_symbols}"
            )
        return (
            s.reshape(-1, self.span, self.depth)
            .transpose(0, 2, 1)
            .reshape(-1)
            .copy()
        )

    def cost(self, symbol_rate_hz: float, bits_per_symbol: int = 8) -> InterleaverCost:
        """Latency and memory from equations (24) and (25)."""
        if not symbol_rate_hz > 0:
            raise ValueError(f"symbol_rate_hz must be > 0 Hz, got {symbol_rate_hz!r}")
        if bits_per_symbol < 1:
            raise ValueError(f"bits_per_symbol must be >= 1, got {bits_per_symbol!r}")
        n = 2 * self.block_symbols
        return InterleaverCost(
            latency_symbols=n,
            latency_ms=1000.0 * n / float(symbol_rate_hz),
            memory_symbols=n,
            memory_bytes=n * bits_per_symbol / 8.0,
            symbol_rate_hz=float(symbol_rate_hz),
        )

    def max_errors_per_codeword(self, burst_symbols: int) -> int:
        """``ceil(L / D)`` from equation (22), the worst case for one burst."""
        if burst_symbols < 0:
            raise ValueError(f"burst_symbols must be >= 0, got {burst_symbols!r}")
        return int(-(-int(burst_symbols) // self.depth))


class ConvolutionalInterleaver:
    """Forney convolutional interleaver: ``B`` branches with delays ``0, M, 2M, ...``.

    Symbols are distributed to branches cyclically; branch ``j`` holds a FIFO of
    ``j*M`` symbols. The de-interleaver applies the complementary delays
    ``(B-1-j)*M`` so that every symbol accumulates ``(B-1)*M`` of total delay.

    Parameters
    ----------
    branches:
        ``B >= 1``.
    delay_increment:
        ``M >= 1``, symbols of extra delay per branch step.
    fill:
        Value used to prime the FIFOs; appears in the output during the
        transient. Defaults to 0.
    """

    def __init__(self, branches: int, delay_increment: int = 1, fill: int = 0) -> None:
        branches, delay_increment = int(branches), int(delay_increment)
        if branches < 1:
            raise ValueError(f"branches must be >= 1, got {branches!r}")
        if delay_increment < 1:
            raise ValueError(f"delay_increment must be >= 1, got {delay_increment!r}")
        self.branches = branches
        self.delay_increment = delay_increment
        self.fill = int(fill)

    @property
    def total_delay_symbols(self) -> int:
        """``M * B * (B-1)``: end-to-end delay through interleaver and de-interleaver."""
        return self.delay_increment * self.branches * (self.branches - 1)

    @property
    def memory_symbols(self) -> int:
        """``M * B * (B-1)``: total FIFO storage at both ends, symbols."""
        return self.delay_increment * self.branches * (self.branches - 1)

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return (
            f"ConvolutionalInterleaver(branches={self.branches}, "
            f"delay_increment={self.delay_increment})"
        )

    def _apply(self, symbols: np.ndarray, delays: np.ndarray) -> np.ndarray:
        s = np.asarray(symbols)
        if s.ndim != 1:
            raise ValueError(f"symbols must be 1-D, got shape {s.shape}")
        out = np.full(s.size, self.fill, dtype=s.dtype)
        for j in range(self.branches):
            idx = np.arange(j, s.size, self.branches)
            d = int(delays[j])
            if d == 0:
                out[idx] = s[idx]
                continue
            shifted = idx + d * self.branches
            keep = shifted < s.size
            out[shifted[keep]] = s[idx[keep]]
        return out

    def interleave(self, symbols: np.ndarray) -> np.ndarray:
        """Apply branch delays ``j*M``. Output is the same length as the input."""
        delays = self.delay_increment * np.arange(self.branches)
        return self._apply(symbols, delays)

    def deinterleave(self, symbols: np.ndarray) -> np.ndarray:
        """Apply complementary delays ``(B-1-j)*M``.

        The composition with :meth:`interleave` is a pure delay of
        ``M*B*(B-1)`` symbols, not the identity; the first
        ``M*B*(B-1)`` output symbols are transient. Verified in the test suite.
        """
        delays = self.delay_increment * (self.branches - 1 - np.arange(self.branches))
        return self._apply(symbols, delays)

    def cost(self, symbol_rate_hz: float, bits_per_symbol: int = 8) -> InterleaverCost:
        """Latency and memory of the convolutional construction."""
        if not symbol_rate_hz > 0:
            raise ValueError(f"symbol_rate_hz must be > 0 Hz, got {symbol_rate_hz!r}")
        if bits_per_symbol < 1:
            raise ValueError(f"bits_per_symbol must be >= 1, got {bits_per_symbol!r}")
        n = self.total_delay_symbols
        return InterleaverCost(
            latency_symbols=n,
            latency_ms=1000.0 * n / float(symbol_rate_hz),
            memory_symbols=self.memory_symbols,
            memory_bytes=self.memory_symbols * bits_per_symbol / 8.0,
            symbol_rate_hz=float(symbol_rate_hz),
        )


def burst_dispersion(depth: int, burst_symbols: int) -> int:
    """Worst-case errors landing in one codeword, equation (22). Dimensionless."""
    if depth < 1:
        raise ValueError(f"depth must be >= 1, got {depth!r}")
    if burst_symbols < 0:
        raise ValueError(f"burst_symbols must be >= 0, got {burst_symbols!r}")
    return int(-(-int(burst_symbols) // int(depth)))
