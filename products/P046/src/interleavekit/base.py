"""Common interleaver interface, permutation conventions and cost records.

Permutation convention used throughout this package
---------------------------------------------------
An interleaver of length ``N`` is a permutation ``pi`` of ``{0, ..., N-1}`` with

    y[i] = x[pi[i]]        for i in 0 .. N-1

so ``pi[i]`` is the *source* index that lands at transmitted position ``i``.
The inverse map, ``position_of_input``, satisfies

    position_of_input[pi[i]] = i

and is therefore ``numpy.argsort(pi)``.  ``position_of_input[j]`` is the
transmitted position of source symbol ``j``; it is the array the burst metrics
in :mod:`interleavekit.metrics` consume, because a channel burst is contiguous
in the *transmitted* stream, not in the source stream.

De-interleaving recovers ``x`` from ``y`` by ``x = y[position_of_input]``.

Units
-----
All latencies and memories in this module are counted in **symbols** (one
symbol = one element of the interleaved sequence), never in bits and never in
seconds.  Convert to milliseconds with :meth:`InterleaverCost.latency_ms`,
which needs the symbol rate in symbols per second.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["Interleaver", "InterleaverCost", "check_length"]


def check_length(n: int, name: str = "length") -> int:
    """Validate a sequence length.

    Parameters
    ----------
    n:
        Candidate length, in symbols.
    name:
        Parameter name used in the error message.

    Returns
    -------
    int
        ``n`` as a plain ``int``.

    Raises
    ------
    TypeError
        If ``n`` is not an integer.
    ValueError
        If ``n < 1``.
    """
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(n).__name__}")
    n = int(n)
    if n < 1:
        raise ValueError(f"{name} must be >= 1 symbol, got {n}")
    return n


@dataclass(frozen=True)
class InterleaverCost:
    """Latency and memory cost of one interleaver and of the matched pair.

    Every field is a count of **symbols**.  ``one_way_*`` is the cost at the
    transmitter alone; ``pair_*`` is the cost of the interleaver plus its
    matched de-interleaver, which is what a link budget actually pays.

    The cost model for each construction is stated in that construction's
    ``cost()`` docstring.  The two models used in this package are:

    * **Full-block buffering** (block, helical, S-random): a fixed permutation
      of a block of ``N`` symbols is realised by buffering the block, so each
      end holds ``N`` symbols and the pair costs ``2N`` symbol times.  An
      implementation that begins reading out before the block completes can do
      better; this package reports the conventional full-block figure and says
      so in the README limitations.
    * **Shift-register bank** (convolutional): the delays are exact, not
      conventional, and are derived in
      :class:`interleavekit.convolutional.ConvolutionalInterleaver`.

    Attributes
    ----------
    one_way_latency_symbols:
        Symbol times from a symbol entering the interleaver to it leaving.
    pair_latency_symbols:
        Symbol times from a symbol entering the interleaver to it leaving the
        matched de-interleaver.  This is the end-to-end latency added to the
        link.
    one_way_memory_symbols:
        Symbol storage held by the interleaver.
    pair_memory_symbols:
        Symbol storage held by the interleaver plus the de-interleaver.
    model:
        Short name of the cost model, for traceability in printed tables.
    """

    one_way_latency_symbols: int
    pair_latency_symbols: int
    one_way_memory_symbols: int
    pair_memory_symbols: int
    model: str

    def latency_ms(self, symbol_rate_hz: float, pair: bool = True) -> float:
        """Latency in milliseconds at a stated symbol rate.

        Parameters
        ----------
        symbol_rate_hz:
            Symbol rate in symbols per second.  Must be strictly positive.
        pair:
            If ``True`` (default) convert :attr:`pair_latency_symbols`,
            otherwise :attr:`one_way_latency_symbols`.

        Returns
        -------
        float
            Latency in milliseconds, ``symbols / symbol_rate_hz * 1e3``.

        Raises
        ------
        ValueError
            If ``symbol_rate_hz <= 0`` or is not finite.
        """
        rate = float(symbol_rate_hz)
        if not np.isfinite(rate) or rate <= 0.0:
            raise ValueError(
                "symbol_rate_hz must be a finite positive rate in symbols/s, got "
                f"{symbol_rate_hz!r}"
            )
        symbols = self.pair_latency_symbols if pair else self.one_way_latency_symbols
        return symbols / rate * 1e3

    def memory_bytes(self, bits_per_symbol: int, pair: bool = True) -> float:
        """Storage in bytes for a stated symbol width.

        Parameters
        ----------
        bits_per_symbol:
            Stored width of one symbol, in bits.  Must be >= 1.
        pair:
            If ``True`` (default) use :attr:`pair_memory_symbols`.

        Returns
        -------
        float
            Storage in bytes, ``symbols * bits_per_symbol / 8``.

        Raises
        ------
        ValueError
            If ``bits_per_symbol < 1``.
        """
        bits = check_length(bits_per_symbol, "bits_per_symbol")
        symbols = self.pair_memory_symbols if pair else self.one_way_memory_symbols
        return symbols * bits / 8.0


class Interleaver(abc.ABC):
    """Abstract base class for a fixed-length block interleaver.

    A subclass supplies :attr:`length`, :meth:`permutation` and :meth:`cost`.
    Everything else -- interleaving, de-interleaving, the inverse map and the
    round-trip guarantee -- follows from the permutation.

    Streaming constructions whose output is not a permutation of a contiguous
    index range (the convolutional interleaver) do **not** subclass this; see
    :class:`interleavekit.convolutional.ConvolutionalInterleaver`.
    """

    @property
    @abc.abstractmethod
    def length(self) -> int:
        """Block length ``N``, in symbols."""

    @abc.abstractmethod
    def permutation(self) -> NDArray[np.int64]:
        """The permutation ``pi`` with ``y[i] = x[pi[i]]``.

        Returns
        -------
        numpy.ndarray
            Shape ``(length,)``, dtype ``int64``, a permutation of
            ``0 .. length-1``.
        """

    @abc.abstractmethod
    def cost(self) -> InterleaverCost:
        """Latency and memory cost of this construction, in symbols."""

    def position_of_input(self) -> NDArray[np.int64]:
        """Transmitted position of each source symbol.

        Returns
        -------
        numpy.ndarray
            Shape ``(length,)``, dtype ``int64``.  Element ``j`` is the
            transmitted position of source symbol ``j``, i.e. the inverse of
            :meth:`permutation`.
        """
        return np.argsort(self.permutation()).astype(np.int64)

    def interleave(self, data: NDArray) -> NDArray:
        """Permute a block into transmission order.

        Parameters
        ----------
        data:
            Array whose first axis has length :attr:`length`.  Any dtype; any
            trailing axes are carried through unchanged.

        Returns
        -------
        numpy.ndarray
            ``data[permutation()]``.

        Raises
        ------
        ValueError
            If the first axis of ``data`` is not exactly :attr:`length`.  The
            message names both lengths, because a length that is not a multiple
            of the block is the most common caller error.
        """
        arr = np.asarray(data)
        self._check_block(arr)
        return arr[self.permutation()]

    def deinterleave(self, data: NDArray) -> NDArray:
        """Invert :meth:`interleave`.

        Parameters
        ----------
        data:
            Array whose first axis has length :attr:`length`.

        Returns
        -------
        numpy.ndarray
            ``data[position_of_input()]``, so that
            ``deinterleave(interleave(x)) is elementwise equal to x``.

        Raises
        ------
        ValueError
            If the first axis of ``data`` is not exactly :attr:`length`.
        """
        arr = np.asarray(data)
        self._check_block(arr)
        return arr[self.position_of_input()]

    def _check_block(self, arr: NDArray) -> None:
        n = self.length
        if arr.ndim == 0:
            raise ValueError(f"expected an array with first axis of {n} symbols, got a scalar")
        got = arr.shape[0]
        if got != n:
            remainder = got % n
            hint = (
                f"; {got} is not a multiple of the block length {n}"
                if remainder
                else f"; split the input into {got // n} blocks of {n} and interleave each"
            )
            raise ValueError(
                f"expected exactly {n} symbols on the first axis, got {got}{hint}"
            )

    def __len__(self) -> int:
        return self.length
