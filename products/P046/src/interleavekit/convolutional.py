"""Convolutional (shift-register bank) interleaver.

This construction is **not** a permutation of a fixed block, so it does not
subclass :class:`interleavekit.base.Interleaver`.  It is a continuous delay
structure, and its index map is an injection of the source stream into a longer
transmitted stream.

Structure
---------
``registers`` shift registers and a commutator.  Register ``k`` (0-based) holds
``k * slope`` symbols.  With each new source symbol the commutator advances to
the next register, shifts the new symbol in and shifts that register's oldest
symbol out; after register ``registers - 1`` it returns to register 0.

A register is therefore clocked **once every ``registers`` symbol times**, so a
register holding ``k * slope`` symbols delays a symbol by
``k * slope * registers`` symbol times, not by ``k * slope``.  That factor of
``registers`` is the one easy thing to get wrong here, and it is why the index
map is

    transmitted_position(i) = i + (i % registers) * slope * registers

De-interleaver and end-to-end delay
-----------------------------------
The matched de-interleaver is the same bank with the delays reversed: its
register ``k`` holds ``(registers - 1 - k) * slope`` symbols.  Because
``slope * registers`` is a multiple of ``registers``, a symbol that entered the
interleaver at time ``i`` reaches the de-interleaver's commutator at a time
congruent to ``i`` modulo ``registers``, so it meets the complementary delay and
the end-to-end delay is the same for every symbol:

    total delay = i + (i % R) * slope * R + (R - 1 - i % R) * slope * R
                = i + R * slope * (R - 1)

independent of ``i``, with ``R = registers``.  The constant
``registers * slope * (registers - 1)`` agrees with the total delay given for
the MathWorks Communications Toolbox ``Convolutional Interleaver`` block,
``N x slope x (N - 1)`` for ``N`` rows of shift registers (documentation read on
2026-10-06).  ``validation/validate_convolutional_delay.py`` checks the constant
delay and the agreement by running the pair.

Memory
------
The interleaver's registers hold ``sum_k k * slope = slope * R * (R - 1) / 2``
symbols; the de-interleaver holds the same, so the pair holds
``slope * R * (R - 1)`` symbols.  For this construction the pair memory in
symbols equals the pair latency in symbol times.

Fill symbols and the steady state
---------------------------------
At start-up the transmitted stream contains fill symbols, because the longer
registers have not yet produced real data; the same happens in reverse while the
bank flushes.  In continuous operation there are none: every transmitted
position in :meth:`ConvolutionalInterleaver.steady_state_range` carries a real
source symbol.  The burst metrics must be evaluated on that range only --
measuring a burst across the start-up fill would credit the interleaver for
dispersing errors that landed on symbols nobody sent.

References
----------
Ramsey, J. L., "Realization of Optimum Interleavers", *IEEE Transactions on
Information Theory*, IT-16 (3), May 1970, pp. 338-345, and Forney, G. D., Jr.,
"Burst-Correcting Codes for the Classic Bursty Channel", *IEEE Transactions on
Communications*, vol. COM-19, October 1971, pp. 772-781, are the standard
references for this class of interleaver.  (Bibliographic details for both read
from the Communications Blockset bibliography on 2026-10-06; neither primary
article was consulted for this build, and no numerical result is attributed to
either.  The delay and memory expressions above are derived in this docstring
and verified by running the construction.)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .base import InterleaverCost, check_length

__all__ = ["ConvolutionalInterleaver"]


class ConvolutionalInterleaver:
    """Shift-register-bank convolutional interleaver.

    Parameters
    ----------
    registers:
        Number of shift registers ``R``, i.e. the commutator period, in symbols.
        Must be >= 1.
    slope:
        Register length step, in symbols: register ``k`` holds ``k * slope``
        symbols.  Must be >= 0.  ``slope = 0`` makes every register empty and the
        interleaver the identity.

    Raises
    ------
    TypeError
        If a parameter is not an integer.
    ValueError
        If ``registers < 1`` or ``slope < 0``.

    Examples
    --------
    >>> import numpy as np
    >>> ci = ConvolutionalInterleaver(registers=3, slope=1)
    >>> ci.register_delays().tolist()
    [0, 1, 2]
    >>> ci.transmitted_position(np.arange(6)).tolist()
    [0, 4, 8, 3, 7, 11]
    >>> x = np.arange(1, 13)
    >>> bool(np.array_equal(ci.deinterleave(ci.interleave(x)), x))
    True
    """

    def __init__(self, registers: int, slope: int = 1) -> None:
        self._registers = check_length(registers, "registers")
        if isinstance(slope, bool) or not isinstance(slope, (int, np.integer)):
            raise TypeError(f"slope must be an integer, got {type(slope).__name__}")
        slope = int(slope)
        if slope < 0:
            raise ValueError(f"slope must be >= 0 symbols, got {slope}")
        self._slope = slope

    @property
    def registers(self) -> int:
        """Number of shift registers ``R``."""
        return self._registers

    @property
    def slope(self) -> int:
        """Register length step, in symbols."""
        return self._slope

    def register_delays(self) -> NDArray[np.int64]:
        """Register contents, in symbols: ``k * slope`` for ``k = 0 .. R-1``."""
        return np.arange(self._registers, dtype=np.int64) * self._slope

    @property
    def max_delay_symbols(self) -> int:
        """Largest per-symbol delay through the interleaver, in symbol times.

        ``(R - 1) * slope * R``.  This is also the length of the start-up region
        of the transmitted stream that contains fill symbols.
        """
        r = self._registers
        return (r - 1) * self._slope * r

    def transmitted_position(self, source_index: NDArray | int) -> NDArray[np.int64]:
        """Transmitted position of each source index.

        Parameters
        ----------
        source_index:
            Source index or array of source indices, >= 0.

        Returns
        -------
        numpy.ndarray
            ``i + (i % registers) * slope * registers``, dtype ``int64``.

        Raises
        ------
        ValueError
            If any index is negative.
        """
        i = np.asarray(source_index, dtype=np.int64)
        if i.size and int(i.min()) < 0:
            raise ValueError("source_index must be >= 0")
        return i + (i % self._registers) * self._slope * self._registers

    def transmitted_length(self, n_symbols: int) -> int:
        """Transmitted-stream length needed to flush ``n_symbols`` source symbols.

        Returns
        -------
        int
            ``n_symbols + max_delay_symbols``, in symbols.
        """
        n = check_length(n_symbols, "n_symbols")
        return n + self.max_delay_symbols

    def steady_state_range(self, n_symbols: int) -> tuple[int, int]:
        """Transmitted positions that are guaranteed to carry real source symbols.

        Parameters
        ----------
        n_symbols:
            Number of source symbols fed in.

        Returns
        -------
        tuple of int
            Half-open range ``(lo, hi)`` of transmitted positions in which every
            position carries a real source symbol: ``lo = max_delay_symbols`` and
            ``hi = n_symbols``.  Empty (``lo >= hi``) when the stream is shorter
            than the start-up transient.

        Notes
        -----
        Burst metrics for this construction must use windows inside this range;
        see the module docstring.
        """
        n = check_length(n_symbols, "n_symbols")
        return self.max_delay_symbols, n

    def cost(self) -> InterleaverCost:
        """Exact shift-register-bank cost, in symbols.

        Returns
        -------
        InterleaverCost
            One-way latency ``max_delay_symbols``; pair latency
            ``registers * slope * (registers - 1)``; one-way memory
            ``slope * registers * (registers - 1) / 2``; pair memory twice that.

        Notes
        -----
        Derived in the module docstring; not a convention.  The one-way latency
        is the worst-case per-symbol delay, which is the longest register's; the
        pair latency is the same for every symbol.
        """
        r, m = self._registers, self._slope
        one_way_mem = m * r * (r - 1) // 2
        pair_latency = r * m * (r - 1)
        return InterleaverCost(
            one_way_latency_symbols=(r - 1) * m * r,
            pair_latency_symbols=pair_latency,
            one_way_memory_symbols=one_way_mem,
            pair_memory_symbols=2 * one_way_mem,
            model="shift-register bank",
        )

    def interleave(self, data: NDArray, fill: object = 0) -> NDArray:
        """Interleave a source stream, returning the padded transmitted stream.

        Parameters
        ----------
        data:
            One-dimensional source stream of >= 1 symbols.
        fill:
            Value written at transmitted positions that no source symbol has
            reached yet.  Cast to ``data``'s dtype.

        Returns
        -------
        numpy.ndarray
            Length :meth:`transmitted_length`, same dtype as ``data``.

        Raises
        ------
        ValueError
            If ``data`` is not one-dimensional or is empty.
        """
        arr = np.asarray(data)
        if arr.ndim != 1:
            raise ValueError(
                f"ConvolutionalInterleaver takes a 1-D stream, got {arr.ndim} dimensions"
            )
        if arr.size == 0:
            raise ValueError("ConvolutionalInterleaver needs at least 1 symbol, got an empty array")
        return self._shift_bank(arr, self.register_delays(), fill)

    def deinterleave(self, stream: NDArray, fill: object = 0) -> NDArray:
        """Invert :meth:`interleave`, returning the recovered source stream.

        Parameters
        ----------
        stream:
            Transmitted stream as produced by :meth:`interleave`, i.e. of length
            ``n_symbols + max_delay_symbols``.
        fill:
            Fill value used inside the matched de-interleaver's registers.

        Returns
        -------
        numpy.ndarray
            The recovered source stream, of length
            ``len(stream) - max_delay_symbols``.  Exactly equal to the original
            ``data`` for a stream produced by :meth:`interleave`.

        Raises
        ------
        ValueError
            If ``stream`` is not one-dimensional, or is no longer than
            ``max_delay_symbols``, in which case no source symbol has completed
            the round trip.
        """
        arr = np.asarray(stream)
        if arr.ndim != 1:
            raise ValueError(
                f"ConvolutionalInterleaver takes a 1-D stream, got {arr.ndim} dimensions"
            )
        total = self.cost().pair_latency_symbols
        n_out = arr.size - self.max_delay_symbols
        if n_out < 1:
            raise ValueError(
                f"stream of {arr.size} symbols is too short to recover anything: the start-up "
                f"transient alone is {self.max_delay_symbols} symbols, so feed at least "
                f"{self.max_delay_symbols + 1}"
            )
        reversed_delays = self.register_delays()[::-1]
        # Pad so the de-interleaver's own longest register can flush.
        padded = np.concatenate([arr, np.full(total, fill, dtype=arr.dtype)])
        out = self._shift_bank(padded, reversed_delays, fill)
        return out[total : total + n_out]

    def _shift_bank(
        self, data: NDArray, delays: NDArray[np.int64], fill: object
    ) -> NDArray:
        """Run a commutated shift-register bank over ``data``.

        Output position ``t`` carries ``data[t - delays[t % R] * R]`` when that
        index is in range, and ``fill`` otherwise.
        """
        r = self._registers
        n = data.size
        out_len = n + int(delays.max(initial=0)) * r
        t = np.arange(out_len, dtype=np.int64)
        src = t - delays[t % r] * r
        valid = (src >= 0) & (src < n)
        out = np.full(out_len, fill, dtype=data.dtype)
        out[valid] = data[src[valid]]
        return out

    def __repr__(self) -> str:
        return f"ConvolutionalInterleaver(registers={self._registers}, slope={self._slope})"
