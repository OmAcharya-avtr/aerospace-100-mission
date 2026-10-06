"""Classical block (matrix) interleaver.

Construction
------------
A block interleaver of ``depth`` rows and ``span`` columns handles blocks of
``N = depth * span`` symbols.  The source block is written into the array **row
by row** and read out **column by column**::

    depth = 4, span = 4, N = 16

    write (source indices)        read order (column by column)
      0  1  2  3                    0  4  8 12  1  5  9 13
      4  5  6  7                    2  6 10 14  3  7 11 15
      8  9 10 11
     12 13 14 15

so the transmitted sequence is source indices
``[0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15]``, which is the
permutation this class returns.

In closed form, writing ``i = q * depth + r`` with ``0 <= r < depth`` and
``0 <= q < span``::

    pi[i] = (i % depth) * span + (i // depth) = r * span + q

Why the two parameters are named this way
-----------------------------------------
* ``depth`` is the number of rows, and it is the **longest channel burst that
  this interleaver fully disperses**: a burst of ``depth`` consecutive
  transmitted symbols takes at most one symbol from each source row, so no two
  errored source symbols are adjacent.  Verified in
  ``validation/validate_burst_dispersion.py``.
* ``span`` is the number of columns, and it is the **source-index separation
  between adjacent transmitted symbols**: ``pi[i+1] - pi[i] = span`` whenever
  ``i`` and ``i+1`` fall in the same column.

Minimum spread
--------------
:meth:`BlockInterleaver.minimum_spread_closed_form` gives the exact minimum
spread of this construction without enumerating pairs.  With the parametrisation
above, a pair of source positions differing by ``(dq, dr)`` has spread
``|depth*dq + dr| + |span*dr + dq|``, which is minimised by one of
``(0, 1)``, ``(1, 0)`` and ``(1, -1)``, giving

    min_spread = min(span + 1, depth + 1, depth + span - 2)   for depth, span >= 2
    min_spread = 2                                            if depth == 1 or span == 1

because ``depth == 1`` or ``span == 1`` makes the permutation the identity, whose
minimum spread is 2.  The closed form is checked against brute-force enumeration
over every ``(depth, span)`` pair with ``depth * span <= 400`` in
``validation/validate_block_spread.py``.

References
----------
Block interleaving with row-wise write and column-wise read is the textbook
construction; see Clark, George C. Jr. and J. Bibb Cain, *Error-Correction
Coding for Digital Communications*, New York: Plenum Press, 1981.  (Reference
details read from the Communications Blockset bibliography on 2026-10-06; the
primary text was not consulted for this build, and no page number is quoted.)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .base import Interleaver, InterleaverCost, check_length

__all__ = ["BlockInterleaver"]


class BlockInterleaver(Interleaver):
    """Row-write, column-read block interleaver.

    Parameters
    ----------
    depth:
        Number of rows, in symbols.  Must be >= 1.  This is the longest channel
        burst the interleaver fully disperses.
    span:
        Number of columns, in symbols.  Must be >= 1.  This is the source-index
        separation between adjacent transmitted symbols.

    Raises
    ------
    TypeError
        If ``depth`` or ``span`` is not an integer.
    ValueError
        If ``depth < 1`` or ``span < 1``.

    Examples
    --------
    >>> import numpy as np
    >>> il = BlockInterleaver(depth=4, span=4)
    >>> il.permutation().tolist()
    [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15]
    >>> x = np.arange(16)
    >>> bool(np.array_equal(il.deinterleave(il.interleave(x)), x))
    True
    """

    def __init__(self, depth: int, span: int) -> None:
        self._depth = check_length(depth, "depth")
        self._span = check_length(span, "span")

    @property
    def depth(self) -> int:
        """Number of rows, in symbols."""
        return self._depth

    @property
    def span(self) -> int:
        """Number of columns, in symbols."""
        return self._span

    @property
    def length(self) -> int:
        """Block length ``depth * span``, in symbols."""
        return self._depth * self._span

    def permutation(self) -> NDArray[np.int64]:
        """``pi[i] = (i % depth) * span + (i // depth)``."""
        i = np.arange(self.length, dtype=np.int64)
        return (i % self._depth) * self._span + (i // self._depth)

    def position_of_input(self) -> NDArray[np.int64]:
        """Closed-form inverse: ``(j % span) * depth + (j // span)``.

        Overrides the base-class ``argsort`` with the exact inverse, which is
        the same construction with ``depth`` and ``span`` exchanged.
        """
        j = np.arange(self.length, dtype=np.int64)
        return (j % self._span) * self._depth + (j // self._span)

    def minimum_spread_closed_form(self) -> int:
        """Exact minimum spread of this construction, without enumeration.

        Returns
        -------
        int
            ``min(span + 1, depth + 1, depth + span - 2)`` for
            ``depth, span >= 2``; ``2`` if either is 1 and ``length >= 2``; ``0``
            for ``length == 1``, where no pair of distinct positions exists.

        Notes
        -----
        Derivation in the module docstring.  Checked against brute force in
        ``validation/validate_block_spread.py``.
        """
        d, s = self._depth, self._span
        if d * s < 2:
            return 0
        if d == 1 or s == 1:
            return 2
        return min(s + 1, d + 1, d + s - 2)

    def cost(self) -> InterleaverCost:
        """Full-block buffering cost, in symbols.

        Returns
        -------
        InterleaverCost
            One-way latency and memory ``depth * span``; pair latency and
            memory ``2 * depth * span``.

        Notes
        -----
        Model: the permutation is realised by buffering a whole block at each
        end, so each end holds ``N = depth * span`` symbols and the pair adds
        ``2N`` symbol times end to end.  This is the conventional figure.  An
        implementation that overlaps read-out with write-in achieves less; the
        README limitations say so.
        """
        n = self.length
        return InterleaverCost(
            one_way_latency_symbols=n,
            pair_latency_symbols=2 * n,
            one_way_memory_symbols=n,
            pair_memory_symbols=2 * n,
            model="full-block buffering",
        )

    def __repr__(self) -> str:
        return f"BlockInterleaver(depth={self._depth}, span={self._span})"
