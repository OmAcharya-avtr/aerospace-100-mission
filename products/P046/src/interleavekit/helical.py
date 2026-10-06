"""Helical (diagonal-read) block interleaver.

Construction
------------
The source block of ``N = rows * columns`` symbols is written into a
``rows x columns`` array **row by row**, exactly as for the block interleaver,
and then read out along **helical diagonals**: transmitted position ``i`` takes
the array cell

    col = i % columns
    row = (i // columns + step * col) mod rows

so each consecutive group of ``columns`` transmitted symbols takes one symbol
from every column, and the row index advances by ``step`` from one column to the
next.  The read sweep therefore walks a diagonal that wraps in the row
direction, which is what makes it helical.

In source indices, ``pi[i] = row * columns + col`` with ``row`` and ``col`` as
above.  Worked example with ``rows = 3``, ``columns = 3``, ``step = 1``::

    source array        read cells (row, col) in transmitted order
      0  1  2            (0,0) (1,1) (2,2) (1,0) (2,1) (0,2) (2,0) (0,1) (1,2)
      3  4  5
      6  7  8          pi = [0, 4, 8, 3, 7, 2, 6, 1, 5]

The map is a bijection for every ``step``, including ``step`` sharing a factor
with ``rows``, because for a fixed column the row index ``(k + step*col) mod
rows`` runs over every row exactly once as ``k = i // columns`` runs over
``0 .. rows-1``.  The ``step`` only rotates each column's read order.

Relation to the MATLAB helical interleaver
-----------------------------------------
The MathWorks Communications Toolbox ``Helical Interleaver`` block is a
*streaming* interleaver: it has ``C`` columns and unbounded rows, partitions the
input into groups of ``N`` symbols, places group ``k`` in column ``k mod C``
starting at row ``1 + (k-1) * s``, and reads the next ``N`` rows out
sequentially, so its output contains initial-condition fill symbols
(documentation read on 2026-10-06).  The construction here is a **fixed-length
block permutation** with a helical read, not that streaming block.  It is not
bit-compatible with ``helintrlv`` and this package does not claim it is.

References
----------
Diagonal and helical read-out of an interleaving array is covered in the
interleaver literature; see Ramsey, J. L., "Realization of Optimum
Interleavers", *IEEE Transactions on Information Theory*, IT-16 (3), May 1970,
pp. 338-345.  (Bibliographic details read from the Communications Blockset
bibliography on 2026-10-06; the primary article was not consulted, and no
specific result from it is quoted or relied on here.)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .base import Interleaver, InterleaverCost, check_length

__all__ = ["HelicalInterleaver"]


class HelicalInterleaver(Interleaver):
    """Block interleaver with a helical (wrapping diagonal) read-out.

    Parameters
    ----------
    rows:
        Number of array rows, in symbols.  Must be >= 1.
    columns:
        Number of array columns, in symbols.  Must be >= 1.
    step:
        Row advance per column during the read sweep, in rows.  Must be >= 0.
        ``step = 0`` degenerates to the plain row-write / row-read identity;
        ``step = 1`` is the usual choice.

    Raises
    ------
    TypeError
        If any parameter is not an integer.
    ValueError
        If ``rows < 1``, ``columns < 1`` or ``step < 0``.

    Examples
    --------
    >>> HelicalInterleaver(rows=3, columns=3, step=1).permutation().tolist()
    [0, 4, 8, 3, 7, 2, 6, 1, 5]
    """

    def __init__(self, rows: int, columns: int, step: int = 1) -> None:
        self._rows = check_length(rows, "rows")
        self._columns = check_length(columns, "columns")
        if isinstance(step, bool) or not isinstance(step, (int, np.integer)):
            raise TypeError(f"step must be an integer, got {type(step).__name__}")
        step = int(step)
        if step < 0:
            raise ValueError(f"step must be >= 0 rows, got {step}")
        self._step = step

    @property
    def rows(self) -> int:
        """Number of array rows, in symbols."""
        return self._rows

    @property
    def columns(self) -> int:
        """Number of array columns, in symbols."""
        return self._columns

    @property
    def step(self) -> int:
        """Row advance per column during the read sweep, in rows."""
        return self._step

    @property
    def length(self) -> int:
        """Block length ``rows * columns``, in symbols."""
        return self._rows * self._columns

    def permutation(self) -> NDArray[np.int64]:
        """``pi[i] = ((i // columns + step * (i % columns)) % rows) * columns + i % columns``."""
        i = np.arange(self.length, dtype=np.int64)
        col = i % self._columns
        row = (i // self._columns + self._step * col) % self._rows
        return row * self._columns + col

    def cost(self) -> InterleaverCost:
        """Full-block buffering cost, in symbols.

        Returns
        -------
        InterleaverCost
            One-way latency and memory ``rows * columns``; pair latency and
            memory ``2 * rows * columns``.  Same model as
            :meth:`interleavekit.block.BlockInterleaver.cost`, because the array
            is the same size and only the read order differs.
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
        return (
            f"HelicalInterleaver(rows={self._rows}, columns={self._columns}, step={self._step})"
        )
