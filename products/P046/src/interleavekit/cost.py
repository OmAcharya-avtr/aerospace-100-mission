"""Latency and memory accounting, and parameter search at a design target.

This module is what turns the metrics in :mod:`interleavekit.metrics` into a
design decision: given a burst length the link has to survive, which
construction and which parameters pay the least latency and memory for it.

Nothing here is a formula lifted from a textbook.  :func:`cheapest_for_burst`
*measures* the burst dispersion of every candidate with
:func:`interleavekit.metrics.max_burst_fully_dispersed` and reports the cheapest
candidate that actually achieves the target.  Rules of thumb about interleaver
depth are exactly what this is meant to replace.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .base import Interleaver, InterleaverCost
from .block import BlockInterleaver
from .convolutional import ConvolutionalInterleaver
from .helical import HelicalInterleaver
from .metrics import max_burst_fully_dispersed, minimum_spread
from .srandom import SRandomInterleaver

__all__ = [
    "CostRow",
    "describe",
    "cost_table",
    "format_cost_table",
    "cheapest_for_burst",
    "row_as_dict",
]


@dataclass(frozen=True)
class CostRow:
    """One row of a construction-comparison table.

    Attributes
    ----------
    name:
        Construction and its parameters.
    block_symbols:
        Block length for a permutation interleaver, or the source-stream length
        used to evaluate the convolutional interleaver, in symbols.
    pair_latency_symbols:
        End-to-end latency of interleaver plus de-interleaver, in symbol times.
    pair_latency_ms:
        The same latency in milliseconds at the stated symbol rate.
    pair_memory_symbols:
        Symbol storage held by interleaver plus de-interleaver.
    pair_memory_bytes:
        The same storage in bytes at the stated symbol width.
    minimum_spread:
        Minimum spread, in index units, or ``None`` when not defined for the
        construction (the convolutional interleaver is not a permutation of a
        contiguous block).
    max_burst_fully_dispersed:
        Largest channel burst, in transmitted symbols, after which every errored
        source symbol is isolated.
    cost_model:
        Which cost model produced the latency and memory figures.
    """

    name: str
    block_symbols: int
    pair_latency_symbols: int
    pair_latency_ms: float
    pair_memory_symbols: int
    pair_memory_bytes: float
    minimum_spread: int | None
    max_burst_fully_dispersed: int
    cost_model: str


def _conv_eval_length(conv: ConvolutionalInterleaver, window: int) -> int:
    """Source-stream length giving a steady-state window of at least ``window``."""
    return conv.max_delay_symbols + max(int(window), 1)


def describe(
    construction: Interleaver | ConvolutionalInterleaver,
    symbol_rate_hz: float,
    bits_per_symbol: int = 8,
    burst_search_limit: int | None = None,
    conv_window: int = 512,
) -> CostRow:
    """Measure one construction's spread, burst dispersion and cost.

    Parameters
    ----------
    construction:
        Any of :class:`~interleavekit.block.BlockInterleaver`,
        :class:`~interleavekit.helical.HelicalInterleaver`,
        :class:`~interleavekit.srandom.SRandomInterleaver` or
        :class:`~interleavekit.convolutional.ConvolutionalInterleaver`.
    symbol_rate_hz:
        Symbol rate in symbols per second, for the millisecond column.
    bits_per_symbol:
        Stored symbol width in bits, for the byte column.
    burst_search_limit:
        Cap on the burst length searched by
        :func:`~interleavekit.metrics.max_burst_fully_dispersed`.
    conv_window:
        Steady-state window width, in transmitted symbols, used when the
        construction is convolutional.  Must be at least
        ``burst_search_limit + 1`` to give the search room.

    Returns
    -------
    CostRow

    Raises
    ------
    TypeError
        If ``construction`` is not one of the supported types.
    """
    if not isinstance(construction, (Interleaver, ConvolutionalInterleaver)):
        raise TypeError(
            "construction must be an Interleaver or a ConvolutionalInterleaver, got "
            f"{type(construction).__name__}"
        )
    cost: InterleaverCost = construction.cost()
    if isinstance(construction, ConvolutionalInterleaver):
        n = _conv_eval_length(construction, conv_window)
        pos = construction.transmitted_position(np.arange(n, dtype=np.int64))
        window = construction.steady_state_range(n)
        spread: int | None = None
        burst = max_burst_fully_dispersed(pos, window, burst_search_limit)
    else:
        n = construction.length
        pos = construction.position_of_input()
        spread = minimum_spread(construction.permutation())
        burst = max_burst_fully_dispersed(pos, None, burst_search_limit)
    return CostRow(
        name=repr(construction),
        block_symbols=n,
        pair_latency_symbols=cost.pair_latency_symbols,
        pair_latency_ms=cost.latency_ms(symbol_rate_hz),
        pair_memory_symbols=cost.pair_memory_symbols,
        pair_memory_bytes=cost.memory_bytes(bits_per_symbol),
        minimum_spread=spread,
        max_burst_fully_dispersed=burst,
        cost_model=cost.model,
    )


def cost_table(
    constructions: list[Interleaver | ConvolutionalInterleaver],
    symbol_rate_hz: float,
    bits_per_symbol: int = 8,
    burst_search_limit: int | None = None,
    conv_window: int = 512,
) -> list[CostRow]:
    """:func:`describe` applied to a list of constructions.

    Returns
    -------
    list of CostRow
        One row per construction, in the order given.
    """
    return [
        describe(c, symbol_rate_hz, bits_per_symbol, burst_search_limit, conv_window)
        for c in constructions
    ]


def format_cost_table(rows: list[CostRow], symbol_rate_hz: float) -> str:
    """Render cost rows as a fixed-width text table.

    Parameters
    ----------
    rows:
        Rows from :func:`cost_table`.
    symbol_rate_hz:
        Echoed in the header so the millisecond column is interpretable.

    Returns
    -------
    str
        A table with no trailing newline.
    """
    head = (
        f"symbol rate {symbol_rate_hz:.6g} sym/s\n"
        f"{'construction':<54} {'N':>7} {'pair lat':>9} {'pair lat':>10} "
        f"{'pair mem':>9} {'min':>5} {'burst':>6}\n"
        f"{'':<54} {'sym':>7} {'sym':>9} {'ms':>10} {'sym':>9} "
        f"{'sprd':>5} {'disp':>6}"
    )
    lines = [head, "-" * 108]
    for r in rows:
        spread = "-" if r.minimum_spread is None else str(r.minimum_spread)
        lines.append(
            f"{r.name:<54} {r.block_symbols:>7d} {r.pair_latency_symbols:>9d} "
            f"{r.pair_latency_ms:>10.4f} {r.pair_memory_symbols:>9d} "
            f"{spread:>5} {r.max_burst_fully_dispersed:>6d}"
        )
    return "\n".join(lines)


def cheapest_for_burst(
    target_burst: int,
    symbol_rate_hz: float = 1.0e6,
    families: tuple[str, ...] = ("block", "helical", "convolutional"),
    max_block_symbols: int = 4096,
    max_registers: int = 48,
    max_slope: int = 48,
    helical_step: int = 1,
) -> dict[str, CostRow | None]:
    """Cheapest parameter set per family that fully disperses a given burst.

    "Cheapest" means the smallest :attr:`CostRow.pair_memory_symbols`, with pair
    latency as the tie-break.  Every candidate's burst dispersion is **measured**,
    not predicted from its depth.

    Parameters
    ----------
    target_burst:
        Burst length in transmitted symbols that must be fully dispersed, i.e.
        the construction must leave every errored source symbol isolated.
        Must be >= 1.
    symbol_rate_hz:
        Symbol rate in symbols per second, for the millisecond column.
    families:
        Which families to search.  ``"srandom"`` is accepted but slow, since each
        candidate needs a fresh greedy search; it is not in the default set.
    max_block_symbols:
        Ceiling on ``depth * span`` for the block and helical searches.
    max_registers, max_slope:
        Ceilings for the convolutional search.
    helical_step:
        ``step`` used for every helical candidate.

    Returns
    -------
    dict
        Family name to its cheapest :class:`CostRow`, or ``None`` if no candidate
        inside the search ceilings achieved the target.

    Raises
    ------
    ValueError
        If ``target_burst < 1`` or a family name is not recognised.
    """
    if isinstance(target_burst, bool) or not isinstance(target_burst, (int, np.integer)):
        raise TypeError(f"target_burst must be an integer, got {type(target_burst).__name__}")
    target = int(target_burst)
    if target < 1:
        raise ValueError(f"target_burst must be >= 1 symbol, got {target}")
    known = {"block", "helical", "convolutional", "srandom"}
    unknown = set(families) - known
    if unknown:
        raise ValueError(f"unknown families {sorted(unknown)}; choose from {sorted(known)}")

    limit = target + 1  # enough to tell "exactly target" from "more than target"
    best: dict[str, CostRow | None] = {f: None for f in families}

    def consider(family: str, row: CostRow) -> None:
        if row.max_burst_fully_dispersed < target:
            return
        cur = best[family]
        key = (row.pair_memory_symbols, row.pair_latency_symbols)
        if cur is None or key < (cur.pair_memory_symbols, cur.pair_latency_symbols):
            best[family] = row

    if "block" in families or "helical" in families:
        for depth in range(1, max_block_symbols + 1):
            if depth > 2 * target + 4:
                break
            for span in range(1, max_block_symbols // depth + 1):
                n = depth * span
                if n > max_block_symbols:
                    break
                cheapest = best.get("block")
                if cheapest is not None and 2 * n > cheapest.pair_memory_symbols:
                    continue
                if "block" in families:
                    consider(
                        "block",
                        describe(
                            BlockInterleaver(depth, span),
                            symbol_rate_hz,
                            burst_search_limit=limit,
                        ),
                    )
                if "helical" in families:
                    consider(
                        "helical",
                        describe(
                            HelicalInterleaver(depth, span, helical_step),
                            symbol_rate_hz,
                            burst_search_limit=limit,
                        ),
                    )

    if "convolutional" in families:
        for registers in range(2, max_registers + 1):
            for slope in range(1, max_slope + 1):
                conv = ConvolutionalInterleaver(registers, slope)
                pair_mem = conv.cost().pair_memory_symbols
                cur = best["convolutional"]
                if cur is not None and pair_mem > cur.pair_memory_symbols:
                    continue
                consider(
                    "convolutional",
                    describe(
                        conv,
                        symbol_rate_hz,
                        burst_search_limit=limit,
                        conv_window=limit + 2,
                    ),
                )

    if "srandom" in families:
        for n in range(max(2, target), max_block_symbols + 1):
            cur = best["srandom"]
            if cur is not None and 2 * n > cur.pair_memory_symbols:
                break
            try:
                il = SRandomInterleaver(n, min(target, int(np.sqrt(n / 2)) or 1), seed=0)
            except ValueError:
                continue
            consider("srandom", describe(il, symbol_rate_hz, burst_search_limit=limit))

    return best


def row_as_dict(row: CostRow) -> dict[str, object]:
    """``dataclasses.asdict`` on a :class:`CostRow`, for printing or serialising."""
    return asdict(row)
