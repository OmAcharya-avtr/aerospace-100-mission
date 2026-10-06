"""Command-line interface: ``python -m interleavekit``.

Subcommands
-----------
``permutation``
    Print a construction's permutation.
``metrics``
    Print minimum spread, S-parameter, normalised dispersion and the burst
    dispersion profile for a construction.
``cost``
    Print the latency and memory cost of one construction at a stated symbol rate.
``compare``
    Print a cost table over several constructions at one block size.
``design``
    Search for the cheapest parameter set in each family that fully disperses a
    given burst length.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import numpy as np

from . import __version__
from .block import BlockInterleaver
from .convolutional import ConvolutionalInterleaver
from .cost import cheapest_for_burst, cost_table, format_cost_table
from .helical import HelicalInterleaver
from .metrics import (
    burst_dispersion_profile,
    dispersion,
    max_burst_fully_dispersed,
    minimum_spread,
    s_parameter,
)
from .srandom import SRandomInterleaver

_DEFAULT_RATE = 1.0e6


def _build(args: argparse.Namespace):
    """Instantiate the construction named on the command line."""
    kind = args.kind
    if kind == "block":
        return BlockInterleaver(args.depth, args.span)
    if kind == "helical":
        return HelicalInterleaver(args.depth, args.span, args.step)
    if kind == "srandom":
        return SRandomInterleaver(args.length, args.spread, args.seed)
    if kind == "convolutional":
        return ConvolutionalInterleaver(args.registers, args.slope)
    raise ValueError(f"unknown construction {kind!r}")


def _add_construction_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "kind",
        choices=["block", "helical", "srandom", "convolutional"],
        help="which construction",
    )
    p.add_argument("--depth", type=int, default=8, help="block/helical rows, symbols")
    p.add_argument("--span", type=int, default=8, help="block/helical columns, symbols")
    p.add_argument("--step", type=int, default=1, help="helical row advance per column")
    p.add_argument("--length", type=int, default=256, help="srandom block length, symbols")
    p.add_argument("--spread", type=int, default=8, help="srandom spread S, symbols")
    p.add_argument("--seed", type=int, default=0, help="srandom seed")
    p.add_argument("--registers", type=int, default=8, help="convolutional registers")
    p.add_argument("--slope", type=int, default=1, help="convolutional register length step")


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m interleavekit",
        description=(
            "Interleaver construction and burst-dispersion metrics. Research-grade; not "
            "flight-qualified, not certified, not approved for operational aerospace use."
        ),
    )
    parser.add_argument("--version", action="version", version=f"interleavekit {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_perm = sub.add_parser("permutation", help="print the permutation")
    _add_construction_args(p_perm)
    p_perm.add_argument(
        "--symbols", type=int, default=24, help="convolutional: source symbols to map"
    )

    p_met = sub.add_parser("metrics", help="print spread, dispersion and burst dispersion")
    _add_construction_args(p_met)
    p_met.add_argument(
        "--bursts", type=int, default=12, help="largest burst length in the profile, symbols"
    )

    p_cost = sub.add_parser("cost", help="print latency and memory cost")
    _add_construction_args(p_cost)
    p_cost.add_argument(
        "--symbol-rate", type=float, default=_DEFAULT_RATE, help="symbol rate, symbols/s"
    )
    p_cost.add_argument("--bits-per-symbol", type=int, default=8, help="stored symbol width, bits")

    p_cmp = sub.add_parser("compare", help="cost table over all four constructions")
    p_cmp.add_argument("--depth", type=int, default=16, help="block/helical rows, symbols")
    p_cmp.add_argument("--span", type=int, default=16, help="block/helical columns, symbols")
    p_cmp.add_argument("--registers", type=int, default=16, help="convolutional registers")
    p_cmp.add_argument("--slope", type=int, default=1, help="convolutional register length step")
    p_cmp.add_argument("--spread", type=int, default=10, help="srandom spread S, symbols")
    p_cmp.add_argument("--seed", type=int, default=0, help="srandom seed")
    p_cmp.add_argument(
        "--symbol-rate", type=float, default=_DEFAULT_RATE, help="symbol rate, symbols/s"
    )

    p_des = sub.add_parser("design", help="cheapest parameters that disperse a given burst")
    p_des.add_argument("burst", type=int, help="burst length to fully disperse, symbols")
    p_des.add_argument(
        "--symbol-rate", type=float, default=_DEFAULT_RATE, help="symbol rate, symbols/s"
    )
    p_des.add_argument(
        "--max-block-symbols", type=int, default=2048, help="search ceiling for depth*span"
    )
    return parser


def _cmd_permutation(args: argparse.Namespace) -> int:
    obj = _build(args)
    if isinstance(obj, ConvolutionalInterleaver):
        n = args.symbols
        pos = obj.transmitted_position(np.arange(n, dtype=np.int64))
        print(f"{obj!r}")
        print(f"register delays (symbols): {obj.register_delays().tolist()}")
        print(f"transmitted position of source 0..{n - 1}: {pos.tolist()}")
        print(f"steady-state transmitted range for {n} symbols: {obj.steady_state_range(n)}")
    else:
        print(f"{obj!r}")
        print(f"pi (source index at each transmitted position): {obj.permutation().tolist()}")
        print(f"position_of_input (transmitted position of each source index): "
              f"{obj.position_of_input().tolist()}")
    return 0


def _cmd_metrics(args: argparse.Namespace) -> int:
    obj = _build(args)
    lengths = np.arange(1, args.bursts + 1, dtype=np.int64)
    print(f"{obj!r}")
    if isinstance(obj, ConvolutionalInterleaver):
        n = obj.max_delay_symbols + args.bursts + 2
        pos = obj.transmitted_position(np.arange(n, dtype=np.int64))
        window = obj.steady_state_range(n)
        print(f"not a block permutation; evaluated on {n} source symbols, "
              f"steady-state transmitted range {window}")
        print("minimum spread: not defined (not a permutation of a contiguous block)")
        profile = burst_dispersion_profile(pos, lengths, window)
        full = max_burst_fully_dispersed(pos, window, args.bursts + 1)
    else:
        pi = obj.permutation()
        pos = obj.position_of_input()
        print(f"block length N: {obj.length} symbols")
        print(f"minimum spread: {minimum_spread(pi)}")
        print(f"S-parameter: {s_parameter(pi)}")
        if obj.length <= 2048:
            print(f"normalised dispersion: {dispersion(pi):.6f}")
        else:
            print("normalised dispersion: skipped (N > 2048)")
        profile = burst_dispersion_profile(pos, lengths)
        full = max_burst_fully_dispersed(pos, None, args.bursts + 1)
    print("burst length -> worst-case surviving run (symbols):")
    for bl, run in zip(lengths.tolist(), profile.tolist(), strict=True):
        print(f"  {bl:>4d} -> {run}")
    print(f"largest fully dispersed burst: {full} symbols")
    return 0


def _cmd_cost(args: argparse.Namespace) -> int:
    obj = _build(args)
    cost = obj.cost()
    print(f"{obj!r}")
    print(f"cost model: {cost.model}")
    print(f"one-way latency: {cost.one_way_latency_symbols} symbols")
    print(f"pair latency:    {cost.pair_latency_symbols} symbols "
          f"= {cost.latency_ms(args.symbol_rate):.6f} ms at {args.symbol_rate:.6g} sym/s")
    print(f"one-way memory:  {cost.one_way_memory_symbols} symbols")
    print(f"pair memory:     {cost.pair_memory_symbols} symbols "
          f"= {cost.memory_bytes(args.bits_per_symbol):.1f} bytes "
          f"at {args.bits_per_symbol} bits/symbol")
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    n = args.depth * args.span
    constructions = [
        BlockInterleaver(args.depth, args.span),
        HelicalInterleaver(args.depth, args.span, 1),
        SRandomInterleaver(n, args.spread, args.seed),
        ConvolutionalInterleaver(args.registers, args.slope),
    ]
    rows = cost_table(
        constructions, args.symbol_rate, burst_search_limit=min(n, 64), conv_window=128
    )
    print(format_cost_table(rows, args.symbol_rate))
    return 0


def _cmd_design(args: argparse.Namespace) -> int:
    best = cheapest_for_burst(
        args.burst,
        symbol_rate_hz=args.symbol_rate,
        max_block_symbols=args.max_block_symbols,
    )
    print(f"cheapest parameters that fully disperse a burst of {args.burst} symbols")
    print(f"symbol rate {args.symbol_rate:.6g} sym/s")
    rows = [r for r in best.values() if r is not None]
    if not rows:
        print("no candidate inside the search ceilings achieved the target")
        return 0
    print(format_cost_table(rows, args.symbol_rate))
    for family, row in best.items():
        if row is None:
            print(f"{family}: no candidate inside the search ceilings")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point.  Returns the process exit status."""
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "permutation": _cmd_permutation,
        "metrics": _cmd_metrics,
        "cost": _cmd_cost,
        "compare": _cmd_compare,
        "design": _cmd_design,
    }
    try:
        return handlers[args.command](args)
    except (ValueError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
