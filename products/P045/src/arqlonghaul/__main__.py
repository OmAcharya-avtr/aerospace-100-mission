"""Command-line interface: ``python -m arqlonghaul <subcommand>``.

Every subcommand prints numbers and exits 0, or prints a message to stderr and
exits 2 on bad input.  Nothing here writes files.
"""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np

from . import __version__
from .channel import GilbertElliottChannel, IndependentFrameChannel, bpsk_ber
from .closedform import gbn_throughput, sr_throughput, sw_throughput
from .crc import CATALOGUE, crc
from .harq import HarqConfig, crossover_rtt, harq_goodput_exact, optimal_first_rate
from .link import PRESETS, LinkParams, preset
from .protocols import simulate
from .window import size_window


def _link_from_args(args: argparse.Namespace) -> LinkParams:
    """Build a :class:`LinkParams` from ``--preset`` or the explicit triple."""
    if args.preset:
        return preset(args.preset)
    if args.rate_bps is None or args.rtt_s is None:
        raise ValueError("give either --preset or both --rate-bps and --rtt-s")
    return LinkParams(
        rate_bps=args.rate_bps,
        rtt_s=args.rtt_s,
        frame_bits=args.frame_bits,
        payload_bits=args.payload_bits,
        name="custom",
    )


def _cmd_link(args: argparse.Namespace) -> int:
    link = _link_from_args(args)
    d = link.describe()
    print(f"link                  {d['name']}")
    print(f"rate                  {d['rate_bps']:.6g} bit/s")
    print(f"round-trip time       {d['rtt_s']:.6g} s")
    print(f"frame                 {d['frame_bits']} bits ({d['frame_bits'] / 8:.0f} bytes)")
    print(f"payload               {d['payload_bits']} bits")
    print(f"frame time T_f        {d['frame_time_s']:.6g} s")
    print(f"N = 1 + RTT/T_f       {d['slots_per_cycle_N']:.4f} slots")
    print(f"bandwidth-delay       {d['bdp_bits']:.6g} bits = {d['bdp_bytes']:.6g} bytes")
    print(f"                      {d['bdp_frames']:.4f} frames")
    print(f"min continuous window {d['min_continuous_window_frames']} frames")
    return 0


def _cmd_closedform(args: argparse.Namespace) -> int:
    link = _link_from_args(args)
    n = link.slots_per_cycle
    w = args.window or link.min_continuous_window
    scale = link.payload / link.frame_time_s
    print(f"N = {n:.4f} slots, frame error rate = {args.fer:g}, window = {w} frames")
    print(f"{'protocol':<18}{'eta':>12}{'bit/s':>16}")
    rows = [
        ("stop_and_wait", sw_throughput(args.fer, n)),
        ("selective_repeat", sr_throughput(args.fer, n, w)),
    ]
    if w >= math.ceil(n):
        rows.insert(1, ("go_back_n", gbn_throughput(args.fer, n, w)))
    else:
        print(
            f"go_back_n omitted: window {w} < ceil(N) = {math.ceil(n)}, "
            "the closed form does not apply",
            file=sys.stderr,
        )
    for name, eta in rows:
        print(f"{name:<18}{eta:>12.6f}{eta * scale:>16.6g}")
    return 0


def _cmd_window(args: argparse.Namespace) -> int:
    link = _link_from_args(args)
    d = size_window(link, args.fer).as_dict()
    print(f"frame                 {d['frame_bits']} bits")
    print(f"N                     {d['N']:.4f} slots")
    print(f"bandwidth-delay       {d['bdp_frames']:.4f} frames = {d['bdp_bytes']:.6g} bytes")
    print(f"knee (window binds)   {d['knee_frames']} frames = {d['knee_bytes']:.6g} bytes")
    print(f"at the knee, fer={d['fer']:g}:")
    print(f"  selective repeat    eta {d['sr_goodput_at_knee']:.6f} -> "
          f"{d['sr_bps_at_knee']:.6g} bit/s")
    print(f"  go-back-N           eta {d['gbn_goodput_at_knee']:.6f} -> "
          f"{d['gbn_bps_at_knee']:.6g} bit/s")
    return 0


def _cmd_simulate(args: argparse.Namespace) -> int:
    rng = np.random.default_rng(args.seed)
    if args.burst > 1.0:
        channel = GilbertElliottChannel.from_mean_and_burst(args.fer, args.burst)
    else:
        channel = IndependentFrameChannel(args.fer)
    errors = channel.errors(args.slots, rng)
    print(f"channel               {channel.name}")
    print(f"realised frame error  {errors.mean():.6f} over {args.slots} slots")
    print(f"{'protocol':<18}{'eta':>12}{'stderr':>12}{'tx/frame':>12}")
    for name in ("stop_and_wait", "go_back_n", "selective_repeat"):
        res = simulate(name, errors, args.n, args.window)
        print(
            f"{name:<18}{res.goodput:>12.6f}{res.goodput_stderr:>12.6f}"
            f"{res.mean_transmissions_per_frame:>12.4f}"
        )
    return 0


def _cmd_harq(args: argparse.Namespace) -> int:
    cfg = HarqConfig(
        k=args.k,
        n1=args.n1,
        delta=args.delta,
        max_rounds=args.max_rounds,
        esn0_db=args.esn0_db,
        rtt_symbols=args.rtt_symbols,
        alpha=args.alpha,
        scheme=args.scheme,
    )
    res = harq_goodput_exact(cfg)
    print(f"scheme                {cfg.scheme}")
    print(f"k / n1 / delta        {cfg.k} / {cfg.n1} / {cfg.delta} symbols")
    print(f"first-transmission R  {cfg.first_rate:.4f}")
    print(f"Es/N0, raw BER        {cfg.esn0_db:g} dB, {bpsk_ber(cfg.esn0_db):.6f}")
    print(f"t after round 1       {cfg.t_after(1)} symbols (alpha = {cfg.alpha:g})")
    print(f"goodput               {res.goodput:.6f} info symbols / symbol time")
    print(f"residual frame error  {res.residual_fer:.6e}")
    print(f"mean rounds           {res.mean_rounds:.4f}")
    print(f"mean elapsed          {res.elapsed_symbols:.2f} symbol times")
    if args.crossover:
        cx = crossover_rtt(
            k=cfg.k,
            esn0_db=cfg.esn0_db,
            delta=cfg.delta,
            retransmit_rate=0.90,
            upfront_rate=0.50,
            max_rounds=cfg.max_rounds,
            alpha=cfg.alpha,
        )
        print(f"crossover D           {cx['crossover_d']:.2f} symbol times "
              "(R=0.90 retransmit vs R=0.50 up front)")
        rate, n1, _ = optimal_first_rate(
            cfg.k, cfg.rtt_symbols, cfg.esn0_db, cfg.delta, cfg.max_rounds, cfg.alpha
        )
        print(f"optimal first rate    {rate:.4f} (n1 = {n1}) at D = {cfg.rtt_symbols:g}")
    return 0


def _cmd_crc(args: argparse.Namespace) -> int:
    spec = CATALOGUE.get(args.name)
    if spec is None:
        print(
            f"unknown CRC {args.name!r}; known: {sorted(CATALOGUE)}", file=sys.stderr
        )
        return 2
    data = args.data.encode("utf-8")
    value = crc(data, spec)
    width = spec.width // 4
    print(f"crc                   {spec.name}")
    print(f"width / poly          {spec.width} / {spec.poly:#x}")
    print(f"init / xorout         {spec.init:#x} / {spec.xorout:#x}")
    print(f"refin / refout        {spec.refin} / {spec.refout}")
    print(f"value                 {value:#0{width + 2}x}")
    print(f"catalogue check       {spec.check:#0{width + 2}x} "
          f"(for the ASCII string 123456789)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for ``python -m arqlonghaul``."""
    parser = argparse.ArgumentParser(
        prog="arqlonghaul",
        description=(
            "ARQ and hybrid-ARQ goodput on links where the round-trip time "
            "dominates. Research-grade; not flight-qualified or certified."
        ),
    )
    parser.add_argument("--version", action="version", version=f"arqlonghaul {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_link_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--preset", choices=sorted(PRESETS), help="named link preset")
        sp.add_argument("--rate-bps", type=float, default=None, help="bit rate, bit/s")
        sp.add_argument("--rtt-s", type=float, default=None, help="round-trip time, s")
        sp.add_argument("--frame-bits", type=int, default=8920, help="frame length, bits")
        sp.add_argument(
            "--payload-bits",
            type=int,
            default=None,
            help="payload per frame, bits; defaults to --frame-bits (no overhead)",
        )

    p_link = sub.add_parser("link", help="link geometry and bandwidth-delay product")
    add_link_args(p_link)
    p_link.set_defaults(func=_cmd_link)

    p_cf = sub.add_parser("closedform", help="classical ARQ throughput expressions")
    add_link_args(p_cf)
    p_cf.add_argument("--fer", type=float, default=0.05, help="frame error rate")
    p_cf.add_argument("--window", type=int, default=None, help="send window, frames")
    p_cf.set_defaults(func=_cmd_closedform)

    p_w = sub.add_parser("window", help="window sizing against the bandwidth-delay product")
    add_link_args(p_w)
    p_w.add_argument("--fer", type=float, default=0.05, help="frame error rate")
    p_w.set_defaults(func=_cmd_window)

    p_s = sub.add_parser("simulate", help="run the three protocol state machines")
    p_s.add_argument("--n", type=int, default=50, help="slots per cycle N")
    p_s.add_argument("--window", type=int, default=50, help="send window, frames")
    p_s.add_argument("--fer", type=float, default=0.05, help="mean frame error rate")
    p_s.add_argument(
        "--burst",
        type=float,
        default=1.0,
        help="mean burst length in slots; <= 1 selects the independent channel",
    )
    p_s.add_argument("--slots", type=int, default=200_000, help="slots to simulate")
    p_s.add_argument("--seed", type=int, default=0, help="random seed")
    p_s.set_defaults(func=_cmd_simulate)

    p_h = sub.add_parser("harq", help="hybrid-ARQ goodput and the crossover")
    p_h.add_argument("--k", type=int, default=200, help="information symbols per frame")
    p_h.add_argument("--n1", type=int, default=300, help="first transmission, symbols")
    p_h.add_argument("--delta", type=int, default=40, help="IR increment, symbols")
    p_h.add_argument("--max-rounds", type=int, default=4, help="transmissions per frame")
    p_h.add_argument("--esn0-db", type=float, default=1.0, help="per-symbol Es/N0, dB")
    p_h.add_argument(
        "--rtt-symbols", type=float, default=500.0, help="feedback latency, symbol times"
    )
    p_h.add_argument("--alpha", type=float, default=0.5, help="code-family efficiency")
    p_h.add_argument(
        "--scheme", choices=("type_i", "type_ii"), default="type_ii", help="HARQ scheme"
    )
    p_h.add_argument(
        "--crossover", action="store_true", help="also locate the retransmit/redundancy crossover"
    )
    p_h.set_defaults(func=_cmd_harq)

    p_c = sub.add_parser("crc", help="compute a frame check sequence")
    p_c.add_argument("--name", default="crc-32", help="catalogue name")
    p_c.add_argument("--data", default="123456789", help="ASCII message")
    p_c.set_defaults(func=_cmd_crc)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point.  Returns the process exit status."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
