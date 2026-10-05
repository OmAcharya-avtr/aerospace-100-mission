"""Command line interface: ``python -m framesync <subcommand>``.

Subcommands
-----------
``fer``        analytic and optionally measured frame error rate over Eb/N0
``falsesync``  false-sync probability against the correlation threshold
``gain``       RS(255,223) coding gain at a target rate
``sync``       drive the acquisition state machine over an observation string
``slip``       slip and re-acquisition behaviour across a bit insertion
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from . import __version__
from .asm import ASM_32_HEX, expected_false_syncs, false_sync_probability
from .channel import bpsk_ber
from .conv import ConvCode, free_distance
from .fer import (
    coding_gain_db,
    measure_conv_fer,
    measure_rs_fer,
    measure_uncoded_fer,
    uncoded_fer,
)
from .frames import FrameGeometry
from .rs import RS_RATE, rs_frame_error_rate
from .sync import FrameSynchroniser, SyncConfig, analyse_slip


def _snr_range(text: str) -> np.ndarray:
    """Parse ``start:stop:step`` or a comma-separated list of dB values."""
    if ":" in text:
        parts = text.split(":")
        if len(parts) != 3:
            raise argparse.ArgumentTypeError(
                f"range must be start:stop:step, got {text!r}"
            )
        start, stop, step = (float(p) for p in parts)
        if step <= 0:
            raise argparse.ArgumentTypeError("step must be positive")
        return np.arange(start, stop + 0.5 * step, step)
    try:
        return np.array([float(p) for p in text.split(",")], dtype=float)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"cannot parse Eb/N0 values from {text!r}") from exc


def _cmd_fer(args: argparse.Namespace) -> int:
    geom = FrameGeometry(data_octets=args.frame_octets, fecf=True)
    snr = args.ebn0
    print(f"framesync {__version__} -- frame error rate (Eb/N0 per information bit)")
    print(f"frame: {geom.frame_octets} octets = {geom.frame_bits} bits, FECF on, ASM 32 bits")
    rs_col = f"FER rs I={args.interleave}"
    print(f"{'Eb/N0 dB':>9}  {'p_bit':>12}  {'FER uncoded':>12}  {rs_col:>14}")
    pu = bpsk_ber(snr, 1.0)
    fu = uncoded_fer(snr, geom.frame_bits)
    prs = bpsk_ber(snr, RS_RATE)
    frs = rs_frame_error_rate(prs, args.interleave)
    for i, s in enumerate(np.atleast_1d(snr)):
        print(f"{s:9.2f}  {pu[i]:12.4e}  {fu[i]:12.4e}  {frs[i]:14.4e}")
    if args.measure:
        rng = np.random.default_rng(args.seed)
        print("\nmeasured (Monte Carlo, binomial standard error):")
        for s in np.atleast_1d(snr):
            print("  " + measure_uncoded_fer(float(s), geom, args.frames, rng).as_row())
    if args.measure_rs:
        rng = np.random.default_rng(args.seed + 1)
        print("\nmeasured RS(255,223):")
        for s in np.atleast_1d(snr):
            print("  " + measure_rs_fer(float(s), args.rs_frames, rng, args.interleave).as_row())
    if args.measure_conv:
        rng = np.random.default_rng(args.seed + 2)
        code = ConvCode()
        print(f"\nmeasured convolutional (171,133) K={code.k}, d_free={free_distance(code)}:")
        for s in np.atleast_1d(snr):
            pt = measure_conv_fer(
                float(s), args.conv_info_bits, args.conv_frames, rng, soft=args.soft
            )
            print("  " + pt.as_row())
    return 0


def _cmd_falsesync(args: argparse.Namespace) -> int:
    print(f"framesync {__version__} -- ASM false-sync probability")
    print(f"marker 0x{ASM_32_HEX:08X}, L = {args.length} bits")
    print("P_fa(T) = 2^-L * sum_{k=0}^{T} C(L, k)   (per window position)")
    print(f"{'T bits':>7}  {'P_fa':>14}  {'expected in ' + str(args.stream_bits) + ' bits':>26}")
    for t in range(args.max_tolerance + 1):
        p = false_sync_probability(t, args.length)
        e = expected_false_syncs(args.stream_bits, t, args.length)
        print(f"{t:7d}  {p:14.6e}  {e:26.6e}")
    return 0


def _cmd_gain(args: argparse.Namespace) -> int:
    print(f"framesync {__version__} -- RS(255,223) coding gain, R = {RS_RATE:.6f}")
    for metric in ("ber", "fer"):
        g = coding_gain_db(args.target, metric=metric, interleave=args.interleave)
        print(
            f"  at {metric.upper()} = {args.target:g}: uncoded needs "
            f"{g['ebn0_uncoded_db']:.4f} dB, RS needs {g['ebn0_rs_db']:.4f} dB, "
            f"gain = {g['gain_db']:.4f} dB"
        )
    print("  (gain is in Eb/N0 per information bit; the 223/255 rate loss is included)")
    return 0


def _cmd_sync(args: argparse.Namespace) -> int:
    obs = ["hit" if c in "1hH" else "miss" for c in args.observations]
    cfg = SyncConfig(
        search_tolerance=args.search_tolerance,
        lock_tolerance=args.lock_tolerance,
        check_required=args.check_required,
        flywheel_max=args.flywheel_max,
    )
    sm = FrameSynchroniser(cfg)
    print(f"framesync {__version__} -- acquisition state machine")
    print(f"config: {cfg}")
    print(f"{'i':>4}  {'obs':<5}  {'from':<9} -> {'to':<9}  miss  check")
    for ev in sm.run(obs):
        print(
            f"{ev.index:4d}  {ev.observation:<5}  {ev.state_before.value:<9} -> "
            f"{ev.state_after.value:<9}  {ev.miss_count:4d}  {ev.check_count:5d}"
        )
    print(f"final state: {sm.state.value}, delivering: {sm.delivering}")
    return 0


def _cmd_slip(args: argparse.Namespace) -> int:
    rep = analyse_slip(
        frame_bits=args.frame_bits,
        n_frames=args.n_frames,
        slip_at_frame=args.slip_at_frame,
        slip_bits=args.slip_bits,
    )
    print(f"framesync {__version__} -- slip behaviour")
    print(f"slip of {rep.slip_bits:+d} bits before frame {args.slip_at_frame}")
    print(f"frames delivered at the correct phase : {rep.frames_before_slip}")
    print(f"frames delivered at the wrong phase   : {rep.frames_delivered_at_wrong_phase}")
    print(f"frames to re-enter SEARCH             : {rep.reacquisition_frames}")
    print(f"final state                           : {rep.final_state.value}")
    print("state trace: " + " ".join(rep.states))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for ``python -m framesync``."""
    p = argparse.ArgumentParser(
        prog="framesync",
        description=(
            "CCSDS telemetry frame-level link performance harness: ASM correlation "
            "detection, sync acquisition state machine, and frame error rate against "
            "Eb/N0 for uncoded, RS(255,223) and convolutionally coded links. Not a "
            "codec and not a packet parser."
        ),
    )
    p.add_argument("--version", action="version", version=f"framesync {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    f = sub.add_parser("fer", help="frame error rate against Eb/N0")
    f.add_argument("--ebn0", type=_snr_range, default=_snr_range("4:12:2"),
                   help="Eb/N0 in dB as start:stop:step or a comma list (default 4:12:2)")
    f.add_argument("--frame-octets", type=int, default=1115,
                   help="transfer frame length in octets, FECF excluded (default 1115 = 223*5)")
    f.add_argument("--interleave", type=int, default=5, help="RS interleaving depth I")
    f.add_argument("--measure", action="store_true", help="also measure the uncoded link")
    f.add_argument("--measure-rs", action="store_true", help="also measure the RS link (slow)")
    f.add_argument("--measure-conv", action="store_true",
                   help="also measure the convolutional link (slow)")
    f.add_argument("--soft", action="store_true", help="soft-decision Viterbi instead of hard")
    f.add_argument("--frames", type=int, default=500, help="frames per measured uncoded point")
    f.add_argument("--rs-frames", type=int, default=50, help="frames per measured RS point")
    f.add_argument("--conv-frames", type=int, default=200, help="frames per measured conv point")
    f.add_argument("--conv-info-bits", type=int, default=512,
                   help="information bits per conv frame")
    f.add_argument("--seed", type=int, default=20261005, help="RNG seed")
    f.set_defaults(func=_cmd_fer)

    fs = sub.add_parser("falsesync", help="false-sync probability against the threshold")
    fs.add_argument("--length", type=int, default=32, help="marker length L in bits")
    fs.add_argument("--max-tolerance", type=int, default=6, help="largest T to tabulate")
    fs.add_argument("--stream-bits", type=int, default=10**9,
                    help="stream length for the expected-count column")
    fs.set_defaults(func=_cmd_falsesync)

    g = sub.add_parser("gain", help="RS(255,223) coding gain at a target rate")
    g.add_argument("--target", type=float, default=1e-5, help="target BER/FER (default 1e-5)")
    g.add_argument("--interleave", type=int, default=5, help="RS interleaving depth I")
    g.set_defaults(func=_cmd_gain)

    s = sub.add_parser("sync", help="drive the acquisition state machine")
    s.add_argument("observations", help="string of 1/h for hit and 0/m for miss, e.g. 1101000")
    s.add_argument("--search-tolerance", type=int, default=0)
    s.add_argument("--lock-tolerance", type=int, default=3)
    s.add_argument("--check-required", type=int, default=1)
    s.add_argument("--flywheel-max", type=int, default=4)
    s.set_defaults(func=_cmd_sync)

    sl = sub.add_parser("slip", help="slip and re-acquisition across a bit insertion")
    sl.add_argument("--frame-bits", type=int, default=1024)
    sl.add_argument("--n-frames", type=int, default=14)
    sl.add_argument("--slip-at-frame", type=int, default=6)
    sl.add_argument("--slip-bits", type=int, default=3)
    sl.set_defaults(func=_cmd_slip)
    return p


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit status."""
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
