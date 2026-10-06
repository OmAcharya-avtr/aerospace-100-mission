"""Command-line interface: ``python -m slotsync <subcommand>``.

Subcommands
-----------
``scurve``  S-curve of a detector on a pulse shape, with the measured gain
``loop``    loop coefficients from a noise bandwidth, damping and gain
``jitter``  closed-form, coloured-noise and Monte Carlo jitter, side by side
``slip``    cycle-slip rate estimate against loop SNR
``ppm``     PPM slot-clock gain, jitter and slot-error rate

Every number printed comes from the library, not from a literal in this file.
"""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np

from . import __version__
from .loop import (
    LoopDesign,
    cycle_slip_rate_rice,
    jitter_variance_closed_form,
    jitter_variance_coloured,
    jitter_variance_exact,
    loop_snr_db,
)
from .ppm import (
    PpmConfig,
    equivalent_slot_bandwidth,
    measure_ppm_slot_statistics,
    ppm_slot_autocovariance,
    ppm_slot_scurve,
    run_ppm_slot_loop,
)
from .pulses import pulse_by_name
from .scurve import default_offsets, scurve
from .simulate import measure_ted_autocovariance, measure_ted_statistics, run_timing_loop
from .ted import TedConfig

_DEFAULT_DAMPING = 1.0 / math.sqrt(2.0)


def _config(args: argparse.Namespace) -> TedConfig:
    return TedConfig(
        detector=args.detector, alphabet=args.alphabet, delta=args.delta, form=args.form
    )


def _print_rows(rows: list[tuple[str, object]]) -> None:
    width = max(len(name) for name, _ in rows)
    for name, value in rows:
        if isinstance(value, float):
            print(f"{name:<{width}}  {value:.6g}")
        else:
            print(f"{name:<{width}}  {value}")


def _cmd_scurve(args: argparse.Namespace) -> int:
    curve = scurve(_config(args), pulse_by_name(args.pulse))
    summary = curve.summary()
    _print_rows(list(summary.items()))
    if curve.gain == 0.0:
        print(
            "\nthe measured gain is zero: this detector carries no timing "
            "information for this pulse shape"
        )
    return 0


def _cmd_loop(args: argparse.Namespace) -> int:
    design = LoopDesign.from_bandwidth(args.bandwidth, args.damping, args.gain)
    _print_rows(list(design.summary().items()))
    recovered = LoopDesign.from_coefficients(
        design.k_proportional, design.k_integral, design.detector_gain
    )
    print()
    _print_rows(
        [
            ("recovered B_n", recovered.noise_bandwidth),
            ("recovered zeta", recovered.damping),
        ]
    )
    return 0


def _cmd_jitter(args: argparse.Namespace) -> int:
    config = _config(args)
    pulse = pulse_by_name(args.pulse)
    curve = scurve(config, pulse)
    gain = curve.gain_central_difference
    if gain == 0.0:
        print("the measured detector gain is zero; no loop can be designed")
        return 1
    stats = measure_ted_statistics(
        config, pulse, sample_snr_db=args.snr, samples=args.open_loop_samples
    )
    autocovariance = measure_ted_autocovariance(
        config, pulse, sample_snr_db=args.snr, max_lag=args.max_lag, samples=args.open_loop_samples
    )
    design = LoopDesign.from_bandwidth(args.bandwidth, args.damping, gain)
    run = run_timing_loop(
        config, pulse, design, n_symbols=args.symbols, sample_snr_db=args.snr
    )
    closed = jitter_variance_closed_form(args.bandwidth, gain, stats.variance)
    exact = jitter_variance_exact(design, stats.variance)
    coloured = jitter_variance_coloured(design, autocovariance)
    _print_rows(
        [
            ("detector", config.label),
            ("pulse", pulse.name),
            ("K_d (measured)", gain),
            ("sigma_n^2 (measured)", stats.variance),
            ("  of which self-noise", stats.self_noise_variance),
            ("B_n", args.bandwidth),
            ("zeta", args.damping),
            ("jitter var, closed form (white)", closed),
            ("jitter var, exact discrete (white)", exact),
            ("jitter var, coloured noise", coloured),
            ("jitter var, Monte Carlo", run.jitter_variance),
            ("  Monte Carlo standard error", run.jitter_variance_standard_error),
            (
                "Monte Carlo / coloured",
                run.jitter_variance / coloured if coloured else float("nan"),
            ),
            (
                "Monte Carlo / closed form",
                run.jitter_variance / closed if closed else float("nan"),
            ),
            ("static lock offset", run.mean_error),
            ("slips observed", run.slip_count),
        ]
    )
    return 0


def _cmd_slip(args: argparse.Namespace) -> int:
    config = _config(args)
    pulse = pulse_by_name(args.pulse)
    curve = scurve(config, pulse)
    gain = curve.gain_central_difference
    if gain == 0.0:
        print("the measured detector gain is zero; no loop can be designed")
        return 1
    boundary = curve.reversal_offset or 0.5
    design = LoopDesign.from_bandwidth(args.bandwidth, args.damping, gain)
    print(f"detector {config.label} on {pulse.name}: K_d = {gain:.6g}, boundary = {boundary:.4g}")
    print(
        f"{'snr_db':>8} {'sigma_n^2':>12} {'jitter_var':>12} "
        f"{'loop_snr_dB':>12} {'slip/symbol':>14}"
    )
    for snr in np.linspace(args.snr_low, args.snr_high, args.points):
        stats = measure_ted_statistics(
            config, pulse, sample_snr_db=float(snr), samples=args.open_loop_samples
        )
        variance = jitter_variance_exact(design, stats.variance)
        rate = cycle_slip_rate_rice(design, stats.variance, boundary)
        print(
            f"{snr:8.2f} {stats.variance:12.6g} {variance:12.6g} "
            f"{loop_snr_db(variance, boundary):12.4f} {rate:14.6e}"
        )
    return 0


def _cmd_ppm(args: argparse.Namespace) -> int:
    config = PpmConfig(order=args.order, delta=args.delta, known_slot=args.known_slot)
    pulse = pulse_by_name(args.pulse)
    curve = ppm_slot_scurve(config, pulse, offsets=default_offsets(0.5, 201))
    gain = curve.gain_central_difference
    stats = measure_ppm_slot_statistics(
        config, pulse, sample_snr_db=args.snr, symbols=args.open_loop_symbols
    )
    design = LoopDesign.from_bandwidth(args.bandwidth, args.damping, gain)
    autocovariance = ppm_slot_autocovariance(
        config, pulse, sample_snr_db=args.snr, max_lag=6, symbols=args.open_loop_symbols
    )
    run = run_ppm_slot_loop(config, pulse, design, n_symbols=args.symbols, sample_snr_db=args.snr)
    rows: list[tuple[str, object]] = list(curve.summary().items())
    rows += [
        ("sigma_n^2 per update", stats["variance"]),
        ("slot error rate (open loop)", stats["slot_error_rate"]),
        ("B_n per update", args.bandwidth),
        ("B_n per slot", equivalent_slot_bandwidth(args.bandwidth, args.order)),
        (
            "jitter var, closed form",
            jitter_variance_closed_form(args.bandwidth, gain, stats["variance"]),
        ),
        ("jitter var, coloured", jitter_variance_coloured(design, autocovariance)),
        ("jitter var, Monte Carlo", run.jitter_variance),
        ("  Monte Carlo standard error", run.jitter_variance_standard_error),
        ("slot error rate (closed loop)", run.slot_error_rate),
        ("slot slips", run.slip_count),
    ]
    _print_rows(rows)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser.  Exposed so the tests can inspect it."""
    parser = argparse.ArgumentParser(
        prog="python -m slotsync",
        description=(
            "Slot and symbol timing recovery for OOK and PPM optical receivers. "
            "Research-grade; not flight-qualified, not certified, not approved for "
            "operational aerospace use."
        ),
    )
    parser.add_argument("--version", action="version", version=f"slotsync {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_detector_arguments(sub: argparse.ArgumentParser) -> None:
        sub.add_argument(
            "--detector",
            default="gardner",
            choices=["early-late", "gardner", "mueller-muller"],
            help="timing-error detector",
        )
        sub.add_argument(
            "--alphabet", default="antipodal", choices=["antipodal", "ook"], help="data alphabet"
        )
        sub.add_argument("--delta", type=float, default=0.25, help="early-late gate half-spacing")
        sub.add_argument(
            "--form",
            default="auto",
            choices=["auto", "dd", "square", "plain"],
            help="early-late arrangement",
        )
        sub.add_argument(
            "--pulse",
            default="nyq-rc",
            choices=["rect", "tri", "rc-time", "half-sine", "nyq-rc"],
            help="pulse shape",
        )

    curve = subparsers.add_parser("scurve", help="S-curve and measured detector gain")
    add_detector_arguments(curve)
    curve.set_defaults(handler=_cmd_scurve)

    loop = subparsers.add_parser("loop", help="loop coefficients from B_n, zeta and K_d")
    loop.add_argument("--bandwidth", type=float, default=0.005, help="B_n, cycles per symbol")
    loop.add_argument("--damping", type=float, default=_DEFAULT_DAMPING, help="zeta")
    loop.add_argument("--gain", type=float, default=1.5, help="K_d, measured")
    loop.set_defaults(handler=_cmd_loop)

    jitter = subparsers.add_parser("jitter", help="jitter by three predictions and Monte Carlo")
    add_detector_arguments(jitter)
    jitter.add_argument("--bandwidth", type=float, default=0.005)
    jitter.add_argument("--damping", type=float, default=_DEFAULT_DAMPING)
    jitter.add_argument("--snr", type=float, default=20.0, help="per-sample SNR, dB")
    jitter.add_argument("--symbols", type=int, default=60000)
    jitter.add_argument("--open-loop-samples", type=int, default=200000)
    jitter.add_argument("--max-lag", type=int, default=24)
    jitter.set_defaults(handler=_cmd_jitter)

    slip = subparsers.add_parser("slip", help="cycle-slip rate estimate against loop SNR")
    add_detector_arguments(slip)
    slip.add_argument("--bandwidth", type=float, default=0.02)
    slip.add_argument("--damping", type=float, default=_DEFAULT_DAMPING)
    slip.add_argument("--snr-low", type=float, default=-2.0)
    slip.add_argument("--snr-high", type=float, default=10.0)
    slip.add_argument("--points", type=int, default=7)
    slip.add_argument("--open-loop-samples", type=int, default=100000)
    slip.set_defaults(handler=_cmd_slip)

    ppm = subparsers.add_parser("ppm", help="PPM slot clock: gain, jitter, slot errors")
    ppm.add_argument("--order", type=int, default=4, help="M, slots per symbol")
    ppm.add_argument("--delta", type=float, default=0.25, help="gate half-spacing, slots")
    ppm.add_argument("--known-slot", action="store_true", help="give the detector the true slot")
    ppm.add_argument(
        "--pulse",
        default="half-sine",
        choices=["rect", "tri", "rc-time", "half-sine", "nyq-rc"],
        help="slot pulse shape",
    )
    ppm.add_argument("--bandwidth", type=float, default=0.005, help="B_n per update")
    ppm.add_argument("--damping", type=float, default=_DEFAULT_DAMPING)
    ppm.add_argument("--snr", type=float, default=20.0)
    ppm.add_argument("--symbols", type=int, default=20000)
    ppm.add_argument("--open-loop-symbols", type=int, default=8000)
    ppm.set_defaults(handler=_cmd_ppm)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point.  Returns a process exit status."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    sys.exit(main())
