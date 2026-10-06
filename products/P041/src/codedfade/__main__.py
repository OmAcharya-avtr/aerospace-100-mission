"""Command-line interface: ``python -m codedfade <command>``.

Commands
--------
``channel``   generate a path and print its marginal and correlation diagnostics
``fade``      print sample fade statistics and the analytic level-crossing result
``depth``     the interleaver-depth sweep with its latency and memory cost
``code``      Reed-Solomon and convolutional code parameters and a round trip
``preflight`` run the hardware-abstraction-layer preflight checks
``dryrun``    rehearse a run through the simulated backend in dry-run mode

Every command writes to stdout only and prints no filesystem path.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from .channel import (
    ChannelConfig,
    gamma_gamma_parameters_from_si,
    generate_amplitude,
    measured_correlation_time,
)
from .convolutional import ConvolutionalCode
from .fade import (
    fade_statistics,
    lognormal_standard_level,
    markov_crossing_rate,
    markov_mean_fade_duration,
    required_interleaver_depth,
    rice_crossing_rate_gauss_kernel,
    rice_mean_fade_duration_gauss_kernel,
)
from .hal import ModemConfig, ModemSession, RunMode, SimulatedModemBackend, run_preflight
from .link import CodedLink, uncoded_bit_error_rate
from .reedsolomon import ReedSolomon


def _channel_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--si", type=float, default=0.6, help="scintillation index (default 0.6)")
    p.add_argument(
        "--tau", type=float, default=2.0e-4, help="correlation time in s (default 2e-4)"
    )
    p.add_argument(
        "--fs", type=float, default=1.0e6, help="symbol rate in Hz (default 1e6)"
    )
    p.add_argument(
        "--marginal", choices=("lognormal", "gammagamma"), default="lognormal"
    )
    p.add_argument("--kernel", choices=("exp", "gauss"), default="exp")
    p.add_argument("--seed", type=int, default=0)


def _config(args: argparse.Namespace) -> ChannelConfig:
    return ChannelConfig(
        scintillation_index=args.si,
        correlation_time_s=args.tau,
        sample_rate_hz=args.fs,
        marginal=args.marginal,
        kernel=args.kernel,
        seed=args.seed,
    )


def cmd_channel(args: argparse.Namespace) -> int:
    cfg = _config(args)
    a = generate_amplitude(cfg, args.samples)
    i = a * a
    print(f"marginal                {cfg.marginal}")
    print(f"kernel                  {cfg.kernel}")
    print(f"scintillation index     {cfg.scintillation_index:.6f} (target)")
    print(f"                        {np.var(i) / np.mean(i) ** 2:.6f} (sample)")
    print(f"mean irradiance         {np.mean(i):.6f} (target 1.0)")
    print(f"correlation time        {cfg.correlation_time_s:.6e} s (target)")
    print(
        f"                        {measured_correlation_time(i, cfg.sample_rate_hz):.6e} s"
        " (sample, log domain)"
    )
    print(f"Lc = tau * Rs           {cfg.samples_per_correlation_time:.3f} symbols")
    if cfg.marginal == "gammagamma":
        alpha, beta = gamma_gamma_parameters_from_si(cfg.scintillation_index)
        print(f"alpha, beta             {alpha:.6f}, {beta:.6f}")
    print(f"samples                 {args.samples}")
    return 0


def cmd_fade(args: argparse.Namespace) -> int:
    cfg = _config(args)
    a = generate_amplitude(cfg, args.samples)
    st = fade_statistics(a, args.threshold, cfg.sample_rate_hz)
    print(f"threshold amplitude     {st.threshold:.6f}")
    print(f"samples                 {st.n_samples}")
    print(f"outage fraction         {st.outage_fraction:.6f}")
    print(f"down-crossings          {st.down_crossings}")
    print(f"level-crossing rate     {st.level_crossing_rate_hz:.6f} s^-1 (sample)")
    print(f"complete fades          {st.complete_fades} (censored {st.censored_fades})")
    print(f"mean fade duration      {st.mean_fade_duration_s:.6e} s (sample)")
    print(f"median fade duration    {st.median_fade_duration_s:.6e} s (sample)")
    print(f"max fade duration       {st.max_fade_duration_s:.6e} s (sample)")
    if cfg.marginal == "lognormal":
        u = lognormal_standard_level(args.threshold, cfg.scintillation_index)
        print(f"standard level u        {u:.6f}")
        if cfg.kernel == "exp":
            print(
                "analytic LCR            "
                f"{markov_crossing_rate(u, cfg.correlation_time_s, cfg.sample_rate_hz):.6f}"
                " s^-1 (sampled Gauss-Markov, exact)"
            )
            print(
                "analytic MFD            "
                f"{markov_mean_fade_duration(u, cfg.correlation_time_s, cfg.sample_rate_hz):.6e}"
                " s"
            )
        else:
            print(
                "analytic LCR            "
                f"{rice_crossing_rate_gauss_kernel(u, cfg.correlation_time_s):.6f}"
                " s^-1 (Rice 1945, continuous time)"
            )
            print(
                "analytic MFD            "
                f"{rice_mean_fade_duration_gauss_kernel(u, cfg.correlation_time_s):.6e} s"
            )
        mfd = st.mean_fade_duration_s
        if np.isfinite(mfd):
            print(
                "depth for mean fade     "
                f"{required_interleaver_depth(mfd, cfg.sample_rate_hz)} symbols"
            )
    return 0


def cmd_depth(args: argparse.Namespace) -> int:
    cfg = _config(args)
    code = ReedSolomon(args.n, args.k, args.m)
    link = CodedLink(code, cfg, mean_snr_db=args.snr_db)
    depths = [int(d) for d in args.depths.split(",")]
    print(f"code                    RS({code.n},{code.k}) over GF(2^{code.m}), t={code.t}")
    print(f"rate                    {code.rate:.6f}")
    print(f"mean SNR                {args.snr_db:.3f} dB")
    print(f"Lc = tau * Rs           {cfg.samples_per_correlation_time:.3f} symbols")
    print(
        "uncoded BER             "
        f"{uncoded_bit_error_rate(cfg, args.snr_db, min(args.codewords * code.n, 200000)):.6e}"
    )
    print()
    header = (
        f"{'depth':>7} {'D/Lc':>8} {'raw SER':>10} {'post BER':>11} {'FER':>10} "
        f"{'FER SE':>10} {'lat ms':>10} {'mem B':>12}"
    )
    print(header)
    print("-" * len(header))
    for r in link.depth_sweep(depths, args.codewords):
        print(
            f"{r.depth:7d} {r.depth / cfg.samples_per_correlation_time:8.3f} "
            f"{r.raw_symbol_error_rate:10.6f} {r.post_bit_error_rate:11.4e} "
            f"{r.frame_error_rate:10.6f} {r.frame_error_rate_standard_error:10.6f} "
            f"{r.latency_ms:10.3f} {r.memory_bytes:12.0f}"
        )
    return 0


def cmd_code(args: argparse.Namespace) -> int:
    code = ReedSolomon(args.n, args.k, args.m)
    rng = np.random.default_rng(args.seed)
    msg = rng.integers(0, 1 << code.m, code.k)
    cw = code.encode(msg)
    received = cw.copy()
    positions = rng.choice(code.n, code.t, replace=False)
    for p in positions:
        received[p] ^= int(rng.integers(1, 1 << code.m))
    out = code.decode(received)
    print(f"RS({code.n},{code.k}) over GF(2^{code.m})")
    print(f"  t                     {code.t} symbols")
    print(f"  rate                  {code.rate:.6f}")
    print(f"  generator degree      {code.generator.size - 1}")
    print(f"  injected errors       {code.t}")
    print(f"  decode success        {out.success}")
    print(f"  symbols corrected     {out.corrected_symbols}")
    print(f"  message recovered     {bool(np.array_equal(out.message, msg))}")
    cc = ConvolutionalCode()
    bits = np.array([1, 0, 1, 1], dtype=np.uint8)
    enc = cc.encode(bits)
    dec = cc.decode(enc, bits.size)
    print(f"{cc!r}")
    print(f"  states                {cc.n_states}")
    print(f"  encode(1011)          {''.join(map(str, enc.tolist()))}")
    print(f"  viterbi recovers      {bool(np.array_equal(dec.message, bits))}")
    print(f"  min terminated weight {cc.minimum_terminated_weight(12)} bits (L=12)")
    return 0


def _modem_config(args: argparse.Namespace) -> ModemConfig:
    return ModemConfig(
        n=args.n, k=args.k, m=args.m, depth=args.depth, symbol_rate_hz=args.fs
    )


def cmd_preflight(args: argparse.Namespace) -> int:
    cfg = _config(args)
    report = run_preflight(_modem_config(args), SimulatedModemBackend(), cfg)
    for r in report.results:
        print(f"[{'PASS' if r.passed else 'FAIL'}] {r.name:24s} {r.detail}")
    print(f"\noverall: {'PASS' if report.passed else 'FAIL'}")
    return 0 if report.passed else 1


def cmd_dryrun(args: argparse.Namespace) -> int:
    cfg = _config(args)
    session = ModemSession(
        SimulatedModemBackend(), _modem_config(args), RunMode.DRY_RUN, cfg
    )
    try:
        session.start()
    except RuntimeError as exc:
        print(f"refused: {exc}")
        print(json.dumps(session.capture.to_dict(), indent=2, sort_keys=True))
        return 1
    payload = session.transfer(args.blocks, seed=args.seed)
    print(f"dry-run payload returned: {payload!r} (discarded by design)")
    capture = session.finish()
    print(json.dumps(capture.to_dict(), indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m codedfade",
        description=(
            "Coded FSO link performance over temporally correlated fading. "
            "Research-grade; not flight-qualified, not certified, not approved for "
            "operational aerospace use."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("channel", help="channel marginal and correlation diagnostics")
    _channel_args(p)
    p.add_argument("--samples", type=int, default=200000)
    p.set_defaults(func=cmd_channel)

    p = sub.add_parser("fade", help="sample fade statistics and the analytic baseline")
    _channel_args(p)
    p.add_argument("--samples", type=int, default=400000)
    p.add_argument("--threshold", type=float, default=0.6, help="amplitude threshold")
    p.set_defaults(func=cmd_fade)

    p = sub.add_parser("depth", help="interleaver-depth sweep with latency and memory cost")
    _channel_args(p)
    p.add_argument("--n", type=int, default=31)
    p.add_argument("--k", type=int, default=21)
    p.add_argument("--m", type=int, default=5)
    p.add_argument("--snr-db", type=float, default=14.0)
    p.add_argument("--codewords", type=int, default=2048)
    p.add_argument("--depths", type=str, default="1,8,64,512,4096")
    p.set_defaults(func=cmd_depth)

    p = sub.add_parser("code", help="code parameters and a round trip")
    p.add_argument("--n", type=int, default=31)
    p.add_argument("--k", type=int, default=21)
    p.add_argument("--m", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_code)

    p = sub.add_parser("preflight", help="hardware-abstraction-layer preflight checks")
    _channel_args(p)
    p.add_argument("--n", type=int, default=31)
    p.add_argument("--k", type=int, default=21)
    p.add_argument("--m", type=int, default=5)
    p.add_argument("--depth", type=int, default=256)
    p.set_defaults(func=cmd_preflight)

    p = sub.add_parser("dryrun", help="rehearse a run in dry-run mode")
    _channel_args(p)
    p.add_argument("--n", type=int, default=31)
    p.add_argument("--k", type=int, default=21)
    p.add_argument("--m", type=int, default=5)
    p.add_argument("--depth", type=int, default=256)
    p.add_argument("--blocks", type=int, default=2)
    p.set_defaults(func=cmd_dryrun)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    sys.exit(main())
