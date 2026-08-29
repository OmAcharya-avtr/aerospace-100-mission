"""Command-line interface: ``python -m waveforge``.

Subcommands
-----------
``noll``      Noll residual-variance table against the published values.
``screen``    Generate a phase screen and report its statistics.
``budget``    Analytic error budget and Strehl for a configuration.
``loop``      Run the closed loop and report residual variance and Strehl.
``predict``   Train the learned predictor and benchmark it against the
              classical integrator and the pure-delay baseline.

Every subcommand prints plain text to stdout; nothing is written to disk.
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np

from . import __version__
from .atmosphere import band_limited_structure_function, phase_screen, structure_function
from .control import noise_variance_gain, stability_limit_gain
from .datasets import make_slope_dataset
from .errorbudget import strehl_marechal
from .loop import AOConfig, AOSystem
from .predictor import LinearSlopePredictor, PureDelayPredictor
from .statistics import (
    NOLL_RESIDUAL_TABLE,
    noll_residual_variance,
    phase_structure_function,
)

__all__ = ["build_parser", "main"]


def _config_from_args(args: argparse.Namespace) -> AOConfig:
    return AOConfig(
        n_pix=args.n_pix,
        diameter_m=args.diameter,
        r0_m=args.r0,
        wind_speed_m_s=args.wind,
        frame_rate_hz=args.frame_rate,
        n_sub=args.n_sub,
        n_act=args.n_act,
        gain=args.gain,
        delay_frames=args.delay,
        photon_flux=args.flux,
        read_noise_e=args.read_noise,
        seed=args.seed,
        screen_pixels=args.screen_pixels,
    )


def _add_system_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--n-pix", type=int, default=64, help="pupil samples across (default 64)")
    parser.add_argument("--diameter", type=float, default=0.5, help="aperture diameter [m]")
    parser.add_argument("--r0", type=float, default=0.10, help="Fried parameter [m]")
    parser.add_argument("--wind", type=float, default=10.0, help="wind speed [m/s]")
    parser.add_argument("--frame-rate", type=float, default=1000.0, help="WFS frame rate [Hz]")
    parser.add_argument("--n-sub", type=int, default=8, help="subapertures across")
    parser.add_argument("--n-act", type=int, default=9, help="actuators across")
    parser.add_argument("--gain", type=float, default=0.4, help="integrator gain")
    parser.add_argument("--delay", type=int, default=2, help="total loop latency [frames]")
    parser.add_argument(
        "--flux", type=float, default=float("inf"), help="photoelectrons per subaperture per frame"
    )
    parser.add_argument("--read-noise", type=float, default=0.0, help="read noise [e- RMS/pixel]")
    parser.add_argument("--seed", type=int, default=0, help="phase-screen seed")
    parser.add_argument("--screen-pixels", type=int, default=1024, help="generating screen size")


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser (exposed for testing)."""
    parser = argparse.ArgumentParser(
        prog="python -m waveforge",
        description="WaveForge — adaptive-optics sizing, simulation and predictive control.",
    )
    parser.add_argument("--version", action="version", version=f"waveforge {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    noll = sub.add_parser("noll", help="Noll residual variance vs the published table")
    noll.add_argument("--j-max", type=int, default=21, help="highest Noll index (default 21)")
    noll.add_argument("--d-over-r0", type=float, default=1.0, help="aperture in r0 units")

    screen = sub.add_parser("screen", help="generate a phase screen and report statistics")
    screen.add_argument("--n-pix", type=int, default=256)
    screen.add_argument("--pixel-scale", type=float, default=0.01, help="sample spacing [m]")
    screen.add_argument("--r0", type=float, default=0.10)
    screen.add_argument("--subharmonics", type=int, default=6)
    screen.add_argument("--seed", type=int, default=0)

    budget = sub.add_parser("budget", help="analytic error budget for a configuration")
    _add_system_arguments(budget)

    loop = sub.add_parser("loop", help="run the closed loop")
    _add_system_arguments(loop)
    loop.add_argument("--frames", type=int, default=400)
    loop.add_argument("--warmup", type=int, default=100)

    predict = sub.add_parser("predict", help="train and benchmark the learned predictor")
    _add_system_arguments(predict)
    predict.add_argument("--frames", type=int, default=400)
    predict.add_argument("--warmup", type=int, default=100)
    predict.add_argument("--history", type=int, default=4, help="predictor input frames")
    predict.add_argument("--train-frames", type=int, default=400)
    return parser


def _cmd_noll(args: argparse.Namespace) -> int:
    print(f"Noll (1976) residual variance Delta_J, D/r0 = {args.d_over_r0:g}")
    print(f"{'J':>4} {'computed':>12} {'published':>12} {'rel. diff':>10}")
    worst = 0.0
    for j in range(1, args.j_max + 1):
        computed = noll_residual_variance(j, args.d_over_r0)
        published = NOLL_RESIDUAL_TABLE.get(j)
        if published is None:
            print(f"{j:>4} {computed:12.6f} {'-':>12} {'-':>10}")
            continue
        scaled = published * args.d_over_r0 ** (5.0 / 3.0)
        rel = (computed - scaled) / scaled
        worst = max(worst, abs(rel))
        print(f"{j:>4} {computed:12.6f} {scaled:12.6f} {rel * 100:9.3f}%")
    print(f"worst relative difference over the published range: {worst * 100:.3f}%")
    return 0


def _cmd_screen(args: argparse.Namespace) -> int:
    screen = phase_screen(
        args.n_pix, args.pixel_scale, args.r0, n_subharmonics=args.subharmonics, rng=args.seed
    )
    lags, measured = structure_function(screen, max_lag=max(1, args.n_pix // 8))
    r = lags * args.pixel_scale
    theory = phase_structure_function(r, args.r0)
    band = band_limited_structure_function(args.n_pix, args.pixel_scale, args.r0, lags)
    print(f"phase screen {args.n_pix}x{args.n_pix}, d = {args.pixel_scale:g} m, r0 = {args.r0:g} m")
    print(f"  subharmonic levels : {args.subharmonics}")
    print(f"  screen extent      : {args.n_pix * args.pixel_scale:.3f} m")
    print(f"  variance           : {np.var(screen):.4f} rad^2")
    print(
        f"{'r [m]':>10} {'D_meas':>12} {'D_bandlim':>12} {'D_theory':>12} "
        f"{'meas/band':>10} {'band/theory':>12}"
    )
    for k in range(0, len(lags), max(1, len(lags) // 8)):
        print(
            f"{r[k]:10.4f} {measured[k]:12.4f} {band[k]:12.4f} {theory[k]:12.4f} "
            f"{measured[k] / band[k]:10.3f} {band[k] / theory[k]:12.3f}"
        )
    print("  D_bandlim is the exact expectation of a Fourier screen on this grid")
    print("  (no subharmonics); the band/theory column is the method's known bias.")
    return 0


def _cmd_budget(args: argparse.Namespace) -> int:
    system = AOSystem(_config_from_args(args))
    budget = system.error_budget()
    print(f"WaveForge error budget  (D/r0 = {system.config.d_over_r0:.3f})")
    print(f"  actuator pitch     : {system.mirror.pitch_m * 1e3:.2f} mm")
    print(f"  subaperture size   : {system.sensor.subaperture_size_m * 1e3:.2f} mm")
    print(f"  valid subapertures : {system.sensor.n_valid}")
    print(f"  actuators          : {system.mirror.n_actuators}")
    print(f"  slope noise sigma  : {system.sensor.slope_noise_sigma():.4g} rad/m")
    print(f"  stability limit    : gain < {stability_limit_gain(system.config.delay_frames):.4f}")
    print(f"  noise variance gain: {noise_variance_gain(args.gain, args.delay):.4f}")
    for key, value in budget.as_dict().items():
        print(f"  {key:<20}: {value:.6g}")
    print(f"  dominant term      : {budget.dominant_term()}")
    return 0


def _cmd_loop(args: argparse.Namespace) -> int:
    system = AOSystem(_config_from_args(args))
    result = system.run(args.frames, warmup_frames=args.warmup)
    print(f"closed loop: {args.frames} frames, gain {args.gain}, latency {args.delay} frames")
    print(f"  open-loop variance : {result.mean_open_loop_variance:.4f} rad^2")
    print(f"  residual variance  : {result.mean_residual_variance:.4f} rad^2")
    print(f"  rejection          : {result.rejection_db:.2f} dB")
    print(f"  Strehl (numerical) : {result.mean_strehl:.4f}")
    print(f"  Strehl (Marechal)  : {float(strehl_marechal(result.mean_residual_variance)):.4f}")
    print(f"  max saturation     : {result.max_saturated_fraction:.3f}")
    print(f"  diverged           : {result.diverged}")
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    sigma = AOSystem(config).sensor.slope_noise_sigma()
    data = make_slope_dataset(
        replace(config, seed=0), n_frames=args.train_frames, noise_sigma=sigma
    )
    predictor = LinearSlopePredictor(n_history=args.history, horizon=config.delay_frames)
    predictor.fit(data.train)
    system = AOSystem(config)
    classical = system.run(args.frames, warmup_frames=args.warmup)
    delayed = system.run(args.frames, warmup_frames=args.warmup, predictor=PureDelayPredictor())
    learned = system.run(args.frames, warmup_frames=args.warmup, predictor=predictor)
    print(f"predictive control benchmark ({args.frames} frames, latency {args.delay})")
    print(f"  training seeds     : {list(data.train_seeds)} ({data.n_train_frames} frames)")
    print(f"  training noise     : {data.noise_sigma:.4g} rad/m")
    print(f"  ridge alpha chosen : {predictor.chosen_alpha:g}")
    print(f"{'controller':<22}{'residual [rad^2]':>18}{'Strehl':>10}")
    for name, result in (
        ("classical integrator", classical),
        ("pure-delay POL", delayed),
        ("learned predictor", learned),
    ):
        print(f"{name:<22}{result.mean_residual_variance:>18.4f}{result.mean_strehl:>10.4f}")
    ratio = learned.mean_residual_variance / classical.mean_residual_variance
    verdict = "learned predictor wins" if ratio < 1.0 else "classical integrator wins"
    print(f"  learned / classical residual variance ratio: {ratio:.3f}  -> {verdict}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point.  Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "noll": _cmd_noll,
        "screen": _cmd_screen,
        "budget": _cmd_budget,
        "loop": _cmd_loop,
        "predict": _cmd_predict,
    }
    try:
        return handlers[args.command](args)
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
        return 2  # pragma: no cover - parser.error raises SystemExit
